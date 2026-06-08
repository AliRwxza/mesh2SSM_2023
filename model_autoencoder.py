"""
Model Autoencoder Module
Implements the core neural architectures of Mesh2SSM:
1. ImNet: An implicit decoder network mapping (template points, shape features) to reconstructed coordinates.
2. DGCNN_AE: Dynamic Graph CNN Autoencoder performing graph-based edge convolutions to extract global latent vectors.
3. Mesh2SSM_AE: The parent autoencoder combining DGCNN encoder and ImNet decoder to establish correspondences relative to a template mesh.
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dictionary mapping activation names to PyTorch activation modules
NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "leakyrelu": nn.LeakyReLU(negative_slope=0.2),
}


class ImNet(nn.Module):
    """
    ImNet: Implicit Representation Network Decoder.
    Maps spatial coordinates of a template shape concatenated with a latent feature representation 
    to output reconstructed 3D points. Contains skip connections to preserve input feature signals.
    """

    def __init__(
        self,
        dim=3,
        in_features=32,
        out_features=3,
        nf=64,
        nonlinearity="tanh",
        device=None
    ):
        """
        Args:
            dim (int): Spatial dimensionality of template coordinate inputs (usually 3 for x, y, z).
            in_features (int): Dimensionality of input latent shape code z (i.e. embedding dims).
            out_features (int): Dimensionality of the predicted points (typically 3).
            nf (int): Network width multiplier scaling channels in fully connected layers.
            nonlinearity (str): Activation function key.
            device (torch.device): Device placement mapping (CPU or GPU).
        """
        super(ImNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        # Combined feature dimensionality: coordinates + latent code size
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.activ = NONLINEARITIES[nonlinearity]
        
        # Dense linear layers with input concatenations (skip connections)
        # Layers concat the output of the prior layer with the original input point features
        self.fc0 = nn.Linear(self.dimz, nf * 16)
        self.fc1 = nn.Linear(nf * 16 + self.dimz, nf * 8)
        self.fc2 = nn.Linear(nf * 8 + self.dimz, nf * 4)
        self.fc3 = nn.Linear(nf * 4 + self.dimz, nf * 2)
        self.fc4 = nn.Linear(nf * 2 + self.dimz, nf * 1)
        self.fc5 = nn.Linear(nf * 1, out_features)

        self.device = device

    def forward(self, z, template):
        """
        Forward pass decoding shape features through the template vertices.
        
        Args:
            z (torch.Tensor): Latent shape embedding features of shape (B, in_features).
            template (np.ndarray): Reference template vertex coordinates of shape (T, 3).
            
        Returns:
            out (torch.Tensor): Reconstructed coordinates for the template points of shape (B, T, 3).
        """
        batch_size = len(z)
        
        # 1. Repeat template array across the batch: shape becomes (B, T, 3)
        template_batch = np.repeat(template[np.newaxis, :, :], batch_size, axis=0)
        template_batch = torch.from_numpy(template_batch).to(self.device)
        
        # 2. Reshape and tile latent code z for each template point: (B, 1, in_features) -> (B, T, in_features)
        zs = z.view(-1, 1, self.in_features).repeat(1, template_batch.shape[1], 1)
        
        # 3. Concatenate template coordinates and shape features: shape (B, T, dim + in_features)
        pointz = torch.cat([template_batch, zs], 2)
        x_tmp = pointz.double()
        
        # 4. Dense layers with residual concatenations of original coordinates + latent features (pointz)
        x_tmp = self.activ(self.fc0(x_tmp))
        x_tmp = torch.cat([x_tmp, pointz], dim=-1)

        x_tmp = self.activ(self.fc1(x_tmp))
        x_tmp = torch.cat([x_tmp, pointz], dim=-1)

        x_tmp = self.activ(self.fc2(x_tmp))
        x_tmp = torch.cat([x_tmp, pointz], dim=-1)

        x_tmp = self.activ(self.fc3(x_tmp))
        x_tmp = torch.cat([x_tmp, pointz], dim=-1)

        x_tmp = self.activ(self.fc4(x_tmp))
        x_tmp = self.fc5(x_tmp)  # Final regression layer mapping to 3D offsets/positions (B, T, 3)
        
        return x_tmp


def knn(x, k):
	"""
	Calculates the indices of the k nearest neighbors for each point in a batch.
	
	Args:
	    x (torch.Tensor): Coordinates tensor of shape (B, 3, N).
	    k (int): Number of nearest neighbors to retrieve.
	    
	Returns:
	    idx (torch.Tensor): Neighborhood indices of shape (B, N, k).
	"""
	# Pairwise distance computation: ||a - b||^2 = ||a||^2 - 2<a,b> + ||b||^2
	inner = -2 * torch.matmul(x.transpose(2, 1), x)
	xx = torch.sum(x**2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
	# Retrieve indices of the top-k closest points (largest negative distances)
	idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
	return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Constructs local neighborhood graph features using edge convolutions.
    For each vertex x_i, extracts differences (x_j - x_i) for its k-nearest neighbors x_j,
    and concatenates with the center feature x_i.
    
    Args:
        x (torch.Tensor): Point coordinates/features tensor of shape (B, C, N).
        k (int): Number of nearest neighbors.
        idx (torch.Tensor, optional): Precomputed neighborhood indices of shape (B, N, k).
        
    Returns:
        feature (torch.Tensor): Local edge features of shape (B, 2*C, N, k).
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    # 1. Compute kNN indices if not pre-provided
    if idx is None:
        idx = knn(x, k=k)   # shape: (B, N, k)
    device = torch.device('cuda')

    # 2. Add batch offset index to flat indices list for broadcasting
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)  # Flattened indices for 1D selection
 
    _, num_dims, _ = x.size()

    # 3. Permute input coordinates to (B, N, C) and reshape to (B * N, C) for indexing
    x = x.transpose(2, 1).contiguous()   
    
    # 4. Gather neighbor features: shape becomes (B, N, k, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    # 5. Tile/repeat center point features across neighbors: (B, N, 1, C) -> (B, N, k, C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # 6. Concatenate local differences (feature - x) and center coordinate features (x)
    # Shape transitions: (B, N, k, 2*C) -> (B, 2*C, N, k)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN_AE(nn.Module):
    """
    DGCNN Autoencoder Encoder Network.
    Uses Dynamic Graph CNN layers with Edge Convolutions to extract local and global geometric shape features.
    """

    def __init__(self, args, output_channels=40):
        """
        Args:
            args: Configuration containing hyperparameters like emb_dims, k, and dropout.
            output_channels (int): Ignored parameter maintained for configuration compatibility.
        """
        super(DGCNN_AE, self).__init__()
        self.args = args
        self.k = args.k
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        # Convolutional layers building up feature hierarchies
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(args.emb_dims + (64 * 3), 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 3, kernel_size=1, bias=False)
        

    def forward(self, x, idx=None):
        """
        Forward pass encoding the point cloud to global features and coarse point-wise reconstructions.
        
        Args:
            x (torch.Tensor): Coordinates of shape (B, 3, N).
            idx (torch.Tensor, optional): Precomputed neighborhood graph indices.
            
        Returns:
            feature (torch.Tensor): Global latent shape embedding of shape (B, emb_dims).
            recon (torch.Tensor): Coarse reconstruction coordinates of shape (B, N, 3).
        """
        batch_size = x.size(0)
        num_points = x.size(2)

        # 1. First EdgeConv block
        # (B, 3, N) -> graph features (B, 6, N, k) -> conv outputs (B, 64, N, k) -> max pool over neighbors -> (B, 64, N)
        x = get_graph_feature(x, k=self.k, idx=idx)      
        x = self.conv1(x)                       
        x = self.conv2(x)                       
        x1 = x.max(dim=-1, keepdim=False)[0]    

        # 2. Second EdgeConv block
        # (B, 64, N) -> graph features (B, 128, N, k) -> conv outputs (B, 64, N, k) -> max pool over neighbors -> (B, 64, N)
        x = get_graph_feature(x1, k=self.k)     
        x = self.conv3(x)                       
        x = self.conv4(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]    

        # 3. Third EdgeConv block
        # (B, 64, N) -> graph features (B, 128, N, k) -> conv outputs (B, 64, N, k) -> max pool over neighbors -> (B, 64, N)
        x = get_graph_feature(x2, k=self.k)     
        x = self.conv5(x)                       
        x3 = x.max(dim=-1, keepdim=False)[0]    

        # 4. Concatenate hierarchical features from all EdgeConv outputs: shape becomes (B, 192, N)
        x = torch.cat((x1, x2, x3), dim=1)      

        # 5. Extract intermediate latent representations: shape (B, emb_dims, N)
        x = self.conv6(x)                       
        
        # 6. Global shape feature extraction using adaptive average pooling: shape (B, emb_dims, 1)
        feature = F.adaptive_avg_pool1d(x, 1)
        
        # 7. Coarse Decoder: replicate global features across all points and concatenate with EdgeConv outputs
        # shape transitions: (B, emb_dims, 1) -> (B, emb_dims, N) -> concatenated: (B, emb_dims + 192, N)
        x = feature.repeat(1, 1, num_points)          
        x = torch.cat((x, x1, x2, x3), dim=1)   

        # 8. Decode to coarse reconstruction coordinates (B, 3, N)
        x = self.conv7(x)                       
        x = self.conv8(x)                       
        x = self.dp1(x)
        x = self.conv9(x)                       
        
        # Permute coordinates to standard shape: (B, N, 3)
        x = x.permute(0, 2, 1)
        
        # Return flattened global latent representation and the coarse reconstruction coordinates
        return feature.view(batch_size, -1), x
   

class Mesh2SSM_AE(nn.Module):
    """
    Combined Mesh2SSM Autoencoder model.
    Encodes surface meshes into global shape feature space and decodes correspondences 
    utilizing a parameterized template model.
    """

    def __init__(self, args):
        super(Mesh2SSM_AE, self).__init__()

        self.args = args
        # Encoder: DGCNN
        self.dgcnn = DGCNN_AE(args).to(args.device).double()
        # Decoder: Implicit mapping relative to template mesh
        self.imnet = ImNet(in_features=args.emb_dims, nf=args.nf, device=args.device).to(args.device).double()

    def set_template(self, args, array=None):
        """
        Configures the reference template points for ImNet decoder mapping.
        
        Args:
            args: Namespace containing template parameters.
            array (np.ndarray, optional): Custom template coordinate array.
            
        Returns:
            num_template_points (int): The number of vertices in the template.
        """
        if array is None:
            # Load template from file system path
            self.template_dir = os.path.join(args.data_directory)
            self.template = np.loadtxt(self.template_dir + args.template + ".particles") / args.scale
        else:
            # Register explicit array
            self.template = array
            
        return self.template.shape[0]

    def get_template(self):
        """
        Retrieves the registered template coordinate vertices.
        """
        return self.template

    def forward(self, x, idx=None):
        """
        Forward pass processing.
        
        Args:
            x (torch.Tensor): Coordinates of shape (B, 3, N).
            idx (torch.Tensor, optional): Precomputed graph neighbor indices.
            
        Returns:
            out (torch.Tensor): Dense correspondence mesh coordinates mapped to template of shape (B, T, 3).
            reconstruction (torch.Tensor): Coarse coordinates reconstructed by DGCNN of shape (B, N, 3).
        """
        # 1. Encode point features using DGCNN encoder
        features, reconstruction = self.dgcnn(x, idx)
        
        # 2. Decode features into structured template matching correspondences
        out = self.imnet(features, self.template)
        
        return out, reconstruction