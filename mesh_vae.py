"""
Mesh VAE Module
This module implements a Variational Autoencoder (VAE) designed to operate on 3D point cloud data.
It maps the high-dimensional point clouds to a continuous latent space representation and generates
reconstructions or new samples from the latent space.
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for 3D point cloud data.
    
    Attributes:
        args: Argument configuration object containing hyperparameter settings.
        latent_dim (int): Dimensionality of the latent bottleneck space.
        num_points (int): The number of points in the input point cloud.
        conv1, conv2, conv3 (nn.Module): 1D Convolutions mapping point features.
        fc1_m, fc3_m (nn.Module): Fully connected layers mapping to mean vector (mu).
        fc1_v, fc3_v (nn.Module): Fully connected layers mapping to log-variance vector (logvar).
        fc1_z, fc2_z, fc3_z (nn.Module): Fully connected layers for decoding latent vectors back to point coordinates.
    """
    def __init__(self, args):
        super(VAE, self).__init__()

        self.args = args
        self.latent_dim = args.latent_dim
        input_dim = 3  # (x, y, z) coordinates for each point
        self.num_points = args.num_points
        
        # Encoder layers: feature extraction on points using Shared MLP (implemented via 1D convolutions with kernel size 1)
        # Input shape: (B, 3, N) -> Output shape: (B, 32, N)
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        # Input shape: (B, 32, N) -> Output shape: (B, 128, N)
        self.conv2 = nn.Conv1d(32, 128, 1)
        # Input shape: (B, 128, N) -> Output shape: (B, 1, N)
        self.conv3 = nn.Conv1d(128, 1, 1)
    
        # Fully connected layers to estimate the mean (mu) of the latent distribution
        self.fc1_m = nn.Linear(self.num_points, 64)
        self.fc3_m = nn.Linear(64, self.latent_dim)
        
        # Fully connected layers to estimate the log-variance (logvar) of the latent distribution
        self.fc1_v = nn.Linear(self.num_points, 64)
        self.fc3_v = nn.Linear(64, self.latent_dim)
        
        # Decoder layers: mapping latent vector z back to reconstructed coordinates
        self.fc1_z = nn.Linear(self.latent_dim, 128)
        self.fc2_z = nn.Linear(128, 256)
        # Output is of shape (B, num_points * 3) representing flattened x,y,z coordinates
        self.fc3_z = nn.Linear(256, self.num_points * 3)

    def encoder(self, x):
        """
        Encodes the input point cloud into the parameters of the latent Gaussian distribution.
        
        Args:
            x (torch.Tensor): Input point cloud tensor of shape (B, N, 3) where B is batch size,
                              N is number of points, and 3 is coordinate dimension (x, y, z).
                              
        Returns:
            m (torch.Tensor): Estimated mean (mu) vector of shape (B, latent_dim).
            v (torch.Tensor): Estimated log-variance (logvar) vector of shape (B, latent_dim).
        """
        # Transpose input for nn.Conv1d from (B, N, 3) to (B, 3, N)
        x = x.transpose(1, 2)
        
        # Apply Shared MLP layers using 1D Convolutions with LeakyReLU activations
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))  # Shape becomes (B, 1, N)
        
        # Flatten features from (B, 1, N) to (B, N) for fully connected layers
        x = x.view(-1, self.num_points)
        
        # Compute latent mean (mu)
        m = F.leaky_relu(self.fc1_m(x))
        m = self.fc3_m(m)
        
        # Compute latent log-variance (logvar)
        v = F.leaky_relu(self.fc1_v(x))
        v = self.fc3_v(v)

        return m, v

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick: samples from the latent distribution N(mu, var) in a differentiable way
        by introducing a random standard normal variable epsilon.
        
        Args:
            mu (torch.Tensor): Mean vector of shape (B, latent_dim).
            logvar (torch.Tensor): Log-variance vector of shape (B, latent_dim).
            
        Returns:
            z (torch.Tensor): Sampled latent vector of shape (B, latent_dim).
        """
        # standard deviation = exp(0.5 * log(variance))
        std = torch.exp(0.5 * logvar)
        
        # Sample random noise epsilon from standard normal distribution N(0, I)
        eps = torch.randn_like(std)
        
        # Reparameterized sample: z = mu + epsilon * std
        return mu + eps * std

    def decoder(self, z):
        """
        Decodes a latent vector z into the reconstructed point cloud representation.
        
        Args:
            z (torch.Tensor): Latent vector of shape (B, latent_dim).
            
        Returns:
            z_out (torch.Tensor): Reconstructed coordinates of shape (B, num_points * 3).
        """
        # Feed-forward decoder layers with LeakyReLU activations
        z = F.leaky_relu(self.fc1_z(z))
        z = F.leaky_relu(self.fc2_z(z))
        z = self.fc3_z(z)  # Linear output layer (no activation)

        return z

    def sample(self, samples_size):
        """
        Generates novel point cloud samples by sampling latent vectors from standard normal distribution
        and passing them through the decoder.
        
        Args:
            samples_size (int): Number of new shapes/samples to generate.
            
        Returns:
            samples (torch.Tensor): Generated point clouds of shape (samples_size, N, 3).
        """
        # Sample latent vectors z ~ N(0, I)
        z = torch.randn(samples_size, self.latent_dim).double()
        z = z.to(self.args.device)

        # Decode latent vectors to shape coordinates and reshape from (B, N*3) to (B, 3, N)
        samples = self.decoder(z).view(-1, 3, self.num_points)
        
        # Permute shape back from (B, 3, N) to (B, N, 3)
        samples = samples.permute(0, 2, 1)
        return samples

    def forward(self, x):
        """
        Forward pass of the VAE: Encoder -> Reparameterize -> Decoder.
        
        Args:
            x (torch.Tensor): Input point clouds of shape (B, N, 3).
            
        Returns:
            mu (torch.Tensor): Mean vector of shape (B, latent_dim).
            logvar (torch.Tensor): Log-variance vector of shape (B, latent_dim).
            z (torch.Tensor): Latent representation vector of shape (B, latent_dim).
            x_recon (torch.Tensor): Reconstructed point clouds of shape (B, N, 3).
        """
        # Dynamically set the number of points based on current input shape
        self.num_points = x.shape[1]
        
        # 1. Encode input to obtain parameters of latent distribution
        mu, logvar = self.encoder(x)
        
        # 2. Sample latent representation using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode latent representation back to 3D point coordinates
        x_recon = self.decoder(z).view(-1, 3, self.num_points)
        
        # 4. Permute reconstruction shape from (B, 3, N) to (B, N, 3)
        x_recon = x_recon.permute(0, 2, 1)
        
        return mu, logvar, z, x_recon