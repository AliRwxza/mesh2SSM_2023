"""
Train Geodesic Module
Training script for the combined Mesh2SSM neural shape correspondence framework.
Implements:
1. Dynamic directory initialization and code backup.
2. Alternating optimization schedule (Autoencoder on even epochs, VAE on odd epochs).
3. Dynamic template update mechanism utilizing shape VAE samples.
4. Experiment checkpoints and text-based epoch progress tracking.
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from data import Meshes, MeshesWithFaces
from mesh_vae import VAE
from model_autoencoder import Mesh2SSM_AE
import numpy as np
from torch.utils.data import DataLoader
from utils import prepare_logger, cd_loss_L1
import sklearn.metrics as metrics
from chamfer_distance import ChamferDistance


def _init_():
    """
    Initializes checkpoint directories for the active experiment and copies 
    critical Python scripts as backups to preserve model history.
    """
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
        
    # Create backups of code configuration files
    os.system('cp train_geodesic.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model_autoencoder.py checkpoints' + '/' + args.exp_name + '/' + 'model_ae.py.backup')
    os.system('cp utils.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args):
    """
    Trains the combined Mesh2SSM_AE and VAE networks.
    
    Optimizers alternate:
    - Even epochs (epoch%2 == 0): Train the Mesh2SSM Autoencoder using Chamfer Distance and reconstruction loss.
    - Odd epochs (epoch%2 == 1): Train the VAE bottleneck using MSE and KL-Divergence loss.
    
    Args:
        args: Namespace configurations containing epochs, batch sizes, lr, and data folders.
    """
    # Initialize logger output directories and TensorBoard summary writers
    epochs_dir, log_fd, train_writer, val_writer = prepare_logger(args)

    # Load and scale input training dataset meshes
    training_data = MeshesWithFaces(directory = args.data_directory, extention=args.extention, k=args.k)
    args.scale = training_data.scale
    print(f"Training data scale: {training_data.scale}")
    
    train_loader = DataLoader(training_data, num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
                              
    # Load and scale validation dataset meshes
    test_data = MeshesWithFaces(directory = args.data_directory, extention=args.extention, partition ='val', k=args.k)
    args.test_scale = test_data.scale
    test_loader = DataLoader(test_data, num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Detect CUDA GPU device support
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    
    # Initialize autoencoder model and read reference template points
    model = Mesh2SSM_AE(args)
    print("Model Mesh2SSM_AE initialized successfully.")
    args.num_points = model.set_template(args)
    
    # Initialize shape VAE model
    model_vae = VAE(args).double().to(device)
    num_steps = int(len(training_data) / args.batch_size)
    print(f"Number of training steps per epoch: {num_steps}")
    print(f"Model VAE: {model_vae}")
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    # Choose optimization algorithms (Adam vs SGD)
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        opt_vae = optim.SGD(model_vae.parameters(), lr=args.vae_lr, momentum=args.momentum, weight_decay=1e-4)    
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        opt_vae = optim.Adam(model_vae.parameters(), lr=args.vae_lr, betas=(0.9, 0.999))
        
    # Learning rate step decay schedulers (decaying rate by 0.1 every 200 epochs)
    scheduler_vae = StepLR(opt_vae, step_size=200, gamma=0.1)
    scheduler = StepLR(opt, step_size=200, gamma=0.1)

    criterion = ChamferDistance()

    best_test_dist = 10e5
    step = 0
    val_step = 0
    
    # Epoch thresholds for scheduling operations
    ae_burnin = 100                 # VAE is only trained after this epoch count to allow autoencoder initialization
    template_update_interval = 200  # Number of epochs between template reconstruction refreshes
    first_update_epoch = 400        # First epoch template updates are triggered

    for epoch in range(args.epochs * 2):
        
        # -----------------------------------------------------------------------
        # Phase 1: Model Training
        # -----------------------------------------------------------------------
        train_loss = 0.0
        count = 0.0
        
        # Hyperparameter alpha schedule: weights L1 Chamfer loss relative to iterations completed
        if step < int((30 * args.epochs * num_steps) / 100):
            alpha = 0.01
        elif step < int((50 * args.epochs * num_steps) / 100):
            alpha = 0.1
        elif step < int((80 * args.epochs * num_steps) / 100):
            alpha = 0.5
        else:
            alpha = 1.0
        
        for data, idx, label, _ in train_loader:
            data, idx, label = data.to(device), idx.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)  # shape: (B, 3, N) for EdgeConv
            batch_size = data.size()[0]
            
            # Forward pass: predict correspondences and coarse reconstructions
            particles, reconstruction = model(data, idx)
            
            # --- VAE Training Epochs (Odd Epochs after Burn-In) ---
            if (epoch % 2 == 1):
                if (epoch > ae_burnin):
                    model.eval()         # Freeze autoencoder parameters
                    model_vae.train()    # Unfreeze VAE parameters
                    opt_vae.zero_grad()
                    
                    # Forward pass through VAE on predicted correspondence particles
                    mu, logvar, z, x_recon = model_vae(particles)
                    
                    # Compute Kullback-Leibler Divergence (KLD) loss
                    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
                    # Compute Mean Squared Error (MSE) reconstruction loss
                    MSE = (torch.sum(torch.sum(torch.sum((particles - x_recon) ** 2, axis=-1), axis=-1))) / batch_size
                    
                    loss = args.vae_mse_weight * MSE + KLD
                    loss.backward()
                    opt_vae.step()
                    scheduler_vae.step()
                    print(f'Epoch: {epoch}  VAE Training Loss: MSE: {MSE.detach().item()}, KLD: {KLD.detach().item()}', flush=True)
            
            # --- Autoencoder Training Epochs (Even Epochs) ---
            if (epoch % 2 == 0):
                model.train()         # Unfreeze autoencoder parameters
                model_vae.eval()         # Freeze VAE parameters
                opt.zero_grad()
                
                # Chamfer Distance between target labels and correspondence outputs
                dist1, dist2, _, _ = criterion(label, particles)
                chamfer_loss = 0.5 * (dist1.mean() + dist2.mean())
                
                # Reconstruction MSE loss comparing original mesh input and DGCNN reconstruction
                mse_loss = F.mse_loss(data.permute(0, 2, 1), reconstruction)
                
                loss1 = chamfer_loss + args.mse_weight * mse_loss
                
                # Combine standard losses with scheduled L1 Chamfer loss (guided by alpha)
                loss = loss1 + alpha * cd_loss_L1(label, particles)
                loss.backward()
                opt.step()         
                train_loss += loss.detach().item() 
                scheduler.step()
                
                # Log metrics to Tensorboard
                train_writer.add_scalar('training loss', loss.detach().item(), step)
                step += 1
                
                print(f'Epoch: {epoch}  Training Loss: {train_loss}')

        # -----------------------------------------------------------------------
        # Phase 2: Template Updates (Re-synthesize template from VAE samples)
        # -----------------------------------------------------------------------
        if (epoch % template_update_interval == 0 and epoch >= first_update_epoch):
            model_vae.eval()
            model.eval()
            with torch.no_grad():
                # Sample novel points from latent space Gaussian distribution
                samples = model_vae.sample(200)
                samples = samples.detach().cpu().numpy()
                
            # Save a subset of samples to outputs directory
            for sid in range(10):
                np.savetxt(args.pred_dir + "sample_" + str(sid) + ".particles", samples[sid] * args.scale)
           
            # Calculate the new mean shape coordinates across VAE samples to define updated template
            template = np.mean(samples, axis=0)
            template = np.reshape(template, (args.num_points, 3))
            np.savetxt(args.pred_dir + "learned_template_" + str(epoch) + ".particles", template * args.scale)
            
            print("Setting Template to new learned average.")
            model.set_template(args, array=template)
            
        # -----------------------------------------------------------------------
        # Phase 3: Validation Checkpoints
        # -----------------------------------------------------------------------
        if (epoch % 10 == 0):
            test_loss = 0.0
            count = 0.0
            model.eval()
            model_vae.eval()
            
            # Iterate through validation loader
            for data, idx, label, names in test_loader:
                data, idx, label = data.to(device), idx.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                
                with torch.no_grad():
                    # Evaluate outputs on the validation mesh set
                    particles, reconstruction = model(data)
                    mu, logvar, z, x_recon = model_vae(particles) 
                    
                dist1, dist2, idx1, idx2 = criterion(label, particles)
                loss = 0.5 * (dist1.mean() + dist2.mean())
                
                val_writer.add_scalar('test chamfer loss', loss, val_step)
                val_step += 1
                test_loss += loss.detach().item()
                
                # Save generated validations meshes to disk for offline evaluation
                for i in range(len(particles)):
					# Rescale normalized points back to original physical coordinates
                    p = particles[i].detach().cpu().numpy() * args.test_scale
                    r = reconstruction[i].detach().cpu().numpy() * args.test_scale
                    n = names[i].split(args.extention)[0] + ".particles"
                    d = data[i].permute(1, 0).detach().cpu().numpy() * args.test_scale
                    vae_r = x_recon[i].detach().cpu().numpy() * args.test_scale

                    np.savetxt(args.pred_dir + n, p)
                    np.savetxt(args.recon_dir + n, r)
                    np.savetxt(args.recon_dir + "og" + n, d)
                    np.savetxt(args.pred_dir + "vae_" + n, vae_r)
                    
            # Checkpoint updates: save best model parameter weights if validation loss improves
            if test_loss <= best_test_dist:
                best_test_dist = test_loss
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
                torch.save(model_vae.state_dict(), 'checkpoints/%s/models/model_vae.t7' % args.exp_name)
                template = model.get_template()
                np.savetxt('checkpoints/%s/models/best_template.txt' % args.exp_name, template)
        
        # Log successful completion of training epoch to checkpoints folder
        print(f"Finished Epoch {epoch} successfully.", flush=True)

        progress_file = os.path.join('checkpoints', args.exp_name, 'last_successful_epoch.txt')
        with open(progress_file, 'w') as f:
            f.write(f"Code ran successfully up to the end of epoch: {epoch}\n")

    # Save final model state checkpoints
    torch.save(model.state_dict(), 'checkpoints/%s/models/model_last.t7' % args.exp_name)
    torch.save(model_vae.state_dict(), 'checkpoints/%s/models/model_vae_last.t7' % args.exp_name)
    template = model.get_template()
    np.savetxt('checkpoints/%s/models/final_template.txt' % args.exp_name, template)


if __name__ == "__main__":
    
    # Initialize command line parser for model configurations
    parser = argparse.ArgumentParser(description='Mesh2SSM: From surface meshes to statistical shape models of anatomy')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N')
    parser.add_argument('--batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--vae_lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=128, metavar='N',
                        help='Dimension of embeddings of the mesh autoencoder for correspondence generation')
    parser.add_argument('--nf', type=int, default=8, metavar='N',
                        help='Dimension of IMnet nf')
    parser.add_argument('--k', type=int, default=10, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data_directory', type=str,
                        help="data directory")
    parser.add_argument('--model_type', type=str, default = 'autoencoder',
						help="model type autoencoder or only encoder")
    parser.add_argument('--mse_weight', type=float, default=0.01, 
                        help="weight for the mesh autoencoder(correspondence generation) mse reconstruction term in the loss")
    parser.add_argument('--template', type=str, default = "template",
                        help="name of the template file")
    parser.add_argument('--extention', type=str, default=".ply",
                        help="extention of the mesh files in the data directory")
    parser.add_argument('--gpuid', type=int, default=0,
                        help="gpuid on which the code should be run")
    parser.add_argument('--vae_mse_weight', type=float, default=10,
                        help="weight for the shape variational autoencoder(analysis) mse reconstruction term in the loss")
    parser.add_argument('--latent_dim', type = int, default = 64,
                        help="latent dimensions of the shape variational autoencoder")
    args = parser.parse_args()

    # Create logs/checkpoints subdirectories
    _init_()

    args.pred_dir = 'checkpoints/' + args.exp_name + "/output/"
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)
    args.recon_dir = 'checkpoints/' + args.exp_name + "/recon/"
    if not os.path.exists(args.recon_dir):
        os.makedirs(args.recon_dir)
    args.log_dir = 'checkpoints/' 
    
    # Device setups
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuid)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    # Execute training loop
    train(args)


'''
This repo reuses code from: https://github.com/WangYueFt/dgcnn/
'''