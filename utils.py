"""
Utils Module
Provides utility functions for file system operations, logging (TensorBoard and plain-text logging),
and Chamfer Distance metric calculation wrappers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from chamfer_distance import ChamferDistance
from logger import get_logger

log = get_logger(__name__)


def make_dir(dir_path):
    """
    Creates a directory if it does not already exist.
    
    Args:
        dir_path (str): Path to the directory to be created.
    """
    if not os.path.exists(dir_path):
        log.debug("Creating directory: %s", dir_path)
        os.makedirs(dir_path)
    else:
        log.debug("Directory already exists: %s", dir_path)

def prepare_logger(params):
    """
    Prepares the output logs directories, standard log file, and TensorBoard writers for training visualization.
    
    Args:
        params (argparse.Namespace): Parameters containing logging and configuration variables:
                                     - log_dir: Base directory for log outputs.
                                     - exp_name: Name of current experiment.
                                     - model_type: Type of model (e.g., autoencoder).
                                     
    Returns:
        epochs_dir (str): Path to directory where model checkpoints per epoch are saved.
        log_fd (file object): File descriptor for plain text logger file, opened in append mode.
        train_writer (SummaryWriter): Tensorboard writer for training runs.
        val_writer (SummaryWriter): Tensorboard writer for validation/test runs.
    """
    # Create the root logs directory (usually 'checkpoints/')
    make_dir(params.log_dir)
    
    # Create the experiment specific directory (e.g., 'checkpoints/exp/')
    make_dir(os.path.join(params.log_dir, params.exp_name))

    # Path to logs specific to the model type under the experiment (e.g., 'checkpoints/exp/autoencoder/')
    logger_path = os.path.join(params.log_dir, params.exp_name, params.model_type)
    
    # Path to epoch checkpoints (e.g., 'checkpoints/exp/autoencoder/epochs/')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, params.model_type, 'epochs')
    make_dir(logger_path)
    make_dir(epochs_dir)

    # File path for the main text log file
    logger_file = os.path.join(params.log_dir, params.exp_name, params.model_type, 'logger.log')
    log_fd = open(logger_file, 'a')
    log.info("Experiment log file: %s", logger_file)

    # Initialize TensorBoard SummaryWriters for training and validation metrics
    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))
    log.info("TensorBoard writers initialised at: %s", logger_path)

    return epochs_dir, log_fd, train_writer, val_writer


# Instantiate the global ChamferDistance loss function calculator
CD = ChamferDistance()


def cd_loss_L1(pcs1, pcs2):
    """
    Calculates L1 Chamfer Distance between two batch point clouds.
    L1 Chamfer Distance computes the average Euclidean distance (using square root) of each point
    in pcs1 to its closest point in pcs2, and vice-versa.
    
    Args:
        pcs1 (torch.Tensor): First batch of point clouds of shape (B, N, 3).
        pcs2 (torch.Tensor): Second batch of point clouds of shape (B, M, 3).
        
    Returns:
        loss (torch.Tensor): Scalar tensor representing the symmetric L1 Chamfer Distance.
    """
    # CD returns squared distances of shape (B, N) for pcs1 to pcs2, and (B, M) for pcs2 to pcs1
    # along with the nearest neighbor indices (which are ignored here)
    dist1, dist2, _, _ = CD(pcs1, pcs2)
    
    # Convert squared distance to Euclidean distance (L1 norm)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    
    # Return symmetric mean distance: (mean(pcs1 -> pcs2) + mean(pcs2 -> pcs1)) / 2.0
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0