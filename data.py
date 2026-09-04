"""
Data Module
Handles loading, augmentation, and preprocessing of point cloud data and 3D meshes.
Supports ModelNet40 point cloud datasets and local ply mesh datasets.
Precomputes geodesic neighborhood matrices using PyTorch Geometric to guide shape correspondences.
"""

import os
import sys
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
import pyvista as pv
from torch_geometric.utils import geodesic_distance
import torch
from logger import get_logger

log = get_logger(__name__)

def download():
	"""
	Downloads the ModelNet40 classification dataset zip file if it does not already exist
	and extracts it into the local 'data' directory.
	"""
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	# Check if dataset directory exists
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		# Download and extract dataset
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))


def load_data(partition):
	"""
	Loads ModelNet40 training or testing h5 files.
	
	Args:
	    partition (str): Data split, either 'train' or 'test'.
	    
	Returns:
	    all_data (np.ndarray): Extracted points of shape (N_samples, 2048, 3).
	    all_label (np.ndarray): Target category labels of shape (N_samples, 1).
	"""
	download()
	import h5py  # Lazy import h5py here to avoid unnecessary dependency import issues
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, 'data')
	all_data = []
	all_label = []
	# Iterate over all h5 parts for current partition
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
		f = h5py.File(h5_name)
		data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label


def translate_pointcloud(pointcloud):
	"""
	Augment point cloud by applying random scaling and translation.
	
	Args:
	    pointcloud (np.ndarray): Input points of shape (N, 3).
	    
	Returns:
	    translated_pointcloud (np.ndarray): Augmented points of shape (N, 3).
	"""
	# Sample scale multiplier along each axis (x, y, z)
	xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
	# Sample translations along each axis (x, y, z)
	xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
	   
	# Apply: point * scale + translation
	translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
	return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
	"""
	Augment point cloud by adding small Gaussian noise to each point.
	
	Args:
	    pointcloud (np.ndarray): Input points of shape (N, 3).
	    sigma (float): Standard deviation of the Gaussian noise.
	    clip (float): Maximum absolute value to clip the noise.
	    
	Returns:
	    pointcloud (np.ndarray): Jittered points of shape (N, 3).
	"""
	N, C = pointcloud.shape
	# Add clipped random normal noise
	pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
	return pointcloud


class ModelNet40(Dataset):
	"""
	PyTorch Dataset class for ModelNet40 dataset.
	"""
	def __init__(self, num_points, partition='train'):
		self.data, self.label = load_data(partition)
		self.num_points = num_points
		self.partition = partition        

	def __getitem__(self, item):
		# Slice points to the specified limit
		pointcloud = self.data[item][:self.num_points]
		label = self.label[item]
		if self.partition == 'train':
			# Apply data augmentation
			pointcloud = translate_pointcloud(pointcloud)
			np.random.shuffle(pointcloud)  # Shuffle point order to enforce permutation invariance
		return pointcloud, label

	def __len__(self):
		return self.data.shape[0]


def save_json_gz(obj, filepath):
	"""
	Utility to serialize an object to JSON and compress it into a .gz file.
	"""
	import gzip
	import json

	json_str = json.dumps(obj)
	json_bytes = json_str.encode()
	with gzip.GzipFile(filepath, mode="w") as f:
		f.write(json_bytes)


def geodescis(pos, face, k):
	"""
	Calculates the k-nearest geodesic neighbors for each vertex in the mesh using Dijkstra's algorithm.
	
	Args:
	    pos (np.ndarray or torch.Tensor): Vertex coordinates of shape (V, 3).
	    face (np.ndarray or torch.Tensor): Face indices of shape (F, 3).
	    k (int): Number of nearest neighbors to retrieve.
	    
	Returns:
	    idx (torch.Tensor): Indices of the k nearest geodesic neighbors of shape (V, k).
	"""
	log.debug("Computing geodesic distances: %d vertices, %d faces, k=%d", len(pos), len(face), k)
	pos = torch.Tensor(pos)
	face = torch.Tensor(face)
	# Compute geodesic distances using torch_geometric. Face transpose transforms shape to (2, 3*F) or (3, F) representation
	dist = -1 * geodesic_distance(pos, face.t(), norm=False, num_workers=8)
	
	# Select indices of the top-k closest (which correspond to largest elements since we multiplied by -1)
	idx = dist.topk(k=k, dim=-1)[1]
	return idx


def load_meshes_with_faces(directory, partition, extention, k):
	"""
	Loads mesh vertices and faces from a directory and either loads cached geodesic neighborhood indices 
	from a pickle file or computes and saves them.
	
	Args:
	    directory (str): Root directory of the dataset.
	    partition (str): Subfolder split (e.g. 'train', 'test', 'val').
	    extention (str): Mesh file extension (e.g. '.ply').
	    k (int): Number of geodesic neighbors to find.
	    
	Returns:
	    vertices_all (list of np.ndarray): List of mesh vertex arrays.
	    idx_all (dict): Mapping from filename to geodesic indices tensor.
	    max_size (int): The maximum number of vertices found in any mesh.
	    max_scale (float): The maximum coordinate scale across all loaded meshes.
	    filename (list of str): List of loaded mesh filenames.
	         """
	log.info("Loading meshes with faces from directory: %s  partition=%s  ext=%s  k=%d",
	         directory, partition, extention, k)
	files = sorted(glob.glob(directory + partition + "/*" + extention))
	log.info("Found %d mesh files", len(files))
	max_size = 0
	vertices_all = []
	
	# Cached geodesic indices file
	pk_filename = directory + 'idx_' + str(k) + '_' + partition + '.pkl'
	try:
		save = False
		with open(pk_filename, 'rb') as f:
			idx_all = pickle.load(f)
		log.info("Loaded cached geodesic indices from: %s", pk_filename)
	except Exception as e:
		log.info("Geodesic cache not found (%s). Computing from scratch for k=%d ...", pk_filename, k)
		save = True
		idx_all = {}
	
	max_scale = 0
	filename = []
	
	# Read meshes and record dimensions
	for i, f in enumerate(files):
		try:
			mesh = pv.read(f)
			name = f.split("/")[-1]
			filename.append(name)
			vertices = np.array(mesh.points).astype('double')
			# Read face array, pyvista prepends number of vertices per face (usually 3 for triangles), so filter out the count indicator
			faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:]
			log.debug("[%d/%d] %s  vertices=%d  faces=%d",
			          i + 1, len(files), name, len(vertices), len(faces))
			
			# Compute geodesic distances if no cache exists
			if save:
				log.debug("Computing geodesics for %s ...", name)
				idx = geodescis(vertices, faces, k)
				idx_all[name] = idx

			# Keep track of absolute scale factor for normalization
			scale = np.max(np.abs(vertices))
			if scale > max_scale:
				max_scale = scale
			vertices_all.append(vertices)
			
			# Record maximum vertex size for pad-to-max padding
			if len(vertices) > max_size:
				max_size = len(vertices)
		except Exception as exc:
			log.exception("Failed to load mesh file %s: %s", f, exc)
			raise
			
	# Save the newly computed indices to pickle file
	if save:
		log.info("Saving geodesic indices to: %s", pk_filename)
		with open(pk_filename, 'wb') as f:
			pickle.dump(idx_all, f)
		log.info("Geodesic indices saved successfully.")

	log.info("Dataset summary: %d meshes  max_vertices=%d  max_scale=%.4f",
	         len(vertices_all), max_size, max_scale)
	return vertices_all, idx_all, max_size, max_scale, filename


class MeshesWithFaces(Dataset):
	"""
	Dataset that loads 3D meshes, vertex coordinates, and their precomputed geodesic indices.
	Resizes and pads point counts to match max_size so they can be loaded in uniform PyTorch batches.
	"""
	def __init__(self, directory, partition='train', extention=".ply", k=10):
		log.info("Initialising MeshesWithFaces: partition=%s  ext=%s  k=%d", partition, extention, k)
		self.data, self.idx_all, self.max_size, self.scale, self.filename = load_meshes_with_faces(directory, partition, extention, k)
		log.info("MeshesWithFaces ready: %d meshes  max_size=%d  scale=%.4f",
		         len(self.data), self.max_size, self.scale)
		self.partition = partition        

	def __getitem__(self, item):
		try:
			name = self.filename[item]
			pointcloud = self.data[item]

			# Padding/duplication check: if mesh vertices are fewer than max_size, duplicate random vertices
			excess = self.max_size - len(pointcloud)
			list_idx = list(range(len(pointcloud)))
			if excess > 0:
				repeat_idx = np.random.randint(0, len(pointcloud), excess)
				list_idx = list_idx + list(repeat_idx)

			# Standardize scales across meshes: pointcloud coordinates / maximum scale
			pointcloud = pointcloud[list_idx, :] / self.scale
			
			# Output label is identical to scaled pointcloud for reconstruction loss
			label = pointcloud.copy()
			
			# Select and pad the geodesic indices for the sampled/padded vertices
			idx = self.idx_all[name]
			idx_extended = idx[list_idx]
			
			return pointcloud, idx_extended, label, name
		except Exception as exc:
			log.exception("MeshesWithFaces.__getitem__ failed at index %d (file: %s): %s",
			              item, self.filename[item] if item < len(self.filename) else '?', exc)
			raise
		
	def __len__(self):
		return len(self.data)


def load_meshes(directory, partition, extention):
	"""
	Loads mesh vertices without face/geodesic information.
	
	Args:
	    directory (str): Root directory of the dataset.
	    partition (str): Subfolder split ('train', 'test', 'val').
	    extention (str): Mesh file extension (e.g. '.ply').
	    
	Returns:
	    vertices_all (list of np.ndarray): Vertex coordinates list.
	    max_size (int): Max number of vertices.
	    max_scale (float): Maximum scaling factor.
	    filename (list of str): Mesh file names.
	"""
	files = sorted(glob.glob(directory + partition + "/*" + extention))
	max_size = 0
	vertices_all = []
	
	max_scale = 0
	filename = []
	for f in files:
		mesh = pv.read(f)
		name = f.split("/")[-1]
		filename.append(name)
		vertices = np.array(mesh.points).astype('double')
		scale = np.max(vertices)
		if scale > max_scale:
			max_scale = scale
		vertices_all.append(vertices)
		if len(vertices) > max_size:
			max_size = len(vertices)
	return vertices_all, max_size, max_scale, filename


class Meshes(Dataset):
	"""
	Dataset that loads 3D meshes and vertex coordinates only (no geodesic features).
	"""
	def __init__(self, directory, partition='train', extention=".ply"):
		self.data, self.max_size, self.scale, self.filename = load_meshes(directory, partition, extention)
		self.partition = partition        

	def __getitem__(self, item):
		name = self.filename[item]
		pointcloud = self.data[item]
		
		# Duplicate points to uniform batch length
		excess = self.max_size - len(pointcloud)
		list_idx = list(range(len(pointcloud)))
		if excess > 0:
			repeat_idx = np.random.randint(0, len(pointcloud), excess)
			list_idx = list_idx + list(repeat_idx)
			
		# Normalize and copy coordinates for reconstruction labels
		pointcloud = pointcloud[list_idx, :] / self.scale
		label = pointcloud.copy()

		return pointcloud, label, name

	def __len__(self):
		return len(self.data)
