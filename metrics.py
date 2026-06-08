"""
Metrics Module
Defines surface-to-surface and point-to-mesh geometric distance metrics for evaluation of reconstructed shape models.
Uses the `trimesh` library for robust 3D mesh proximity queries.
"""

# import shapeworks as sw
import glob 
import os
import trimesh
import time
import numpy as np
import multiprocessing as mp

class SurfaceDistance():
	"""
	Calculates the symmetric vertex-to-surface distance between two trimesh meshes.
	This is computed by finding the distance from all vertices of mesh A to the surface of mesh B,
	and from all vertices of mesh B to the surface of mesh A, then taking a weighted average.
	"""

	def __init__(self):
		pass

	def __call__(self, A, B):
		"""
		Computes symmetric vertex-to-surface distance.
		
		Args:
		    A (trimesh.Trimesh): The first 3D mesh.
		    B (trimesh.Trimesh): The second 3D mesh.
		  
		Returns:
		    distance (np.ndarray): Single-element array containing the symmetric mean distance.
		"""
		# Find closest points on surface A to vertices of B. Returns (closest_points, distances, triangle_indices)
		_, A_B_dist, _ = trimesh.proximity.closest_point(A, B.vertices)
		
		# Find closest points on surface B to vertices of A
		_, B_A_dist, _ = trimesh.proximity.closest_point(B, A.vertices)
		
		# Compute the symmetric mean distance
		distance = .5 * np.array(A_B_dist).mean() + .5 * np.array(B_A_dist).mean()

		return np.array([distance])

def calculate_surface_to_surface_distance(m, r):
	"""
	Loads two meshes from paths and calculates their surface-to-surface distance.
	
	Args:
	    m (str): File path to the first mesh (e.g., target mesh).
	    r (str): File path to the second mesh (e.g., reconstructed mesh).
	    
	Returns:
	    dist (float): Mean surface-to-surface distance.
	"""
	mesh = trimesh.load(m)
	recon_mesh = trimesh.load(r)
	
	# Compute symmetric surface distance between loaded meshes
	s2sDist = SurfaceDistance()(mesh, recon_mesh)
	
	return np.mean(s2sDist)


def calculate_point_to_mesh_distance(m, p):
	"""
	Loads a mesh and a point cloud (particles/points file) and calculates
	the signed distance from the points to the surface of the mesh.
	
	Args:
	    m (str): File path to the reference surface mesh.
	    p (str): File path to the predicted particles/coordinates txt file.
	    
	Returns:
	    dist (float): Mean point-to-mesh distance.
	"""
	# Load target mesh
	mesh = trimesh.load(m)
	# Load predicted particle/point coordinates (shape: [N, 3])
	points = np.loadtxt(p)

	# Set up proximity query solver on target mesh
	c = trimesh.proximity.ProximityQuery(mesh)
	
	# Calculate signed distances from the points to the mesh surface.
	# Positive distances represent points outside, negative represent points inside.
	p2mDist = c.signed_distance(points)

	# Return the mean distance (using np.mean on the array of signed distances)
	return np.mean(p2mDist)