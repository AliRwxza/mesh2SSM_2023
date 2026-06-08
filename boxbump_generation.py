"""
Box-Bump Generation Script
Generates a synthetic dataset of 3D meshes for training and validating statistical shape models.
The script constructs a box mesh and a sphere mesh, smooths them using ShapeWorks mesh tools,
and iteratively positions/translates the sphere along the box surface.
Uses PyVista Boolean union operations to create a 'bump' on the box surface.
"""

import os
import vtk 
import numpy as np
import pyvista as pv
import shapeworks as sw
import glob

# ==============================================================================
# Configuration Parameters
# ==============================================================================
# To generate train/test/val datasets, customize the save_dir and num_samples variables.
# Typical partition splits: Train = 500, Test = 100, Val = 100 samples.
# ==============================================================================

save_dir = "box_bump_100_test/"
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# 1. Initialize Box mesh with subdivisions (level=6) to ensure high vertex density
mesh = pv.Box(level=6)
scale_x = 6
scale_y = 12
scale_z = 6
radius = 4.5

# Scale box coordinates and convert quad faces to triangular faces for compatibility
box = mesh.scale([scale_x, scale_y, scale_z], inplace=False).triangulate()
box.save(save_dir + "box.ply")

# 2. Initialize Sphere mesh centered at coordinate origin (0, 0, 0)
sphere = pv.Sphere(radius=radius, center=[0, 0, 0]).triangulate()
sphere.save(save_dir + "sphere.ply")

# ---------------------------------------------------------------------------
# Mesh Grooming: Smooth and remesh using ShapeWorks APIs
# ---------------------------------------------------------------------------
# Load box mesh into ShapeWorks and resample to exactly 2000 vertices,
# apply Laplacian smoothing (100 iterations, relaxation factor 0.19), then remesh again
sw_box = sw.Mesh(save_dir + "box.ply")
sw_box.remesh(numVertices=2000, adaptivity=0.0).smooth(100, 0.19).remesh(numVertices=2000, adaptivity=0.0)
sw_box.write(save_dir + "box_smoothed.ply")

# Load sphere mesh, resample to 1000 vertices, apply smoothing, and remesh
sw_sphere = sw.Mesh(save_dir + "sphere.ply")
sw_sphere.remesh(numVertices=1000, adaptivity=0.0).smooth(50, 0.01).remesh(numVertices=1000, adaptivity=0.0)
sw_sphere.write(save_dir + "sphere_smoothed.ply")


# ---------------------------------------------------------------------------
# Synthetic Shape Sequence Generation
# ---------------------------------------------------------------------------
# Load smoothed meshes back into PyVista for translation and boolean combination
pv_box = pv.read(save_dir + "box_smoothed.ply")
pv_sphere = pv.read(save_dir + "sphere_smoothed.ply")

margin = 1
num_samples = 100

# Calculate translation bounds and step increment along the Y-axis.
# Step bounds keep the sphere centered on the box with a boundary margin.
step = 2 * (scale_y - radius - margin) / num_samples
starting_point = -1 * (scale_y - radius - margin)

for i in range(num_samples):
	# Calculate translation coordinates: (x=0, y=translated position, z=perched on top of box)
	y_translation = starting_point + (i * step)
	sphere = pv_sphere.translate((0, y_translation, scale_z), inplace=False)
	
	# Compute Boolean union to merge box and sphere into a single continuous mesh
	boxbump = sphere.boolean_union(pv_box)
	
	# Save the generated sample
	filename = "sample_" + str(i) + ".ply"
	boxbump.save(save_dir + filename)

	# Clean up PyVista mesh references to avoid memory leaks
	del boxbump
	del sphere
