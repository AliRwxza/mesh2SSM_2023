"""
Generate Project Module
Integrates the prediction outputs of Mesh2SSM with ShapeWorks.
Creates Excel spreadsheet projects containing groomed meshes, original segmentations,
and correspondence particles for visualizations or analysis in ShapeWorks Studio/Cloud.
"""

import shapeworks as sw 
import glob
import os 


def make_project_files(meshes, particles, data_dir, proj_name):
    """
    Creates a ShapeWorks spreadsheet (.xlsx) project file registering meshes and correspondence particle files.
    
    Args:
        meshes (list of str): List of sorted paths to the input meshes.
        particles (list of str): List of sorted paths to the predicted particle/landmark coordinate files.
        data_dir (str): Root directory path containing the outputs where the project folder will be created.
        proj_name (str): Name of the generated Excel project spreadsheet file (excluding extension).
    """
    # Create the shape_models project directory
    project_location = os.path.join(data_dir, "shape_models/")
    if not os.path.exists(project_location):
        os.makedirs(project_location)
        
    subjects = []
    print(f"Creating project with {len(meshes)} meshes and {len(particles)} particles.")
    number_domains = 1  # Standard single-domain shape representation
    
    # Iterate through each subject to set up mesh and particle file mappings
    for i in range(len(particles)):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        
        # Get paths relative to the Excel sheet location to ensure portability of project folder
        rel_seg_files = sw.utils.get_relative_paths([meshes[i]], project_location)
        
        # Set original and groomed (aligned/processed) segmentation filenames
        subject.set_original_filenames(rel_seg_files)
        subject.set_groomed_filenames(rel_seg_files)
        
        # Get relative paths for predicted particles/landmarks
        f = sw.utils.get_relative_paths([particles[i]], project_location)
        
        # Set particle files for local and world coordinates
        subject.set_local_particle_filenames(f)
        subject.set_world_particle_filenames(f)
        
        subjects.append(subject)
        
    # Instantiate a ShapeWorks Project object and register subjects
    project = sw.Project()
    project.set_subjects(subjects)
    
    # Save the Excel spreadsheet project
    spreadsheet_file = os.path.join(project_location, f"{proj_name}.xlsx")
    project.save(spreadsheet_file) 
    print(f"Project successfully saved to: {spreadsheet_file}")


def main():
	# Example path templates. Update these when running on a specific dataset.
	meshes = sorted(glob.glob("path_to_mesh_files"))
	data_dir = "original_dir_containing_outputs"
	particles = sorted(glob.glob("path_to_predicted_particles_from_Mesh2SSM"))
	project_name = "output_project_name"
	
	# Execute project generation
	make_project_files(meshes, particles, data_dir, project_name)
 

if __name__ == '__main__':
	main()
