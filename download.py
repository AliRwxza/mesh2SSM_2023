"""
Download Module
Provides interface to ShapeWorks Cloud to authenticate and download the pre-processed MedDecathalon Pancreas dataset.
The dataset is groomed using ShapeWorks mesh grooming tools and contains the train, test, and validation meshes.
"""

import getpass
from pathlib import Path
from swcc.api import swcc_session
from swcc.models import Dataset, Project

# ==============================================================================
# Dataset Licensing and Citations
# ==============================================================================
# Med Decathalon Dataset: Pancreas 
# Source: http://medicaldecathlon.com/
# License: CC-BY-SA 4.0 (permitting sharing, distribution, and modification).
# Verified by expert human raters for clinical training.
# 
# Citation for Dataset:
# https://arxiv.org/abs/1902.09063
#
# Citation for ShapeWorks:
# Joshua Cates, Shireen Elhabian, Ross Whitaker. "Shapeworks: particle-based shape 
# correspondence and visualization software." Statistical Shape and Deformation Analysis. 
# Academic Press, 2017. 257-298.
# ==============================================================================

def public_server_download(download_dir):
	"""
	Prompts the user for ShapeWorks Cloud credentials, logs in to the public server,
	and downloads the 'MedDecathalon_Pancreas' dataset to the specified folder path.
	
	Args:
	    download_dir (str): Relative or absolute directory path where the dataset should be saved.
	"""
	# Query user credentials via standard terminal inputs
	username = input('Enter username: ')
	password = getpass.getpass('Enter password: ')
	
	print("---------------------------------------------------------------------------------------")
	print("Please read the comments in the code about the license information before processing")
	print("---------------------------------------------------------------------------------------")
	input("Press Enter to proceed")
	
	# Open an API session wrapper to shapeworks-cloud.org
	with swcc_session() as public_server_session:
		try:
			# Log in using swcc API
			public_server_session.login(username, password)
			dataset_name = 'MedDecathalon_Pancreas'

			# Search for dataset entry by name
			dataset = Dataset.from_name(dataset_name)
		
			# Convert directory path string to Path object
			download_path = Path(download_dir)
			
			# Download the first project associated with the dataset
			for project in dataset.projects:
				print(f"Downloading project {project.name} to {download_path}...")
				project.download(Path(download_path))
				break
		except Exception as e:
			# Handles bad credentials or API connection errors
			print(f"Error occurred during download: {e}")
			print("---------------------------------------------------------------------------------------")
			print("Please create an account on https://www.shapeworks-cloud.org/#/ to download the dataset")
			print("---------------------------------------------------------------------------------------")
			input("Press Enter to proceed")
			
		print("\n")
		print("You can also visualize the samples on the ShapeWorks Cloud portal https://www.shapeworks-cloud.org/#/ once you login")
		print("---------------------------------------------------------------------------------------")
		print("Please clone the dataset to modify the dataset. Do not edit the existing dataset")


# Automatically trigger download to local pancreas directory when script is run
if __name__ == '__main__':
	public_server_download("./pancreas/")