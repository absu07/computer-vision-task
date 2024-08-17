# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 23:40:52 2024

@author: ASUS
"""

import os

def delete_images_with_output(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # List all files in the directory
    files = os.listdir(folder_path)

    # Iterate over the files
    for file_name in files:
        # Check if '_output' is in the file name
        if '_output' in file_name:
            file_path = os.path.join(folder_path, file_name)
            try:
                # Remove the file
                os.remove(file_path)
                print(f"Deleted: {file_name}")
            except Exception as e:
                print(f"Error deleting {file_name}: {e}")

# Example usage
folder_path = "D:\Personal\Interviews\ML_SmartCowTakeHomeAssignment_2024\ML_SmartCowTakeHomeAssignment\Take_Home_Computer_Vision\computer_vision\computer-vision-task\WIDERFACE_Validation\images"  # Replace this with the path to your folder
delete_images_with_output(folder_path)
