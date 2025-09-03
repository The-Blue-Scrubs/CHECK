import os

def create_file_manifest(root_dir, manifest_filename="file_manifest.txt"):
    """
    Scans a directory for .csv and .json files and writes their
    relative paths to a manifest file.

    Args:
        root_dir (str): The root directory of the project to scan.
        manifest_filename (str): The name of the output manifest file.
    """
    # Exclude the script's own directory and any virtual environment folders
    excluded_dirs = {'.git', '.idea', 'venv', '__pycache__'}
    
    print(f"Scanning for .csv and .json files in '{root_dir}'...")
    
    with open(manifest_filename, "w") as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Remove excluded directories from the walk
            dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
            
            for filename in filenames:
                if filename.endswith(".csv") or filename.endswith(".json"):
                    # Get the full path of the file
                    full_path = os.path.join(dirpath, filename)
                    # Get the relative path from the root directory
                    relative_path = os.path.relpath(full_path, root_dir)
                    # Use forward slashes for cross-platform compatibility
                    normalized_path = relative_path.replace(os.sep, '/')
                    f.write(normalized_path + "\n")
                    print(f"  Found: {normalized_path}")
                    
    print(f"\nManifest created successfully: '{manifest_filename}'")
    print("This file now contains the paths of all your data files.")

if __name__ == "__main__":
    # Run the script in the current directory
    project_directory = "." 
    create_file_manifest(project_directory)