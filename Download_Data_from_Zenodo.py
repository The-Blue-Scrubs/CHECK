import os
import requests
from tqdm import tqdm

def download_files_from_zenodo(zenodo_base_url, manifest_filename="file_manifest.txt"):
    """
    Downloads files listed in a manifest from a Zenodo repository and
    places them in their correct local directories.

    Args:
        zenodo_base_url (str): The base URL for the file downloads from Zenodo.
        manifest_filename (str): The path to the manifest file.
    """
    if not os.path.exists(manifest_filename):
        print(f"Error: Manifest file '{manifest_filename}' not found.")
        print("Please ensure you are in the root directory of the repository.")
        return

    with open(manifest_filename, "r") as f:
        files_to_download = [line.strip() for line in f if line.strip()]

    print(f"Found {len(files_to_download)} files to download from Zenodo.")

    for relative_path in files_to_download:
        # Construct the full download URL
        # Zenodo URLs typically have filenames at the end
        filename = os.path.basename(relative_path)
        download_url = f"{zenodo_base_url}{filename}"
        
        # Ensure the local directory exists
        local_dir = os.path.dirname(relative_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            
        print(f"\nDownloading '{filename}' to '{relative_path}'...")

        try:
            # Make the request with streaming to handle large files and get total size
            response = requests.get(download_url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            with open(relative_path, "wb") as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)

            if total_size != 0 and bar.n != total_size:
                print(f"Error: Download failed for {filename}. Size mismatch.")
            else:
                print(f"Successfully downloaded {filename}.")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            print("Please check the Zenodo URL and your internet connection.")
            print(f"Attempted URL: {download_url}")


if __name__ == "__main__":
    # --- IMPORTANT ---
    # YOU MUST REPLACE THIS URL WITH YOUR ACTUAL ZENODO RECORD URL.
    # Go to your Zenodo upload, and in the "Files" section, right-click
    # on any file and "Copy Link Address". Paste it here and remove the
    # filename at the end. It should end with a forward slash '/'.
    ZENODO_DOWNLOAD_URL = "https://zenodo.org/records/17048677/files/"

    if "YOUR_RECORD_ID" in ZENODO_DOWNLOAD_URL:
        print("="*60)
        print("!!! PLEASE EDIT THE SCRIPT !!!")
        print("You must replace 'YOUR_RECORD_ID' in the ZENODO_DOWNLOAD_URL")
        print("variable inside the 'download_data.py' script.")
        print("="*60)
    else:
        download_files_from_zenodo(ZENODO_DOWNLOAD_URL)
