# 1_download_data.py

import os
import kagglehub
import shutil

print("--- Step 1: Setting up Kaggle API credentials ---")

# The user must place their kaggle.json file in the root of this project folder
# This script will move it to the correct location.
kaggle_json_path = 'kaggle.json'
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

if os.path.exists(kaggle_json_path):
    shutil.move(kaggle_json_path, os.path.join(kaggle_dir, 'kaggle.json'))
    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 600)
    print("kaggle.json successfully moved and configured.")
else:
    print("WARNING: kaggle.json not found in the project root.")
    print("Please download it from your Kaggle account and place it here.")
    # Exit if credentials are not available
    if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
        exit("Exiting: Kaggle credentials are required to download data.")

print("\n--- Step 2: Downloading the dataset ---")
# Define the path for the raw data
raw_data_path = 'data/raw'
os.makedirs(raw_data_path, exist_ok=True)

# Download the dataset using kagglehub
# The library automatically handles caching, so it won't re-download if already present.
path = kagglehub.dataset_download(
    "snap/amazon-fine-food-reviews",
    path=raw_data_path # Specify download location
)

print(f"\nDataset downloaded and unpacked to: {raw_data_path}")
print("Setup complete. You can now run 2_train_model.py")