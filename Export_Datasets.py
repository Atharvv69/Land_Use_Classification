import zipfile
import os

zip_path = "earth-observation-delhi-airshed.zip"
extract_path = "Datasets"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted Successfully!")