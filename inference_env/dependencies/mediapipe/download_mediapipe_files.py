import re
import os
import urllib.request

def parse_model_files(filepath):
    pattern = re.compile(r'urls = \["(.*?(?:\.pb|\.tflite)\?generation=\d+)"\],')
    model_files = []

    try:
        with open(filepath, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    model_files.append(match.group(1))
    except FileNotFoundError:
        print("The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return model_files

def download_files(model_files, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # Create target directory if it doesn't exist
    
    for file_url in model_files:
        file_name = file_url.split('/')[-1].split('?')[0]  # Get the file name from URL
        save_path = os.path.join(target_dir, file_name)
        
        try:
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, save_path)
            print(f"Saved to {save_path}")
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")

# Path to the .bzl file
file_path = "/Users/nikny/mediapipe/third_party/external_files.bzl"
# Directory to save the model files
download_dir = "/Users/nikny/mediapipe/third_party/model_files"

# Parse the model files from the .bzl file
model_files = parse_model_files(file_path)
# Download the model files
download_files(model_files, download_dir)
