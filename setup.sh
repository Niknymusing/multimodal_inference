#!/bin/bash

# Define the base directory as the current working directory
BASE_DIR=$(pwd)

# Directory where libtorch will be downloaded and extracted
LIBTORCH_DIR="$BASE_DIR/inference_env/dependencies"

# Ensure the dependencies directory exists
mkdir -p "$LIBTORCH_DIR"

# Download libtorch for mac arm64
echo "Downloading libtorch..."
wget -q -O libtorch.zip "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.1.zip"

# Extract libtorch to the desired directory
echo "Extracting libtorch..."
unzip -qo libtorch.zip -d "$LIBTORCH_DIR"

# Check if libtorch was extracted correctly
if [ -d "$LIBTORCH_DIR/libtorch" ]; then
    echo "Libtorch installed successfully."
else
    echo "Failed to install Libtorch."
    exit 1
fi

# Remove the downloaded zip file to clean up
rm libtorch.zip


echo "Libtorch setup completed successfully."
