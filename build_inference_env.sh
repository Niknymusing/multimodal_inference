#!/bin/bash

# Get the current working directory
CWD=$(pwd)

# Define the build directory
BUILD_DIRECTORY="$CWD/inference_env/inference_cpp/build"

# Check if the build directory exists
if [ ! -d "$BUILD_DIRECTORY" ]; then
    echo "Build directory $BUILD_DIRECTORY does not exist."
    exit 1
fi

# Change to the build directory
cd "$BUILD_DIRECTORY" || { echo "Failed to change directory to $BUILD_DIRECTORY"; exit 1; }

# Run cmake and make
echo "Running cmake and make in $BUILD_DIRECTORY"
cmake .. && make

# Check the exit status of the make command
if [ $? -eq 0 ]; then
    echo "Build completed successfully."
else
    echo "Build failed."
    exit 1
fi
