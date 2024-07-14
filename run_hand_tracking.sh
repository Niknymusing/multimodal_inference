#!/bin/bash

# Get the current working directory
CWD=$(pwd)

# Define the directory where the shell script is located
SCRIPT_DIRECTORY="$CWD/inference_env/dependencies/mediapipe"

# Change the current working directory to the script directory
cd "$SCRIPT_DIRECTORY" || { echo "Failed to change directory to $SCRIPT_DIRECTORY"; exit 1; }

# Define the path to the shell script
SCRIPT_PATH="./execute_hand_tracking.sh"

# Run the shell script
if [ -f "$SCRIPT_PATH" ]; then
    echo "Running script: $SCRIPT_PATH"
    bash "$SCRIPT_PATH"
    
    # Check the exit status of the script
    if [ $? -eq 0 ]; then
        echo "Script executed successfully."
    else
        echo "Script executed with errors."
    fi
else
    echo "Script $SCRIPT_PATH not found."
    exit 1
fi
