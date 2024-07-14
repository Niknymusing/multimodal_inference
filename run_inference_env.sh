#!/bin/bash

# Set the base directory for the repository
BASE_DIR=$(pwd)

# Set the paths to the models
DUMMY_AUDIO_ENC_PATH="$BASE_DIR/model_files/dummy_audio_enc.pt"
SPIRALNET_TEST_PATH="$BASE_DIR/model_files/spiralnet_test.pt"
DUMMY_DEC_PATH="$BASE_DIR/model_files/dummy_dec.pt"

# Set the path to the compiled binary
INFERENCE_ENV_BINARY="$BASE_DIR/inference_env/inference_cpp/build/inference_env"

# Check if the binary exists
if [ ! -f "$INFERENCE_ENV_BINARY" ]; then
    echo "Error: Compiled binary $INFERENCE_ENV_BINARY does not exist."
    exit 1
fi

# Check if the models exist
if [ ! -f "$DUMMY_AUDIO_ENC_PATH" ]; then
    echo "Error: Model $DUMMY_AUDIO_ENC_PATH does not exist."
    exit 1
fi

if [ ! -f "$SPIRALNET_TEST_PATH" ]; then
    echo "Error: Model $SPIRALNET_TEST_PATH does not exist."
    exit 1
fi

if [ ! -f "$DUMMY_DEC_PATH" ]; then
    echo "Error: Model $DUMMY_DEC_PATH does not exist."
    exit 1
fi

# Run the compiled application with the provided models
"$INFERENCE_ENV_BINARY" "$DUMMY_AUDIO_ENC_PATH" "$SPIRALNET_TEST_PATH" "$DUMMY_DEC_PATH"

# Check the exit status of the application
if [ $? -ne 0 ]; then
    echo "Error: Application exited with a non-zero status."
    exit 1
else
    echo "Application ran successfully."
fi
