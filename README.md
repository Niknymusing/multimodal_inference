
# Project Setup Instructions

## Prerequisites
Ensure you have **Bazel**, **CMake**, and **Miniconda** (or Anaconda) installed on your system.

## Environment Setup
To create and activate the conda environment with all required Python dependencies, run:

```bash
conda env create -f environment.yml
conda activate mmm_dev
```

## Building and Running the Bazel Project

First download libtorch to the right folder for the project by running

```bash
chmod +x setup.sh
./setup.sh
```

Then to build the Bazel project with the MediaPipe hand-tracking application:

```bash
chmod +x build_hand_tracking_OSC.sh
./build_hand_tracking_OSC.sh
```

You can then run the hand tracking application using:

```bash
./run_hand_tracking.sh
```

This application tracks hand landmarks from the device camera and sends detected landmark coordinates as OSC messages to port 7770 using the liblo library.

## Building and Running the CMake Project
To build the CMake project containing the `inference_env` app, execute the shell script:

```bash
chmod +x build_inference_env.sh
./build_inference_env.sh
```

Ensure all dependencies (PortAudio, libtorch, liblo (lo), spdlog, sndfile, Threads) are correctly installed and linked on your system from the `CMakeLists.txt` file located in `mmm/inference_env/inference_cpp`. Adjustments might be needed if dependencies are not properly configured.

## Running the Inference Environment
You can run the `inference_env` application from:

```bash
./run_inference_env.sh
```

This application records live audio from a selected input device, starts an OSC server on port 7770 (to handle incoming hand tracking OSC messages from the hand tracking app), encodes the input signals using the loaded PyTorch models, and sends the output from the models as OSC messages to port 7773.

### Loading Models
For the `inference_env` app, you need to load some JIT-compiled PyTorch models:
- Encoder model to process the audio stream
- Encoder model process the motion stream
- Decoder model producing the output

To instantiate untrained dummy models for testing, run:

```bash
python instantiate_models.py
```

This script is located in the top directory of the project.



