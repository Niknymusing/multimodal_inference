cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(inference_env)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#list(APPEND CMAKE_PREFIX_PATH "/Users/nikny/libtorch_4")
#list(APPEND CMAKE_PREFIX_PATH "/Users/nikny/nilsrepo/mmm/inference_env/dependencies/libtorch 4")
# Append the specific path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../inference_env/dependencies/libtorch")


find_package(Torch REQUIRED)

# Add TSan flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -g")


find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)

# Include directories for PortAudio and libsndfile
include_directories(/opt/homebrew/include)

# Link directories for PortAudio and libsndfile
link_directories(/opt/homebrew/lib)

add_executable(inference_env main.cpp)

# Link with Torch, PortAudio, liblo, libsndfile, and Threads
target_link_libraries(inference_env "${TORCH_LIBRARIES}" portaudio lo sndfile Threads::Threads spdlog::spdlog)

