#include <torch/script.h>
#include <iostream>
#include <memory>
#include <portaudio.h>
#include <lo/lo.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <array>
#include <string>
#include <sstream>
#include <signal.h>
#include <csignal>
#include <atomic>

// Application constants
const int SAMPLE_RATE = 44100;
const int FRAMES_PER_BUFFER = 2048;

// Pose data structure and synchronization
struct PoseData {
    std::array<std::array<float, 3>, 21> joints;  // 21 joints, 3 coordinates each
};

std::mutex poseMutex;   // Mutex for pose data synchronization
PoseData poseBuffer1, poseBuffer2;  // Two buffers for pose data
PoseData* activePoseBuffer = &poseBuffer1;
PoseData* processingPoseBuffer = &poseBuffer2;
std::array<bool, 21> jointUpdated;  // Flags for each joint update
std::atomic<bool> allJointsUpdated(false);  // Flag to indicate all joints have been updated

// Audio data structure and synchronization
std::atomic<bool> newAudioData(false);  // Flag to indicate new audio data is available
std::mutex audioMutex;
std::vector<float> audioBuffer1, audioBuffer2;  // Two buffers for audio data
std::vector<float>* activeAudioBuffer = &audioBuffer1;
std::vector<float>* processingAudioBuffer = &audioBuffer2;
std::condition_variable audioDataAvailable;  // Condition variable to signal audio data availability

std::atomic<bool> runApp(true);  // Atomic flag to control thread execution

// Signal handling to gracefully exit
void signalHandler(int signal) {
    std::cout << "Signal received: " << signal << std::endl;
    runApp = false;
    audioDataAvailable.notify_all();  // Notify all waiting threads
}

class PoseThread {
public:
    PoseThread(torch::jit::script::Module* poseEncoderModule)
    : st(lo_server_thread_new("7771", nullptr)), poseEncoderModule(poseEncoderModule) {
        lo_server_thread_add_method(st, "/pose/joint/*", "fff", poseHandler, this);
    }

    ~PoseThread() {
        if (st) {
            lo_server_thread_stop(st);
            lo_server_thread_free(st);
        }
    }

    void start() {
        lo_server_thread_start(st);
        std::cout << "OSC Server started, listening for messages..." << std::endl;
        runningThread = std::thread([this] {
            while (runApp) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    void stop() {
        runApp = false;
        if (runningThread.joinable()) {
            runningThread.join();
        }
    }

    static int poseHandler(const char *path, const char *types, lo_arg **argv, int argc, lo_message msg, void *user_data) {
        PoseThread* self = static_cast<PoseThread*>(user_data);
        int jointIndex = -1;
        if (sscanf(path, "/pose/joint/%d", &jointIndex) == 1 && jointIndex >= 0 && jointIndex < 21) {
            std::lock_guard<std::mutex> lock(poseMutex);
            (*activePoseBuffer).joints[jointIndex][0] = argv[0]->f;
            (*activePoseBuffer).joints[jointIndex][1] = argv[1]->f;
            (*activePoseBuffer).joints[jointIndex][2] = argv[2]->f;
            jointUpdated[jointIndex] = true;

            if (std::all_of(jointUpdated.begin(), jointUpdated.end(), [](bool v) { return v; }) && !allJointsUpdated) {
                allJointsUpdated = true;  // Set all joints updated flag
            }
            return 0;
        }
        return 1;
    }

    void encodePoseIfNeeded() {
        if (allJointsUpdated && newAudioData) {
            std::lock_guard<std::mutex> lock(poseMutex);
            auto poseTensor = torch::from_blob(activePoseBuffer->joints.data(), {1, 21, 3}, torch::kFloat32);

            // Encode pose data
            auto poseEncoding = poseEncoderModule->forward({poseTensor}).toTensor();
            std::cout << "Pose Encoding Output: " << poseEncoding << std::endl;

            newAudioData = false;
            allJointsUpdated = false;  // Reset for next update cycle
        }
    }

private:
    lo_server_thread st;
    std::thread runningThread;
    torch::jit::script::Module* poseEncoderModule;
};

void listAudioDevices() {
    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "ERROR: Pa_CountDevices returned " << numDevices << std::endl;
        return;
    }

    std::cout << "Number of audio devices: " << numDevices << std::endl;
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo) {
            std::cout << "Device " << i << ": " << deviceInfo->name
                      << " (Input Channels: " << deviceInfo->maxInputChannels
                      << ", Output Channels: " << deviceInfo->maxOutputChannels
                      << ", Sample Rate: " << deviceInfo->defaultSampleRate << ")" << std::endl;
        } else {
            std::cout << "Failed to get info for device " << i << std::endl;
        }
    }
}

// Audio callback function: Producer
static int audioCallback(const void* inputBuffer, void* outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo* timeInfo,
                         PaStreamCallbackFlags statusFlags,
                         void* userData) {
    if (!inputBuffer) {
        std::cerr << "Error: Input buffer was NULL." << std::endl;
        return paAbort;
    }
    // Get the input data
    auto* in = static_cast<const float*>(inputBuffer);
    std::lock_guard<std::mutex> audioLock(audioMutex);
    // Clear the active buffer and insert the new data
    activeAudioBuffer->clear();
    activeAudioBuffer->insert(activeAudioBuffer->end(), in, in + framesPerBuffer);

    // Notify the main thread that new audio data is available
    audioDataAvailable.notify_one();

    return paContinue; // Continue recording
}

void audioInputThread(int deviceIndex, torch::jit::script::Module* audioEncoderModule, PoseThread* poseThread) {
    PaError paErr = Pa_Initialize();
    if (paErr != paNoError) {
        std::cerr << "Failed to initialize PortAudio: " << Pa_GetErrorText(paErr) << std::endl;
        return;
    }

    PaStreamParameters inputParameters;
    memset(&inputParameters, 0, sizeof(inputParameters));
    inputParameters.device = deviceIndex;
    inputParameters.channelCount = 1; // Mono input
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    PaStream* stream;
    paErr = Pa_OpenStream(
        &stream,
        &inputParameters,
        NULL, // No output
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        audioCallback,
        nullptr // No user data needed
    );
    if (paErr != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(paErr) << std::endl;
        Pa_Terminate();
        return;
    }

    paErr = Pa_StartStream(stream);
    if (paErr != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(paErr) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return;
    }

    while (runApp) {
        std::unique_lock<std::mutex> lock(audioMutex);
        audioDataAvailable.wait(lock, [] { return !activeAudioBuffer->empty() || !runApp; });

        if (!runApp) break;

        // Process audio data
        auto audioTensor = torch::from_blob(activeAudioBuffer->data(), {1, 1, static_cast<long>(activeAudioBuffer->size())}, torch::kFloat32);

        // Encode audio data
        auto audioEncoding = audioEncoderModule->forward({audioTensor}).toTensor();
        //std::cout << "Audio Encoding Output: " << audioEncoding << std::endl;

        
        // Ensure the tensor data is contiguous and get the data pointer
        auto audioEncodingContiguous = audioEncoding.contiguous();
        float* encodingData = audioEncodingContiguous.data_ptr<float>();



        // Ensure we are sending exactly 128 floats
        /*
        // Send a single OSC message with an array of 128 floats as value
        lo_address osc_target = lo_address_new("localhost", "7773");
        lo_message msg = lo_message_new();
        lo_blob b = lo_blob_new(128 * sizeof(float), encodingData);
        lo_message_add_blob(msg, b);
        lo_send_message(osc_target, "/audio/encoding", msg);
        lo_blob_free(b);
        lo_message_free(msg);
        lo_address_free(osc_target);
        */
       // Send a single OSC message with an array of 128 floats as value
        
        lo_address osc_target = lo_address_new("localhost", "7773");
        lo_message msg = lo_message_new();
        for (int i = 0; i < 128; ++i) {
            lo_message_add_float(msg, encodingData[i]);
        }
        lo_send_message(osc_target, "/audio/encoding", msg);
        lo_message_free(msg);
        lo_address_free(osc_target);
        
        
        newAudioData = true;
        poseThread->encodePoseIfNeeded();  // Ensure pose encoding happens after audio is processed
    }


    // Stop and close the stream
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    std::cout << "Stream closed and PortAudio terminated." << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path-to-audio-encoder-model> <path-to-pose-encoder-model>\n";
        return -1;
    }

    signal(SIGINT, signalHandler);  // Handle Ctrl-C
    signal(SIGTERM, signalHandler); // Handle termination signal

    // Load models
    torch::jit::script::Module audioEncoderModule, poseEncoderModule;
    try {
        audioEncoderModule = torch::jit::load(argv[1]);
        poseEncoderModule = torch::jit::load(argv[2]);
        std::cout << "Models loaded successfully.\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the models: " << e.what() << std::endl;
        return -1;
    }

    // Initialize PortAudio
    PaError paErr = Pa_Initialize();
    if (paErr != paNoError) {
        std::cerr << "Failed to initialize PortAudio: " << Pa_GetErrorText(paErr) << std::endl;
        return -1;
    }
    std::cout << "PortAudio initialized successfully.\n";
    listAudioDevices();
    std::cout << "Enter device index to use: ";
    int deviceIndex;
    std::cin >> deviceIndex;
    std::cin.ignore();  // Ignore leftover newline

    runApp = true;  // Set flag to control thread execution

    PoseThread poseThread(&poseEncoderModule); // Thread for handling pose data
    poseThread.start();

    std::thread audioThread(audioInputThread, deviceIndex, &audioEncoderModule, &poseThread); // Audio input thread

    std::cout << "Press Ctrl+C to stop the application.\n";
    std::cin.get();  // Wait for user input to stop the application

    // Signal threads to stop
    runApp = false;
    audioDataAvailable.notify_all();

    // Join all threads
    if (audioThread.joinable())
        audioThread.join();
    
    poseThread.stop(); // Stop the PoseThread properly

    // Clean up PortAudio
    Pa_Terminate();
    std::cout << "Application terminated successfully.\n";
    return 0;
}

