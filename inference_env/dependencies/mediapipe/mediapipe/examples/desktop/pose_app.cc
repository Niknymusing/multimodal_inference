#include <cstdlib>
#include <vector>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kLandmarksStream[] = "landmarks";


ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 7771, "Local port for sending OSC messages.");

class OscSender {
public:
    OscSender(const std::string& address, const std::string& port) {
        osc_address = lo_address_new(address.c_str(), port.c_str());
        if (!osc_address) {
            ABSL_LOG(ERROR) << "Failed to create OSC address.";
            throw std::runtime_error("Failed to create OSC address.");
        }
    }

    ~OscSender() {
        if (osc_address) {
            lo_address_free(osc_address);
        }
    }

    

    void sendJoint(float x, float y, float z, int jointId) {
        lo_message msg = lo_message_new();
        lo_message_add_float(msg, x);
        lo_message_add_float(msg, y);
        lo_message_add_float(msg, z);
        std::string address = "/pose/joint/" + std::to_string(jointId);  // Adjusted to match receiver
        lo_send_message(osc_address, address.c_str(), msg);
        lo_message_free(msg);
    }
private:
    lo_address osc_address;
};

void SendLandmarksFromPoller(mediapipe::OutputStreamPoller& poller, OscSender& sender) {
    mediapipe::Packet packet;
    if (poller.Next(&packet)) {
        auto landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        for (const auto& landmark_list : landmarks) {
            for (int i = 0; i < landmark_list.landmark_size(); ++i) {
                const auto& landmark = landmark_list.landmark(i);
                sender.sendJoint(landmark.x(), landmark.y(), landmark.z(), i);
            }
        }
    }
}

absl::Status RunMPPGraph(OscSender& osc_sender) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    if (!absl::GetFlag(FLAGS_input_video_path).empty()) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
        capture.set(cv::CAP_PROP_FPS, 60);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            continue;
        }

        if (camera_frame_raw.empty()) {
            ABSL_LOG(INFO) << "Received an empty frame.";
            continue;
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (video_poller.QueueSize() > 0) {
            if (!video_poller.Next(&video_packet)) {
                ABSL_LOG(ERROR) << "Failed to fetch a video packet.";
                continue;
            }
            auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);
            if (cv::waitKey(5) >= 0) break;
        }

        mediapipe::Packet landmarks_packet;
        if (landmarks_poller.QueueSize() > 0) {
            if (landmarks_poller.Next(&landmarks_packet)) {
                auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                for (const auto& landmarks : landmark_lists) {
                    for (int i = 0; i < landmarks.landmark_size(); ++i) {
                        const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                        osc_sender.sendJoint(landmark.x(), landmark.y(), landmark.z(), i);
                    }
                }
            } else {
                ABSL_LOG(INFO) << "No landmarks detected.";
            }
        } else {
            continue; // Handle gracefully when no landmarks are detected
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}


int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    google::InitGoogleLogging(argv[0]);

    try {
        OscSender osc_sender("127.0.0.1", std::to_string(absl::GetFlag(FLAGS_osc_port)));
        absl::Status run_status = RunMPPGraph(osc_sender);
        if (!run_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Exception caught in main: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/*
#include <cstdlib>
#include <vector>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 8082, "Local port for sending OSC messages.");

class OscSender {
public:
    OscSender(const std::string& address, const std::string& port) {
        osc_address = lo_address_new(address.c_str(), port.c_str());
        if (!osc_address) {
            ABSL_LOG(ERROR) << "Failed to create OSC address.";
            throw std::runtime_error("Failed to create OSC address.");
        }
    }

    ~OscSender() {
        if (osc_address) {
            lo_address_free(osc_address);
        }
    }

    void send(const std::vector<std::tuple<float, float, float>>& landmarks) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, "/hand_landmarks", msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};


absl::Status RunMPPGraph(OscSender& osc_sender) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    if (!absl::GetFlag(FLAGS_input_video_path).empty()) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
        // Set the frame rate to 60 fps
        capture.set(cv::CAP_PROP_FPS, 60); // Optional: Setting FPS directly after opening
        // Set resolution directly after opening the capture device
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);  // Set width
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480); // Set height
    }



    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            continue; // Handle missing frames more gracefully
        }

        if (camera_frame_raw.empty()) {
            ABSL_LOG(INFO) << "Received an empty frame.";
            continue; // Skip processing if the frame is empty
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (video_poller.QueueSize() > 0) {
            if (!video_poller.Next(&video_packet)) {
                ABSL_LOG(ERROR) << "Failed to fetch a video packet.";
                continue; // Handle polling failure
            }
            auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);
            if (cv::waitKey(5) >= 0) break; // Check if user wants to close the window
        }

        mediapipe::Packet landmarks_packet;
        if (landmarks_poller.QueueSize() > 0) {
            if (landmarks_poller.Next(&landmarks_packet)) {
                auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                std::vector<std::tuple<float, float, float>> all_landmarks;
                for (const auto& landmarks : landmark_lists) {
                    for (int i = 0; i < landmarks.landmark_size(); ++i) {
                        const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                        all_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
                    }
                }
                if (!all_landmarks.empty()) {
                    osc_sender.send(all_landmarks);
                }
            } else {
                ABSL_LOG(INFO) << "No landmarks detected.";
            }
        } else {
            //ABSL_LOG(INFO) << "Waiting for landmarks.";
            continue; // Handle gracefully when no landmarks are detected
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone(); // Correct syntax here
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    google::InitGoogleLogging(argv[0]);

    try {
        OscSender osc_sender("127.0.0.1", std::to_string(absl::GetFlag(FLAGS_osc_port)));
        absl::Status run_status = RunMPPGraph(osc_sender);
        if (!run_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Exception caught in main: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
*/

/*
// seems to work correctly!
#include <cstdlib>
#include <vector>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 8082, "Local port for sending OSC messages.");

class OscSender {
public:
    OscSender(const std::string& address, const std::string& port) {
        osc_address = lo_address_new(address.c_str(), port.c_str());
        if (!osc_address) {
            ABSL_LOG(ERROR) << "Failed to create OSC address.";
            throw std::runtime_error("Failed to create OSC address.");
        }
    }

    ~OscSender() {
        if (osc_address) {
            lo_address_free(osc_address);
        }
    }

    void send(const std::vector<std::tuple<float, float, float>>& landmarks) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, "/hand_landmarks", msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};


absl::Status RunMPPGraph(OscSender& osc_sender) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    if (!absl::GetFlag(FLAGS_input_video_path).empty()) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            continue; // Handle missing frames more gracefully
        }

        if (camera_frame_raw.empty()) {
            ABSL_LOG(INFO) << "Received an empty frame.";
            continue; // Skip processing if the frame is empty
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (video_poller.QueueSize() > 0) {
            if (!video_poller.Next(&video_packet)) {
                ABSL_LOG(ERROR) << "Failed to fetch a video packet.";
                continue; // Handle polling failure
            }
            auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);
            if (cv::waitKey(5) >= 0) break; // Check if user wants to close the window
        }

        mediapipe::Packet landmarks_packet;
        if (landmarks_poller.QueueSize() > 0) {
            if (landmarks_poller.Next(&landmarks_packet)) {
                auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                std::vector<std::tuple<float, float, float>> all_landmarks;
                for (const auto& landmarks : landmark_lists) {
                    for (int i = 0; i < landmarks.landmark_size(); ++i) {
                        const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                        all_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
                    }
                }
                if (!all_landmarks.empty()) {
                    osc_sender.send(all_landmarks);
                }
            } else {
                ABSL_LOG(INFO) << "No landmarks detected.";
            }
        } else {
            ABSL_LOG(INFO) << "Waiting for landmarks.";
            continue; // Handle gracefully when no landmarks are detected
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone(); // Correct syntax here
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    google::InitGoogleLogging(argv[0]);

    try {
        OscSender osc_sender("127.0.0.1", std::to_string(absl::GetFlag(FLAGS_osc_port)));
        absl::Status run_status = RunMPPGraph(osc_sender);
        if (!run_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Exception caught in main: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

*/





/*
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 8082, "Local port for sending OSC messages.");

class OscSender {
public:
    OscSender(const std::string& address, const std::string& port) {
        osc_address = lo_address_new(address.c_str(), port.c_str());
        if (!osc_address) {
            ABSL_LOG(ERROR) << "Failed to create OSC address.";
            throw std::runtime_error("Failed to create OSC address.");
        }
    }

    ~OscSender() {
        if (osc_address) {
            lo_address_free(osc_address);
        }
    }

    void send(const std::vector<std::tuple<float, float, float>>& landmarks) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, "/hand_landmarks", msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};

absl::Status RunMPPGraph(OscSender& osc_sender) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    if (!absl::GetFlag(FLAGS_input_video_path).empty()) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            continue; // Handle missing frames more gracefully
        }

        if (camera_frame_raw.empty()) {
            ABSL_LOG(INFO) << "Received an empty frame.";
            continue; // Skip processing if the frame is empty
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (!video_poller.Next(&video_packet)) {
            ABSL_LOG(ERROR) << "Failed to fetch a video packet.";
            continue; // Handle polling failure
        }
        auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break; // Check if user wants to close the window

        mediapipe::Packet landmarks_packet;
        if (landmarks_poller.Next(&landmarks_packet)) {
            auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            std::vector<std::tuple<float, float, float>> all_landmarks;
            for (const auto& landmarks : landmark_lists) {
                for (int i = 0; i < landmarks.landmark_size(); ++i) {
                    const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                    all_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
                }
            }
            if (!all_landmarks.empty()) {
                osc_sender.send(all_landmarks);
            }
        } else {
            ABSL_LOG(INFO) << "Waiting for landmarks.";
            continue; // Handle gracefully when no landmarks are detected
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone(); // Correct syntax here
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    google::InitGoogleLogging(argv[0]);

    try {
        OscSender osc_sender("127.0.0.1", std::to_string(absl::GetFlag(FLAGS_osc_port)));
        absl::Status run_status = RunMPPGraph(osc_sender);
        if (!run_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Exception caught in main: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
*/





/*
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 8082, "Local port for sending OSC messages.");

class OscSender {
public:
    OscSender(const std::string& address, const std::string& port) {
        osc_address = lo_address_new(address.c_str(), port.c_str());
        if (!osc_address) {
            ABSL_LOG(ERROR) << "Failed to create OSC address.";
            throw std::runtime_error("Failed to create OSC address.");
        }
    }

    ~OscSender() {
        if (osc_address) {
            lo_address_free(osc_address);
        }
    }

    void send(const std::vector<std::tuple<float, float, float>>& landmarks) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, "/hand_landmarks", msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};

absl::Status RunMPPGraph(OscSender& osc_sender) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    if (!absl::GetFlag(FLAGS_input_video_path).empty()) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            continue; // Handle missing frames more gracefully
        }

        if (camera_frame_raw.empty()) {
            ABSL_LOG(INFO) << "Received an empty frame.";
            continue; // Skip processing if the frame is empty
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (!video_poller.Next(&video_packet)) {
            ABSL_LOG(ERROR) << "Failed to fetch a video packet.";
            continue; // Handle polling failure
        }
        auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break; // Check if user wants to close the window

        mediapipe::Packet landmarks_packet;
        if (landmarks_poller.Next(&landmarks_packet)) {
            auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            std::vector<std::tuple<float, float, float>> all_landmarks;
            for (const auto& landmarks : landmark_lists) {
                for (int i = 0; i < landmarks.landmark_size(); ++i) {
                    const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                    all_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
                }
            }
            if (!all_landmarks.empty()) {
                osc_sender.send(all_landmarks);
            }
        } else {
            ABSL_LOG(INFO) << "No landmarks detected.";
            continue; // Handle no landmarks scenario
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    google::InitGoogleLogging(argv[0]);

    try {
        OscSender osc_sender("127.0.0.1", std::to_string(absl::GetFlag(FLAGS_osc_port)));
        absl::Status run_status = RunMPPGraph(osc_sender);
        if (!run_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Exception caught in main: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

*/


/*
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 8082, "Local port for sending OSC messages.");

bool pollWithTimeout(mediapipe::OutputStreamPoller& poller, mediapipe::Packet& packet, int timeout_ms) {
    std::future<bool> future = std::async(std::launch::async, [&poller, &packet]() {
        return poller.Next(&packet);
    });
    if (future.wait_for(std::chrono::milliseconds(timeout_ms)) == std::future_status::ready) {
        return future.get(); 
    } else {
        return false; 
    }
}

class OscSender {
public:
    OscSender(const std::string& address, const std::string& port) {
        osc_address = lo_address_new(address.c_str(), port.c_str());
        if (!osc_address) {
            ABSL_LOG(ERROR) << "Failed to create OSC address.";
            throw std::runtime_error("Failed to create OSC address.");
        }
    }

    ~OscSender() {
        if (osc_address) {
            lo_address_free(osc_address);
        }
    }

    void send(const std::vector<std::tuple<float, float, float>>& landmarks) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, "/hand_landmarks", msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};

absl::Status RunMPPGraph(OscSender& osc_sender) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    if (!absl::GetFlag(FLAGS_input_video_path).empty()) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            continue;
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (!video_poller.Next(&video_packet)) {
            continue;
        }
        auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break;

        mediapipe::Packet landmarks_packet;
        if (!pollWithTimeout(landmarks_poller, landmarks_packet, 1000)) {
            ABSL_LOG(INFO) << "No landmarks detected or timeout reached.";
            continue;
        }
        auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        std::vector<std::tuple<float, float, float>> all_landmarks;
        for (const auto& landmarks : landmark_lists) {
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
                const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                all_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
            }
        }
        if (!all_landmarks.empty()) {
            osc_sender.send(all_landmarks);
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    google::InitGoogleLogging(argv[0]);

    try {
        OscSender osc_sender("127.0.0.1", std::to_string(absl::GetFlag(FLAGS_osc_port)));
        absl::Status run_status = RunMPPGraph(osc_sender);
        if (!run_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Exception caught in main: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


*/





// works to send OSC messages, sends all landmarks with one OSC message
/*
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <thread>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 8082, "Local port for sending OSC messages.");

class OscSender {
public:
    OscSender(const std::string& address, const std::string& port) {
        osc_address = lo_address_new(address.c_str(), port.c_str());
        if (!osc_address) {
            ABSL_LOG(ERROR) << "Failed to create OSC address.";
            throw std::runtime_error("Failed to create OSC address.");
        }
    }

    ~OscSender() {
        if (osc_address) {
            lo_address_free(osc_address);
        }
    }

    void send(const std::vector<std::tuple<float, float, float>>& landmarks) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            float x = std::get<0>(landmark);
            float y = std::get<1>(landmark);
            float z = std::get<2>(landmark);
            lo_message_add_float(msg, x);
            lo_message_add_float(msg, y);
            lo_message_add_float(msg, z);
        }
        lo_send_message(osc_address, "/hand_landmarks", msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};

absl::Status RunMPPGraph(OscSender& osc_sender) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            break;
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1); // Depending on camera orientation

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (!video_poller.Next(&video_packet)) break;
        auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();

        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break;

        mediapipe::Packet landmarks_packet;
        if (!landmarks_poller.Next(&landmarks_packet)) break;
        auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        std::vector<std::tuple<float, float, float>> all_landmarks;
        for (const auto& landmarks : landmark_lists) {
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
                const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                all_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
            }
        }
        if (!all_landmarks.empty()) {
            osc_sender.send(all_landmarks);
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}


int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    google::InitGoogleLogging(argv[0]);

    try {
        OscSender osc_sender("127.0.0.1", std::to_string(absl::GetFlag(FLAGS_osc_port)));
        absl::Status run_status = RunMPPGraph(osc_sender);
        if (!run_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Exception caught in main: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
*/





/*// this version runs and sends OSC messages correctly, but freezes when hands are removed from the camera view
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 8082, "Local port for sending OSC messages.");

class OscSender {
public:
    OscSender(const std::string& address, const std::string& port) {
        osc_address = lo_address_new(address.c_str(), port.c_str());
        if (!osc_address) {
            ABSL_LOG(ERROR) << "Failed to create OSC address.";
            throw std::runtime_error("Failed to create OSC address.");
        }
        sender_thread = std::thread(&OscSender::processMessages, this);
    }

    ~OscSender() {
        stop();
    }

    void send(int index, float x, float y, float z) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        osc_queue.emplace(index, x, y, z);
        cv.notify_one();
    }

    void stop() {
        is_running = false;
        cv.notify_one();
        if (sender_thread.joinable())
            sender_thread.join();
        if (osc_address)
            lo_address_free(osc_address);
    }

private:
    lo_address osc_address;
    std::queue<std::tuple<int, float, float, float>> osc_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::thread sender_thread;
    bool is_running = true;

    void processMessages() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        while (is_running || !osc_queue.empty()) {
            cv.wait(lock, [this]{ return !osc_queue.empty() || !is_running; });
            while (!osc_queue.empty()) {
                auto [index, x, y, z] = osc_queue.front();
                if (lo_send(osc_address, "/hand_landmark", "ifff", index, x, y, z) == -1) {
                    ABSL_LOG(ERROR) << "Failed to send OSC message: " << lo_address_errstr(osc_address);
                }
                osc_queue.pop();
            }
        }
    }
};

absl::Status RunMPPGraph(OscSender& osc_sender) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;

    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            break;
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1); // Depending on camera orientation

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = static_cast<size_t>(cv::getTickCount() / cv::getTickFrequency() * 1e6);
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (!video_poller.Next(&video_packet)) break;
        auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();

        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break;

        mediapipe::Packet landmarks_packet;
        if (!landmarks_poller.Next(&landmarks_packet)) break;
        auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

        for (const auto& landmarks : landmark_lists) {
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
                const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                osc_sender.send(i, landmark.x(), landmark.y(), landmark.z());
            }
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    google::InitGoogleLogging(argv[0]);

    try {
        OscSender osc_sender("127.0.0.1", std::to_string(absl::GetFlag(FLAGS_osc_port)));
        absl::Status run_status = RunMPPGraph(osc_sender);
        if (!run_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Exception caught in main: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
*/


/* // runs and detect poses but osc messages dont work and crashes
#include <cstdlib>
#include <queue>
#include <mutex>
#include <thread>
#include <vector>
#include "absl/synchronization/notification.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 8082, "Local port for sending OSC messages.");

std::queue<std::tuple<int, float, float, float>> osc_queue;
std::mutex queue_mutex;
absl::Notification shutdown_notifier;

void send_osc_messages(const char* address) {
    lo_address osc_address = lo_address_new("127.0.0.1", address);
    if (!osc_address) {
        ABSL_LOG(ERROR) << "Failed to create OSC address.";
        return;
    }

    while (!shutdown_notifier.HasBeenNotified()) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        while (!osc_queue.empty()) {
            auto [index, x, y, z] = osc_queue.front();
            if (lo_send(osc_address, "/hand_landmark", "ifff", index, x, y, z) == -1) {
                ABSL_LOG(ERROR) << "Failed to send OSC message: " << lo_address_errstr(osc_address);
            }
            osc_queue.pop();
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Reduce CPU usage
    }
    lo_address_free(osc_address);
}


absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    std::thread osc_thread(send_osc_messages, std::to_string(absl::GetFlag(FLAGS_osc_port)).c_str());
    
    while (capture.isOpened()) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            break;
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (!video_poller.Next(&video_packet)) break;
        auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();

        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break;

        mediapipe::Packet landmarks_packet;
        if (!landmarks_poller.Next(&landmarks_packet)) break;
        auto& landmark_lists = landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

        for (const auto& landmarks : landmark_lists) {
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
                const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
                std::lock_guard<std::mutex> lock(queue_mutex);
                osc_queue.emplace(i, landmark.x(), landmark.y(), landmark.z());
            }
        }
    }

    shutdown_notifier.Notify();
    osc_thread.join();
    ABSL_LOG(INFO) << "Shutting down.";
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);  // Initializes Google logging library
    absl::ParseCommandLine(argc, argv);  // Parses the command line arguments

    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

*/




/* // compiled but with bugs and threading issues
#include <cstdlib>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 7000, "Local port for sending OSC messages.");

absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video) {
        if (!capture.open(absl::GetFlag(FLAGS_input_video_path))) {
            return absl::NotFoundError("Failed to open video file.");
        }
    } else {
        if (!capture.open(0)) {
            return absl::NotFoundError("Failed to open webcam.");
        }
    }

    cv::VideoWriter writer;
    if (!absl::GetFlag(FLAGS_output_video_path).empty()) {
        cv::namedWindow(kWindowName, 1);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture.set(cv::CAP_PROP_FPS, 30);
    }

    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto landmarks_poller, graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    lo_address osc_address = lo_address_new(NULL, std::to_string(absl::GetFlag(FLAGS_osc_port)).c_str());
    bool grab_frames = true;
    while (grab_frames) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw)) {
            ABSL_LOG(ERROR) << "Failed to grab a frame from the camera.";
            break;
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet video_packet;
        if (!video_poller.Next(&video_packet)) break;
        auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();

        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break;

        mediapipe::Packet landmarks_packet;
        if (!landmarks_poller.Next(&landmarks_packet)) break;
        auto& landmarks = landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();

        for (int i = 0; i < landmarks.landmark_size(); ++i) {
            const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
            lo_send(osc_address, "/hand_landmark", "ifff", i, landmark.x(), landmark.y(), landmark.z());
        }
    }

    ABSL_LOG(INFO) << "Shutting down.";
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

*/




/*
#include <cstdlib>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");
ABSL_FLAG(int, osc_port, 7000, "Local port for sending OSC messages.");

absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    ABSL_LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    ABSL_LOG(INFO) << "Initialize the camera or load the video.";
    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video) {
        capture.open(absl::GetFlag(FLAGS_input_video_path));
    } else {
        capture.open(0);
    }
    RET_CHECK(capture.isOpened());

    cv::VideoWriter writer;
    const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
    if (!save_video) {
        cv::namedWindow(kWindowName, 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture.set(cv::CAP_PROP_FPS, 30);
#endif
    }

    ABSL_LOG(INFO) << "Start running the calculator graph.";
    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller video_poller,
                        graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_poller,
                        graph.AddOutputStreamPoller(kLandmarksStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    ABSL_LOG(INFO) << "Start grabbing and processing frames.";
    lo_address osc_address = lo_address_new(NULL, std::to_string(absl::GetFlag(FLAGS_osc_port)).c_str());
    bool grab_frames = true;
    while (grab_frames) {
        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty()) {
            if (!load_video) {
                ABSL_LOG(INFO) << "Ignore empty frames from camera.";
                continue;
            }
            ABSL_LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        if (!load_video) {
            cv::flip(camera_frame, camera_frame, 1);
        }

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        // Send image packet into the graph.
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release())
                              .At(mediapipe::Timestamp(frame_timestamp_us))));

        // Get the graph result packet, or stop if that fails.
        mediapipe::Packet video_packet;
        if (!video_poller.Next(&video_packet)) break;
        auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();

        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        if (save_video) {
            if (!writer.isOpened()) {
                ABSL_LOG(INFO) << "Prepare video writer.";
                writer.open(absl::GetFlag(FLAGS_output_video_path),
                            cv::VideoWriter::fourcc('a', 'v', 'c', '1'),  // .mp4
                            capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
                RET_CHECK(writer.isOpened());
            }
            writer.write(output_frame_mat);
        } else {
            cv::imshow(kWindowName, output_frame_mat);
            // Press any key to exit.
            const int pressed_key = cv::waitKey(5);
            if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
        }

        // Get the graph landmarks packet, or stop if that fails.
        mediapipe::Packet landmarks_packet;
        if (!landmarks_poller.Next(&landmarks_packet)) break;
        auto& landmarks = landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();

        // Send landmarks via OSC
        for (int i = 0; i < landmarks.landmark_size(); ++i) {
            const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
            lo_send(osc_address, "/hand_landmark", "ifff", i, landmark.x(), landmark.y(), landmark.z());
        }
    }

    ABSL_LOG(INFO) << "Shutting down.";
    if (writer.isOpened()) writer.release();
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    } else {
        ABSL_LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
*/






/*
#include <cstdlib>
#include <memory>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(int, osc_port, 7000, "Local port for sending OSC messages.");

absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    auto poller_status = graph.AddOutputStreamPoller(kOutputStream);
    if (!poller_status.ok()) {
        return poller_status.status();
    }
    mediapipe::OutputStreamPoller poller = std::move(poller_status.value());

    lo_address osc_address = lo_address_new(NULL, std::to_string(absl::GetFlag(FLAGS_osc_port)).c_str());
    cv::VideoCapture capture(absl::GetFlag(FLAGS_input_video_path).empty() ? 0 : std::stoi(absl::GetFlag(FLAGS_input_video_path)));
    if (!capture.isOpened()) {
        return absl::NotFoundError("Failed to open video capture");
    }

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (true) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw) || camera_frame_raw.empty()) break;

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame_raw.cols, camera_frame_raw.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        cv::cvtColor(camera_frame_raw, input_frame_mat, cv::COLOR_BGR2RGB);

        if (absl::GetFlag(FLAGS_input_video_path).empty()) {
            cv::flip(input_frame_mat, input_frame_mat, 1);
        }

        size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet packet;
        if (!poller.Next(&packet)) break;
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();

        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        // Show the output frame
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break;

        // Send landmarks via OSC
        auto& landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
        for (int i = 0; i < landmarks.landmark_size(); ++i) {
            const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
            lo_send(osc_address, "/hand_landmark", "ifff", i, landmark.x(), landmark.y(), landmark.z());
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    LOG(INFO) << "Success!";
    return EXIT_SUCCESS;
}

*/
/*
#include <cstdlib>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "lo/lo.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(int, osc_port, 7000, "Local port for sending OSC messages.");

absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    auto poller_status = graph.AddOutputStreamPoller(kOutputStream);
    if (!poller_status.ok()) {
        return poller_status.status();
    }
    mediapipe::OutputStreamPoller poller = std::move(poller_status.value());

    lo_address osc_address = lo_address_new(NULL, std::to_string(absl::GetFlag(FLAGS_osc_port)).c_str());
    cv::VideoCapture capture(absl::GetFlag(FLAGS_input_video_path).empty() ? 0 : std::stoi(absl::GetFlag(FLAGS_input_video_path)));
    if (!capture.isOpened()) {
        return absl::NotFoundError("Failed to open video capture");
    }

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    while (true) {
        cv::Mat camera_frame_raw;
        if (!capture.read(camera_frame_raw) || camera_frame_raw.empty()) break;

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame_raw.cols, camera_frame_raw.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        cv::cvtColor(camera_frame_raw, input_frame_mat, cv::COLOR_BGR2RGB);

        if (absl::GetFlag(FLAGS_input_video_path).empty()) {
            cv::flip(input_frame_mat, input_frame_mat, 1);
        }

        size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet packet;
        if (!poller.Next(&packet)) break;
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();

        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        // Show the output frame
        cv::imshow(kWindowName, output_frame_mat);
        if (cv::waitKey(5) >= 0) break;

        // Send landmarks via OSC
        auto& landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
        for (int i = 0; i < landmarks.landmark_size(); ++i) {
            const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
            lo_send(osc_address, "/hand_landmark", "ifff", i, landmark.x(), landmark.y(), landmark.z());
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    ABSL_LOG(INFO) << "Success!";
    return EXIT_SUCCESS;
}

*/

