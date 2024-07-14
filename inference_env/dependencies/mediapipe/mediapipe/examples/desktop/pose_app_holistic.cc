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

    void sendLandmarks(const std::vector<std::tuple<float, float, float>>& landmarks, const char* path) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, path, msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};

void SendLandmarksFromPoller(mediapipe::OutputStreamPoller& poller, OscSender& sender, const std::string& osc_address) {
    mediapipe::Packet packet;
    if (poller.QueueSize() > 0) {
        if (poller.Next(&packet)) {
            auto landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            std::vector<std::tuple<float, float, float>> osc_landmarks;
            for (const auto& landmark_list : landmarks) {
                for (const auto& landmark : landmark_list.landmark()) {
                    osc_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
                }
            }
            if (!osc_landmarks.empty()) {
                sender.sendLandmarks(osc_landmarks, osc_address.c_str());
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

    // Ensure the stream names are correctly matched with your graph's output streams
    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto left_hand_landmarks_poller, graph.AddOutputStreamPoller("LEFT_HAND_LANDMARKS"));
    MP_ASSIGN_OR_RETURN(auto right_hand_landmarks_poller, graph.AddOutputStreamPoller("RIGHT_HAND_LANDMARKS"));
    MP_ASSIGN_OR_RETURN(auto face_landmarks_poller, graph.AddOutputStreamPoller("FACE_LANDMARKS"));
    MP_ASSIGN_OR_RETURN(auto pose_landmarks_poller, graph.AddOutputStreamPoller("POSE_LANDMARKS"));
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

        // Handling different types of landmarks and sending them via OSC
        SendLandmarksFromPoller(left_hand_landmarks_poller, osc_sender, "/left_hand_landmarks");
        SendLandmarksFromPoller(right_hand_landmarks_poller, osc_sender, "/right_hand_landmarks");
        SendLandmarksFromPoller(face_landmarks_poller, osc_sender, "/face_landmarks");
        SendLandmarksFromPoller(pose_landmarks_poller, osc_sender, "/pose_landmarks");

        // Display output video
        mediapipe::Packet video_packet;
        if (video_poller.Next(&video_packet)) {
            auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);
            if (cv::waitKey(5) >= 0) break;
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

bool InitializeCapture(cv::VideoCapture& capture, const std::string& video_path) {
    if (!video_path.empty()) {
        if (!capture.open(video_path)) {
            ABSL_LOG(ERROR) << "Failed to open video file.";
            return false;
        }
    } else {
        if (!capture.open(0)) {
            ABSL_LOG(ERROR) << "Failed to open webcam.";
            return false;
        }
        capture.set(cv::CAP_PROP_FPS, 60);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }
    return true;
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


/* gpto gen ...
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

    void sendLandmarks(const std::vector<std::tuple<float, float, float>>& landmarks, const char* path) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, path, msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};

void SendLandmarksFromPoller(mediapipe::OutputStreamPoller& poller, OscSender& sender, const std::string& osc_address) {
    mediapipe::Packet packet;
    if (poller.QueueSize() > 0) {
        if (poller.Next(&packet)) {
            auto landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            std::vector<std::tuple<float, float, float>> osc_landmarks;
            for (const auto& landmark_list : landmarks) {
                for (const auto& landmark : landmark_list.landmark()) {
                    osc_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
                }
            }
            if (!osc_landmarks.empty()) {
                sender.sendLandmarks(osc_landmarks, osc_address.c_str());
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

    // Ensure the stream names are correctly matched with your graph's output streams
    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto left_hand_landmarks_poller, graph.AddOutputStreamPoller("LEFT_HAND_LANDMARKS"));
    MP_ASSIGN_OR_RETURN(auto right_hand_landmarks_poller, graph.AddOutputStreamPoller("RIGHT_HAND_LANDMARKS")); 
    MP_ASSIGN_OR_RETURN(auto face_landmarks_poller, graph.AddOutputStreamPoller("FACE_LANDMARKS"));
    MP_ASSIGN_OR_RETURN(auto pose_landmarks_poller, graph.AddOutputStreamPoller("POSE_LANDMARKS"));
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

        // Handling different types of landmarks and sending them via OSC
        SendLandmarksFromPoller(left_hand_landmarks_poller, osc_sender, "/left_hand_landmarks");
        SendLandmarksFromPoller(right_hand_landmarks_poller, osc_sender, "/right_hand_landmarks");
        SendLandmarksFromPoller(face_landmarks_poller, osc_sender, "/face_landmarks");
        SendLandmarksFromPoller(pose_landmarks_poller, osc_sender, "/pose_landmarks");


        // Display output video
        mediapipe::Packet video_packet;
        if (video_poller.Next(&video_packet)) {
            auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);
            if (cv::waitKey(5) >= 0) break;
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

bool InitializeCapture(cv::VideoCapture& capture, const std::string& video_path) {
    if (!video_path.empty()) {
        if (!capture.open(video_path)) {
            ABSL_LOG(ERROR) << "Failed to open video file.";
            return false;
        }
    } else {
        if (!capture.open(0)) {
            ABSL_LOG(ERROR) << "Failed to open webcam.";
            return false;
        }
        capture.set(cv::CAP_PROP_FPS, 60);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }
    return true;
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


// current dev version
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

    void sendLandmarks(const std::vector<std::tuple<float, float, float>>& landmarks, const char* path) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, path, msg);
        lo_message_free(msg);
    }

private:
    lo_address osc_address;
};

void SendLandmarksFromPoller(mediapipe::OutputStreamPoller& poller, OscSender& sender, const std::string& osc_address) {
    mediapipe::Packet packet;
    if (poller.QueueSize() > 0) {
        if (poller.Next(&packet)) {
            auto landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            std::vector<std::tuple<float, float, float>> osc_landmarks;
            for (const auto& landmark_list : landmarks) {
                for (const auto& landmark : landmark_list.landmark()) {
                    osc_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
                }
            }
            if (!osc_landmarks.empty()) {
                sender.sendLandmarks(osc_landmarks, osc_address.c_str());
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
    MP_ASSIGN_OR_RETURN(auto left_hand_landmarks_poller, graph.AddOutputStreamPoller("LEFT_HAND_LANDMARKS"));
    MP_ASSIGN_OR_RETURN(auto right_hand_landmarks_poller, graph.AddOutputStreamPoller("RIGHT_HAND_LANDMARKS"));
    MP_ASSIGN_OR_RETURN(auto face_landmarks_poller, graph.AddOutputStreamPoller("FACE_LANDMARKS"));
    MP_ASSIGN_OR_RETURN(auto pose_landmarks_poller, graph.AddOutputStreamPoller("POSE_LANDMARKS"));
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

        // Handling different types of landmarks and sending them via OSC
        SendLandmarksFromPoller(left_hand_landmarks_poller, osc_sender, "/left_hand_landmarks");
        SendLandmarksFromPoller(right_hand_landmarks_poller, osc_sender, "/right_hand_landmarks");
        SendLandmarksFromPoller(face_landmarks_poller, osc_sender, "/face_landmarks");
        SendLandmarksFromPoller(pose_landmarks_poller, osc_sender, "/pose_landmarks");


        // Display output video
        mediapipe::Packet video_packet;
        if (video_poller.Next(&video_packet)) {
            auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);
            if (cv::waitKey(5) >= 0) break;
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}



bool InitializeCapture(cv::VideoCapture& capture, const std::string& video_path) {
    if (!video_path.empty()) {
        if (!capture.open(video_path)) {
            ABSL_LOG(ERROR) << "Failed to open video file.";
            return false;
        }
    } else {
        if (!capture.open(0)) {
            ABSL_LOG(ERROR) << "Failed to open webcam.";
            return false;
        }
        capture.set(cv::CAP_PROP_FPS, 60);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }
    return true;
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

    void sendHandLandmarks(const std::vector<std::tuple<float, float, float>>& landmarks) {
        sendLandmarks(landmarks, "/hand_landmarks");
    }

    void sendFaceLandmarks(const std::vector<std::tuple<float, float, float>>& landmarks) {
        sendLandmarks(landmarks, "/face_landmarks");
    }

    void sendPoseLandmarks(const std::vector<std::tuple<float, float, float>>& landmarks) {
        sendLandmarks(landmarks, "/pose_landmarks");
    }

public:
    void sendLandmarks(const std::vector<std::tuple<float, float, float>>& landmarks, const char* path) {
        lo_message msg = lo_message_new();
        for (const auto& landmark : landmarks) {
            lo_message_add_float(msg, std::get<0>(landmark));
            lo_message_add_float(msg, std::get<1>(landmark));
            lo_message_add_float(msg, std::get<2>(landmark));
        }
        lo_send_message(osc_address, path, msg);
        lo_message_free(msg);
    }
private:
    lo_address osc_address;
};


void SendLandmarksFromPoller(mediapipe::OutputStreamPoller& poller, OscSender& sender, const std::string& osc_address) {
    mediapipe::Packet packet;
    if (poller.QueueSize() > 0) {
        if (poller.Next(&packet)) {
            auto landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            std::vector<std::tuple<float, float, float>> osc_landmarks;
            for (const auto& landmark_list : landmarks) {
                for (const auto& landmark : landmark_list.landmark()) {
                    osc_landmarks.emplace_back(landmark.x(), landmark.y(), landmark.z());
                }
            }
            if (!osc_landmarks.empty()) {
                sender.sendLandmarks(osc_landmarks, osc_address.c_str());
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

    // Adding more pollers for different types of landmarks
    MP_ASSIGN_OR_RETURN(auto video_poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(auto hand_landmarks_poller, graph.AddOutputStreamPoller("hand_landmarks"));
    MP_ASSIGN_OR_RETURN(auto face_landmarks_poller, graph.AddOutputStreamPoller("face_landmarks"));
    MP_ASSIGN_OR_RETURN(auto pose_landmarks_poller, graph.AddOutputStreamPoller("pose_landmarks"));
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

        // Handling different types of landmarks and sending them via OSC
        SendLandmarksFromPoller(hand_landmarks_poller, osc_sender, "/hand_landmarks");
        SendLandmarksFromPoller(face_landmarks_poller, osc_sender, "/face_landmarks");
        SendLandmarksFromPoller(pose_landmarks_poller, osc_sender, "/pose_landmarks");

        // Display output video
        mediapipe::Packet video_packet;
        if (video_poller.Next(&video_packet)) {
            auto& output_frame = video_packet.Get<mediapipe::ImageFrame>();
            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);
            if (cv::waitKey(5) >= 0) break;
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}



bool InitializeCapture(cv::VideoCapture& capture, const std::string& video_path) {
    if (!video_path.empty()) {
        if (!capture.open(video_path)) {
            ABSL_LOG(ERROR) << "Failed to open video file.";
            return false;
        }
    } else {
        if (!capture.open(0)) {
            ABSL_LOG(ERROR) << "Failed to open webcam.";
            return false;
        }
        capture.set(cv::CAP_PROP_FPS, 60);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }
    return true;
}

// Ensure this function is well-tested and any issues during initialization or execution are logged.



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