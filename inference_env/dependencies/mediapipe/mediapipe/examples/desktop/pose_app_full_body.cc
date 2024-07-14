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
constexpr char kLandmarksStream[] = "pose_landmarks";



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

    void sendJoint(float x, float y, float z, int jointId) {
        lo_message msg = lo_message_new();
        lo_message_add_float(msg, x);
        lo_message_add_float(msg, y);
        lo_message_add_float(msg, z);
        std::string address = "/joint_" + std::to_string(jointId);
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