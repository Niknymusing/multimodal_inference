cc_binary(
    name = "pose_app",
    srcs = ["pose_app.cc"],
    copts = ["-std=c++17"],  # Correct C++ version.
    deps = [
        "@liblo//:liblo",  # Assuming liblo is correctly configured in the WORKSPACE.
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_base",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/util:resource_util",
        "@opencv//:core",  # Correct OpenCV dependency, ensure it's set up in the WORKSPACE.
        "@opencv//:highgui",
        "@opencv//:imgproc",
    ],
    linkopts = [
        "-L/opt/homebrew/lib",  # Ensure the path to liblo and OpenCV if not standard.
        "-llo",  # Linking liblo for OSC messages.
        "-lopencv_core", 
        "-lopencv_imgproc", 
        "-lopencv_highgui",
    ],
)

