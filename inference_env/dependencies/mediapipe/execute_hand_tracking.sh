#!/bin/zsh
export GLOG_logtostderr=1

/Users/nikny/nilsrepo/mmm/inference_env/dependencies/mediapipe/bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_with_osc \
--calculator_graph_config_file=/Users/nikny/nilsrepo/mmm/inference_env/dependencies/mediapipe/mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt



 