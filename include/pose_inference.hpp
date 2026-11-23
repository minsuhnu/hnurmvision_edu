#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace training {
// RM_POSE_SOCKET selects the local CUDA worker; otherwise use the existing CPU ONNX backend.
std::vector<cv::Mat> pose_forward(const cv::Mat& blob);
}
