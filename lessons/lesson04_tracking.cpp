#include "lessons.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <utility>

namespace training {
namespace {

bool finite_vec(const cv::Vec3d& value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool finite_rotation(const cv::Matx33d& value) {
  for (double element : value.val) {
    if (!std::isfinite(element)) return false;
  }
  return true;
}

bool proper_rotation(const cv::Matx33d& value) {
  if (!finite_rotation(value)) return false;
  const cv::Matx33d difference = value.t() * value - cv::Matx33d::eye();
  return cv::norm(cv::Mat(difference), cv::NORM_INF) <= 1e-9 &&
         std::abs(cv::determinant(cv::Mat(value)) - 1.0) <= 1e-9;
}

std::string validate_calibration(const CameraCalibration& calibration) {
  const cv::Mat& matrix = calibration.matrix;
  const cv::Mat& distortion = calibration.distortion;
  const auto floating_scalar = [](const cv::Mat& value) {
    return value.channels() == 1 && (value.depth() == CV_32F || value.depth() == CV_64F);
  };
  if (matrix.dims != 2 || matrix.rows != 3 || matrix.cols != 3 || !floating_scalar(matrix) ||
      !cv::checkRange(matrix, true)) return "camera matrix must be finite 3x3 float/double";
  if (calibration.width == 0 || calibration.height == 0) return "image resolution is unknown";
  cv::Mat camera;
  matrix.convertTo(camera, CV_64F);
  if (camera.at<double>(0, 0) <= 0 || camera.at<double>(1, 1) <= 0 ||
      std::abs(camera.at<double>(0, 1)) > 1e-12 ||
      std::abs(camera.at<double>(1, 0)) > 1e-12 ||
      std::abs(camera.at<double>(2, 0)) > 1e-12 ||
      std::abs(camera.at<double>(2, 1)) > 1e-12 ||
      std::abs(camera.at<double>(2, 2) - 1.0) > 1e-12)
    return "camera matrix is not a supported pinhole intrinsic matrix";
  if (distortion.dims != 2 || (distortion.rows != 1 && distortion.cols != 1) ||
      !floating_scalar(distortion) || !cv::checkRange(distortion, true))
    return "distortion must be an explicit finite float/double vector";
  const auto n = distortion.total();
  if (n != 4 && n != 5 && n != 8 && n != 12 && n != 14)
    return "unsupported distortion coefficient count";
  return {};
}

std::string validate_geometry(const PnpInput& input) {
  const bool fixed = (input.source == PnpInputSource::OracleProjected ||
                      input.source == PnpInputSource::VerifiedPhysicalKeypoints) && input.correspondence_verified;
  const bool approximate = input.source == PnpInputSource::ApproximateLightbarEndpoints &&
                           input.approximate_pairing_declared && !input.correspondence_verified;
  if (!fixed && !approximate)
    return "physical correspondence or oracle pairing is unverified";
  if (input.object_length_unit != PnpLengthUnit::Metres || input.image_unit != PnpImageUnit::Pixels)
    return "object units must be metres and image units must be pixels";
  if ((input.model_size != PnpModelSize::Small && input.model_size != PnpModelSize::Large) ||
      input.model_name.empty() ||
      input.model_version.empty() || input.asset_sha256.size() != 64 || input.object_frame.empty() ||
      !std::all_of(input.asset_sha256.begin(), input.asset_sha256.end(),
                   [](unsigned char c) { return std::isxdigit(c) != 0; }))
    return "model size, provenance, version or object frame is missing";

  std::set<int> identities;
  cv::Mat object(4, 3, CV_64F), image(4, 2, CV_64F);
  cv::Vec3d object_center{};
  cv::Vec2d image_center{};
  for (int i = 0; i < 4; ++i) {
    const auto& point = input.points[static_cast<std::size_t>(i)];
    if (point.feature_id < 0 || !identities.insert(point.feature_id).second)
      return "feature identities must be distinct nonnegative model IDs";
    if (!std::isfinite(point.object_point_m.x) || !std::isfinite(point.object_point_m.y) ||
        !std::isfinite(point.object_point_m.z) || !std::isfinite(point.image_point_px.x) ||
        !std::isfinite(point.image_point_px.y)) return "point coordinates must be finite";
    object_center += cv::Vec3d(point.object_point_m.x, point.object_point_m.y, point.object_point_m.z);
    image_center += cv::Vec2d(point.image_point_px.x, point.image_point_px.y);
    for (int j = 0; j < i; ++j) {
      const auto& other = input.points[static_cast<std::size_t>(j)];
      if (cv::norm(point.object_point_m - other.object_point_m) <= 1e-12)
        return "duplicate object points";
      if (cv::norm(point.image_point_px - other.image_point_px) <= 1e-9)
        return "duplicate image points";
    }
  }
  object_center *= 0.25;
  image_center *= 0.25;
  for (int i = 0; i < 4; ++i) {
    const auto& p = input.points[static_cast<std::size_t>(i)];
    object.at<double>(i, 0) = p.object_point_m.x - object_center[0];
    object.at<double>(i, 1) = p.object_point_m.y - object_center[1];
    object.at<double>(i, 2) = p.object_point_m.z - object_center[2];
    image.at<double>(i, 0) = p.image_point_px.x - image_center[0];
    image.at<double>(i, 1) = p.image_point_px.y - image_center[1];
  }
  cv::Mat os, is;
  cv::SVD::compute(object, os);
  cv::SVD::compute(image, is);
  // 数值退化检查；阈值用于判断共线、共面。
  if (os.at<double>(0) <= 1e-12 || os.at<double>(1) <= os.at<double>(0) * 1e-8)
    return "object points are collinear or numerically degenerate";
  if (is.at<double>(0) <= 1e-9 || is.at<double>(1) <= is.at<double>(0) * 1e-8)
    return "image points are collinear or numerically degenerate";
  if (os.at<double>(2) > std::max(1e-12, os.at<double>(0) * 1e-8))
    return "object points are nonplanar; flatten explicitly before using IPPE";
  return {};
}

bool mat_to_vec3(const cv::Mat& input, cv::Vec3d& value) {
  if (input.empty() || input.channels() != 1 || input.total() != 3 ||
      (input.depth() != CV_32F && input.depth() != CV_64F)) return false;
  cv::Mat data;
  input.reshape(1, 3).convertTo(data, CV_64F);
  value = cv::Vec3d(data.at<double>(0), data.at<double>(1), data.at<double>(2));
  return true;
}
}  // namespace

std::string pnp_candidate_count_error(int reported_count, std::size_t rotation_count,
                                      std::size_t translation_count) {
  if (reported_count < 0 || static_cast<std::size_t>(reported_count) != rotation_count ||
      rotation_count != translation_count) return "inconsistent IPPE candidate counts";
  if (reported_count != 2) return "IPPE returned an unexpected number of candidates";
  return {};
}

PnpCandidate evaluate_pnp_candidate(const PnpInput& input, const CameraCalibration& calibration,
                                    const cv::Vec3d& rvec, const cv::Vec3d& tvec, int solver_index) {
  PnpCandidate result;
  result.solver_index = solver_index;
  result.rvec_camera = rvec;
  result.tvec_camera_m = tvec;
  if (!finite_vec(rvec) || !finite_vec(tvec)) {
    result.rejection_reason = "non-finite candidate pose";
    return result;
  }
  const std::string calibration_error = validate_calibration(calibration);
  if (!calibration_error.empty()) {
    result.rejection_reason = calibration_error;
    return result;
  }
  try {
    cv::Rodrigues(rvec, result.rotation_camera_object);
    if (!proper_rotation(result.rotation_camera_object)) {
      result.rejection_reason = "candidate rotation is not finite and orthonormal";
      return result;
    }
    result.finite = true;
    std::vector<cv::Point3d> object_points;
    object_points.reserve(4);
    result.positive_depth = true;
    bool has_zero_depth = false;
    for (const auto& point : input.points) {
      const cv::Vec3d object(point.object_point_m.x, point.object_point_m.y, point.object_point_m.z);
      if (!finite_vec(object) || !std::isfinite(point.image_point_px.x) ||
          !std::isfinite(point.image_point_px.y)) {
        result.finite = false;
        result.positive_depth = false;
        result.rejection_reason = "candidate evaluation received non-finite points";
        return result;
      }
      const cv::Vec3d camera = result.rotation_camera_object * object + tvec;
      if (!finite_vec(camera)) {
        result.finite = false;
        result.positive_depth = false;
        result.rejection_reason = "non-finite transformed point";
        return result;
      }
      result.positive_depth = result.positive_depth && camera[2] > 0.0;
      has_zero_depth = has_zero_depth || camera[2] == 0.0;
      object_points.push_back(point.object_point_m);
    }
    if (!result.positive_depth) {
      result.rejection_reason = "at least one object point has non-positive camera depth";
    }
    // 负深度候选保留代数残差供排查；零深度无法投影。
    if (has_zero_depth) {
      return result;
    }
    std::vector<cv::Point2d> projected;
    cv::projectPoints(object_points, rvec, tvec, calibration.matrix, calibration.distortion, projected);
    if (projected.size() != 4) {
      result.rejection_reason = "unexpected projection count";
      return result;
    }
    double squared_sum = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
      const cv::Point2d residual = projected[i] - input.points[i].image_point_px;
      if (!std::isfinite(residual.x) || !std::isfinite(residual.y)) {
        result.rejection_reason = "non-finite reprojection";
        return result;
      }
      result.reprojection_residuals_px[i] = residual;
      squared_sum += residual.dot(residual);
    }
    result.reprojection_rmse_px = std::sqrt(squared_sum / 4.0);
    result.reprojection_computed = std::isfinite(result.reprojection_rmse_px);
    if (!result.reprojection_computed) result.rejection_reason = "non-finite reprojection RMSE";
  } catch (const cv::Exception& error) {
    result.rejection_reason = std::string("OpenCV candidate evaluation: ") + error.what();
  }
  return result;
}

PnpResult solve_pnp_observation(const PnpInput& input, const CameraCalibration& calibration,
                                const PnpOptions& options) {
  PnpResult result;
  result.input = input;
  result.applied_reprojection_limit_px = options.max_reprojection_rmse_px;
  try {
    if (options.max_reprojection_rmse_px &&
        (!std::isfinite(*options.max_reprojection_rmse_px) || *options.max_reprojection_rmse_px <= 0.0)) {
      result.error = "caller reprojection limit must be finite and positive";
      return result;
    }
    result.error = validate_calibration(calibration);
    if (!result.error.empty()) return result;
    result.error = validate_geometry(input);
    if (!result.error.empty()) return result;
    result.geometry_checked = true;
    std::vector<cv::Point3d> objects;
    std::vector<cv::Point2d> images;
    for (const auto& point : input.points) {
      objects.push_back(point.object_point_m);
      images.push_back(point.image_point_px);
    }
    std::vector<cv::Mat> rotations, translations;
    result.solver_reported_count = cv::solvePnPGeneric(objects, images, calibration.matrix,
        calibration.distortion, rotations, translations, false, cv::SOLVEPNP_IPPE);
    result.solver_completed = true;
    result.error = pnp_candidate_count_error(result.solver_reported_count, rotations.size(), translations.size());
    const std::size_t count = std::max(rotations.size(), translations.size());
    for (std::size_t i = 0; i < count; ++i) {
      cv::Vec3d rvec, tvec;
      if (i >= rotations.size() || i >= translations.size() ||
          !mat_to_vec3(rotations[i], rvec) || !mat_to_vec3(translations[i], tvec)) {
        PnpCandidate invalid;
        invalid.solver_index = static_cast<int>(i);
        invalid.rejection_reason = "missing or malformed candidate vector";
        result.candidates.push_back(std::move(invalid));
      } else {
        result.candidates.push_back(evaluate_pnp_candidate(input, calibration, rvec, tvec, static_cast<int>(i)));
      }
    }
    std::vector<const PnpCandidate*> usable;
    result.fit_quality_assessed = options.max_reprojection_rmse_px.has_value();
    for (auto& candidate : result.candidates) {
      if (candidate.finite && candidate.positive_depth && candidate.reprojection_computed) {
        usable.push_back(&candidate);
        if (options.max_reprojection_rmse_px) {
          candidate.fit_quality_assessed = true;
          candidate.passes_caller_reprojection_limit =
              candidate.reprojection_rmse_px <= *options.max_reprojection_rmse_px;
          if (!candidate.passes_caller_reprojection_limit)
            candidate.rejection_reason = "reprojection exceeds caller-provided limit";
          result.candidate_passes_caller_reprojection_limit =
              result.candidate_passes_caller_reprojection_limit || candidate.passes_caller_reprojection_limit;
        }
      }
    }
    result.numerical_candidate_available = result.error.empty() && !usable.empty();
    if (usable.empty() && result.error.empty()) result.error = "no finite positive-depth candidate";
    if (result.fit_quality_assessed && !result.error.empty()) {
      result.candidate_passes_caller_reprojection_limit = false;
      result.fit_quality_reason = result.error;
    } else if (result.fit_quality_assessed && !result.candidate_passes_caller_reprojection_limit) {
      result.fit_quality_reason = usable.empty() ? "no numerical candidate to assess" :
                                                  "no candidate meets caller-provided reprojection limit";
    }
    if (usable.size() == 1) result.ambiguity = PnpAmbiguity::OneNumericallyUsableCandidate;
    if (usable.size() == 2) {
      const auto& a = *usable[0];
      const auto& b = *usable[1];
      const cv::Matx33d relative = a.rotation_camera_object.t() * b.rotation_camera_object;
      const double cosine = std::clamp((cv::trace(cv::Mat(relative))[0] - 1.0) / 2.0, -1.0, 1.0);
      result.rotation_difference_rad = std::acos(cosine);
      result.translation_difference_m = cv::norm(a.tvec_camera_m - b.tvec_camera_m);
      result.reprojection_rmse_difference_px = std::abs(a.reprojection_rmse_px - b.reprojection_rmse_px);
      // 这里只合并数值上近乎重合的解，明显不同的平面双解仍保留。
      result.ambiguity = *result.rotation_difference_rad <= 1e-6 &&
                         *result.translation_difference_m <= 1e-9
          ? PnpAmbiguity::NumericallyNearIdentical : PnpAmbiguity::MultipleUnresolved;
    }
  } catch (const cv::Exception& error) {
    result.error = std::string("OpenCV PnP: ") + error.what();
  }
  return result;
}

std::optional<PnpPose> marker_pose_to_center(const PnpPose& camera_from_marker,
    const cv::Matx33d& B_center_from_marker, const cv::Vec3d& c_center_m) {
  if (!proper_rotation(camera_from_marker.rotation_camera_object) ||
      !proper_rotation(B_center_from_marker) || !finite_vec(c_center_m) ||
      !finite_vec(camera_from_marker.translation_camera_object_m)) return std::nullopt;
  PnpPose result;
  result.rotation_camera_object = camera_from_marker.rotation_camera_object * B_center_from_marker.t();
  result.translation_camera_object_m = camera_from_marker.translation_camera_object_m -
                                       result.rotation_camera_object * c_center_m;
  return result;
}

// 第四讲：位姿、坐标与基础跟踪
//
// 输入格式：
//   armors     : const std::vector<Armor>&，第三讲输出的本帧装甲板观测。
//   timestamp  : TimePoint，本帧图像时刻；只能与其他 TimePoint 求时间差。
//   imu_q      : const Eigen::Quaterniond&，图像时刻的归一化姿态四元数。
//   calibration: const CameraCalibration&：
//                matrix=3x3 CV_64F 相机内参；distortion=1x5 CV_64F 畸变；
//                camera_to_gimbal=相机到云台的 3x3 旋转矩阵；
//                camera_to_gimbal_translation=相机到云台平移，单位 m。
//   state      : TrackingState&，仅本讲使用的跨帧状态；函数需要原地更新。
// 输出格式：
//   std::optional<Target>。有目标时：
//   - position : IMU 稳定参考系中、相对云台原点的目标向量，单位 m；
//   - velocity : 同一参考坐标系中的目标速度，单位 m/s。
//   本课程统一使用 x 向前、y 向左、z 向上的右手坐标系。
//   无有效观测、PnP 失败或跟踪未建立时返回 std::nullopt。
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration, TrackingState& state) {
  // ===== 实现区 BEGIN =====
  return update_lightbar_track(armors, timestamp, imu_q, calibration, state);
  // ===== 实现区 END =====
}
}  // namespace training
