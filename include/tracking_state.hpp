#pragma once

#include "types.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace training {

// X_camera = R_camera_object * X_object + t_camera_object。
// 物点和平移用米，像素点使用原图坐标；IPPE 要求物点共面。
enum class PnpInputSource {
  Unverified,
  OracleProjected,
  VerifiedPhysicalKeypoints,
  ApproximateLightbarEndpoints
};
enum class PnpModelSize { Unknown, Small, Large };
enum class PnpLengthUnit { Unknown, Metres };
enum class PnpImageUnit { Unknown, Pixels };
enum class PnpAmbiguity {
  NotAvailable,
  OneNumericallyUsableCandidate,
  NumericallyNearIdentical,
  MultipleUnresolved
};

struct PnpCorrespondence {
  cv::Point3d object_point_m{};
  cv::Point2d image_point_px{};
  int feature_id = -1;  // 模型点编号；近似模型使用虚拟端点编号。
};

struct PnpOptions {
  // 像素 RMSE 上限；不设置时只计算残差。
  std::optional<double> max_reprojection_rmse_px;
};

struct PnpInput {
  std::array<PnpCorrespondence, 4> points{};
  PnpInputSource source = PnpInputSource::Unverified;
  PnpModelSize model_size = PnpModelSize::Unknown;
  PnpLengthUnit object_length_unit = PnpLengthUnit::Unknown;
  PnpImageUnit image_unit = PnpImageUnit::Unknown;
  std::string model_name;
  std::string model_version;
  std::string asset_sha256;
  std::string object_frame;
  TimePoint timestamp{};
  // 固定物点与像素点的对应关系已确认。
  bool correspondence_verified = false;
  // 灯条轮廓端点采用近似模型，与固定物点对应分开记录。
  bool approximate_pairing_declared = false;
};

struct PnpCandidate {
  int solver_index = -1;
  cv::Vec3d rvec_camera{};
  cv::Vec3d tvec_camera_m{};
  cv::Matx33d rotation_camera_object = cv::Matx33d::eye();
  bool finite = false;
  bool positive_depth = false;
  bool reprojection_computed = false;
  bool fit_quality_assessed = false;
  bool passes_caller_reprojection_limit = false;
  std::array<cv::Point2d, 4> reprojection_residuals_px{};
  double reprojection_rmse_px = std::numeric_limits<double>::quiet_NaN();
  std::string rejection_reason;
};

struct PnpResult {
  PnpInput input;
  bool geometry_checked = false;
  bool solver_completed = false;
  bool numerical_candidate_available = false;
  bool fit_quality_assessed = false;
  bool candidate_passes_caller_reprojection_limit = false;
  std::optional<double> applied_reprojection_limit_px;
  std::string fit_quality_reason;
  // 独立真值实验的验证状态；求解器不会仅凭残差设置此值。
  bool physical_accuracy_verified = false;
  int solver_reported_count = 0;
  std::vector<PnpCandidate> candidates;
  PnpAmbiguity ambiguity = PnpAmbiguity::NotAvailable;
  std::optional<double> rotation_difference_rad;
  std::optional<double> translation_difference_m;
  std::optional<double> reprojection_rmse_difference_px;
  std::string error;
};

struct PnpPose {
  cv::Matx33d rotation_camera_object = cv::Matx33d::eye();
  cv::Vec3d translation_camera_object_m{};
};

// 返回全部 IPPE 候选及各自的深度、重投影残差。
PnpResult solve_pnp_observation(const PnpInput& input, const CameraCalibration& calibration,
                                const PnpOptions& options = {});

// 独立检查候选和返回数量，便于覆盖负深度等异常情况。
PnpCandidate evaluate_pnp_candidate(const PnpInput& input, const CameraCalibration& calibration,
                                    const cv::Vec3d& rvec, const cv::Vec3d& tvec, int solver_index);
std::string pnp_candidate_count_error(int reported_count, std::size_t rotation_count,
                                      std::size_t translation_count);

// X_center = B * X_marker + c；由 marker 位姿换算 CENTER 位姿。
std::optional<PnpPose> marker_pose_to_center(const PnpPose& camera_from_marker,
                                             const cv::Matx33d& B_center_from_marker,
                                             const cv::Vec3d& c_center_m);

enum class TrackPhase { Lost, Detecting, Tracking, TempLost };
struct LightbarModel {
  PnpInput prototype;
  cv::Matx33d center_from_model_rotation = cv::Matx33d::eye();
  cv::Vec3d center_from_model_translation_m{};
  std::array<std::vector<cv::Point3d>, 2> lamp_meshes;
};
struct TrackingParameters {
  double prediction_horizon_ms = 250;
  double reprojection_limit_px = 3;
  double association_gate_px = 100;
  double max_speed_mps = 8;
  double max_gap_s = 0.25;
  double temp_lost_s = 0.20;
  double velocity_window_s = 0.12;
  std::optional<double> expected_model_normal_z;
  double normal_prior_weight_px = 0;
  bool mesh_refinement_enabled = false;
  double mounting_constraint_weight_px = 20;
  int confirm_frames = 3;
  bool continuous_association = true;
  std::array<std::optional<LightbarModel>, 2> models;
};
struct TrackSample {
  TimePoint time;
  Eigen::Vector3d position;
};
struct TrackingState {
  TrackingParameters parameters;
  std::vector<PnpResult> current_pnp_observations;
  TrackPhase phase = TrackPhase::Lost;
  std::optional<Armor> selected;
  std::optional<PnpPose> center_pose_camera;
  std::optional<PnpPose> model_pose_camera;
  std::optional<Target> target;
  Eigen::Matrix3d reference_from_camera = Eigen::Matrix3d::Identity();
  Eigen::Vector3d camera_origin_reference = Eigen::Vector3d::Zero();
  std::deque<TrackSample> history;
  TimePoint last_call{}, last_seen{};
  std::size_t generation = 0;
  int confirmations = 0;
  bool observed = false;
  bool velocity_valid = false;
  bool mesh_refinement_used = false;
  double mesh_reprojection_rmse_px = std::numeric_limits<double>::quiet_NaN();
  std::string diagnostic;
};
TrackingState load_tracking_state(const std::string& path);
const char* track_phase_name(TrackPhase phase);
std::optional<Target> update_lightbar_track(const std::vector<Armor>& armors, TimePoint timestamp,
                                            const Eigen::Quaterniond& imu_q,
                                            const CameraCalibration& calibration,
                                            TrackingState& state);
std::optional<Eigen::Vector3d> predict_target(const TrackingState& state, double horizon_ms);
std::vector<cv::Point2d> project_prediction(const TrackingState& state,
                                            const CameraCalibration& calibration,
                                            double horizon_ms);
}  // namespace training
