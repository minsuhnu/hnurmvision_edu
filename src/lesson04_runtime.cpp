#include "tracking_state.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <filesystem>

namespace training {
namespace {
double seconds(TimePoint a, TimePoint b) {
  return std::chrono::duration<double>(a - b).count();
}
cv::Point2f center(const Armor& a) {
  return (a.corners[0] + a.corners[1] + a.corners[2] + a.corners[3]) * 0.25f;
}
bool valid_armor(const Armor& a, const CameraCalibration& c) {
  for (auto p : a.corners)
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 || p.x >= c.width ||
        p.y >= c.height)
      return false;
  const std::vector<cv::Point2f> p(a.corners.begin(), a.corners.end());
  return cv::isContourConvex(p) && cv::contourArea(p) > 4 && a.corners[0].x < a.corners[1].x &&
         a.corners[3].x < a.corners[2].x && a.corners[0].y < a.corners[3].y &&
         a.corners[1].y < a.corners[2].y;
}
Eigen::Vector3d eigen(const cv::Vec3d& a) {
  return {a[0], a[1], a[2]};
}
cv::Vec3d cvvec(const Eigen::Vector3d& a) {
  return {a.x(), a.y(), a.z()};
}
void lose(TrackingState& s) {
  s.phase = TrackPhase::Lost;
  s.selected.reset();
  s.target.reset();
  s.history.clear();
  s.center_pose_camera.reset();
  s.model_pose_camera.reset();
  s.confirmations = 0;
  s.velocity_valid = false;
}
std::optional<Target> miss(TrackingState& s, TimePoint t) {
  if (s.phase == TrackPhase::Detecting || s.phase == TrackPhase::Lost ||
      seconds(t, s.last_seen) > s.parameters.temp_lost_s)
    lose(s);
  else
    s.phase = TrackPhase::TempLost;
  // 保留历史供重关联，本帧不输出观测。
  return std::nullopt;
}
struct MeshFit {
  PnpPose pose;
  double cost = 0, rmse = 0;
};
// PnP 后投影灯条网格，拟合与第三讲相同定义的长轴端点，修正轮廓拟合偏差。
std::optional<MeshFit> refine_mesh(const LightbarModel& model, const Armor& armor,
                                   const CameraCalibration& c,
                                   const Eigen::Matrix3d& reference_from_camera,
                                   const TrackingParameters& parameters, const PnpCandidate& seed) {
  if (model.lamp_meshes[0].empty() || model.lamp_meshes[1].empty())
    return std::nullopt;
  using Parameters = Eigen::Matrix<double, 6, 1>;
  using Residual = Eigen::Matrix<double, 10, 1>;
  Parameters p;
  for (int i = 0; i < 3; ++i) {
    p[i] = seed.rvec_camera[i];
    p[i + 3] = seed.tvec_camera_m[i];
  }
  const auto residual = [&](const Parameters& x) -> std::optional<Residual> {
    if (!x.allFinite() || x[5] <= 0.1)
      return std::nullopt;
    cv::Vec3d rv(x[0], x[1], x[2]), tv(x[3], x[4], x[5]);
    cv::Matx33d R;
    cv::Rodrigues(rv, R);
    struct Bar {
      cv::Point2f mid, top, bottom;
    };
    std::array<Bar, 2> bars;
    for (int side = 0; side < 2; ++side) {
      for (const auto& pt : model.lamp_meshes[side])
        if ((R * cv::Vec3d(pt.x, pt.y, pt.z) + tv)[2] <= 0.05)
          return std::nullopt;
      std::vector<cv::Point2d> projected;
      cv::projectPoints(model.lamp_meshes[side], rv, tv, c.matrix, c.distortion, projected);
      std::vector<cv::Point2f> uv;
      uv.reserve(projected.size());
      for (const auto& point : projected)
        uv.emplace_back(point);
      const auto box = cv::minAreaRect(uv);
      cv::Point2f vertices[4];
      box.points(vertices);
      auto d = vertices[1] - vertices[0];
      const auto other = vertices[2] - vertices[1];
      if (cv::norm(other) > cv::norm(d))
        d = other;
      const double length = cv::norm(d);
      if (length < 1e-6)
        return std::nullopt;
      d *= 1.0 / length;
      if (d.y < 0)
        d *= -1;
      bars[side] = {box.center, box.center - d * static_cast<float>(length * .5),
                    box.center + d * static_cast<float>(length * .5)};
    }
    if (bars[0].mid.x > bars[1].mid.x)
      std::swap(bars[0], bars[1]);
    const std::array<cv::Point2f, 4> points{bars[0].top, bars[1].top, bars[1].bottom,
                                            bars[0].bottom};
    Residual r = Residual::Zero();
    for (int i = 0; i < 4; ++i) {
      r[i * 2] = points[i].x - armor.corners[i].x;
      r[i * 2 + 1] = points[i].y - armor.corners[i].y;
    }
    if (parameters.expected_model_normal_z) {
      const Eigen::Vector3d model_x =
          reference_from_camera * Eigen::Vector3d(R(0, 0), R(1, 0), R(2, 0));
      const Eigen::Vector3d model_z =
          reference_from_camera * Eigen::Vector3d(R(0, 2), R(1, 2), R(2, 2));
      r[8] = parameters.mounting_constraint_weight_px * model_x.z();
      r[9] = parameters.mounting_constraint_weight_px *
             (model_z.z() - *parameters.expected_model_normal_z);
    }
    return r;
  };
  auto r = residual(p);
  if (!r)
    return std::nullopt;
  double lambda = .01;
  for (int iter = 0; iter < 20; ++iter) {
    Eigen::Matrix<double, 10, 6> J;
    bool valid = true;
    for (int k = 0; k < 6; ++k) {
      Parameters delta = Parameters::Zero();
      delta[k] = 1e-4;
      auto plus = residual(p + delta), minus = residual(p - delta);
      if (!plus || !minus) {
        valid = false;
        break;
      }
      J.col(k) = (*plus - *minus) / 2e-4;
    }
    if (!valid)
      break;
    const Eigen::Matrix<double, 6, 6> H = J.transpose() * J;
    Eigen::Matrix<double, 6, 6> regularized = H;
    for (int k = 0; k < 6; ++k)
      regularized(k, k) += lambda * std::max(1e-6, H(k, k));
    const Parameters step = regularized.ldlt().solve(-J.transpose() * (*r));
    if (!step.allFinite() || step.norm() < 1e-8)
      break;
    auto next = residual(p + step);
    if (next && next->squaredNorm() < r->squaredNorm()) {
      p += step;
      r = next;
      lambda = std::max(1e-8, lambda / 3);
    } else
      lambda = std::min(1e10, lambda * 5);
  }
  MeshFit fit;
  cv::Rodrigues(cv::Vec3d(p[0], p[1], p[2]), fit.pose.rotation_camera_object);
  fit.pose.translation_camera_object_m = {p[3], p[4], p[5]};
  fit.cost = r->squaredNorm();
  fit.rmse = std::sqrt(r->head<8>().squaredNorm() / 4);
  return fit;
}
}  // namespace

TrackingState load_tracking_state(const std::string& path) {
  const auto y = YAML::LoadFile(path);
  YAML::Node meshes;
  if (y["lamp_mesh_file"])
    meshes = YAML::LoadFile(
        (std::filesystem::path(path).parent_path() / y["lamp_mesh_file"].as<std::string>())
            .string());
  TrackingState s;
  auto& p = s.parameters;
  p.prediction_horizon_ms = y["prediction_horizon_ms"].as<double>();
  p.reprojection_limit_px = y["reprojection_limit_px"].as<double>();
  p.association_gate_px = y["association_gate_px"].as<double>();
  p.max_speed_mps = y["max_speed_mps"].as<double>();
  p.max_gap_s = y["max_gap_s"].as<double>();
  p.temp_lost_s = y["temp_lost_s"].as<double>();
  p.velocity_window_s = y["velocity_window_s"].as<double>();
  p.confirm_frames = y["confirm_frames"].as<int>();
  p.mesh_refinement_enabled = y["mesh_refinement_enabled"].as<bool>(false);
  if (y["expected_model_normal_z"]) {
    p.expected_model_normal_z = y["expected_model_normal_z"].as<double>();
    p.normal_prior_weight_px = y["normal_prior_weight_px"].as<double>();
    if (!std::isfinite(*p.expected_model_normal_z) || std::abs(*p.expected_model_normal_z) > 1 ||
        !std::isfinite(p.normal_prior_weight_px) || p.normal_prior_weight_px < 0)
      throw std::runtime_error("invalid mounting prior");
  }
  for (double v : {p.prediction_horizon_ms, p.reprojection_limit_px, p.association_gate_px,
                   p.max_speed_mps, p.max_gap_s, p.temp_lost_s, p.velocity_window_s})
    if (!std::isfinite(v) || v <= 0)
      throw std::runtime_error("invalid lesson04 parameter");
  if (p.prediction_horizon_ms > 5000 || p.confirm_frames < 1 || p.confirm_frames > 100)
    throw std::runtime_error("invalid lesson04 horizon or confirmation count");
  for (int size = 0; size < 2; ++size) {
    auto m = y["models"][size == 0 ? "small" : "large"];
    if (!m)
      continue;
    LightbarModel model;
    auto& in = model.prototype;
    in.source = PnpInputSource::ApproximateLightbarEndpoints;
    in.approximate_pairing_declared = true;
    in.model_size = size == 0 ? PnpModelSize::Small : PnpModelSize::Large;
    in.object_length_unit = PnpLengthUnit::Metres;
    in.image_unit = PnpImageUnit::Pixels;
    in.model_name = m["name"].as<std::string>();
    in.model_version = m["version"].as<std::string>();
    in.asset_sha256 = m["asset_sha256"].as<std::string>();
    in.object_frame = "virtual_lightbar_endpoint_plane";
    if (m["points_m"].size() != 4 || m["center_from_model_rotation"].size() != 9 ||
        m["center_from_model_translation_m"].size() != 3)
      throw std::runtime_error("invalid lightbar model dimensions");
    for (int i = 0; i < 4; ++i) {
      const auto xyz = m["points_m"][i].as<std::vector<double>>();
      if (xyz.size() != 3)
        throw std::runtime_error("invalid model point");
      in.points[i].object_point_m = {xyz[0], xyz[1], xyz[2]};
      in.points[i].feature_id = i;
    }
    for (int i = 0; i < 9; ++i)
      model.center_from_model_rotation.val[i] = m["center_from_model_rotation"][i].as<double>();
    for (int i = 0; i < 3; ++i)
      model.center_from_model_translation_m[i] =
          m["center_from_model_translation_m"][i].as<double>();
    if (!marker_pose_to_center(PnpPose{}, model.center_from_model_rotation,
                               model.center_from_model_translation_m))
      throw std::runtime_error("invalid lightbar to CENTER transform");
    p.models[size] = model;
    const auto vertices = meshes ? meshes[size == 0 ? "small" : "large"] : m["lamp_meshes_m"];
    if (vertices) {
      if (vertices.size() != 2)
        throw std::runtime_error("need exactly two lamp meshes");
      for (int side = 0; side < 2; ++side)
        for (const auto& xyz : vertices[side]) {
          if (xyz.size() != 3)
            throw std::runtime_error("invalid mesh vertex");
          cv::Point3d point(xyz[0].as<double>(), xyz[1].as<double>(), xyz[2].as<double>());
          if (!std::isfinite(point.x + point.y + point.z))
            throw std::runtime_error("nonfinite mesh vertex");
          p.models[size]->lamp_meshes[side].push_back(point);
        }
    }
  }
  return s;
}

const char* track_phase_name(TrackPhase p) {
  switch (p) {
    case TrackPhase::Lost:
      return "LOST";
    case TrackPhase::Detecting:
      return "DETECTING";
    case TrackPhase::Tracking:
      return "TRACKING";
    case TrackPhase::TempLost:
      return "TEMP_LOST";
  }
  return "INVALID";
}

std::optional<Target> update_lightbar_track(const std::vector<Armor>& armors, TimePoint t,
                                            const Eigen::Quaterniond& imu,
                                            const CameraCalibration& c, TrackingState& s) {
  s.current_pnp_observations.clear();
  s.observed = false;
  s.diagnostic.clear();
  s.mesh_refinement_used = false;
  if (s.last_call != TimePoint{} && t <= s.last_call) {
    s.diagnostic = "non-increasing timestamp; frame ignored";
    return std::nullopt;
  }
  if (t == TimePoint{} || !imu.coeffs().allFinite() || imu.norm() < 1e-6 ||
      !c.camera_to_gimbal.allFinite() || !c.camera_to_gimbal_translation.allFinite() ||
      (c.camera_to_gimbal.transpose() * c.camera_to_gimbal - Eigen::Matrix3d::Identity()).norm() >
          1e-6 ||
      std::abs(c.camera_to_gimbal.determinant() - 1) > 1e-6) {
    lose(s);
    s.diagnostic = "invalid time or extrinsics";
    return std::nullopt;
  }
  if (s.last_call != TimePoint{} && seconds(t, s.last_call) > s.parameters.max_gap_s)
    lose(s);
  s.last_call = t;
  if (s.phase != TrackPhase::Lost && seconds(t, s.last_seen) > s.parameters.temp_lost_s)
    lose(s);
  // 先选板；已有目标时，将匀速预测投影到当前图像作为关联中心。
  const Eigen::Matrix3d R = imu.normalized().toRotationMatrix() * c.camera_to_gimbal;
  const Eigen::Vector3d offset = imu.normalized() * c.camera_to_gimbal_translation;
  cv::Point2f anchor(c.width * 0.5f, c.height * 0.5f);
  const bool locked = s.parameters.continuous_association && s.selected && s.target;
  if (locked) {
    const auto predicted = s.target->position + s.target->velocity * seconds(t, s.last_seen);
    const Eigen::Vector3d pc = R.transpose() * (predicted - offset);
    if (pc.z() > 0) {
      std::vector<cv::Point2d> uv;
      cv::projectPoints(std::vector<cv::Point3d>{{pc.x(), pc.y(), pc.z()}}, cv::Vec3d{},
                        cv::Vec3d{}, c.matrix, c.distortion, uv);
      anchor = cv::Point2f(uv[0]);
    } else
      anchor = center(*s.selected);
  }
  const Armor* chosen = nullptr;
  double best = std::numeric_limits<double>::infinity();
  for (const auto& a : armors) {
    if (!valid_armor(a, c))
      continue;
    if (locked &&
        (a.large_armor != s.selected->large_armor ||
         (a.class_id >= 0 && s.selected->class_id >= 0 && a.class_id != s.selected->class_id)))
      continue;
    const double d = cv::norm(center(a) - anchor);
    if (locked && d > s.parameters.association_gate_px)
      continue;
    if (d < best) {
      best = d;
      chosen = &a;
    }
  }
  if (!chosen) {
    s.diagnostic = "no associated detection";
    return miss(s, t);
  }
  const auto& model = s.parameters.models[chosen->large_armor ? 1 : 0];
  if (!model) {
    s.diagnostic = "no declared observation model";
    return miss(s, t);
  }
  auto in = model->prototype;
  in.timestamp = t;
  for (int i = 0; i < 4; ++i)
    in.points[i].image_point_px = chosen->corners[i];
  PnpOptions opt;
  opt.max_reprojection_rmse_px = s.parameters.reprojection_limit_px;
  auto result = solve_pnp_observation(in, c, opt);
  const PnpCandidate* selected = nullptr;
  double score = std::numeric_limits<double>::infinity();
  for (const auto& candidate : result.candidates) {
    if (!candidate.finite || !candidate.positive_depth ||
        !candidate.passes_caller_reprojection_limit)
      continue;
    // 用残差、安装角度和上一帧位置对候选排序。
    double rank = candidate.reprojection_rmse_px;
    if (s.parameters.expected_model_normal_z) {
      const Eigen::Vector3d normal = R * Eigen::Vector3d(candidate.rotation_camera_object(0, 2),
                                                         candidate.rotation_camera_object(1, 2),
                                                         candidate.rotation_camera_object(2, 2));
      rank += s.parameters.normal_prior_weight_px *
              std::abs(normal.z() - *s.parameters.expected_model_normal_z);
    }
    if (s.model_pose_camera && rank < 0.5)
      rank += 0.02 *
              cv::norm(candidate.tvec_camera_m - s.model_pose_camera->translation_camera_object_m);
    if (rank < score) {
      score = rank;
      selected = &candidate;
    }
  }
  if (!selected) {
    s.diagnostic = result.error.empty() ? result.fit_quality_reason : result.error;
    s.current_pnp_observations.push_back(std::move(result));
    return miss(s, t);
  }
  PnpPose model_pose{selected->rotation_camera_object, selected->tvec_camera_m};
  if (s.parameters.mesh_refinement_enabled) {
    std::optional<MeshFit> fit;
    for (const auto& candidate : result.candidates) {
      if (!candidate.finite || !candidate.positive_depth ||
          !candidate.passes_caller_reprojection_limit)
        continue;
      auto next = refine_mesh(*model, *chosen, c, R, s.parameters, candidate);
      if (next && (!fit || next->cost < fit->cost))
        fit = next;
    }
    if (!fit || fit->rmse > s.parameters.reprojection_limit_px) {
      s.diagnostic = "mesh observable refinement rejected";
      s.current_pnp_observations.push_back(std::move(result));
      return miss(s, t);
    }
    model_pose = fit->pose;
    s.mesh_refinement_used = true;
    s.mesh_reprojection_rmse_px = fit->rmse;
  }
  const auto center_pose = marker_pose_to_center(model_pose, model->center_from_model_rotation,
                                                 model->center_from_model_translation_m);
  s.current_pnp_observations.push_back(std::move(result));
  if (!center_pose) {
    s.diagnostic = "invalid CENTER conversion";
    return miss(s, t);
  }
  Target next;
  next.position = R * eigen(center_pose->translation_camera_object_m) + offset;
  if (!next.position.allFinite())
    return miss(s, t);
  const bool acquiring = s.phase == TrackPhase::Lost;
  if (!acquiring && s.target) {
    const double dt = seconds(t, s.last_seen);
    if (dt <= 1e-5 ||
        (next.position - s.target->position).norm() / dt > s.parameters.max_speed_mps) {
      s.diagnostic = "position/velocity innovation rejected";
      return miss(s, t);
    }
  }
  if (acquiring) {
    s.history.clear();
    s.confirmations = 0;
    s.velocity_valid = false;
    ++s.generation;
  }
  s.history.push_back({t, next.position});
  while (s.history.size() > 2 && seconds(t, s.history[1].time) > s.parameters.velocity_window_s)
    s.history.pop_front();
  // 短窗拟合 p(t)=a+v*t，降低 PnP 抖动对速度的影响；两点时即差分。
  s.velocity_valid = false;
  if (s.history.size() >= 2 && seconds(t, s.history.front().time) >= 0.02) {
    double mean_t = 0;
    Eigen::Vector3d mean_p = Eigen::Vector3d::Zero();
    for (auto& x : s.history) {
      mean_t += seconds(x.time, t);
      mean_p += x.position;
    }
    mean_t /= s.history.size();
    mean_p /= s.history.size();
    double den = 0;
    Eigen::Vector3d num = Eigen::Vector3d::Zero();
    for (auto& x : s.history) {
      double d = seconds(x.time, t) - mean_t;
      den += d * d;
      num += d * (x.position - mean_p);
    }
    if (den > 1e-10) {
      next.velocity = num / den;
      s.velocity_valid =
          next.velocity.allFinite() && next.velocity.norm() <= s.parameters.max_speed_mps;
    }
    if (!s.velocity_valid)
      next.velocity.setZero();
  }
  ++s.confirmations;
  s.phase =
      s.confirmations >= s.parameters.confirm_frames ? TrackPhase::Tracking : TrackPhase::Detecting;
  s.target = next;
  s.last_seen = t;
  s.selected = *chosen;
  s.center_pose_camera = center_pose;
  s.model_pose_camera = model_pose;
  s.reference_from_camera = R;
  s.camera_origin_reference = offset;
  s.observed = true;
  s.diagnostic = s.mesh_refinement_used
                     ? "PnP + statistical mesh observable fit; mounting assumed"
                     : "approximate lightbar model; planar alternatives retained";
  return s.phase == TrackPhase::Tracking ? std::optional<Target>(next) : std::nullopt;
}

std::optional<Eigen::Vector3d> predict_target(const TrackingState& s, double horizon_ms) {
  if (!std::isfinite(horizon_ms) || horizon_ms < 0 || horizon_ms > 5000 || !s.observed ||
      !s.target || !s.velocity_valid || s.phase != TrackPhase::Tracking)
    return std::nullopt;
  return s.target->position + s.target->velocity * (horizon_ms / 1000.0);
}
std::vector<cv::Point2d> project_prediction(const TrackingState& s, const CameraCalibration& c,
                                            double horizon_ms) {
  const auto pred = predict_target(s, horizon_ms);
  if (!pred || !s.selected || !s.model_pose_camera || !s.center_pose_camera)
    return {};
  const auto& model = s.parameters.models[s.selected->large_armor ? 1 : 0];
  if (!model)
    return {};
  const Eigen::Vector3d delta = s.reference_from_camera.transpose() * (*pred - s.target->position);
  const cv::Vec3d tv = s.model_pose_camera->translation_camera_object_m + cvvec(delta);
  std::vector<cv::Point3d> points;
  for (auto& p : model->prototype.points) {
    const auto pc = s.model_pose_camera->rotation_camera_object *
                        cv::Vec3d(p.object_point_m.x, p.object_point_m.y, p.object_point_m.z) +
                    tv;
    if (pc[2] <= 0)
      return {};
    points.push_back(p.object_point_m);
  }
  cv::Vec3d rv;
  cv::Rodrigues(s.model_pose_camera->rotation_camera_object, rv);
  std::vector<cv::Point2d> uv;
  cv::projectPoints(points, rv, tv, c.matrix, c.distortion, uv);
  return uv;
}
}  // namespace training
