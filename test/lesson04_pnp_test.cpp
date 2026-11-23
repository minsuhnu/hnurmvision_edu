#include "lessons.hpp"
#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace training;

namespace {
struct Model {
  const char* name;
  const char* asset_sha256;
  PnpModelSize size;
  int marker_node;
  int position_accessor;
  std::array<cv::Point3d, 4> points;
  std::array<cv::Point3d, 4> raw_points;
  cv::Matx33d B;
  cv::Vec3d c;
  double max_planarization_m;
};
// 测试物点来自 GLB 的 MARKER 顶点；与检测器灯条轮廓端点使用的模型不同。
std::vector<Model> models() {
  return {
    {"small", "1cc0a3cd1ab05bc9822b616271db3afb64d078e56b9bbf452a8acc6d9bad0a6f", PnpModelSize::Small, 15, 54,
     {cv::Point3d(-0.066876361093695963, -0.026943659104384735, 0), cv::Point3d(-0.066894190426344327, 0.026941615183105342, 0), cv::Point3d(0.066891142695345532, -0.026943659104384739, 0), cv::Point3d(0.06687940882469473, 0.026945703025664135, 0)},
     {cv::Point3d(-0.066876361093695963, -0.026943659104384735, 1.7550270898382997e-06), cv::Point3d(-0.066894190426344327, 0.026941615183105342, -1.7549471094606669e-06), cv::Point3d(0.066891142695345532, -0.026943659104384739, -1.7548939662086279e-06), cv::Point3d(0.06687940882469473, 0.026945703025664135, 1.7548139858309952e-06)},
     cv::Matx33d(0.99999997562573839, -0.00022077888468832976, 2.2817929524915647e-06, -2.284591465763629e-06, -1.2675386869728496e-05, 0.99999999991705757, -0.0002207788557474093, -0.99999997554800912, -1.2675890950331079e-05),
     cv::Vec3d(3.7252902984619141e-09, 0.0027041137218473768, -6.301328539836358e-05), 1.7550270898382997e-06},
    {"large", "e10fe48e41bbd4ffd89e5fc8359fa0ba5df656ce178d750b750c30eb7e2d1a4c", PnpModelSize::Large, 13, 39,
     {cv::Point3d(-0.11437368019260262, -0.026943662290816104, 0), cv::Point3d(-0.11439681755808195, 0.026941615167533371, 0), cv::Point3d(0.11439377701804948, -0.026943662290816104, 0), cv::Point3d(0.11437672073263511, 0.026945709414098844, 0)},
     {cv::Point3d(-0.11437368019260262, -0.026943662290816104, 1.7552319770154939e-06), cv::Point3d(-0.11439681755808195, 0.026941615167533371, -1.7551853108646472e-06), cv::Point3d(0.11439377701804948, -0.026943662290816104, -1.7550986267601573e-06), cv::Point3d(0.11437672073263511, 0.026945709414098844, 1.7550519606075758e-06)},
     cv::Matx33d(0.99999999252346727, -0.00012209464840094909, 6.7795562856092047e-06, -6.7811072407035051e-06, -1.2702482199394915e-05, 0.99999999989633193, -0.00012209456227109885, -0.99999999246577187, -1.2703310041328714e-05),
     cv::Vec3d(-1.0809807080569556e-06, 0.0027046760522674001, 1.5802278048759888e-05), 1.7552319770154939e-06},
  };
}

struct Pose { cv::Matx33d rotation; cv::Vec3d translation; };
struct TestRunner {
  int passed = 0;
  int failed = 0;
  void run(const std::string& name, const std::function<void()>& test) {
    try { test(); ++passed; std::cout << "[PASS] " << name << '\n'; }
    catch (const std::exception& error) {
      ++failed;
      std::cout << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
};

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

CameraCalibration calibration_from_yaml(const std::string& path) {
  const auto yaml = YAML::LoadFile(path);
  const auto K = yaml["camera_matrix"].as<std::vector<double>>();
  const auto D = yaml["distort_coeffs"].as<std::vector<double>>();
  require(K.size() == 9 && D.size() == 5, "unexpected simulation configuration");
  CameraCalibration c;
  c.matrix = cv::Mat(K).reshape(1, 3).clone();
  c.distortion = cv::Mat(D).reshape(1, 1).clone();
  // Matches io/config.cpp load_calibration; not inferred from the test image.
  c.width = 1440;
  c.height = 1080;
  return c;
}

cv::Matx33d rotation(double yaw_deg, double pitch_deg, double roll_deg) {
  const double y = yaw_deg * kPi / 180.0, p = pitch_deg * kPi / 180.0, r = roll_deg * kPi / 180.0;
  const double sy = std::sin(y), cy = std::cos(y), sp = std::sin(p), cp = std::cos(p);
  const double sr = std::sin(r), cr = std::cos(r);
  return cv::Matx33d(cr, -sr, 0, sr, cr, 0, 0, 0, 1) *
         cv::Matx33d(cy, 0, sy, 0, 1, 0, -sy, 0, cy) *
         cv::Matx33d(1, 0, 0, 0, cp, -sp, 0, sp, cp);
}

double rotation_error_deg(const cv::Matx33d& expected, const cv::Matx33d& actual) {
  const cv::Matx33d delta = expected.t() * actual;
  const double cosine = std::clamp((cv::trace(cv::Mat(delta))[0] - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cosine) * 180.0 / kPi;
}

PnpInput projected_input(const Model& model, const Pose& pose, const CameraCalibration& camera) {
  PnpInput input;
  input.source = PnpInputSource::OracleProjected;
  input.model_size = model.size;
  input.object_length_unit = PnpLengthUnit::Metres;
  input.image_unit = PnpImageUnit::Pixels;
  input.model_name = std::string("MARKER_") + model.name;
  input.model_version = "part2b_planarized_node_" + std::to_string(model.marker_node) +
                        "_accessor_" + std::to_string(model.position_accessor);
  input.asset_sha256 = model.asset_sha256;
  input.object_frame = "MARKER_centroid_O";
  input.timestamp = TimePoint(std::chrono::milliseconds(1234));
  input.correspondence_verified = true;
  std::vector<cv::Point3d> points(model.points.begin(), model.points.end());
  cv::Vec3d rvec;
  cv::Rodrigues(pose.rotation, rvec);
  std::vector<cv::Point2d> pixels;
  cv::projectPoints(points, rvec, pose.translation, camera.matrix, camera.distortion, pixels);
  for (std::size_t i = 0; i < 4; ++i) {
    input.points[i] = {model.points[i], pixels[i], static_cast<int>(i)};
  }
  return input;
}

const PnpCandidate& best_by_reprojection(const PnpResult& result) {
  const PnpCandidate* best = nullptr;
  for (const auto& candidate : result.candidates) {
    if (candidate.finite && candidate.positive_depth && candidate.reprojection_computed &&
        (!best || candidate.reprojection_rmse_px < best->reprojection_rmse_px)) best = &candidate;
  }
  require(best != nullptr, "no numerically usable candidate: " + result.error);
  return *best;
}

void assert_rejected(const PnpInput& input, const CameraCalibration& calibration,
                     const std::string& error_piece) {
  const auto result = solve_pnp_observation(input, calibration);
  require(!result.solver_completed && !result.numerical_candidate_available,
          "invalid input reached solver or exposed a numerical candidate");
  require(result.error.find(error_piece) != std::string::npos, "unexpected rejection: " + result.error);
  require(!result.physical_accuracy_verified, "physical accuracy incorrectly asserted");
}

double percentile(std::vector<double> values, double q) {
  std::sort(values.begin(), values.end());
  const double index = q * (values.size() - 1);
  const auto lo = static_cast<std::size_t>(index);
  const auto hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (index - lo);
}

void run_normal_cases(TestRunner& runner, const std::vector<Model>& available,
                      const CameraCalibration& camera) {
  struct Case { const char* name; double yaw, pitch, roll, z; };
  const std::vector<Case> cases{
      {"front_1.5m", 0, 0, 0, 1.5}, {"front_3m", 0, 0, 0, 3}, {"front_6m", 0, 0, 0, 6},
      {"yaw_plus30", 30, 0, 0, 3}, {"yaw_minus30", -30, 0, 0, 3},
      {"pitch_plus20", 0, 20, 0, 3}, {"pitch_minus20", 0, -20, 0, 3},
      {"roll5", 0, 0, 5, 3}, {"combined", 25, 15, 5, 3}};
  for (const auto& model : available) for (const auto& c : cases) {
    runner.run(std::string("oracle_") + model.name + "_" + c.name, [&] {
      const Pose gt{rotation(c.yaw, c.pitch, c.roll), cv::Vec3d(0, 0, c.z)};
      const auto input = projected_input(model, gt, camera);
      const auto result = solve_pnp_observation(input, camera);
      require(result.error.empty(), result.error);
      require(result.geometry_checked && result.solver_completed && result.numerical_candidate_available,
              "solver did not produce usable candidates");
      require(result.solver_reported_count == 2 && result.candidates.size() == 2, "lost IPPE candidates");
      require(!result.physical_accuracy_verified && result.input.source == PnpInputSource::OracleProjected,
              "oracle was promoted to a physical-accuracy claim");
      require(!result.fit_quality_assessed && !result.candidate_passes_caller_reprojection_limit,
              "an unprovided pixel limit was treated as an accuracy criterion");
      require(result.input.timestamp == input.timestamp && result.input.points[3].feature_id == 3,
              "observation metadata was not retained");
      const auto& best = best_by_reprojection(result); // Selection never uses ground truth.
      const double et = cv::norm(best.tvec_camera_m - gt.translation);
      const double er = rotation_error_deg(gt.rotation, best.rotation_camera_object);
      std::cout << "[METRIC] " << model.name << '_' << c.name << " et_m=" << et
                << " er_deg=" << er << " rmse_px=" << best.reprojection_rmse_px
                << " candidates=" << result.candidates.size() << '\n';
      for (const auto& candidate : result.candidates) {
        std::cout << "[CANDIDATE] " << model.name << '_' << c.name << " id=" << candidate.solver_index
                  << " finite=" << candidate.finite << " positive=" << candidate.positive_depth
                  << " rmse_px=" << candidate.reprojection_rmse_px;
        if (candidate.finite) std::cout << " et_m=" << cv::norm(candidate.tvec_camera_m - gt.translation)
                                        << " er_deg=" << rotation_error_deg(gt.rotation, candidate.rotation_camera_object);
        std::cout << '\n';
      }
      require(et < 1e-5 && er < 0.001 && best.reprojection_rmse_px < 1e-4,
              "normal noiseless numerical accuracy outside test tolerance");
    });
  }
}

void run_noisy_case(TestRunner& runner, const Model& model, const CameraCalibration& camera) {
  runner.run(std::string("noise_") + model.name + "_seed_20261114", [&] {
    std::mt19937 generator(20261114);
    std::normal_distribution<double> noise(0, 0.5);
    const Pose gt{rotation(30, 10, 3), cv::Vec3d(0, 0, 3)};
    std::vector<double> translation_errors, rotation_errors, rmse;
    int alternate_closer_to_truth = 0;
    for (int trial = 0; trial < 32; ++trial) {
      auto input = projected_input(model, gt, camera);
      for (auto& point : input.points) { point.image_point_px.x += noise(generator); point.image_point_px.y += noise(generator); }
      const auto result = solve_pnp_observation(input, camera);
      require(result.numerical_candidate_available && result.candidates.size() == 2, "noisy solve lost candidates");
      require(!result.physical_accuracy_verified, "noise implied physical accuracy");
      const auto& chosen = best_by_reprojection(result);
      const double chosen_rotation = rotation_error_deg(gt.rotation, chosen.rotation_camera_object);
      double closest_rotation = chosen_rotation;
      for (const auto& candidate : result.candidates) {
        if (candidate.finite && candidate.positive_depth && candidate.reprojection_computed)
          closest_rotation = std::min(closest_rotation, rotation_error_deg(gt.rotation, candidate.rotation_camera_object));
      }
      if (closest_rotation + 1e-8 < chosen_rotation) ++alternate_closer_to_truth;
      std::cout << "[NOISE_TRIAL] model=" << model.name << " trial=" << trial
                << " selected_by_rmse=" << chosen.solver_index << " selected_rmse_px=" << chosen.reprojection_rmse_px
                << " et_m=" << cv::norm(chosen.tvec_camera_m - gt.translation)
                << " er_deg=" << chosen_rotation << " closest_candidate_er_deg=" << closest_rotation << '\n';
      translation_errors.push_back(cv::norm(chosen.tvec_camera_m - gt.translation));
      rotation_errors.push_back(chosen_rotation);
      rmse.push_back(chosen.reprojection_rmse_px);
    }
    std::cout << "[NOISE] " << model.name << " trials=32 sigma_px=0.5 seed=20261114"
              << " et_p50_m=" << percentile(translation_errors, 0.5)
              << " et_p95_m=" << percentile(translation_errors, 0.95)
              << " er_p50_deg=" << percentile(rotation_errors, 0.5)
              << " er_p95_deg=" << percentile(rotation_errors, 0.95)
              << " rmse_p50_px=" << percentile(rmse, 0.5)
              << " alternate_rotation_closer=" << alternate_closer_to_truth << '\n';
    // 带噪样本统计所有候选，不用真值替求解器选解。
  });
}

void run_rejection_cases(TestRunner& runner, const Model& model, const CameraCalibration& camera) {
  const Pose gt{rotation(30, 10, 3), cv::Vec3d(0, 0, 3)};
  const auto good = projected_input(model, gt, camera);
  runner.run("reject_unverified_source_even_if_boolean_true", [&] {
    auto x = good; x.source = PnpInputSource::Unverified; assert_rejected(x, camera, "unverified");
  });
  runner.run("reject_unverified_correspondence", [&] {
    auto x = good; x.correspondence_verified = false; assert_rejected(x, camera, "unverified");
  });
  runner.run("verified_source_is_declaration_not_accuracy", [&] {
    auto x = good; x.source = PnpInputSource::VerifiedPhysicalKeypoints;
    const auto r = solve_pnp_observation(x, camera);
    require(r.numerical_candidate_available && !r.physical_accuracy_verified,
            "declared physical correspondence promoted to accuracy verification");
  });
  runner.run("reject_unknown_model", [&] {
    auto x = good; x.model_size = PnpModelSize::Unknown; assert_rejected(x, camera, "model");
  });
  runner.run("reject_unspecified_coordinate_units", [&] {
    auto x = good; x.object_length_unit = PnpLengthUnit::Unknown; assert_rejected(x, camera, "units");
    x = good; x.image_unit = PnpImageUnit::Unknown; assert_rejected(x, camera, "units");
  });
  runner.run("reject_missing_provenance", [&] {
    auto x = good; x.asset_sha256.clear(); assert_rejected(x, camera, "provenance");
  });
  runner.run("reject_duplicate_feature_id", [&] {
    auto x = good; x.points[1].feature_id = x.points[0].feature_id; assert_rejected(x, camera, "identities");
  });
  runner.run("reject_duplicate_image_points", [&] {
    auto x = good; x.points[1].image_point_px = x.points[0].image_point_px; assert_rejected(x, camera, "duplicate image");
  });
  runner.run("reject_duplicate_object_points", [&] {
    auto x = good; x.points[1].object_point_m = x.points[0].object_point_m; assert_rejected(x, camera, "duplicate object");
  });
  runner.run("reject_collinear_image_points", [&] {
    auto x = good;
    for (int i = 0; i < 4; ++i) x.points[static_cast<std::size_t>(i)].image_point_px = cv::Point2d(700 + 10 * i, 500);
    assert_rejected(x, camera, "image points are collinear");
  });
  runner.run("reject_collinear_object_points", [&] {
    auto x = good;
    for (int i = 0; i < 4; ++i) x.points[static_cast<std::size_t>(i)].object_point_m = cv::Point3d(0.01 * i, 0, 0);
    assert_rejected(x, camera, "object points are collinear");
  });
  runner.run("reject_nan_point", [&] {
    auto x = good; x.points[0].image_point_px.x = std::numeric_limits<double>::quiet_NaN(); assert_rejected(x, camera, "finite");
  });
  runner.run("reject_inf_point", [&] {
    auto x = good; x.points[0].object_point_m.z = std::numeric_limits<double>::infinity(); assert_rejected(x, camera, "finite");
  });
  runner.run("reject_raw_nonplanar_marker", [&] {
    auto x = good;
    for (std::size_t i = 0; i < 4; ++i) x.points[i].object_point_m = model.raw_points[i];
    assert_rejected(x, camera, "nonplanar");
  });
  runner.run("reject_empty_camera_matrix", [&] {
    auto x = camera; x.matrix.release(); assert_rejected(good, x, "camera matrix");
  });
  runner.run("reject_wrong_camera_dimensions", [&] {
    auto x = camera; x.matrix = cv::Mat::eye(2, 2, CV_64F); assert_rejected(good, x, "3x3");
  });
  runner.run("reject_negative_focal_length", [&] {
    auto x = camera; x.matrix = camera.matrix.clone(); x.matrix.at<double>(0, 0) = -1; assert_rejected(good, x, "pinhole");
  });
  runner.run("reject_nan_intrinsics", [&] {
    auto x = camera; x.matrix = camera.matrix.clone(); x.matrix.at<double>(1, 1) = std::numeric_limits<double>::quiet_NaN(); assert_rejected(good, x, "finite");
  });
  runner.run("reject_unsupported_distortion", [&] {
    auto x = camera; x.distortion = cv::Mat::zeros(1, 3, CV_64F); assert_rejected(good, x, "coefficient count");
  });
  runner.run("reject_missing_distortion", [&] {
    auto x = camera; x.distortion.release(); assert_rejected(good, x, "distortion");
  });
  runner.run("reject_unknown_image_size", [&] {
    auto x = camera; x.width = 0; assert_rejected(good, x, "resolution");
  });
  runner.run("negative_depth_candidate_is_rejected", [&] {
    const auto candidate = evaluate_pnp_candidate(good, camera, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, -3), 0);
    require(candidate.finite && !candidate.positive_depth && candidate.reprojection_computed,
            "negative-depth candidate or its diagnostic residual was not retained");
    require(candidate.rejection_reason.find("depth") != std::string::npos, candidate.rejection_reason);
  });
  runner.run("nonfinite_candidate_is_rejected", [&] {
    const auto candidate = evaluate_pnp_candidate(good, camera, cv::Vec3d(0, 0, 0),
        cv::Vec3d(0, 0, std::numeric_limits<double>::quiet_NaN()), 0);
    require(!candidate.finite && !candidate.rejection_reason.empty(), "non-finite candidate accepted");
  });
  runner.run("candidate_count_anomalies", [&] {
    require(pnp_candidate_count_error(2, 2, 2).empty(), "normal count rejected");
    require(!pnp_candidate_count_error(0, 0, 0).empty(), "zero count accepted");
    require(!pnp_candidate_count_error(1, 1, 1).empty(), "one count accepted");
    require(!pnp_candidate_count_error(2, 2, 1).empty(), "inconsistent count accepted");
    require(!pnp_candidate_count_error(-1, 0, 0).empty(), "negative count accepted");
  });
  runner.run("invalid_caller_reprojection_limit_rejected", [&] {
    PnpOptions options;
    options.max_reprojection_rmse_px = std::numeric_limits<double>::quiet_NaN();
    const auto result = solve_pnp_observation(good, camera, options);
    require(!result.solver_completed && result.error.find("limit") != std::string::npos,
            "invalid caller-selected threshold was accepted");
  });
}

void run_model_and_reference_cases(TestRunner& runner, const std::vector<Model>& available,
                                   const CameraCalibration& camera) {
  for (const auto& model : available) {
    runner.run(std::string("marker_geometry_identity_") + model.name, [&] {
      require(std::string(model.asset_sha256).size() == 64, "missing asset SHA-256");
      double largest = 0;
      for (std::size_t i = 0; i < 4; ++i) {
        require(model.points[i].z == 0.0, "IPPE fixture is not planar");
        require(std::abs(model.raw_points[i].x - model.points[i].x) < 1e-15 &&
                std::abs(model.raw_points[i].y - model.points[i].y) < 1e-15, "flattening changed x/y");
        largest = std::max(largest, std::abs(model.raw_points[i].z));
      }
      require(std::abs(largest - model.max_planarization_m) < 1e-15, "raw planarization record differs");
      const auto orthogonal = model.B.t() * model.B - cv::Matx33d::eye();
      require(cv::norm(cv::Mat(orthogonal), cv::NORM_INF) < 1e-12 &&
              std::abs(cv::determinant(cv::Mat(model.B)) - 1.0) < 1e-12, "B is not a proper rotation");
      std::cout << "[MODEL] " << model.name << " SHA256=" << model.asset_sha256
                << " node=" << model.marker_node << " accessor=" << model.position_accessor
                << " planarization_max_m=" << largest << '\n';
    });
    runner.run(std::string("center_transform_roundtrip_") + model.name, [&] {
      const PnpPose co{rotation(20, -12, 7), cv::Vec3d(0.1, -0.2, 3)};
      const auto cs = marker_pose_to_center(co, model.B, model.c);
      require(cs.has_value(), "valid transform rejected");
      const cv::Matx33d back_rotation = cs->rotation_camera_object * model.B;
      const cv::Vec3d back_translation = cs->rotation_camera_object * model.c + cs->translation_camera_object_m;
      require(cv::norm(cv::Mat(back_rotation - co.rotation_camera_object), cv::NORM_INF) < 1e-12,
              "inverse rotation does not recover T_CO");
      require(cv::norm(back_translation - co.translation_camera_object_m) < 1e-12,
              "inverse translation does not recover T_CO");
      for (const auto& p : model.points) {
        const cv::Vec3d o(p.x, p.y, p.z);
        const cv::Vec3d s = model.B * o + model.c;
        const cv::Vec3d direct = co.rotation_camera_object * o + co.translation_camera_object_m;
        const cv::Vec3d via_center = cs->rotation_camera_object * s + cs->translation_camera_object_m;
        require(cv::norm(direct - via_center) < 1e-12, "same 3D point differs across reference frames");
      }
    });
    runner.run(std::string("reject_invalid_center_rotation_") + model.name, [&] {
      cv::Matx33d invalid = model.B; invalid(0, 0) += 0.01;
      require(!marker_pose_to_center(PnpPose{}, invalid, model.c), "invalid B was accepted");
    });
  }
  runner.run("symmetric_wrong_point_order_cannot_be_proven_by_rmse", [&] {
    const auto& m = available[0];
    Model rectangle = m;
    const double w = (cv::norm(m.points[2] - m.points[0]) + cv::norm(m.points[3] - m.points[1])) / 2;
    const double h = (cv::norm(m.points[1] - m.points[0]) + cv::norm(m.points[3] - m.points[2])) / 2;
    rectangle.points = {cv::Point3d(-w/2, -h/2, 0), cv::Point3d(-w/2, h/2, 0),
                        cv::Point3d(w/2, -h/2, 0), cv::Point3d(w/2, h/2, 0)};
    const Pose gt{rotation(25, 12, 4), cv::Vec3d(0, 0, 3)};
    auto input = projected_input(rectangle, gt, camera);
    const auto original = input.points;
    const std::array<int, 4> wrong{{3, 2, 1, 0}};
    for (std::size_t i = 0; i < 4; ++i) input.points[i].image_point_px = original[wrong[i]].image_point_px;
    const auto result = solve_pnp_observation(input, camera);
    const auto& best = best_by_reprojection(result);
    const double er = rotation_error_deg(gt.rotation, best.rotation_camera_object);
    std::cout << "[COUNTEREXAMPLE] wrong_order rmse_px=" << best.reprojection_rmse_px
              << " er_deg=" << er << '\n';
    require(best.reprojection_rmse_px < 1e-5 && er > 179.0, "symmetric correspondence demonstration changed");
    require(!result.physical_accuracy_verified, "wrong pairing incorrectly verified");
  });
  runner.run("wrong_scale_can_have_zero_reprojection_error", [&] {
    const auto& m = available[0];
    const Pose gt{rotation(30, 10, 3), cv::Vec3d(0, 0, 3)};
    auto input = projected_input(m, gt, camera);
    for (auto& p : input.points) p.object_point_m *= 0.9;
    const auto result = solve_pnp_observation(input, camera);
    const auto& best = best_by_reprojection(result);
    const double et = cv::norm(best.tvec_camera_m - gt.translation);
    std::cout << "[COUNTEREXAMPLE] wrong_scale rmse_px=" << best.reprojection_rmse_px << " et_m=" << et << '\n';
    require(best.reprojection_rmse_px < 1e-5 && std::abs(et - 0.3) < 1e-4,
            "scale ambiguity demonstration changed");
    require(!result.physical_accuracy_verified, "wrong scale incorrectly verified");
  });
  runner.run("mismatched_small_large_models_do_not_gain_accuracy_claim", [&] {
    const Pose gt{rotation(0, 10, 3), cv::Vec3d(0, 0, 3)};
    auto input = projected_input(available[1], gt, camera);
    for (std::size_t i = 0; i < 4; ++i) input.points[i].object_point_m = available[0].points[i];
    input.model_size = PnpModelSize::Small;
    input.asset_sha256 = available[0].asset_sha256;
    input.model_name = std::string("MARKER_") + available[0].name;
    input.model_version = "part2b_planarized_node_" + std::to_string(available[0].marker_node) +
                          "_accessor_" + std::to_string(available[0].position_accessor);
    const auto result = solve_pnp_observation(input, camera);
    const auto& best = best_by_reprojection(result);
    std::cout << "[COUNTEREXAMPLE] large_as_small rmse_px=" << best.reprojection_rmse_px
              << " et_m=" << cv::norm(best.tvec_camera_m - gt.translation) << '\n';
    require(!result.physical_accuracy_verified, "size mismatch incorrectly verified");
  });
}

void run_ambiguity_and_boundary_cases(TestRunner& runner, const Model& model,
                                      const CameraCalibration& camera) {
  runner.run(std::string("two_distinct_planar_candidates_remain_unresolved_") + model.name, [&] {
    const Pose gt{rotation(30, 0, 0), cv::Vec3d(0, 0, 3)};
    const auto result = solve_pnp_observation(projected_input(model, gt, camera), camera);
    require(result.candidates.size() == 2 && result.ambiguity == PnpAmbiguity::MultipleUnresolved,
            "distinct candidates not retained as unresolved");
    require(result.rotation_difference_rad && *result.rotation_difference_rad > 0.5 &&
            result.translation_difference_m && result.reprojection_rmse_difference_px,
            "candidate differences missing");
    std::cout << "[AMBIGUITY] rotation_delta_deg=" << *result.rotation_difference_rad * 180 / kPi
              << " translation_delta_m=" << *result.translation_difference_m
              << " rmse_delta_px=" << *result.reprojection_rmse_difference_px << '\n';
  });
  runner.run(std::string("nearly_identical_candidates_numerically_grouped_") + model.name, [&] {
    const Pose gt{rotation(0, 0, 0), cv::Vec3d(0, 0, 3)};
    const auto result = solve_pnp_observation(projected_input(model, gt, camera), camera);
    require(result.candidates.size() == 2 && result.rotation_difference_rad && result.translation_difference_m,
            "near-identical candidate differences missing");
    require(result.ambiguity == PnpAmbiguity::NumericallyNearIdentical,
            "near-identical numerical poses not distinguished from separated solutions");
  });
  for (const double angle : {179.0, 179.9, 179.999, 180.0, 180.001, 180.1, 181.0}) {
    runner.run(std::string("ippe_") + model.name + "_roll_" + std::to_string(angle) + "_preserves_all_candidates", [&] {
      const Pose gt{rotation(0, 0, angle), cv::Vec3d(0, 0, 3)};
      const auto input = projected_input(model, gt, camera);
      PnpOptions test_only_limit;
      test_only_limit.max_reprojection_rmse_px = 1e-3; // Noiseless synthetic regression only.
      const auto result = solve_pnp_observation(input, camera, test_only_limit);
      require(result.solver_completed && result.solver_reported_count == 2 && result.candidates.size() == 2,
              "IPPE candidates not preserved");
      require(!result.physical_accuracy_verified, "pressure case claimed verified accuracy");
      std::cout << "[BOUNDARY] model=" << model.name << " roll_deg=" << angle << " candidate_count=" << result.candidates.size()
                << " numerical_candidate=" << result.numerical_candidate_available
                << " passes_test_only_1e-3px_limit=" << result.candidate_passes_caller_reprojection_limit
                << " error=" << (result.error.empty() ? "none" : result.error);
      for (const auto& c : result.candidates) {
        std::cout << " {id=" << c.solver_index << ",finite=" << c.finite
                  << ",positive=" << c.positive_depth << ",rmse_px=" << c.reprojection_rmse_px;
        if (c.finite) std::cout << ",er_deg=" << rotation_error_deg(gt.rotation, c.rotation_camera_object);
        std::cout << '}';
      }
      std::cout << '\n';
      bool any_below_limit = false;
      for (const auto& c : result.candidates) {
        if (c.finite && c.positive_depth && c.reprojection_computed && c.reprojection_rmse_px <= 1e-3)
          any_below_limit = true;
      }
      require(result.fit_quality_assessed && result.candidate_passes_caller_reprojection_limit == any_below_limit,
              "caller test-only residual limit was not applied consistently");
      require(result.applied_reprojection_limit_px && *result.applied_reprojection_limit_px == 1e-3,
              "caller limit was not retained");
      if (!any_below_limit) require(!result.fit_quality_reason.empty(), "rejection reason missing");
      // 记录 IPPE 在接近 180° 时的异常输出。
    });
  }
}
}  // namespace

int main(int argc, char** argv) {
  cv::setNumThreads(1);
  std::cout << std::setprecision(12);
  TestRunner runner;
  try {
    require(argc == 2, "usage: lesson04_pnp_test /absolute/path/to/simulation.yaml");
    const auto camera = calibration_from_yaml(argv[1]);
    const auto available = models();
    std::cout << "[CONFIG] path=" << argv[1] << " fx=" << camera.matrix.at<double>(0, 0)
              << " fy=" << camera.matrix.at<double>(1, 1) << " resolution=" << camera.width << 'x' << camera.height
              << " opencv=" << CV_VERSION << '\n';
    run_normal_cases(runner, available, camera);
    for (const auto& m : available) run_noisy_case(runner, m, camera);
    run_rejection_cases(runner, available[0], camera);
    run_model_and_reference_cases(runner, available, camera);
    for (const auto& m : available) run_ambiguity_and_boundary_cases(runner, m, camera);
    runner.run("lesson04_keeps_unverified_armor_out_of_target", [&] {
      TrackingState state;
      state.current_pnp_observations.emplace_back();
      const auto target = lesson04_tracking(std::vector<Armor>(1), TimePoint{}, Eigen::Quaterniond::Identity(), camera, state);
      require(!target && state.current_pnp_observations.empty(), "unverified Armor was promoted or stale data retained");
    });
  } catch (const std::exception& error) {
    ++runner.failed; std::cout << "[FATAL] " << error.what() << '\n';
  }
  std::cout << "[SUMMARY] passed=" << runner.passed << " failed=" << runner.failed << '\n';
  return runner.failed == 0 ? 0 : 1;
}
