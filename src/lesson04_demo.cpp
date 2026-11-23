// 读取仿真采集的 PNG/JSON，用于实时显示、回放和真值比较。
#include "lessons.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/SVD>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include <thread>
#include <deque>

namespace fs = std::filesystem;
using namespace training;
namespace {
struct Truth {
  Eigen::Vector3d p;
  Eigen::Matrix3d rotation;
  cv::Point2d uv;
};
struct Frame {
  YAML::Node metadata;
  cv::Mat image;
  CameraCalibration camera;
  TimePoint time;
  double t = 0;
  int scene = 0;
  std::map<int, Truth> truth;
};
Eigen::Matrix4d matrix(const YAML::Node& n) {
  if (n.size() != 4)
    throw std::runtime_error("invalid GT transform");
  Eigen::Matrix4d m;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      m(i, j) = n[i][j].as<double>();
  if (!m.allFinite())
    throw std::runtime_error("nonfinite GT transform");
  return m;
}
Eigen::Matrix3d rotation(Eigen::Matrix3d m) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d fix = Eigen::Matrix3d::Identity();
  fix(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
  return svd.matrixU() * fix * svd.matrixV().transpose();
}
const Eigen::Matrix3d A = (Eigen::Matrix3d() << 1, 0, 0, 0, 0, -1, 0, 1, 0).finished();
Frame load_frame(const fs::path& file, std::optional<Eigen::Vector3d>& origin) {
  Frame f;
  f.metadata = YAML::LoadFile(file.string());
  const auto& n = f.metadata;
  if (n["evidence_kind"].as<std::string>() != "actual_capture" ||
      !n["camera"]["extracted_view_compared"].as<bool>() ||
      !n["camera"]["copy_source_compared"].as<bool>())
    throw std::runtime_error("not an actual structurally bound capture");
  const auto source = file.parent_path().parent_path() / n["image"]["image_path"].as<std::string>();
  f.image = cv::imread(source.string());
  if (f.image.empty())
    throw std::runtime_error("missing capture PNG");
  auto c = n["camera"];
  f.camera.width = c["width"].as<unsigned>();
  f.camera.height = c["height"].as<unsigned>();
  if (f.image.cols != static_cast<int>(f.camera.width) ||
      f.image.rows != static_cast<int>(f.camera.height))
    throw std::runtime_error("PNG/calibration size mismatch");
  f.camera.matrix = cv::Mat(3, 3, CV_64F);
  f.camera.distortion = cv::Mat(1, 5, CV_64F);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      f.camera.matrix.at<double>(i, j) = c["K_actual_from_projection"][i][j].as<double>();
  for (int i = 0; i < 5; ++i)
    f.camera.distortion.at<double>(i) = c["D_rendered"][i].as<double>();
  const auto bc = matrix(c["T_B_from_C"]["matrix"]);
  if (!origin)
    origin = bc.block<3, 1>(0, 3);
  f.camera.camera_to_gimbal = A * rotation(bc.block<3, 3>(0, 0));
  f.camera.camera_to_gimbal_translation = A * (bc.block<3, 1>(0, 3) - *origin);
  const auto ns = n["capture_timestamp"]["fixed_time_ns"].as<std::int64_t>();
  f.time = TimePoint(std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(ns)));
  f.t = ns * 1e-9;
  f.scene = n["scene"]["selected_training_target_kind"].as<int>();
  const auto cb = bc.inverse().eval();
  for (const auto& target : n["targets"]) {
    auto identity = target["training_target"];
    if (!identity || identity.IsNull() || identity["kind"].as<int>() != f.scene)
      continue;
    const auto bs = matrix(target["T_B_from_S"]["matrix"]);
    const auto bo = matrix(target["T_B_from_O"]["matrix"]);
    const Eigen::Vector4d pc = cb * bo.col(3);
    if (pc.z() <= 0)
      continue;
    Truth gt;
    gt.p = A * (bs.block<3, 1>(0, 3) - *origin);
    gt.rotation = A * rotation(bs.block<3, 3>(0, 0));
    gt.uv = {f.camera.matrix.at<double>(0, 0) * pc.x() / pc.z() + f.camera.matrix.at<double>(0, 2),
             f.camera.matrix.at<double>(1, 1) * pc.y() / pc.z() + f.camera.matrix.at<double>(1, 2)};
    f.truth.emplace(target["armor_id"].as<int>(), gt);
  }
  return f;
}
struct Pending {
  double issued, due;
  int id, scene;
  std::size_t generation;
  Eigen::Vector3d p;
};
struct Evaluator {
  std::deque<Pending> pending;
  std::optional<Frame> previous;
  std::ofstream poses, predictions, candidates;
  std::vector<double> position_errors, rotation_errors, prediction_errors;
  std::optional<double> last_error;
  std::optional<Truth> current_gt;
  int frames = 0, observed = 0, tracked = 0, unmatched = 0, rejected_predictions = 0;
  explicit Evaluator(const std::string& prefix) {
    poses.open(prefix + "_poses.csv");
    predictions.open(prefix + "_predictions.csv");
    candidates.open(prefix + "_candidates.csv");
    if (!poses || !predictions || !candidates)
      throw std::runtime_error("cannot open evidence logs");
    candidates << std::setprecision(12)
               << "frame_uid,candidate_index,finite,positive_depth,rmse_px,rx,ry,rz,tx,ty,tz\n";
    poses
        << std::setprecision(12)
        << "frame_uid,time_s,scene,phase,observed,generation,gt_id,position_error_m,rotation_error_deg,est_x,est_y,est_z,gt_x,gt_y,gt_z,vx,vy,vz,pnp_candidates,raw_pnp_min_rmse_px,mesh_refinement_used,mesh_observable_rmse_px\n";
    predictions
        << std::setprecision(12)
        << "issued_s,due_s,lower_s,upper_s,id,scene,generation,pred_x,pred_y,pred_z,gt_x,gt_y,gt_z,error_m\n";
  }
  int associate_truth(const Frame& f, const TrackingState& s) {
    if (!s.observed || !s.selected)
      return -1;
    const auto& p = s.selected->corners;
    const auto middle = (p[0] + p[1] + p[2] + p[3]) * 0.25f;
    int id = -1;
    double best = 10, second = 1e9;
    for (const auto& [key, gt] : f.truth) {
      double d = cv::norm(gt.uv - cv::Point2d(middle));
      if (d < best) {
        second = best;
        best = d;
        id = key;
      } else
        second = std::min(second, d);
    }
    // 仅用于误差统计；匹配结果不回传给跟踪器。
    return second - best > 3 ? id : -1;
  }
  void update(const Frame& f, const TrackingState& s) {
    current_gt.reset();
    ++frames;
    observed += s.observed;
    tracked += s.phase == TrackPhase::Tracking;
    if (previous && f.scene != previous->scene) {
      rejected_predictions += pending.size();
      pending.clear();
      last_error.reset();
    }
    while (!pending.empty() && pending.front().due <= f.t) {
      const auto p = pending.front();
      pending.pop_front();
      if (!previous || previous->scene != p.scene || f.scene != p.scene || previous->t > p.due ||
          f.t - previous->t > 0.30 || !previous->truth.count(p.id) || !f.truth.count(p.id)) {
        ++rejected_predictions;
        continue;
      }
      const double alpha = (p.due - previous->t) / (f.t - previous->t);
      const Eigen::Vector3d gt =
          (1 - alpha) * previous->truth.at(p.id).p + alpha * f.truth.at(p.id).p;
      const double error = (gt - p.p).norm();
      prediction_errors.push_back(error);
      last_error = error;
      predictions << p.issued << ',' << p.due << ',' << previous->t << ',' << f.t << ',' << p.id
                  << ',' << p.scene << ',' << p.generation << ',' << p.p.x() << ',' << p.p.y()
                  << ',' << p.p.z() << ',' << gt.x() << ',' << gt.y() << ',' << gt.z() << ','
                  << error << '\n';
    }
    const int id = associate_truth(f, s);
    poses << f.metadata["frame_uid"].as<std::string>() << ',' << f.t << ',' << f.scene << ','
          << track_phase_name(s.phase) << ',' << s.observed << ',' << s.generation << ',' << id
          << ',';
    if (id >= 0 && s.target && s.center_pose_camera) {
      const auto& gt = f.truth.at(id);
      current_gt = gt;
      const double ep = (s.target->position - gt.p).norm();
      Eigen::Matrix3d rc;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          rc(i, j) = s.center_pose_camera->rotation_camera_object(i, j);
      const auto re = f.camera.camera_to_gimbal * rc;
      const double er =
          std::acos(std::clamp(((gt.rotation.transpose() * re).trace() - 1) * 0.5, -1.0, 1.0)) *
          180 / kPi;
      position_errors.push_back(ep);
      rotation_errors.push_back(er);
      const auto& p = s.target->position;
      const auto& v = s.target->velocity;
      poses << ep << ',' << er << ',' << p.x() << ',' << p.y() << ',' << p.z() << ',' << gt.p.x()
            << ',' << gt.p.y() << ',' << gt.p.z() << ',' << v.x() << ',' << v.y() << ',' << v.z()
            << ',';
      const auto pred = predict_target(s, s.parameters.prediction_horizon_ms);
      if (pred)
        pending.push_back({f.t, f.t + s.parameters.prediction_horizon_ms / 1000, id, f.scene,
                           s.generation, *pred});
    } else {
      ++unmatched;
      poses << ",,,,,,,,,,,";
    }
    if (!s.current_pnp_observations.empty()) {
      const auto& r = s.current_pnp_observations.front();
      double rmse = 1e9;
      for (auto& candidate : r.candidates) {
        if (candidate.finite && candidate.positive_depth)
          rmse = std::min(rmse, candidate.reprojection_rmse_px);
        candidates << f.metadata["frame_uid"].as<std::string>() << ',' << candidate.solver_index
                   << ',' << candidate.finite << ',' << candidate.positive_depth << ','
                   << candidate.reprojection_rmse_px;
        for (double v : candidate.rvec_camera.val)
          candidates << ',' << v;
        for (double v : candidate.tvec_camera_m.val)
          candidates << ',' << v;
        candidates << '\n';
      }
      poses << r.candidates.size() << ',' << rmse;
    } else
      poses << "0,";
    poses << ',' << s.mesh_refinement_used << ',';
    if (s.mesh_refinement_used)
      poses << s.mesh_reprojection_rmse_px;
    poses << '\n';
    poses.flush();
    predictions.flush();
    // 插值只需上一帧时间和真值，不保留图像与完整元数据。
    previous = f;
    previous->image.release();
    previous->metadata = YAML::Node{};
  }
  static void stats(std::ofstream& out, const std::string& key, const std::vector<double>& xs) {
    out << key << ":\n  count: " << xs.size() << '\n';
    if (xs.empty())
      return;
    double mean = 0, var = 0;
    for (auto x : xs)
      mean += x;
    mean /= xs.size();
    for (auto x : xs)
      var += (x - mean) * (x - mean);
    var /= xs.size();
    auto sorted = xs;
    std::sort(sorted.begin(), sorted.end());
    out << "  mean: " << mean << "\n  min: " << sorted.front() << "\n  max: " << sorted.back()
        << "\n  stddev: " << std::sqrt(var) << "\n  median: " << sorted[sorted.size() / 2] << '\n';
  }
  void finish(const std::string& prefix) {
    std::ofstream out(prefix + "_summary.yaml");
    out << std::setprecision(12);
    out << "frames: " << frames << "\nobserved: " << observed << "\ntracked: " << tracked
        << "\nunmatched_evaluation_frames: " << unmatched
        << "\nrejected_predictions: " << rejected_predictions
        << "\nnot_yet_due_at_end: " << pending.size() << '\n';
    stats(out, "position_error_m", position_errors);
    stats(out, "rotation_error_deg", rotation_errors);
    stats(out, "prediction_error_m", prediction_errors);
  }
};
void label(cv::Mat& image, const std::string& text, int y,
           const cv::Scalar& color = {235, 235, 235}) {
  cv::putText(image, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv::LINE_AA);
}
cv::Mat draw(const Frame& f, const TrackingState& s, const Evaluator& e) {
  cv::Mat output;
  cv::resize(f.image, output, {1280, 960});
  const double scale = 1280.0 / f.image.cols;
  if (s.observed && s.selected)
    for (int k = 0; k < 4; ++k)
      cv::line(output, s.selected->corners[k] * scale, s.selected->corners[(k + 1) % 4] * scale,
               {0, 255, 0}, 2, cv::LINE_AA);
  const auto predicted = project_prediction(s, f.camera, s.parameters.prediction_horizon_ms);
  if (predicted.size() == 4)
    for (int k = 0; k < 4; ++k)
      cv::line(output, predicted[k] * scale, predicted[(k + 1) % 4] * scale, {255, 100, 30}, 2,
               cv::LINE_AA);
  if (s.observed && s.selected)
    for (auto p : s.selected->corners)
      cv::circle(output, p * scale, 3, {0, 255, 0}, cv::FILLED, cv::LINE_AA);
  cv::rectangle(output, {8, 8, 1100, 296}, {20, 20, 20}, cv::FILLED);
  label(output,
        "LESSON 04 | " + std::string(f.scene == 0 ? "STATIC" : "TRANSLATION") +
            " | actual simulator capture",
        35, {0, 255, 0});
  label(output,
        cv::format("t=%.3f s   %s   track=%zu   Green=current / Blue=future", f.t,
                   track_phase_name(s.phase), s.generation),
        65);
  label(output,
        s.mesh_refinement_used
            ? "IPPE + mesh observable fit | upright mounting assumed | raw candidates retained"
            : "PnP: IPPE (both candidates retained), virtual lightbar model / approximate",
        95, {100, 220, 255});
  if (s.center_pose_camera) {
    const auto p = s.center_pose_camera->translation_camera_object_m;
    cv::Vec3d rv;
    cv::Rodrigues(s.center_pose_camera->rotation_camera_object, rv);
    label(output, cv::format("CENTER camera XYZ m: [%.4f, %.4f, %.4f]", p[0], p[1], p[2]), 125);
    label(output, cv::format("CENTER camera rvec rad: [%.3f, %.3f, %.3f]", rv[0], rv[1], rv[2]),
          153);
  }
  if (s.target)
    label(output,
          cv::format("Velocity W m/s: [%.3f, %.3f, %.3f]", s.target->velocity.x(),
                     s.target->velocity.y(), s.target->velocity.z()),
          182);
  if (e.current_gt) {
    const Eigen::Vector3d pc = f.camera.camera_to_gimbal.transpose() *
                               (e.current_gt->p - f.camera.camera_to_gimbal_translation);
    label(output, cv::format("GT CENTER camera XYZ m: [%.4f, %.4f, %.4f]", pc.x(), pc.y(), pc.z()),
          212);
    const Eigen::Matrix3d rc = f.camera.camera_to_gimbal.transpose() * e.current_gt->rotation;
    cv::Matx33d rotation_cv;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        rotation_cv(i, j) = rc(i, j);
    cv::Vec3d rv;
    cv::Rodrigues(rotation_cv, rv);
    label(output, cv::format("GT CENTER camera rvec rad: [%.3f, %.3f, %.3f]", rv[0], rv[1], rv[2]),
          242);
    label(output,
          cv::format("Independent GT errors: position %.2f cm / orientation %.2f deg",
                     e.position_errors.back() * 100, e.rotation_errors.back()),
          272);
  }
  cv::rectangle(output, {8, output.rows - 126, 1060, 118}, {20, 20, 20}, cv::FILLED);
  label(output,
        cv::format("Prediction horizon n = %.0f ms  |  evaluated after due time",
                   s.parameters.prediction_horizon_ms),
        output.rows - 94, {255, 150, 90});
  label(output,
        e.last_error ? cv::format("Prediction error = %.2f cm   matched comparisons: %zu",
                                  *e.last_error * 100, e.prediction_errors.size())
                     : "Prediction error: waiting for matching future GT (metres internally)",
        output.rows - 64, {255, 150, 90});
  label(output, "GT: same extraction packet; W=(Bevy X, -Z, Y), fixed first-camera origin",
        output.rows - 34);
  return output;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc < 4)
      throw std::runtime_error(
          "usage: lesson04_demo CAPTURE_SESSION MODEL_YAML OUTPUT_PREFIX [--watch] [--no-gui] [--center-only] [--horizon-ms=N]");
    cv::setNumThreads(2);
    bool watch = false, gui = true;
    auto s = load_tracking_state(argv[2]);
    for (int i = 4; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--watch")
        watch = true;
      else if (a == "--no-gui")
        gui = false;
      else if (a == "--center-only")
        s.parameters.continuous_association = false;
      else if (a.rfind("--horizon-ms=", 0) == 0)
        s.parameters.prediction_horizon_ms = std::stod(a.substr(13));
      else
        throw std::runtime_error("unknown option " + a);
    }
    if (!std::isfinite(s.parameters.prediction_horizon_ms) ||
        s.parameters.prediction_horizon_ms <= 0 || s.parameters.prediction_horizon_ms > 5000)
      throw std::runtime_error("prediction horizon must be in (0,5000] ms");
    const fs::path root = argv[1];
    const std::string prefix = argv[3];
    Evaluator evaluator(prefix);
    if (gui) {
      cv::namedWindow("Lesson04 acceptance", cv::WINDOW_AUTOSIZE);
      cv::moveWindow("Lesson04 acceptance", 0, 0);
      cv::Mat startup(960, 1280, CV_8UC3, cv::Scalar(20, 20, 20));
      label(startup, "Lesson 04 | Starting / waiting for actual simulator capture", 100);
      cv::imshow("Lesson04 acceptance", startup);
      cv::waitKey(1);
    }
    std::set<fs::path> done;
    std::optional<Eigen::Vector3d> origin;
    std::optional<int> last_scene;
    bool stop = false;
    while (!stop) {
      std::vector<fs::path> files;
      if (fs::exists(root / "metadata"))
        for (const auto& item : fs::directory_iterator(root / "metadata"))
          if (item.path().extension() == ".json" && !done.count(item.path()))
            files.push_back(item.path());
      std::sort(files.begin(), files.end());
      for (const auto& file : files) {
        // JSON 在 PNG 之后写入；实时读取时避开尚未写完的文件。
        if (watch && fs::file_time_type::clock::now() - fs::last_write_time(file) <
                         std::chrono::milliseconds(50))
          continue;
        auto f = load_frame(file, origin);
        if (last_scene && f.scene != *last_scene) {
          auto params = s.parameters;
          s = TrackingState{};
          s.parameters = params;
        }
        last_scene = f.scene;
        const auto detections = lesson03_detection(f.image);
        lesson04_tracking(detections, f.time, Eigen::Quaterniond::Identity(), f.camera, s);
        evaluator.update(f, s);
        auto output = draw(f, s, evaluator);
        done.insert(file);
        if (gui) {
          cv::imshow("Lesson04 acceptance", output);
          int key = cv::waitKey(watch ? 1 : 15);
          if (key == 'q' || key == 27)
            stop = true;
        }
        if (evaluator.frames == 1 || (f.scene == 1 && evaluator.frames % 40 == 0))
          cv::imwrite(prefix + "_" + std::to_string(evaluator.frames) + ".png", output);
        if (evaluator.frames % 25 == 0)
          std::cout << "processed=" << evaluator.frames << " phase=" << track_phase_name(s.phase)
                    << " prediction_evaluations=" << evaluator.prediction_errors.size()
                    << std::endl;
        if (stop)
          break;
      }
      if (!watch || fs::exists(prefix + ".stop"))
        break;
      if (gui) {
        int key = cv::waitKey(10);
        if (key == 'q' || key == 27)
          break;
      } else
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    evaluator.finish(prefix);
    if (gui)
      cv::destroyAllWindows();
    std::cout << "frames=" << evaluator.frames
              << " pose_comparisons=" << evaluator.position_errors.size()
              << " prediction_comparisons=" << evaluator.prediction_errors.size() << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Lesson04: " << e.what() << std::endl;
    return 1;
  }
}
