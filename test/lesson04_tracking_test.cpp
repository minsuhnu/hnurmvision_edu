#include "lessons.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <stdexcept>
using namespace training;
namespace {
void check(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
TimePoint at(double t) {
  return TimePoint(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(t)));
}
Armor projected(const TrackingState& state, const CameraCalibration& camera, double x) {
  Armor a;
  std::vector<cv::Point3d> pts;
  for (auto& p : state.parameters.models[0]->prototype.points)
    pts.push_back(p.object_point_m);
  std::vector<cv::Point2d> uv;
  cv::projectPoints(pts, cv::Vec3d(-.3, .2, .01), cv::Vec3d(x, 0, 3), camera.matrix,
                    camera.distortion, uv);
  for (int i = 0; i < 4; ++i)
    a.corners[i] = uv[i];
  a.class_id = 1;
  a.confidence = 1;
  return a;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    check(argc == 2, "need lesson04.yaml");
    auto state = load_tracking_state(argv[1]);
    state.parameters.expected_model_normal_z.reset();
    state.parameters.mesh_refinement_enabled = false;  // 本测试投影固定模型点，因此关闭网格修正。
    CameraCalibration c;
    c.width = 1440;
    c.height = 1080;
    c.matrix = (cv::Mat_<double>(3, 3) << 1303.6753, 0, 720, 0, 1303.6753, 540, 0, 0, 1);
    c.distortion = cv::Mat::zeros(1, 5, CV_64F);
    for (int i = 0; i < 12; ++i) {
      const double t = 1 + i * .05;
      const auto a = projected(state, c, .2 * (t - 1));
      lesson04_tracking({a}, at(t), Eigen::Quaterniond::Identity(), c, state);
      check(state.current_pnp_observations.size() == 1,
            "PnP must follow selection of exactly one armor");
      check(!state.current_pnp_observations[0].physical_accuracy_verified,
            "approximation promoted to physical accuracy");
    }
    check(state.phase == TrackPhase::Tracking && state.velocity_valid, "CV track not established");
    check(std::abs(state.target->velocity.x() - .2) < 1e-5, "incorrect CV velocity");
    const auto pred = predict_target(state, 250);
    check(pred.has_value(), "prediction unavailable");
    check(std::abs((*pred - state.target->position).x() - .05) < 1e-5, "250ms conversion wrong");
    check(project_prediction(state, c, 250).size() == 4, "3D prediction not projected");
    auto generation = state.generation;
    auto time = state.last_call;
    auto position = state.target->position;
    check(!lesson04_tracking({projected(state, c, 0)}, time, Eigen::Quaterniond::Identity(), c,
                             state),
          "duplicate timestamp accepted");
    check(state.last_call == time && (state.target->position - position).norm() == 0,
          "duplicate mutated history");
    lesson04_tracking({}, at(1.60), Eigen::Quaterniond::Identity(), c, state);
    check(state.phase == TrackPhase::TempLost && !state.observed,
          "short disappearance not handled");
    lesson04_tracking({projected(state, c, .13)}, at(1.65), Eigen::Quaterniond::Identity(), c,
                      state);
    check(state.phase == TrackPhase::Tracking && state.generation == generation,
          "same-target recovery reset track");
    lesson04_tracking({}, at(2), Eigen::Quaterniond::Identity(), c, state);
    check(state.phase == TrackPhase::Lost && !state.target, "timeout retained stale target");
    lesson04_tracking({projected(state, c, .2)}, at(2.05), Eigen::Quaterniond::Identity(), c,
                      state);
    check(state.phase == TrackPhase::Detecting && !state.velocity_valid &&
              state.generation > generation,
          "reacquisition reused old velocity");
    auto bad = projected(state, c, .2);
    bad.corners[0].x = std::numeric_limits<float>::quiet_NaN();
    lesson04_tracking({bad}, at(2.1), Eigen::Quaterniond::Identity(), c, state);
    check(!state.observed && state.current_pnp_observations.empty(), "invalid points reached PnP");
    // 同类双目标：比较逐帧中心选板与连续关联的切换次数。
    int switches[2] = {0, 0};
    for (int mode = 0; mode < 2; ++mode) {
      auto s = load_tracking_state(argv[1]);
      s.parameters.expected_model_normal_z.reset();
      s.parameters.continuous_association = mode == 1;
      s.parameters.mesh_refinement_enabled = false;
      int last = -1;
      for (int i = 0; i < 8; ++i) {
        auto a = projected(s, c, -.04);
        auto b = projected(s, c, i == 0 ? .12 : 0.0);
        lesson04_tracking({a, b}, at(3 + i * .05), Eigen::Quaterniond::Identity(), c, s);
        check(s.selected.has_value(), "selection failed");
        int id = cv::norm(s.selected->corners[0] - a.corners[0]) < 1e-5 ? 0 : 1;
        if (last >= 0 && last != id)
          ++switches[mode];
        last = id;
      }
    }
    check(switches[0] == 1 && switches[1] == 0, "continuous association did not retain target");
    std::cout << "PASS CV/timestamps/loss/recovery/projection/approximation contract\n"
              << "controlled_two_armor_test center_only_switches=" << switches[0]
              << " associated_switches=" << switches[1] << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
