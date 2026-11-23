#include "lessons.hpp"
#include "ballistics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace training {
BallisticSolution solve_ballistic(double speed_mps, double horizontal_m, double height_m) {
  if (!std::isfinite(speed_mps) || speed_mps <= 0 || !std::isfinite(horizontal_m) ||
      horizontal_m < 1e-6 || !std::isfinite(height_m)) return {};
  constexpr double gravity = 9.81;
  const double ratio = horizontal_m / speed_mps;
  const double a = gravity * ratio * ratio / 2;
  const double disc = horizontal_m * horizontal_m - 4 * a * (a + height_m);
  const double tolerance = 32 * std::numeric_limits<double>::epsilon() *
                           (horizontal_m * horizontal_m + std::abs(4 * a * (a + height_m)));
  if (!std::isfinite(a) || a <= 0 || !std::isfinite(disc) || disc < -tolerance) return {};
  const double root = std::sqrt(std::max(0.0, disc));
  // 有理化低伸解，避免近距离时两个相近数相减丢失精度。
  const double theta = std::atan(2 * (a + height_m) / (horizontal_m + root));
  const double flight = horizontal_m / (speed_mps * std::cos(theta));
  if (!std::isfinite(flight) || flight <= 0) return {};
  return {theta, flight, true};
}

// 第五讲：弹道解算
//
// 输入格式：
//   target       : const Target&；position 单位 m，velocity 单位 m/s，二者均采用
//                  第四讲规定的 x 向前、y 向左、z 向上坐标系。
//   timestamp    : TimePoint，target 对应的图像时刻，用于计算预测时间。
//   bullet_speed : double，弹丸初速度，单位 m/s，必须为有限正数。
// 输出格式：
//   Command：
//   - control : true 表示请求云台跟随；无法产生安全解时必须为 false；
//   - shoot   : true 表示请求发射，最终仍受 IO 层 allow_fire 和有效性门控；
//   - yaw     : 世界/IMU 参考系下的绝对偏航角，单位 rad；
//   - pitch   : 世界/IMU 参考系下的绝对俯仰角，单位 rad。
// 验收：仿真器完成静止靶、平移靶任务。
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed) {
  Command command;
  const double dt = std::chrono::duration<double>(Clock::now() - timestamp).count();
  if (timestamp == TimePoint{} || !std::isfinite(dt) || dt < 0 || dt > 0.5 ||
      !std::isfinite(bullet_speed) || bullet_speed <= 0 ||
      !target.position.allFinite() || !target.velocity.allFinite())
    return command;

  double flight = 0;
  for (int i = 0; i < 30; ++i) {
    const Eigen::Vector3d hit = target.position + target.velocity * (dt + flight);
    const double d = std::hypot(hit.x(), hit.y());
    if (!hit.allFinite() || !std::isfinite(d) || d < 1e-6)
      return {};
    const auto ballistic = solve_ballistic(bullet_speed, d, hit.z());
    if (!ballistic.solvable) return {};
    if (std::abs(ballistic.flight_time - flight) < 0.001) {
      const double distance = hit.norm();
      if (!std::isfinite(distance)) return {};
      command.control = true;
      command.yaw = std::atan2(hit.y(), hit.x());
      command.pitch = 25 * kPi / 180 - ballistic.theta;
      command.target_distance = distance;
      return command;
    }
    flight = ballistic.flight_time;
  }
  return {};
}
}  // namespace training
