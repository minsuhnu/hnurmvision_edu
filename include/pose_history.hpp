#pragma once
#include "types.hpp"
#include <deque>
#include <optional>

namespace training {
struct TimedPose {
  TimePoint time;
  Eigen::Quaterniond rotation;
};

inline std::optional<Eigen::Quaterniond> pose_at(const std::deque<TimedPose>& poses,
                                               TimePoint time) {
  if (poses.empty()) return std::nullopt;
  constexpr auto boundary = std::chrono::milliseconds(30);
  if (time <= poses.front().time)
    return poses.front().time - time <= boundary
               ? std::optional<Eigen::Quaterniond>(poses.front().rotation) : std::nullopt;
  for (std::size_t i = 1; i < poses.size(); ++i) {
    if (time > poses[i].time) continue;
    const double span = std::chrono::duration<double>(poses[i].time - poses[i-1].time).count();
    if (span <= 0 || span > 0.1) return std::nullopt;
    const double fraction = std::chrono::duration<double>(time - poses[i-1].time).count() / span;
    return poses[i-1].rotation.slerp(fraction, poses[i].rotation).normalized();
  }
  return time - poses.back().time <= boundary
             ? std::optional<Eigen::Quaterniond>(poses.back().rotation) : std::nullopt;
}
}  // namespace training
