#pragma once
#include "types.hpp"
#include <cmath>

namespace training {
inline bool gimbal_settled(const Command& current, const Command& previous,
                           double measured_yaw, double measured_pitch, double tolerance) {
  if (!std::isfinite(tolerance) || tolerance <= 0) return false;
  const auto near = [tolerance](double a, double b) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(std::remainder(a-b, 2*kPi)) < tolerance;
  };
  return near(measured_yaw, previous.yaw) && near(measured_pitch, previous.pitch) &&
         near(current.yaw, previous.yaw) && near(current.pitch, previous.pitch);
}
}  // namespace training
