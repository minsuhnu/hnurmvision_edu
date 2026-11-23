#pragma once

namespace training {
struct BallisticSolution {
  double theta = 0;  // rad，抬头为正
  double flight_time = 0;  // s
  bool solvable = false;
};

BallisticSolution solve_ballistic(double speed_mps, double horizontal_m, double height_m);
}  // namespace training
