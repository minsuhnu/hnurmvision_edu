#include "lessons.hpp"
#include "pose_history.hpp"
#include "ballistics.hpp"
#include "fire_control.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace training;

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  try {
    Command previous, current;
    previous.yaw=kPi-0.002;
    current.yaw=-kPi+0.002;
    check(gimbal_settled(current,previous,-kPi+0.001,0,0.01),
          "fire control angle wrap");
    check(!gimbal_settled(current,previous,0,0,0.01) &&
          !gimbal_settled(current,previous,-kPi,0.1,0.01) &&
          !gimbal_settled(current,previous,std::numeric_limits<double>::quiet_NaN(),0,0.01),
          "fire control invalid or unsettled angles");
    const double max_range=25.0*25.0/9.81;
    const auto limit=solve_ballistic(25,max_range,0);
    check(limit.solvable && std::abs(limit.theta-kPi/4)<1e-6,
          "maximum range tangent solution");
    check(!solve_ballistic(25,max_range+0.01,0).solvable &&
          !solve_ballistic(25,-1,0).solvable &&
          !solve_ballistic(std::numeric_limits<double>::infinity(),3,0).solvable,
          "ballistic reachability and invalid inputs");
    const auto base = Clock::now();
    const Eigen::Quaterniond left(Eigen::AngleAxisd(179*kPi/180, Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond right(Eigen::AngleAxisd(-179*kPi/180, Eigen::Vector3d::UnitZ()));
    const std::deque<TimedPose> poses{{base,left},{base+std::chrono::milliseconds(20),right}};
    const auto middle=pose_at(poses,base+std::chrono::milliseconds(10));
    check(middle && std::abs(((*middle)*Eigen::Vector3d::UnitX()).x()+1)<1e-10,
          "pose interpolation crossed angle wrap incorrectly");
    check(!pose_at(poses,base-std::chrono::seconds(1)) &&
          !pose_at(poses,base+std::chrono::seconds(1)), "stale pose accepted");
    auto aim = [](Eigen::Vector3d p, Eigen::Vector3d v, double delay, double speed = 25) {
      Target t;
      t.position = p;
      t.velocity = v;
      return lesson05_aiming(t, Clock::now() - std::chrono::duration_cast<Clock::duration>(
                                 std::chrono::duration<double>(delay)), speed);
    };
    for (double d : {0.1, 1.0, 3.0, 10.0}) {
      for (double h : {-0.5, 0.0, 0.5}) {
        const auto cmd = aim({d, 0, h}, {0, 0, 0}, 0.01);
        check(cmd.control && !cmd.shoot, "stationary solution");
        const double theta = 25 * kPi / 180 - cmd.pitch;
        const double t = d / (25 * std::cos(theta));
        check(std::abs(25 * std::sin(theta) * t - 4.905 * t * t - h) < 1e-8,
              "trajectory residual");
        check(t < 1, "low trajectory");
      }
    }
    const auto positive = aim({3, 0, 0}, {0, 1, 0}, 0.02);
    const auto negative = aim({3, 0, 0}, {0, -1, 0}, 0.02);
    const auto slower = aim({3, 0, 0}, {0, 0.5, 0}, 0.02);
    const auto later = aim({3, 0, 0}, {0, 1, 0}, 0.10);
    check(!aim({3,0,0},{0,0,0},-0.1).control &&
          !lesson05_aiming(Target{},TimePoint{},25).control,
          "future or missing observation time");
    check(!aim({0.1,0,0},{22.9,0,0},0.01).control,
          "iteration limit must reject a finite but slowly converging intercept");
    const double distance=3/std::cos(positive.yaw);
    const auto lateral_ballistic=solve_ballistic(25,distance,0);
    check(lateral_ballistic.solvable &&
          std::abs(3*std::tan(positive.yaw)-(0.02+lateral_ballistic.flight_time))<0.0011,
          "fixed point intercept residual");
    check(positive.control && negative.control && positive.yaw > slower.yaw && slower.yaw > 0 &&
              negative.yaw < 0 && later.yaw > positive.yaw,
          "motion lead");
    check(!aim({3, 0, 0}, {0, 0, 0}, 0, 0).control &&
              !aim({3, 0, 0}, {0, 0, 0}, 0, -1).control &&
              !aim({100, 0, 100}, {0, 0, 0}, 0, 5).control &&
              !aim({0, 0, 0}, {0, 0, 0}, 0).control &&
              !aim({1e-9, 0, 0}, {0, 0, 0}, 0).control &&
              !aim({3, 0, 0}, {10000, 0, 0}, 0).control &&
              !aim({std::numeric_limits<double>::quiet_NaN(), 0, 0}, {0, 0, 0}, 0).control &&
              !aim({3, 0, 0}, {0, std::numeric_limits<double>::infinity(), 0}, 0).control &&
              !aim({3, 0, 0}, {0, 0, 0}, 0.6).control,
          "invalid or unsolvable input");
    std::cout << "PASS ballistic residual, low arc, lead and invalid inputs\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
