#include "realtime_processor.hpp"
#include <utility>
#include <cstdlib>
#include <fstream>
#include <iomanip>

namespace training {
namespace {
TrackingState realtime_tracking_state() {
  auto state =
      load_tracking_state(std::getenv("RM_TRACKING_CONFIG") ? std::getenv("RM_TRACKING_CONFIG")
                                                            : "configs/lesson04.yaml");
  const std::string backend = std::getenv("RM_DETECTOR") ? std::getenv("RM_DETECTOR") : "opencv";
  // Hybrid 沿用灯条几何；纯 Pose 关键点尚未与该网格模型建立对应。
  if (backend != "opencv" && backend != "hybrid")
    state.parameters.models = {};
  return state;
}
}  // namespace
RealtimeProcessor::RealtimeProcessor(CameraCalibration calibration)
    : calibration_(std::move(calibration)),
      tracking_(realtime_tracking_state()),
      thread_(&RealtimeProcessor::run, this) {}
RealtimeProcessor::~RealtimeProcessor() {
  stop();
}

void RealtimeProcessor::submit(std::uint64_t sequence, const SensorData& input) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return;
  if (pending_)
    ++overwritten_;
  pending_ = Job{sequence, input};  // 图像共享只读像素，任务持有引用。
  ready_.notify_one();
}

std::shared_ptr<const ProcessedFrame> RealtimeProcessor::latest() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_;
}

std::uint64_t RealtimeProcessor::overwritten() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return overwritten_;
}

void RealtimeProcessor::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    pending_.reset();
  }
  ready_.notify_one();
  if (thread_.joinable())
    thread_.join();
}

void RealtimeProcessor::run() {
  std::ofstream aim_log;
  if (const char* path = std::getenv("RM_AIM_LOG")) {
    aim_log.open(path, std::ios::app);
    aim_log << std::setprecision(12);
  }
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [&] { return stopping_ || pending_.has_value(); });
      if (stopping_)
        return;
      job = std::move(*pending_);
      pending_.reset();
    }
    auto result = std::make_shared<ProcessedFrame>();
    result->sequence = job.sequence;
    result->input = std::move(job.input);
    const auto started = Clock::now();
    try {
      result->armors = lesson03_detection(result->input.image);
      const auto target = lesson04_tracking(result->armors, result->input.timestamp,
                                            result->input.imu_q, calibration_, tracking_);
      result->tracking = tracking_;
      if (target && tracking_.observed && tracking_.velocity_valid) {
        result->command =
            lesson05_aiming(*target, result->input.timestamp, result->input.bullet_speed);
        result->command.shoot = result->command.control;
      }
    } catch (const std::exception& error) {
      result->error = error.what();
      result->armors.clear();
      result->command = {};
    }
    result->processing_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    if (aim_log.is_open()) {
      const auto& s = result->tracking;
      aim_log << "{\"t\":" << std::chrono::duration<double>(Clock::now().time_since_epoch()).count()
              << ",\"frame\":" << result->sequence << ",\"processing_ms\":" << result->processing_ms
              << ",\"age_ms\":" << std::chrono::duration<double,std::milli>(Clock::now()-result->input.timestamp).count()
              << ",\"observed\":" << s.observed << ",\"velocity_valid\":" << s.velocity_valid
              << ",\"control\":" << result->command.control << ",\"shoot_request\":" << result->command.shoot
              << ",\"yaw\":" << result->command.yaw << ",\"pitch\":" << result->command.pitch;
      if (s.target) {
        const auto& p=s.target->position; const auto& v=s.target->velocity;
        aim_log << ",\"position\":[" << p.x() << ',' << p.y() << ',' << p.z()
                << "],\"velocity\":[" << v.x() << ',' << v.y() << ',' << v.z() << ']';
      }
      aim_log << "}\n";
      if (result->sequence % 100 == 0) aim_log.flush();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      completed_ = std::move(result);
    }
  }
}
}  // namespace training
