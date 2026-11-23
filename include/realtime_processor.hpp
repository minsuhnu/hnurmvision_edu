#pragma once
#include "lessons.hpp"
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace training {
std::vector<Armor> lesson03_lightbars(const cv::Mat& image);

struct ProcessedFrame {
  std::uint64_t sequence = 0;
  SensorData input;
  std::vector<Armor> armors;
  TrackingState tracking;
  Command command;
  double processing_ms = 0;
  std::string error;
};

class RealtimeProcessor {
 public:
  explicit RealtimeProcessor(CameraCalibration calibration);
  ~RealtimeProcessor();
  void submit(std::uint64_t sequence, const SensorData& input);
  std::shared_ptr<const ProcessedFrame> latest() const;
  std::uint64_t overwritten() const;
  void stop();

 private:
  struct Job {
    std::uint64_t sequence;
    SensorData input;
  };
  void run();
  CameraCalibration calibration_;
  TrackingState tracking_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::optional<Job> pending_;
  std::shared_ptr<const ProcessedFrame> completed_;
  bool stopping_ = false;
  std::uint64_t overwritten_ = 0;
  std::thread thread_;
};
}  // namespace training
