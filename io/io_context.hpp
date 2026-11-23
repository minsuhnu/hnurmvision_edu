
#pragma once

#include "config.hpp"
#include "types.hpp"
#include "pose_history.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>

namespace training::io {

struct IoContext {
  explicit IoContext(const Config& cfg) : config(cfg) {}
  ~IoContext();

  IoContext(const IoContext&) = delete;
  IoContext& operator=(const IoContext&) = delete;

  Config config;

  void open();
  std::optional<SensorData> read();
  void write(const Command& command);
  void close();

 private:
  std::shared_ptr<rclcpp::Context> context_;
  rclcpp::Node::SharedPtr node_;

  std::unique_ptr<
      rclcpp::executors::SingleThreadedExecutor> executor_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      image_sub_;

  rclcpp::Subscription<
      hnurm_interfaces::msg::VisionRecvData>::SharedPtr
      feedback_sub_;

  rclcpp::Publisher<
      hnurm_interfaces::msg::VisionSendData>::SharedPtr
      command_pub_;

  std::thread spin_thread_;

  std::mutex mutex_;
  std::condition_variable data_cv_;

  sensor_msgs::msg::Image::ConstSharedPtr latest_image_;
  hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr
      latest_feedback_;

  TimePoint last_feedback_time_{};
  TimePoint latest_image_time_{};
  TimePoint last_valid_image_time_{};
  TimePoint last_pose_log_{};
  std::int64_t last_image_stamp_ns_ = 0;
  bool input_valid_ = false;

  // open/read/write/close 由主线程顺序调用；回调共享状态均由 mutex_ 保护。

  // 一张图像只交给主循环处理一次。
  bool image_pending_ = false;
  bool opened_ = false;
  std::optional<Command> previous_control_;
  std::deque<TimedPose> poses_;
};

}  // namespace training::io
