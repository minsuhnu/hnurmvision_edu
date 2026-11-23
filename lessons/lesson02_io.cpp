
#include "io_context.hpp"
#include "fire_control.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cmath>
#include <exception>
#include <limits>
#include <thread>

namespace training::io {

namespace {

constexpr auto kFeedbackTimeout = std::chrono::milliseconds(500);
constexpr auto kReadWait = std::chrono::milliseconds(10);

bool finite_command(const Command& command) {
  return std::isfinite(command.yaw) &&
         std::isfinite(command.pitch) &&
         std::abs(command.yaw * 180.0 / kPi) <= std::numeric_limits<float>::max() &&
         std::abs(command.pitch * 180.0 / kPi) <= std::numeric_limits<float>::max();
}

}  // namespace

IoContext::~IoContext() {
  try {
    close();
  } catch (...) {
    // 析构时不允许异常继续传播。
  }
}

void IoContext::open() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (opened_) {
      return;
    }

    image_pending_ = false;
    latest_image_.reset();
    latest_feedback_.reset();
    last_feedback_time_ = TimePoint{};
    latest_image_time_ = TimePoint{};
    last_valid_image_time_ = TimePoint{};
    last_pose_log_ = TimePoint{};
    last_image_stamp_ns_ = 0;
    input_valid_ = false;
    previous_control_.reset();
    poses_.clear();
  }

  // 1. 初始化独立 ROS 2 Context。
  context_ = std::make_shared<rclcpp::Context>();
  context_->init(0, nullptr);

  // 2. 创建培训程序节点。
  rclcpp::NodeOptions node_options;
  node_options.context(context_);

  node_ = std::make_shared<rclcpp::Node>(
      "training_io", node_options);

  // 3. 传感器使用适合实时数据的 QoS。
  const auto qos = rclcpp::SensorDataQoS().keep_last(1);

  // 4. 订阅相机图像。
  image_sub_ =
      node_->create_subscription<sensor_msgs::msg::Image>(
          "/image_raw", qos,
          [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
            // 在接收时配对两种时钟，避免把 ROS epoch 当作 steady epoch。
            const auto received = Clock::now();
            auto captured = received;
            const std::int64_t stamp_ns =
                static_cast<std::int64_t>(msg->header.stamp.sec) * 1'000'000'000LL +
                msg->header.stamp.nanosec;
            if (stamp_ns > 0) {
              const auto age_ns = node_->now().nanoseconds() - stamp_ns;
              // 明显不同的时基退回接收时刻；正常时间戳保留传输延迟。
              if (std::abs(age_ns) < 5'000'000'000LL) {
                captured -= std::chrono::duration_cast<Clock::duration>(
                    std::chrono::nanoseconds(age_ns));
              } else {
                RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                                     "Image clock differs; using reception time.");
              }
            }
            {
              std::lock_guard<std::mutex> lock(mutex_);
              latest_image_time_ = captured;
              latest_image_ = std::move(msg);
              image_pending_ = true;
            }
            data_cv_.notify_one();
          });

  // 5. 订阅云台反馈。
  feedback_sub_ =
      node_->create_subscription<
          hnurm_interfaces::msg::VisionRecvData>(
          "/vision_receive_data", qos,
          [this](
              hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr
                  msg) {
            const auto received = Clock::now();
            auto sampled = received;
            const std::int64_t stamp_ns = static_cast<std::int64_t>(msg->header.stamp.sec) *
                                           1'000'000'000LL + msg->header.stamp.nanosec;
            const auto age_ns = node_->now().nanoseconds() - stamp_ns;
            if (stamp_ns > 0 && std::abs(age_ns) < 5'000'000'000LL)
              sampled -= std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(age_ns));
            {
              std::lock_guard<std::mutex> lock(mutex_);
              if (std::isfinite(msg->yaw) && std::isfinite(msg->pitch) && std::isfinite(msg->roll) &&
                  (poses_.empty() || sampled > poses_.back().time)) {
                const Eigen::Quaterniond q =
                    Eigen::AngleAxisd(msg->yaw * kPi / 180, Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(msg->pitch * kPi / 180, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(-msg->roll * kPi / 180, Eigen::Vector3d::UnitX());
                poses_.push_back({sampled, q.normalized()});
                while (poses_.size() > 2 && received - poses_.front().time > std::chrono::seconds(1))
                  poses_.pop_front();
              }
              latest_feedback_ = std::move(msg);
              last_feedback_time_ = received;
            }
            data_cv_.notify_one();
          });

  // 6. 创建控制指令发布器。
  command_pub_ =
      node_->create_publisher<
          hnurm_interfaces::msg::VisionSendData>(
          "/vision_send_data", qos);

  // 7. 使用独立线程执行回调。
  rclcpp::ExecutorOptions executor_options;
  executor_options.context = context_;

  executor_ =
      std::make_unique<
          rclcpp::executors::SingleThreadedExecutor>(
          executor_options);

  executor_->add_node(node_);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    opened_ = true;
  }

  spin_thread_ = std::thread([this]() {
    try {
      executor_->spin();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node_->get_logger(), "IO executor stopped: %s", e.what());
      std::lock_guard<std::mutex> lock(mutex_);
      opened_ = false;
      input_valid_ = false;
      data_cv_.notify_all();
    }
  });

  RCLCPP_INFO(
      node_->get_logger(),
      "IO opened: subscribed to image and feedback topics.");
}

std::optional<SensorData> IoContext::read() {
  sensor_msgs::msg::Image::ConstSharedPtr image;
  hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr feedback;

  TimePoint timestamp{};
  TimePoint feedback_time{};
  std::optional<Eigen::Quaterniond> image_pose;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    input_valid_ = false;
  }

  // 等待最多 10 ms，避免无数据时主循环持续占用 CPU。
  {
    std::unique_lock<std::mutex> lock(mutex_);

    data_cv_.wait_for(lock, kReadWait, [this]() {
      return !opened_ ||
             (image_pending_ &&
              latest_image_ &&
              latest_feedback_);
    });

    if (!opened_ ||
        !image_pending_ ||
        !latest_image_ ||
        !latest_feedback_) {
      return std::nullopt;
    }

    // 不使用过期反馈生成算法输入。
    if (Clock::now() - last_feedback_time_ >
        kFeedbackTimeout) {
      image_pending_ = false;
      return std::nullopt;
    }

    image = latest_image_;
    feedback = latest_feedback_;
    timestamp = latest_image_time_;
    feedback_time = last_feedback_time_;
    image_pending_ = false;
    if (image->header.stamp.sec < 0 || image->header.stamp.nanosec >= 1'000'000'000u)
      return std::nullopt;
    const std::int64_t stamp_ns =
        static_cast<std::int64_t>(image->header.stamp.sec) * 1'000'000'000LL +
        image->header.stamp.nanosec;
    if (stamp_ns != 0 && stamp_ns == last_image_stamp_ns_) return std::nullopt;
    last_image_stamp_ns_ = stamp_ns;
    const auto age = Clock::now() - timestamp;
    if (age > kFeedbackTimeout || age < -std::chrono::milliseconds(100))
      return std::nullopt;
    image_pose = pose_at(poses_, timestamp);
    if (!image_pose) return std::nullopt;
  }

  // 1. 检查图像编码。
  const auto& msg = *image;

  if (msg.width == 0 || msg.height == 0 ||
      msg.width > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
      msg.height > static_cast<unsigned>(std::numeric_limits<int>::max())) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000, "Empty camera image.");
    return std::nullopt;
  }

  const bool is_rgb = msg.encoding == "rgb8";
  const bool is_bgr = msg.encoding == "bgr8";
  const bool is_mono = msg.encoding == "mono8";

  if (!is_rgb && !is_bgr && !is_mono) {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 2000,
        "Unsupported image encoding: %s",
        msg.encoding.c_str());
    return std::nullopt;
  }

  const std::size_t channels = is_mono ? 1u : 3u;
  const std::size_t min_step =
      static_cast<std::size_t>(msg.width) * channels;

  if (msg.step < min_step ||
      msg.data.size() <
          static_cast<std::size_t>(msg.step) * msg.height) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000, "Invalid image buffer.");
    return std::nullopt;
  }

  // 2. 构造只读图像视图，再转换为独立拥有内存的 BGR 图像。
  cv::Mat source(
      static_cast<int>(msg.height),
      static_cast<int>(msg.width),
      is_mono ? CV_8UC1 : CV_8UC3,
      const_cast<unsigned char*>(msg.data.data()),
      static_cast<std::size_t>(msg.step));

  cv::Mat bgr;

  if (is_rgb) {
    cv::cvtColor(source, bgr, cv::COLOR_RGB2BGR);
  } else if (is_bgr) {
    bgr = source.clone();
  } else {
    cv::cvtColor(source, bgr, cv::COLOR_GRAY2BGR);
  }

  // 3. 检查姿态数据。
  if (!std::isfinite(feedback->roll) ||
      !std::isfinite(feedback->pitch) ||
      !std::isfinite(feedback->yaw) || feedback->bullet_speed.data == 0) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000, "Invalid gimbal angles or bullet speed.");
    return std::nullopt;
  }

  // 图像和反馈映射到同一时钟后，用曝光时刻的姿态抵消相机转动。
  SensorData result;
  result.image = std::move(bgr);
  result.timestamp = timestamp;
  result.imu_q = *image_pose;
  result.bullet_speed =
      static_cast<double>(feedback->bullet_speed.data);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    input_valid_ = true;
    last_valid_image_time_ = timestamp;
  }
  const auto now = Clock::now();
  if (now - last_pose_log_ >= std::chrono::seconds(1)) {
    RCLCPP_INFO(node_->get_logger(),
                "yaw=%.2f deg pitch=%.2f deg bullet_speed=%.1f m/s image_age=%.1f ms feedback_rx_delta=%.1f ms",
                feedback->yaw, feedback->pitch, result.bullet_speed,
                std::chrono::duration<double, std::milli>(now - timestamp).count(),
                std::chrono::duration<double, std::milli>(feedback_time - timestamp).count());
    last_pose_log_ = now;
  }
  return result;
}

void IoContext::write(const Command& command) {
  if (!node_ || !command_pub_) {
    return;
  }

  bool auto_aim_mode = false;
  bool feedback_fresh = false;
  bool valid_input = false;
  double measured_yaw = 0;
  double measured_pitch = 0;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    valid_input = opened_ && input_valid_ &&
        Clock::now() - last_valid_image_time_ <= kFeedbackTimeout;
    if (latest_feedback_) {
      measured_yaw = latest_feedback_->yaw * kPi / 180.0;
      measured_pitch = latest_feedback_->pitch * kPi / 180.0;
      auto_aim_mode =
          latest_feedback_->work_mode.data == hnurm_interfaces::msg::WorkMode::AUTO_AIM;

      feedback_fresh =
          Clock::now() - last_feedback_time_ <=
          kFeedbackTimeout &&
          std::isfinite(latest_feedback_->roll) &&
          std::isfinite(latest_feedback_->pitch) &&
          std::isfinite(latest_feedback_->yaw) &&
          latest_feedback_->bullet_speed.data > 0;
    }
  }

  // 最后一道安全门控：没有有效控制请求、
  // 反馈过期、模式不正确或角度非法时停止控制。
  const bool can_control =
      command.control &&
      valid_input &&
      auto_aim_mode &&
      feedback_fresh &&
      finite_command(command) && std::isfinite(command.target_distance) &&
      command.target_distance > 0 && command.target_distance <= std::numeric_limits<float>::max();

  const bool can_fire =
      can_control &&
      command.shoot &&
      config.allow_fire && previous_control_ &&
      gimbal_settled(command, *previous_control_, measured_yaw, measured_pitch,
                     config.fire_tolerance_rad);

  hnurm_interfaces::msg::VisionSendData msg;

  msg.header.stamp = node_->get_clock()->now();
  msg.header.frame_id = "gimbal_link";

  msg.target_type.data = 0;  // NONE
  msg.target_state.data =
      can_fire ? 2 : (can_control ? 1 : 0);

  if (can_control) {
    // Command 使用 rad，协议消息使用 degree。
    // 仿真器负责处理其内部 pitch 轴的方向差异。
    msg.yaw = static_cast<float>(
        command.yaw * 180.0 / kPi);

    msg.pitch = static_cast<float>(
        command.pitch * 180.0 / kPi);
  } else {
    msg.yaw = 0.0f;
    msg.pitch = 0.0f;
  }

  // 仿真器将 -1 视作停止，这里使用当前预测位置计算的距离。
  msg.target_distance = can_control ? static_cast<float>(command.target_distance) : 0.0f;

  msg.delta_pitch = 0.0f;
  msg.delta_yaw = 0.0f;
  msg.original_yout = 0.0f;

  msg.vel_x = 0.0f;
  msg.vel_y = 0.0f;
  msg.vel_yaw = 0.0f;
  msg.control_id = 0.0f;

  command_pub_->publish(msg);
  if (can_control) previous_control_ = command;
  else previous_control_.reset();
}

void IoContext::close() {
  if (!context_ && !node_ && !executor_) {
    return;
  }

  bool was_open = false;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    was_open = opened_;
  }

  // 先发布停止指令，再关闭通信。
  if (was_open && command_pub_) {
    try {
      write(Command{});
      RCLCPP_INFO(node_->get_logger(), "IO closing: STOP published (state=0 distance=0).");
      std::this_thread::sleep_for(
          std::chrono::milliseconds(30));
    } catch (const std::exception&) {
      // 继续释放资源。
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    opened_ = false;
    image_pending_ = false;
    input_valid_ = false;
  }

  data_cv_.notify_all();

  if (executor_) {
    executor_->cancel();
  }

  if (context_ && context_->is_valid()) {
    try {
      context_->shutdown("IoContext closing");
    } catch (const std::exception&) {
      // 继续清理资源。
    }
  }

  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }

  if (executor_ && node_) {
    try {
      executor_->remove_node(node_);
    } catch (const std::exception&) {
      // 节点可能已随 Executor 清理。
    }
  }

  image_sub_.reset();
  feedback_sub_.reset();
  command_pub_.reset();

  executor_.reset();
  node_.reset();
  context_.reset();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_image_.reset();
    latest_feedback_.reset();
    poses_.clear();
  }
  std::fprintf(stderr, "IO closed: executor joined; ROS resources released.\n");
}

}  // namespace training::io
