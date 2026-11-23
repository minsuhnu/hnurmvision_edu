#include "pose_inference.hpp"
#include <opencv2/dnn.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace training {
namespace {
class PoseConnection {
 public:
  ~PoseConnection() { reset(); }
  std::vector<cv::Mat> infer(const std::string& path, const cv::Mat& blob) {
    if (blob.dims != 4 || blob.size[0] != 1 || blob.size[1] != 3 ||
        blob.size[2] != 640 || blob.size[3] != 640 || blob.type() != CV_32F || !blob.isContinuous())
      throw std::runtime_error("CUDA Pose input must be contiguous 1x3x640x640 float32");
    try {
      if (fd_ < 0) {
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (path.size() >= sizeof(address.sun_path)) throw std::runtime_error("Pose socket path is too long");
        std::memcpy(address.sun_path, path.c_str(), path.size()+1);
        fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd_ < 0) throw std::runtime_error("Cannot create Pose socket");
        timeval timeout{2, 0};
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        if (connect(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
          throw std::runtime_error("Cannot connect to CUDA Pose worker: " + std::string(std::strerror(errno)));
      }
      const std::uint32_t bytes = htonl(static_cast<std::uint32_t>(blob.total()*sizeof(float)));
      send_all("RMP1", 4);
      send_all(&bytes, sizeof(bytes));
      send_all(blob.data, blob.total()*sizeof(float));
      char magic[4];
      std::uint32_t status;
      std::uint64_t elapsed_bits;
      receive_all(magic, sizeof(magic));
      receive_all(&status, sizeof(status));
      receive_all(&elapsed_bits, sizeof(elapsed_bits));  // Server timing is retained by the server log.
      if (std::memcmp(magic, "RMR1", 4) != 0) throw std::runtime_error("Invalid CUDA Pose response");
      if (ntohl(status) != 0) {
        std::uint32_t length;
        receive_all(&length, sizeof(length));
        length = ntohl(length);
        if (length > 4096) throw std::runtime_error("Invalid CUDA Pose error length");
        std::string message(length, '\0');
        receive_all(message.data(), message.size());
        throw std::runtime_error("CUDA Pose worker: " + message);
      }
      std::vector<cv::Mat> outputs;
      for (int side : {80, 40, 20}) {
        const int sizes[]{1, 26, side, side};
        cv::Mat output(4, sizes, CV_32F);
        receive_all(output.data, output.total()*sizeof(float));
        outputs.push_back(std::move(output));
      }
      return outputs;
    } catch (...) {
      reset();
      throw;
    }
  }
 private:
  int fd_ = -1;
  void reset() { if (fd_ >= 0) close(fd_); fd_ = -1; }
  void send_all(const void* data, std::size_t size) {
    const auto* cursor = static_cast<const char*>(data);
    while (size) {
      const auto sent = send(fd_, cursor, size, MSG_NOSIGNAL);
      if (sent < 0 && errno == EINTR) continue;
      if (sent <= 0) throw std::runtime_error("CUDA Pose send failed");
      cursor += sent;
      size -= sent;
    }
  }
  void receive_all(void* data, std::size_t size) {
    auto* cursor = static_cast<char*>(data);
    while (size) {
      const auto received = recv(fd_, cursor, size, 0);
      if (received < 0 && errno == EINTR) continue;
      if (received <= 0) throw std::runtime_error("CUDA Pose response timed out or worker disconnected");
      cursor += received;
      size -= received;
    }
  }
};
}

std::vector<cv::Mat> pose_forward(const cv::Mat& blob) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (const char* socket = std::getenv("RM_POSE_SOCKET")) {
    if (*socket) {
      static PoseConnection connection;
      return connection.infer(socket, blob);
    }
  }
  static cv::dnn::Net network = [] {
    const char* path = std::getenv("RM_POSE_MODEL");
    if (!path || !*path) throw std::runtime_error("Pose mode needs RM_POSE_MODEL or RM_POSE_SOCKET");
    auto net = cv::dnn::readNetFromONNX(path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return net;
  }();
  network.setInput(blob);
  std::vector<cv::Mat> outputs;
  network.forward(outputs, std::vector<std::string>{"stride8", "stride16", "stride32"});
  return outputs;
}
}
