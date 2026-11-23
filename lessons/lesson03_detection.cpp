#include "lessons.hpp"
#include "pose_inference.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

namespace training {
namespace {
struct LightBar {
  cv::Point2f center;
  cv::Point2f top;
  cv::Point2f bottom;
  float length;
  cv::Point2f direction;
  int color;
};

struct ArmorCandidate {
  Armor armor;
  std::size_t left;
  std::size_t right;
};

std::vector<Armor> detect_opencv(const cv::Mat& image) {
  if (image.empty() || image.type() != CV_8UC3) return {};

  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  std::vector<LightBar> lights;
  for (int color = 0; color < 2; ++color) {
    cv::Mat difference, mask;
    cv::subtract(channels[color == 0 ? 2 : 0], channels[color == 0 ? 0 : 2], difference);
    cv::threshold(difference, mask, 70, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
      if (cv::contourArea(contour) < 8.0) continue;
      const cv::RotatedRect rect = cv::minAreaRect(contour);
      const float length = std::max(rect.size.width, rect.size.height);
      const float width = std::min(rect.size.width, rect.size.height);
      if (length < 6.0f || width < 1.0f || length / width < 2.0f || length / width > 35.0f)
        continue;

      cv::Point2f corners[4];
      rect.points(corners);
      const cv::Point2f edge1 = corners[1] - corners[0];
      const cv::Point2f edge2 = corners[2] - corners[1];
      cv::Point2f direction = cv::norm(edge1) > cv::norm(edge2) ? edge1 : edge2;
      direction *= 1.0f / static_cast<float>(cv::norm(direction));
      if (direction.y < 0.0f) direction *= -1.0f;
      if (direction.y < 0.866f) continue;
      lights.push_back({rect.center, rect.center - direction * (length * 0.5f),
                        rect.center + direction * (length * 0.5f), length, direction, color});
    }
  }

  std::sort(lights.begin(), lights.end(), [](const LightBar& a, const LightBar& b) {
    return a.center.x < b.center.x;
  });
  std::vector<ArmorCandidate> candidates;
  for (std::size_t i = 0; i < lights.size(); ++i) {
    for (std::size_t j = i + 1; j < lights.size(); ++j) {
      const auto& left = lights[i];
      const auto& right = lights[j];
      if (left.color != right.color) continue;
      const float average_length = (left.length + right.length) * 0.5f;
      const float length_ratio = std::min(left.length, right.length) / std::max(left.length, right.length);
      const float horizontal = (right.center.x - left.center.x) / average_length;
      const float vertical = std::abs(right.center.y - left.center.y) / average_length;
      const float alignment = left.direction.dot(right.direction);
      if (length_ratio < 0.5f || horizontal < 1.2f || horizontal > 4.5f ||
          vertical > 0.5f || alignment < 0.906f)
        continue;

      Armor armor;
      armor.corners = {left.top, right.top, right.bottom, left.bottom};
      armor.large_armor = horizontal > 3.2f;
      armor.class_id = -1;
      // 几何匹配评分，不是数字分类或神经网络的概率。
      armor.confidence = std::clamp(static_cast<double>(length_ratio * alignment * (1.0f - vertical)), 0.0, 1.0);
      candidates.push_back({armor, i, j});
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const ArmorCandidate& a, const ArmorCandidate& b) {
    return a.armor.confidence > b.armor.confidence;
  });
  std::vector<bool> used(lights.size(), false);
  std::vector<Armor> armors;
  for (const auto& candidate : candidates) {
    if (used[candidate.left] || used[candidate.right]) continue;
    used[candidate.left] = used[candidate.right] = true;
    armors.push_back(candidate.armor);
  }
  return armors;
}

std::vector<Armor> detect_pose(const cv::Mat& image) {
  constexpr int input_size = 640;
  static const float score_threshold = [] {
    const char* value = std::getenv("RM_POSE_CONF");
    const float threshold = value ? std::stof(value) : 0.25f;
    if (!std::isfinite(threshold) || threshold <= 0.0f || threshold >= 1.0f)
      throw std::runtime_error("RM_POSE_CONF 必须在 0 和 1 之间");
    return threshold;
  }();
  if (image.empty() || image.type() != CV_8UC3) return {};

  const double scale = std::min(static_cast<double>(input_size) / image.cols,
                                static_cast<double>(input_size) / image.rows);
  const int width = cvRound(image.cols * scale);
  const int height = cvRound(image.rows * scale);
  const int pad_x = (input_size - width) / 2;
  const int pad_y = (input_size - height) / 2;
  cv::Mat resized, padded(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(image, resized, {width, height});
  resized.copyTo(padded(cv::Rect(pad_x, pad_y, width, height)));
  const auto outputs = pose_forward(cv::dnn::blobFromImage(padded, 1.0 / 255.0, {}, {}, true, false));

  std::vector<Armor> candidates;
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  for (const auto& output : outputs) {
    if (output.dims != 4 || output.size[0] != 1 || output.size[1] != 26 ||
        output.size[2] != output.size[3])
      throw std::runtime_error("ONNX 输出格式错误；请使用配套 export_pose.py 导出的 640 模型");
    const int side = output.size[2];
    const float stride = static_cast<float>(input_size) / side;
    for (int y = 0; y < side; ++y) {
      for (int x = 0; x < side; ++x) {
        const int offset = y * side + x;
        int class_id = 0;
        float confidence = 0.0f;
        for (int c = 0; c < 14; ++c) {
          const float probability = output.ptr<float>(0, 4 + c)[offset];
          if (probability > confidence) { confidence = probability; class_id = c; }
        }
        if (confidence < score_threshold) continue;
        auto value = [&](int channel) { return output.ptr<float>(0, channel)[offset]; };
        const float x1 = ((x + 0.5f - value(0)) * stride - pad_x) / scale;
        const float y1 = ((y + 0.5f - value(1)) * stride - pad_y) / scale;
        const float x2 = ((x + 0.5f + value(2)) * stride - pad_x) / scale;
        const float y2 = ((y + 0.5f + value(3)) * stride - pad_y) / scale;
        if (!std::isfinite(x1 + y1 + x2 + y2) || x2 <= x1 || y2 <= y1) continue;
        Armor armor;
        armor.class_id = class_id;
        armor.confidence = confidence;
        // 数据标签为左上、左下、右下、右上，转换为培训接口的顺序。
        constexpr int point_order[4] = {0, 3, 2, 1};
        bool valid = true;
        for (int k = 0; k < 4; ++k) {
          const int point = point_order[k];
          const float px = ((value(18 + point * 2) + x + 0.5f) * stride - pad_x) / scale;
          const float py = ((value(19 + point * 2) + y + 0.5f) * stride - pad_y) / scale;
          valid = valid && std::isfinite(px) && std::isfinite(py);
          armor.corners[k] = {px, py};
        }
        if (!valid) continue;
        const double plate_width = (cv::norm(armor.corners[1] - armor.corners[0]) +
                                    cv::norm(armor.corners[2] - armor.corners[3])) * 0.5;
        const double plate_height = (cv::norm(armor.corners[3] - armor.corners[0]) +
                                     cv::norm(armor.corners[2] - armor.corners[1])) * 0.5;
        const std::vector<cv::Point2f> polygon(armor.corners.begin(), armor.corners.end());
        if (plate_height < 2.0 || plate_width / plate_height < 1.2 ||
            plate_width / plate_height > 6.0 || !cv::isContourConvex(polygon) ||
            cv::contourArea(polygon) < 4.0)
          continue;
        armor.large_armor = plate_height > 0.0 && plate_width / plate_height > 3.2;
        candidates.push_back(armor);
        boxes.emplace_back(cv::Point(cvRound(x1), cvRound(y1)), cv::Point(cvRound(x2), cvRound(y2)));
        scores.push_back(confidence);
      }
    }
  }
  std::vector<Armor> armors;
  for (int class_id = 0; class_id < 14; ++class_id) {
    std::vector<cv::Rect> class_boxes;
    std::vector<float> class_scores;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].class_id != class_id) continue;
      class_boxes.push_back(boxes[i]); class_scores.push_back(scores[i]); indices.push_back(i);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(class_boxes, class_scores, score_threshold, 0.45f, keep);
    for (int index : keep) armors.push_back(candidates[indices[index]]);
  }
  return armors;
}

std::vector<Armor> detect_hybrid(const cv::Mat& image) {
  struct Classification {
    cv::Rect plate;
    bool blue;
    int class_id;
    double confidence;
    std::chrono::steady_clock::time_point classified_at;
  };
  static std::vector<Classification> recent;
  const auto now = std::chrono::steady_clock::now();
  recent.erase(std::remove_if(recent.begin(), recent.end(), [&](const auto& value) {
    return std::getenv("RM_POSE_SOCKET") || now - value.classified_at >= std::chrono::milliseconds(250);
  }), recent.end());
  std::vector<Armor> verified;
  const auto proposals = detect_opencv(image);
  if (proposals.empty()) recent.clear();
  for (auto proposal : proposals) {
    const std::vector<cv::Point2f> points(proposal.corners.begin(), proposal.corners.end());
    const cv::Rect plate = cv::boundingRect(points);
    const cv::Point2f center = (proposal.corners[0] + proposal.corners[1] +
                                proposal.corners[2] + proposal.corners[3]) * 0.25f;
    // Keep context around the proposal; a tight crop removes useful classification features.
    const double lamp_span = (cv::norm(proposal.corners[1] - proposal.corners[0]) +
                              cv::norm(proposal.corners[2] - proposal.corners[3])) * 0.5;

    cv::Mat lamp_mask = cv::Mat::zeros(image.size(), CV_8UC1);
    const int thickness = std::max(2, cvRound(plate.height * 0.15));
    cv::line(lamp_mask, proposal.corners[0], proposal.corners[3], cv::Scalar(255), thickness);
    cv::line(lamp_mask, proposal.corners[1], proposal.corners[2], cv::Scalar(255), thickness);
    const cv::Scalar lamp_color = cv::mean(image, lamp_mask);
    if (std::abs(lamp_color[0] - lamp_color[2]) < 25.0) continue;
    const bool blue = lamp_color[0] > lamp_color[2];

    const auto cached = std::find_if(recent.begin(), recent.end(), [&](const auto& value) {
      const double intersection = (value.plate & plate).area();
      const double overlap = intersection / (value.plate.area() + plate.area() - intersection);
      return value.blue == blue && overlap >= 0.6;
    });
    if (cached != recent.end()) {
      cached->plate = plate;
      proposal.class_id = cached->class_id;
      proposal.confidence = cached->confidence;
      verified.push_back(proposal);
      continue;
    }

    int best_class = -1;
    float best_score = 0.0f, alternative_score = 0.0f;
    std::array<float, 14> class_scores{};
    for (double context : {3.75, 3.0}) {
      const int width = std::max(96, cvRound(lamp_span * context));
      const int height = cvRound(width * 0.75);
      const cv::Rect region = cv::Rect(cvRound(center.x) - width / 2,
                                      cvRound(center.y) - height / 2, width, height) &
                              cv::Rect(0, 0, image.cols, image.rows);
      if (region.width < 32 || region.height < 32) continue;
      for (const auto& prediction : detect_pose(image(region))) {
        if (prediction.class_id < 0 || prediction.class_id >= 14 || (prediction.class_id < 7) != blue) continue;
        std::vector<cv::Point2f> predicted_points;
        for (auto point : prediction.corners)
          predicted_points.push_back(point + cv::Point2f(region.x, region.y));
        const cv::Rect predicted_plate = cv::boundingRect(predicted_points);
        const double intersection = (plate & predicted_plate).area();
        const double overlap = intersection / (plate.area() + predicted_plate.area() - intersection);
        if (overlap < 0.5) continue;
        class_scores[prediction.class_id] = std::max(class_scores[prediction.class_id],
                                                     static_cast<float>(prediction.confidence));
      }
      best_score = alternative_score = 0.0f;
      for (int id = 0; id < 14; ++id) {
        if (class_scores[id] > best_score) {
          alternative_score = best_score;
          best_score = class_scores[id];
          best_class = id;
        } else {
          alternative_score = std::max(alternative_score, class_scores[id]);
        }
      }
      if (best_score >= 0.55f && best_score - alternative_score >= 0.10f) break;
    }
    // An uncertain digit or a neural prediction on a neighbouring lamp is not a verified armor.
    if (best_class < 0 || best_score < 0.55f || best_score - alternative_score < 0.10f) {
      continue;
    }
    proposal.class_id = best_class;
    proposal.confidence = best_score;
    recent.push_back({plate, blue, best_class, best_score, std::chrono::steady_clock::now()});
    verified.push_back(proposal);
  }
  return verified;
}
}  // namespace

std::vector<Armor> lesson03_lightbars(const cv::Mat& image) {
  return detect_opencv(image);
}

std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  if (image.empty() || image.type() != CV_8UC3) return {};
  const char* selected = std::getenv("RM_DETECTOR");
  const std::string mode = selected ? selected : "opencv";
  if (mode == "opencv") return detect_opencv(image);
  if (mode == "pose") return detect_pose(image);
  if (mode == "hybrid") return detect_hybrid(image);
  throw std::runtime_error("RM_DETECTOR 只能设为 opencv、pose 或 hybrid");
}
}  // namespace training
