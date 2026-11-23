#include "lessons.hpp"
#include "io_context.hpp"
#include "realtime_processor.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
using training::Armor;

std::string class_name(int id) {
  static const std::array<const char*, 14> names{
      "B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS"};
  return id >= 0 && id < static_cast<int>(names.size()) ? names[id] : "UNKNOWN";
}
void text_line(cv::Mat& image, const std::string& text, int y,
               const cv::Scalar& color = cv::Scalar(235, 235, 235)) {
  cv::putText(image, text, {28, y}, cv::FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv::LINE_AA);
}
cv::Rect bounds(const Armor& armor) {
  return cv::boundingRect(std::vector<cv::Point2f>(armor.corners.begin(), armor.corners.end()));
}
bool blue_lamps(const cv::Mat& image, const Armor& armor) {
  double difference = 0;
  for (int side : {0, 1}) {
    cv::LineIterator pixels(image, armor.corners[side], armor.corners[side == 0 ? 3 : 2], 8);
    for (int k = 0; k < pixels.count; ++k, ++pixels) {
      const auto* value = *pixels;
      difference += static_cast<int>(value[0])-value[2];
    }
  }
  return difference > 0;
}
std::vector<Armor> verified_current_lights(const cv::Mat& image, const std::vector<Armor>& lights,
                                          const std::vector<Armor>& predictions) {
  std::vector<Armor> verified;
  std::vector<bool> used(predictions.size(), false);
  for (auto light : lights) {
    const auto plate = bounds(light);
    const bool blue = blue_lamps(image, light);
    double best_overlap = 0.5;
    int selected = -1;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
      if (used[i] || predictions[i].class_id < 0 || (predictions[i].class_id < 7) != blue) continue;
      const auto predicted = bounds(predictions[i]);
      const double intersection = (plate & predicted).area();
      const double overlap = intersection/(plate.area()+predicted.area()-intersection);
      if (overlap > best_overlap) { best_overlap = overlap; selected = static_cast<int>(i); }
    }
    if (selected >= 0) {
      used[selected] = true;
      light.class_id = predictions[selected].class_id;
      light.confidence = predictions[selected].confidence;
      verified.push_back(light);  // Corners are measured on the current image, not copied from an old result.
    }
  }
  return verified;
}
std::vector<Armor> align_pose(const training::ProcessedFrame& source, const cv::Mat& current) {
  if (source.input.image.size() != current.size() || source.armors.empty()) return {};
  cv::Mat before, after;
  cv::cvtColor(source.input.image, before, cv::COLOR_BGR2GRAY);
  cv::cvtColor(current, after, cv::COLOR_BGR2GRAY);
  std::vector<cv::Point2f> original, moved, back;
  for (const auto& armor : source.armors)
    original.insert(original.end(), armor.corners.begin(), armor.corners.end());
  std::vector<uchar> forward_ok, backward_ok;
  std::vector<float> errors, back_errors;
  cv::calcOpticalFlowPyrLK(before, after, original, moved, forward_ok, errors);
  cv::calcOpticalFlowPyrLK(after, before, moved, back, backward_ok, back_errors);
  std::vector<Armor> aligned;
  for (std::size_t i = 0; i < source.armors.size(); ++i) {
    auto armor = source.armors[i];
    bool valid = true;
    for (int k = 0; k < 4; ++k) {
      const auto j = i*4+k;
      const auto point = moved[j];
      valid = valid && forward_ok[j] && backward_ok[j] && errors[j] < 25 &&
              cv::norm(back[j]-original[j]) < 2 && std::isfinite(point.x+point.y) &&
              point.x >= 0 && point.x < current.cols && point.y >= 0 && point.y < current.rows;
      armor.corners[k] = point;
    }
    const std::vector<cv::Point2f> polygon(armor.corners.begin(), armor.corners.end());
    if (valid && cv::isContourConvex(polygon) && cv::contourArea(polygon) > 4) aligned.push_back(armor);
  }
  return aligned;
}
void log_detection(std::ofstream& log, const training::ProcessedFrame& result) {
  if (!log.is_open()) return;
  const double timestamp = std::chrono::duration<double>(training::Clock::now().time_since_epoch()).count();
  log << "{\"t\":" << timestamp << ",\"frame\":" << result.sequence
      << ",\"processing_ms\":" << result.processing_ms << ",\"detections\":[";
  for (std::size_t i = 0; i < result.armors.size(); ++i) {
    if (i) log << ',';
    const auto& armor = result.armors[i];
    log << "{\"class_id\":" << armor.class_id << ",\"confidence\":" << armor.confidence << ",\"corners\":[";
    for (int k = 0; k < 4; ++k) {
      if (k) log << ',';
      log << '[' << armor.corners[k].x << ',' << armor.corners[k].y << ']';
    }
    log << "]}";
  }
  log << "]}\n";
}
}

int main(int argc, char** argv) {
  cv::setNumThreads(2);
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::RealtimeProcessor processor(calibration);
  const std::string backend = std::getenv("RM_DETECTOR") ? std::getenv("RM_DETECTOR") : "opencv";
  const bool model_score = backend != "opencv";
  const bool cuda = model_score && std::getenv("RM_POSE_SOCKET");
  const double preview_fps = std::getenv("RM_PREVIEW_FPS") ? std::stod(std::getenv("RM_PREVIEW_FPS")) : 60;
  if (!std::isfinite(preview_fps) || preview_fps < 10 || preview_fps > 120)
    throw std::runtime_error("RM_PREVIEW_FPS must be between 10 and 120");
  std::ofstream detection_log, timing_log;
  for (auto entry : {std::make_pair("RM_DETECTION_LOG", &detection_log), std::make_pair("RM_TIMING_LOG", &timing_log)}) {
    if (const char* path = std::getenv(entry.first)) {
      entry.second->open(path, std::ios::app);
      if (!*entry.second) throw std::runtime_error("Cannot open runtime log");
      *entry.second << std::setprecision(12);
    }
  }
  devices.open();
  if (config.display) {
    cv::namedWindow("RoboMaster training", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("RoboMaster training", 20, 40);
  }
  std::optional<training::SensorData> current;
  std::vector<Armor> lights;
  std::uint64_t sequence = 0, logged = 0, rounds = 0;
  auto last_preview = training::TimePoint{};
  auto last_write = training::TimePoint{};
  auto last_flush = training::Clock::now();
  auto last_error = training::TimePoint{};
  while (!interrupted) {
    const auto input = devices.read();
    if (input) {
      current = *input;
      ++sequence;
      if (backend == "hybrid" || backend == "opencv") lights = training::lesson03_lightbars(current->image);
      processor.submit(sequence, *current);
    }
    const auto result = processor.latest();
    const auto now = training::Clock::now();
    const double result_age = result ? std::chrono::duration<double, std::milli>(now-result->input.timestamp).count() : 1e9;
    if (result && result->sequence != logged) {
      logged = result->sequence;
      log_detection(detection_log, *result);
      if (!result->error.empty() && now-last_error >= std::chrono::seconds(2)) {
        std::cerr << "Detector unavailable: " << result->error << std::endl;
        last_error = now;
      }
    }
    if (interrupted) break;
    if (input || now-last_write >= std::chrono::milliseconds(50)) {
      training::Command command{};
      if (input && result && result->error.empty() && result_age <= 150) command = result->command;
      devices.write(command);
      last_write = now;
    }
    if (++rounds % 500 == 0)
      std::cout << "IO frames=" << sequence << " processed=" << (result ? result->sequence : 0)
                << " overwritten=" << processor.overwritten() << " allow_fire=" << config.allow_fire << std::endl;
    if (config.display) {
      if (current && now-last_preview >= std::chrono::duration<double>(1/preview_fps)) {
        const double camera_age = std::chrono::duration<double, std::milli>(now-current->timestamp).count();
        std::vector<Armor> armors;
        if (camera_age <= 200) {
          if (backend == "opencv") armors = lights;
          else if (result && result->error.empty() && result_age <= 200) {
            if (backend == "hybrid") {
              const auto aligned = result->sequence == sequence ? result->armors : align_pose(*result, current->image);
              armors = verified_current_lights(current->image, lights, aligned);
            }
            else if (result->sequence == sequence) armors = result->armors;
            else if (result_age <= 150) armors = align_pose(*result, current->image);
          }
        }
        cv::Mat preview;
        const double scale = std::min(1.0, 1280.0/current->image.cols);
        cv::resize(current->image, preview, {}, scale, scale, cv::INTER_AREA);
        for (const auto& armor : armors) {
          for (int i = 0; i < 4; ++i)
            cv::line(preview, armor.corners[i]*scale, armor.corners[(i+1)%4]*scale, {0,255,0}, 2);
          const auto anchor = armor.corners[0]*scale;
          const cv::Point origin(std::clamp(cvRound(anchor.x),4,std::max(4,preview.cols-145)),std::max(22,cvRound(anchor.y)-12));
          cv::rectangle(preview, {origin.x-3,origin.y-22,146,29}, {24,24,24}, cv::FILLED);
          cv::putText(preview,class_name(armor.class_id)+cv::format("  %.2f",armor.confidence),origin,
                      cv::FONT_HERSHEY_SIMPLEX,.65,{0,255,0},2,cv::LINE_AA);
        }
        cv::rectangle(preview,{14,14,490,245},{25,25,25},cv::FILLED);
        text_line(preview,"ARMOR | "+std::string(backend=="hybrid"?"CV + POSE":backend)+(cuda?" CUDA":" CPU"),43,{0,255,0});
        text_line(preview,cv::format("Objects: %zu    Image: %d x %d",armors.size(),current->image.cols,current->image.rows),73);
        if (!armors.empty()) {
          const auto& selected=*std::max_element(armors.begin(),armors.end(),[](const auto& a,const auto& b){return a.confidence<b.confidence;});
          const auto center=(selected.corners[0]+selected.corners[1]+selected.corners[2]+selected.corners[3])*.25f;
          const double width=(cv::norm(selected.corners[1]-selected.corners[0])+cv::norm(selected.corners[2]-selected.corners[3]))*.5;
          const double height=(cv::norm(selected.corners[3]-selected.corners[0])+cv::norm(selected.corners[2]-selected.corners[1]))*.5;
          const std::string color=selected.class_id<0?"UNKNOWN":selected.class_id<7?"BLUE":"RED";
          text_line(preview,"ID: "+class_name(selected.class_id)+"  Color: "+color+"  "+(selected.large_armor?"LARGE":"SMALL")+" (est.)",103);
          text_line(preview,cv::format("%s: %.2f",model_score?"Model confidence":"Geometry score",selected.confidence),133);
          text_line(preview,cv::format("Center: (%.0f, %.0f) px",center.x,center.y),163);
          text_line(preview,cv::format("Lamp span: %.0f x %.0f px",width,height),193);
        } else text_line(preview, result && !result->error.empty()?"Classifier unavailable":"No verified armor / waiting",113,{0,200,255});
        text_line(preview,cv::format("Camera age: %.0f ms  Result: %.0f ms",camera_age,result?result_age:0),223,
                  camera_age>200?cv::Scalar(0,100,255):cv::Scalar(190,190,190));
        if (result && result_age<=150 && result->tracking.observed) {
          const auto& track=result->tracking;
          if (track.target) {
            const auto& p=track.target->position;const auto& v=track.target->velocity;
            cv::rectangle(preview,{14,preview.rows-95,1000,85},{25,25,25},cv::FILLED);
            text_line(preview,cv::format("%s  position m: [%.3f %.3f %.3f]  speed: %.3f m/s",
                training::track_phase_name(track.phase),p.x(),p.y(),p.z(),v.norm()),preview.rows-66,{0,255,0});
            text_line(preview,cv::format("n=%.0f ms | approximate lightbar model | GT evaluation: lesson04_demo",
                track.parameters.prediction_horizon_ms),preview.rows-36,{255,160,100});
          }
          if (result->sequence==sequence) {
            const auto projected=training::project_prediction(track,calibration,track.parameters.prediction_horizon_ms);
            if (projected.size()==4)for(int k=0;k<4;++k)
              cv::line(preview,projected[k]*scale,projected[(k+1)%4]*scale,{255,100,30},2,cv::LINE_AA);
          }
        }
        cv::imshow("RoboMaster training",preview);
        if (timing_log.is_open()) {
          const double timestamp=std::chrono::duration<double>(training::Clock::now().time_since_epoch()).count();
          timing_log<<"{\"t\":"<<timestamp<<",\"frame\":"<<sequence<<",\"capture_age_ms\":"<<camera_age
                    <<",\"result_age_ms\":"<<(result?result_age:-1)<<",\"processing_ms\":"<<(result?result->processing_ms:0)
                    <<",\"objects\":"<<armors.size()<<",\"overwritten\":"<<processor.overwritten()<<"}\n";
        }
        last_preview=now;
      }
      const int key=cv::waitKey(1);
      if (key==27 || key=='q') break;
    }
    if (now-last_flush>=std::chrono::seconds(1)) {
      detection_log.flush();timing_log.flush();last_flush=now;
    }
  }
  devices.write(training::Command{});
  processor.stop();
  devices.close();
  if (config.display) cv::destroyAllWindows();
}
