#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 视觉链路预览与原始数据落盘
constructor_args:
  runtime:
    enabled: false
    record_raw: false
    realtime_preview: false
    overlay:
      detector: true
      tracker: true
      candidate_debug: false
      model_faces: false
    output_dir: "/tmp/autoaim_preview"
    raw_video_name: "raw.avi"
    overlay_video_name: "overlay.avi"
    preview_window_name: "autoaim_preview"
    preview_scale: 0.5
    preview_wait_key_ms: 1
    record_fps: 100.0
  sync: '@camera_frame_sync'
template_args:
  - Info:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
      camera_matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
      distortion_model: CameraTypes::DistortionModel::PLUMB_BOB
      distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
      rectification_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection_matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
required_hardware: []
depends:
  - qdu-future/CameraFrameSync
  - qdu-future/ArmorDetector
  - qdu-future/ArmorTracker
=== END MANIFEST === */
// clang-format on

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "ArmorTracker.hpp"
#include "CameraFrameSync.hpp"
#include "app_framework.hpp"
#include "armor.hpp"
#include "libxr.hpp"
#include "linux_shared_topic.hpp"
#include "logger.hpp"
#include "message.hpp"

template <CameraTypes::CameraInfo CameraInfoV>
class VisionPreview : public LibXR::Application
{
 public:
  using Self = VisionPreview<CameraInfoV>;
  using Sync = CameraFrameSync<CameraInfoV>;
  using ImageFrame = typename Sync::ImageFrame;
  using ImageTopic = typename Sync::ImageTopic;
  using ImageData = typename Sync::ImageData;
  using DetectorMessage = ArmorDetectionsMessage;
  using DetectorMetrics = ArmorDetectorMetrics;
  using Tracker = ArmorTracker<CameraInfoV>;
  using TargetMessage = SolveTrajectory::Target;
  using EkfPointsMessage = typename Tracker::EkfPointsMsg;
  using CandidateDebugMessage = typename Tracker::CandidateDebugMsg;

  static inline constexpr auto camera_info = CameraInfoV;
  static constexpr std::size_t image_queue_capacity = 2;
  static constexpr std::size_t history_capacity = 128;
  static constexpr std::size_t record_queue_capacity = 256;
  static constexpr std::size_t worker_stack_bytes = 256U * 1024U;
  static constexpr uint32_t image_wait_timeout_ms = 100;

  struct OverlayConfig
  {
    bool detector = true;       // 绘制 detector 原始识别框、角点和置信度。
    bool tracker = true;        // 绘制 tracker 中心和匹配面 EKF 投影。
    bool candidate_debug = false;  // 显示候选统计。
    bool model_faces = false;      // 额外绘制 EKF 模型补全的全部装甲板点。
  };

  struct RuntimeParam
  {
    bool enabled = false;            // 总开关；关闭时不注册回调、不启动线程。
    bool record_raw = false;         // 原始视频和 topic 数据落盘。
    bool realtime_preview = false;   // 实时窗口预览。
    OverlayConfig overlay{};
    std::string_view output_dir = "/tmp/autoaim_preview";
    std::string_view raw_video_name = "raw.avi";
    std::string_view overlay_video_name = "overlay.avi";
    std::string_view preview_window_name = "autoaim_preview";
    double preview_scale = 0.5;
    int preview_wait_key_ms = 1;
    double record_fps = 100.0;
  };

  VisionPreview(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                RuntimeParam runtime, Sync& sync)
      : runtime_(runtime),
        image_topic_name_(sync.ImageTopicName()),
        output_dir_(runtime.output_dir),
        raw_video_name_(runtime.raw_video_name),
        overlay_video_name_(runtime.overlay_video_name),
        preview_window_name_(runtime.preview_window_name)
  {
    (void)hw;
    if (!ShouldRun())
    {
      XR_LOG_INFO("VisionPreview disabled");
      return;
    }

    realtime_preview_enabled_ = runtime_.realtime_preview && UiAvailable();
    if (runtime_.realtime_preview && !realtime_preview_enabled_)
    {
      XR_LOG_WARN("VisionPreview realtime preview disabled: display backend unavailable");
    }

    if (runtime_.record_raw)
    {
      OpenRecordFiles();
      if (!record_ready_ && !realtime_preview_enabled_)
      {
        XR_LOG_ERROR("VisionPreview disabled: record output is unavailable");
        return;
      }
    }

    RegisterCallbacks();
    running_ = true;
    image_thread_.Create(this, ImageThreadMain, "VisionPrevImg", worker_stack_bytes,
                         LibXR::Thread::Priority::LOW);
    worker_thread_.Create(this, WorkerThreadMain, "VisionPreview", worker_stack_bytes,
                          LibXR::Thread::Priority::LOW);
    app.Register(*this);
  }

  void OnMonitor() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    XR_LOG_INFO("VisionPreview monitor: image_dropped=%u detector_dropped=%u tracker_dropped=%u",
                static_cast<unsigned>(image_dropped_),
                static_cast<unsigned>(detector_dropped_),
                static_cast<unsigned>(tracker_dropped_));
  }

 private:
  template <typename T, std::size_t capacity>
  struct Ring
  {
    std::array<T, capacity> items{};
    std::size_t head = 0;
    std::size_t count = 0;
    uint32_t dropped = 0;

    void Push(const T& value)
    {
      if (count == capacity)
      {
        items[head] = value;
        head = (head + 1U) % capacity;
        ++dropped;
        return;
      }

      items[(head + count) % capacity] = value;
      ++count;
    }

    bool PopOldest(T& out)
    {
      if (count == 0)
      {
        return false;
      }

      out = std::move(items[head]);
      head = (head + 1U) % capacity;
      --count;
      return true;
    }

    bool FindByTimestamp(uint64_t timestamp_us, T& out) const
    {
      for (std::size_t offset = 0; offset < count; ++offset)
      {
        const std::size_t reverse_index = count - 1U - offset;
        const std::size_t index = (head + reverse_index) % capacity;
        if (TimestampOf(items[index]) == timestamp_us)
        {
          out = items[index];
          return true;
        }
      }
      return false;
    }

    bool Empty() const { return count == 0; }
  };

  struct FrameSnapshot
  {
    bool detector_valid = false;
    bool target_valid = false;
    bool ekf_valid = false;
    bool candidate_valid = false;
    DetectorMessage detector{};
    TargetMessage target{};
    EkfPointsMessage ekf{};
    CandidateDebugMessage candidate{};
  };

  static uint64_t TimestampOf(const DetectorMessage& message)
  {
    return message.image_timestamp_us;
  }

  static uint64_t TimestampOf(const DetectorMetrics& message)
  {
    return message.image_timestamp_us;
  }

  static uint64_t TimestampOf(const TargetMessage& message)
  {
    return message.image_timestamp_us;
  }

  static uint64_t TimestampOf(const EkfPointsMessage& message)
  {
    return message.image_timestamp_us;
  }

  static uint64_t TimestampOf(const CandidateDebugMessage& message)
  {
    return message.image_timestamp_us;
  }

  static bool UiAvailable()
  {
    const char* display = std::getenv("DISPLAY");
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    return (display != nullptr && display[0] != '\0') ||
           (wayland_display != nullptr && wayland_display[0] != '\0');
  }

  bool ShouldRun() const
  {
    return runtime_.enabled && (runtime_.record_raw || runtime_.realtime_preview);
  }

  bool OverlayEnabled() const
  {
    return runtime_.overlay.detector || runtime_.overlay.tracker ||
           runtime_.overlay.candidate_debug;
  }

  void RegisterCallbacks()
  {
    LibXR::Topic::Domain detector_domain("armor_detector");
    LibXR::Topic detector_topic(LibXR::Topic::WaitTopic("armors_result", UINT32_MAX,
                                                        &detector_domain));
    auto detector_callback = LibXR::Topic::Callback::Create(
        [](bool, Self* self, LibXR::RawData& data)
        {
          const auto* message = reinterpret_cast<const DetectorMessage*>(data.addr_);
          self->PushDetector(*message);
        },
        this);
    detector_topic.RegisterCallback(detector_callback);

    LibXR::Topic metrics_topic(LibXR::Topic::WaitTopic("metrics", UINT32_MAX,
                                                       &detector_domain));
    auto metrics_callback = LibXR::Topic::Callback::Create(
        [](bool, Self* self, LibXR::RawData& data)
        {
          const auto* message = reinterpret_cast<const DetectorMetrics*>(data.addr_);
          self->PushMetrics(*message);
        },
        this);
    metrics_topic.RegisterCallback(metrics_callback);

    LibXR::Topic::Domain tracker_domain("tracker");
    LibXR::Topic target_topic(LibXR::Topic::WaitTopic("target", UINT32_MAX,
                                                      &tracker_domain));
    auto target_callback = LibXR::Topic::Callback::Create(
        [](bool, Self* self, LibXR::RawData& data)
        {
          const auto* message = reinterpret_cast<const TargetMessage*>(data.addr_);
          self->PushTarget(*message);
        },
        this);
    target_topic.RegisterCallback(target_callback);

    LibXR::Topic ekf_topic(LibXR::Topic::WaitTopic("ekf_points", UINT32_MAX,
                                                   &tracker_domain));
    auto ekf_callback = LibXR::Topic::Callback::Create(
        [](bool, Self* self, LibXR::RawData& data)
        {
          const auto* message = reinterpret_cast<const EkfPointsMessage*>(data.addr_);
          self->PushEkf(*message);
        },
        this);
    ekf_topic.RegisterCallback(ekf_callback);

    LibXR::Topic candidate_topic(LibXR::Topic::WaitTopic("candidate_debug", UINT32_MAX,
                                                         &tracker_domain));
    auto candidate_callback = LibXR::Topic::Callback::Create(
        [](bool, Self* self, LibXR::RawData& data)
        {
          const auto* message = reinterpret_cast<const CandidateDebugMessage*>(data.addr_);
          self->PushCandidate(*message);
        },
        this);
    candidate_topic.RegisterCallback(candidate_callback);
  }

  void PushDetector(const DetectorMessage& message)
  {
    // topic 回调只拷贝消息并入队，绘制和文件写入都放到 worker 线程。
    std::lock_guard<std::mutex> lock(mutex_);
    detector_history_.Push(message);
    if (RecordEnabled())
    {
      detector_record_queue_.Push(message);
      detector_dropped_ = detector_record_queue_.dropped;
    }
    work_cv_.notify_one();
  }

  void PushMetrics(const DetectorMetrics& message)
  {
    // metrics 不参与 overlay 对齐，只在落盘模式下排队写出。
    std::lock_guard<std::mutex> lock(mutex_);
    if (RecordEnabled())
    {
      metrics_record_queue_.Push(message);
    }
    work_cv_.notify_one();
  }

  void PushTarget(const TargetMessage& message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    target_history_.Push(message);
    if (RecordEnabled())
    {
      target_record_queue_.Push(message);
      tracker_dropped_ = target_record_queue_.dropped;
    }
    work_cv_.notify_one();
  }

  void PushEkf(const EkfPointsMessage& message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ekf_history_.Push(message);
    if (RecordEnabled())
    {
      ekf_record_queue_.Push(message);
    }
    work_cv_.notify_one();
  }

  void PushCandidate(const CandidateDebugMessage& message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    candidate_history_.Push(message);
    if (RecordEnabled())
    {
      candidate_record_queue_.Push(message);
    }
    work_cv_.notify_one();
  }

  static void ImageThreadMain(Self* self)
  {
    typename ImageTopic::Subscriber image_sub(self->image_topic_name_.c_str(),
                                              LibXR::LinuxSharedSubscriberMode::BROADCAST_DROP_OLD);
    if (!image_sub.Valid())
    {
      XR_LOG_ERROR("VisionPreview failed to attach image topic: %s",
                   self->image_topic_name_.c_str());
      return;
    }

    while (self->running_)
    {
      ImageData image;
      const auto wait_ans = image_sub.Wait(image, image_wait_timeout_ms);
      if (wait_ans == LibXR::ErrorCode::TIMEOUT)
      {
        continue;
      }
      if (wait_ans != LibXR::ErrorCode::OK)
      {
        XR_LOG_WARN("VisionPreview image wait failed: %d", static_cast<int>(wait_ans));
        continue;
      }

      self->PushImage(std::move(image));
    }
  }

  void PushImage(ImageData&& image)
  {
    // 图像数据只移动共享槽位句柄；队列满时释放旧句柄，避免反压主链路。
    std::lock_guard<std::mutex> lock(mutex_);

    if (image_count_ == image_queue_capacity)
    {
      image_queue_[image_head_].Reset();
      image_head_ = (image_head_ + 1U) % image_queue_capacity;
      --image_count_;
      ++image_dropped_;
    }

    const std::size_t tail = (image_head_ + image_count_) % image_queue_capacity;
    image_queue_[tail] = std::move(image);
    ++image_count_;
    work_cv_.notify_one();
  }

  bool PopLatestImageLocked(ImageData& out)
  {
    if (image_count_ == 0)
    {
      return false;
    }

    while (image_count_ > 1)
    {
      image_queue_[image_head_].Reset();
      image_head_ = (image_head_ + 1U) % image_queue_capacity;
      --image_count_;
      ++image_dropped_;
    }

    out = std::move(image_queue_[image_head_]);
    image_head_ = (image_head_ + 1U) % image_queue_capacity;
    image_count_ = 0;
    return true;
  }

  bool HasWorkLocked() const
  {
    return image_count_ > 0 ||
           !detector_record_queue_.Empty() ||
           !metrics_record_queue_.Empty() ||
           !target_record_queue_.Empty() ||
           !ekf_record_queue_.Empty() ||
           !candidate_record_queue_.Empty();
  }

  static void WorkerThreadMain(Self* self)
  {
    while (self->running_)
    {
      ImageData image;
      std::vector<DetectorMessage> detector_records;
      std::vector<DetectorMetrics> metrics_records;
      std::vector<TargetMessage> target_records;
      std::vector<EkfPointsMessage> ekf_records;
      std::vector<CandidateDebugMessage> candidate_records;

      {
        std::unique_lock<std::mutex> lock(self->mutex_);
        self->work_cv_.wait(lock, [self]()
        {
          return !self->running_ || self->HasWorkLocked();
        });
        if (!self->running_)
        {
          return;
        }

        (void)self->PopLatestImageLocked(image);
        self->DrainRecordQueuesLocked(detector_records, metrics_records, target_records,
                                      ekf_records, candidate_records);
      }

      self->WriteRecords(detector_records, metrics_records, target_records, ekf_records,
                         candidate_records);
      if (image.Valid())
      {
        self->ProcessImage(image);
      }
    }
  }

  void DrainRecordQueuesLocked(std::vector<DetectorMessage>& detector_records,
                               std::vector<DetectorMetrics>& metrics_records,
                               std::vector<TargetMessage>& target_records,
                               std::vector<EkfPointsMessage>& ekf_records,
                               std::vector<CandidateDebugMessage>& candidate_records)
  {
    DetectorMessage detector;
    while (detector_record_queue_.PopOldest(detector))
    {
      detector_records.push_back(std::move(detector));
    }

    DetectorMetrics metrics;
    while (metrics_record_queue_.PopOldest(metrics))
    {
      metrics_records.push_back(metrics);
    }

    TargetMessage target;
    while (target_record_queue_.PopOldest(target))
    {
      target_records.push_back(target);
    }

    EkfPointsMessage ekf;
    while (ekf_record_queue_.PopOldest(ekf))
    {
      ekf_records.push_back(ekf);
    }

    CandidateDebugMessage candidate;
    while (candidate_record_queue_.PopOldest(candidate))
    {
      candidate_records.push_back(candidate);
    }
  }

  FrameSnapshot SnapshotFor(uint64_t timestamp_us)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    FrameSnapshot snapshot;
    snapshot.detector_valid =
        detector_history_.FindByTimestamp(timestamp_us, snapshot.detector);
    snapshot.target_valid =
        target_history_.FindByTimestamp(timestamp_us, snapshot.target);
    snapshot.ekf_valid = ekf_history_.FindByTimestamp(timestamp_us, snapshot.ekf);
    snapshot.candidate_valid =
        candidate_history_.FindByTimestamp(timestamp_us, snapshot.candidate);
    return snapshot;
  }

  void ProcessImage(const ImageData& image)
  {
    const ImageFrame* image_frame = image.GetData();
    if (image_frame == nullptr)
    {
      return;
    }

    const int cv_type = CvTypeFromEncoding(camera_info.encoding);
    if (cv_type < 0)
    {
      XR_LOG_WARN("VisionPreview unsupported image encoding: %u",
                  static_cast<unsigned>(camera_info.encoding));
      return;
    }

    cv::Mat raw(static_cast<int>(camera_info.height),
                static_cast<int>(camera_info.width), cv_type,
                const_cast<uint8_t*>(image_frame->data.data()),
                static_cast<std::size_t>(camera_info.step));
    cv::Mat bgr = ConvertToBgr(raw, camera_info.encoding);
    if (bgr.empty())
    {
      return;
    }

    if (RecordEnabled())
    {
      WriteRawVideo(bgr);
    }

    if (!realtime_preview_enabled_ && !(RecordEnabled() && OverlayEnabled()))
    {
      return;
    }

    const uint64_t timestamp_us = static_cast<uint64_t>(image_frame->timestamp_us);
    FrameSnapshot snapshot = SnapshotFor(timestamp_us);
    cv::Mat canvas = bgr.clone();

    if (runtime_.overlay.detector && snapshot.detector_valid)
    {
      DrawDetector(canvas, snapshot.detector,
                   snapshot.candidate_valid ? &snapshot.candidate : nullptr);
    }
    if (runtime_.overlay.tracker && snapshot.ekf_valid)
    {
      DrawTracker(canvas, snapshot.ekf,
                  snapshot.detector_valid ? &snapshot.detector : nullptr,
                  snapshot.candidate_valid ? &snapshot.candidate : nullptr);
    }
    DrawStatus(canvas, timestamp_us, snapshot);

    if (RecordEnabled() && OverlayEnabled())
    {
      WriteOverlayVideo(canvas);
    }

    if (!realtime_preview_enabled_)
    {
      return;
    }

    if (runtime_.preview_scale > 0.0 &&
        std::abs(runtime_.preview_scale - 1.0) > 1e-6)
    {
      cv::Mat scaled;
      cv::resize(canvas, scaled, cv::Size(), runtime_.preview_scale,
                 runtime_.preview_scale);
      cv::imshow(preview_window_name_, scaled);
    }
    else
    {
      cv::imshow(preview_window_name_, canvas);
    }
    cv::waitKey(std::max(runtime_.preview_wait_key_ms, 1));
  }

  static int CvTypeFromEncoding(CameraTypes::Encoding encoding)
  {
    switch (encoding)
    {
      case CameraTypes::Encoding::RGB8:
      case CameraTypes::Encoding::BGR8:
        return CV_8UC3;
      case CameraTypes::Encoding::RGBA8:
      case CameraTypes::Encoding::BGRA8:
        return CV_8UC4;
      case CameraTypes::Encoding::MONO8:
        return CV_8UC1;
      default:
        return -1;
    }
  }

  static cv::Mat ConvertToBgr(const cv::Mat& input, CameraTypes::Encoding encoding)
  {
    if (input.empty())
    {
      return {};
    }

    switch (encoding)
    {
      case CameraTypes::Encoding::RGB8:
      {
        cv::Mat output;
        cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
        return output;
      }
      case CameraTypes::Encoding::BGRA8:
      {
        cv::Mat output;
        cv::cvtColor(input, output, cv::COLOR_BGRA2BGR);
        return output;
      }
      case CameraTypes::Encoding::RGBA8:
      {
        cv::Mat output;
        cv::cvtColor(input, output, cv::COLOR_RGBA2BGR);
        return output;
      }
      case CameraTypes::Encoding::MONO8:
      {
        cv::Mat output;
        cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);
        return output;
      }
      case CameraTypes::Encoding::BGR8:
        return input;
      default:
        return {};
    }
  }

  static cv::Scalar ArmorColorToScalar(ArmorColor color)
  {
    switch (color)
    {
      case ArmorColor::BLUE:
        return cv::Scalar(255, 180, 40);
      case ArmorColor::RED:
        return cv::Scalar(60, 90, 255);
      case ArmorColor::EXTINGUISH:
        return cv::Scalar(180, 180, 180);
      default:
        return cv::Scalar(90, 220, 120);
    }
  }

  static std::string_view ArmorNumberName(ArmorNumber number)
  {
    const std::size_t index = static_cast<std::size_t>(number);
    if (index >= ARMOR_NUMBER_NAMES.size())
    {
      return "invalid";
    }
    return ARMOR_NUMBER_NAMES[index];
  }

  std::array<int, CandidateDebugMessage::kMaxDetections> MatchedFaces(
      const CandidateDebugMessage* candidate) const
  {
    std::array<int, CandidateDebugMessage::kMaxDetections> matched_faces{};
    matched_faces.fill(-1);
    if (candidate == nullptr || candidate->matched == 0U)
    {
      return matched_faces;
    }

    const uint8_t count = std::min<uint8_t>(candidate->count,
                                            CandidateDebugMessage::kMaxItems);
    for (uint8_t i = 0; i < count; ++i)
    {
      const auto& item = candidate->items[i];
      if (item.armor_index < matched_faces.size())
      {
        matched_faces[item.armor_index] = static_cast<int>(item.face_index);
      }
    }
    return matched_faces;
  }

  void DrawDetector(cv::Mat& canvas, const DetectorMessage& detector,
                    const CandidateDebugMessage* candidate)
  {
    const auto matched_faces = MatchedFaces(candidate);
    for (std::size_t index = 0; index < detector.results.size(); ++index)
    {
      const auto& armor = detector.results[index];
      const bool is_matched =
          index < matched_faces.size() && matched_faces[index] >= 0;
      const cv::Scalar color = ArmorColorToScalar(armor.color);
      const cv::Scalar draw_color = is_matched ? cv::Scalar(255, 0, 255) : color;
      std::array<cv::Point, 4> points{};
      for (std::size_t i = 0; i < armor.points.size(); ++i)
      {
        points[i] = cv::Point(cvRound(armor.points[i].x), cvRound(armor.points[i].y));
      }
      const cv::Point* polygon = points.data();
      const int point_count = static_cast<int>(points.size());
      cv::polylines(canvas, &polygon, &point_count, 1, true, draw_color, 2,
                    cv::LINE_AA);
      cv::rectangle(canvas, armor.box, draw_color, 1, cv::LINE_AA);

      std::ostringstream label;
      if (is_matched)
      {
        label << "M" << index << " f=" << matched_faces[index] << " ";
      }
      label << ArmorNumberName(armor.number) << " " << std::fixed
            << std::setprecision(2) << armor.confidence;
      cv::putText(canvas, label.str(),
                  cv::Point(std::max(armor.box.x, 4), std::max(armor.box.y - 6, 18)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 1, cv::LINE_AA);
    }
  }

  bool ProjectPoint(const cv::Mat& canvas, const LibXR::Position<double>& point,
                    cv::Point2d& uv) const
  {
    const cv::Mat camera_matrix = ScaledCameraMatrix(canvas);
    const cv::Mat dist_coeffs = DistCoeffs();
    const Eigen::Vector3d pc(point.x(), point.y(), point.z());
    if (!(pc.z() > 1e-6) || !std::isfinite(pc.x()) || !std::isfinite(pc.y()) ||
        !std::isfinite(pc.z()))
    {
      return false;
    }

    std::vector<cv::Point3d> object_points{cv::Point3d(pc.x(), pc.y(), pc.z())};
    cv::Mat rvec = cv::Mat::zeros(1, 3, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(1, 3, CV_64F);
    std::vector<cv::Point2d> image_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs,
                      image_points);
    uv = image_points[0];
    return uv.x >= 0.0 && uv.x < canvas.cols && uv.y >= 0.0 && uv.y < canvas.rows;
  }

  void DrawTracker(cv::Mat& canvas, const EkfPointsMessage& ekf,
                   const DetectorMessage* detector,
                   const CandidateDebugMessage* candidate)
  {
    cv::Point2d center_uv;
    const bool center_visible =
        ekf.valid[0] && ProjectPoint(canvas, ekf.center_cam, center_uv);
    if (center_visible)
    {
      cv::circle(canvas, center_uv, 5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
      cv::putText(canvas, "TC", center_uv + cv::Point2d(6, -6),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1,
                  cv::LINE_AA);
    }

    DrawMatchedEkfFaces(canvas, ekf, detector, candidate);
    if (!runtime_.overlay.model_faces)
    {
      return;
    }

    for (int i = 0; i < std::min<int>(ekf.count, 4); ++i)
    {
      cv::Point2d armor_uv;
      if (!ekf.valid[i + 1] || !ProjectPoint(canvas, ekf.armors_cam[i], armor_uv))
      {
        continue;
      }

      cv::circle(canvas, armor_uv, 3, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
      std::ostringstream label;
      label << "model" << i;
      cv::putText(canvas, label.str(), armor_uv + cv::Point2d(7, -7),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(180, 180, 180), 1,
                  cv::LINE_AA);
      if (center_visible)
      {
        cv::line(canvas, center_uv, armor_uv, cv::Scalar(120, 120, 120), 1,
                 cv::LINE_AA);
      }
    }
  }

  void DrawMatchedEkfFaces(cv::Mat& canvas, const EkfPointsMessage& ekf,
                           const DetectorMessage* detector,
                           const CandidateDebugMessage* candidate)
  {
    if (detector == nullptr || candidate == nullptr || candidate->matched == 0U)
    {
      return;
    }

    const double sx = static_cast<double>(canvas.cols) /
                      static_cast<double>(std::max<uint32_t>(camera_info.width, 1));
    const double sy = static_cast<double>(canvas.rows) /
                      static_cast<double>(std::max<uint32_t>(camera_info.height, 1));
    const uint8_t count = std::min<uint8_t>(candidate->count,
                                            CandidateDebugMessage::kMaxItems);
    for (uint8_t i = 0; i < count; ++i)
    {
      const auto& item = candidate->items[i];
      if (item.armor_index >= detector->results.size())
      {
        continue;
      }
      const int face_index = static_cast<int>(item.face_index);
      if (face_index < 0 || face_index >= 4 || face_index >= ekf.count ||
          !ekf.valid[face_index + 1])
      {
        continue;
      }

      cv::Point2d ef_uv;
      if (!ProjectPoint(canvas, ekf.armors_cam[face_index], ef_uv))
      {
        continue;
      }

      const auto& armor = detector->results[item.armor_index];
      const cv::Point2d det_uv(armor.center.x * sx, armor.center.y * sy);
      const double err_px = std::hypot(ef_uv.x - det_uv.x, ef_uv.y - det_uv.y);
      cv::circle(canvas, ef_uv, 6, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
      cv::line(canvas, det_uv, ef_uv, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

      std::ostringstream label;
      label << "EF f=" << face_index << " e=" << std::fixed
            << std::setprecision(0) << err_px;
      cv::putText(canvas, label.str(), ef_uv + cv::Point2d(8, -8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 0), 1,
                  cv::LINE_AA);
    }
  }

  cv::Mat ScaledCameraMatrix(const cv::Mat& canvas) const
  {
    const auto& k = camera_info.camera_matrix;
    cv::Mat matrix = (cv::Mat_<double>(3, 3) << k[0], k[1], k[2], k[3], k[4],
                      k[5], k[6], k[7], k[8]);
    const double sx = static_cast<double>(canvas.cols) /
                      static_cast<double>(std::max<uint32_t>(camera_info.width, 1));
    const double sy = static_cast<double>(canvas.rows) /
                      static_cast<double>(std::max<uint32_t>(camera_info.height, 1));
    matrix.at<double>(0, 0) *= sx;
    matrix.at<double>(1, 1) *= sy;
    matrix.at<double>(0, 2) *= sx;
    matrix.at<double>(1, 2) *= sy;
    return matrix;
  }

  cv::Mat DistCoeffs() const
  {
    if (camera_info.distortion_model == CameraTypes::DistortionModel::PLUMB_BOB)
    {
      std::vector<double> coeffs = {
          camera_info.distortion_coefficients[0],
          camera_info.distortion_coefficients[1],
          camera_info.distortion_coefficients[2],
          camera_info.distortion_coefficients[3],
          camera_info.distortion_coefficients[4],
      };
      return cv::Mat(coeffs).reshape(1, 1).clone();
    }
    if (camera_info.distortion_model == CameraTypes::DistortionModel::RATIONAL_POLYNOMIAL)
    {
      std::vector<double> coeffs = {
          camera_info.distortion_coefficients[0],
          camera_info.distortion_coefficients[1],
          camera_info.distortion_coefficients[2],
          camera_info.distortion_coefficients[3],
          camera_info.distortion_coefficients[4],
          camera_info.distortion_coefficients[5],
          camera_info.distortion_coefficients[6],
          camera_info.distortion_coefficients[7],
      };
      return cv::Mat(coeffs).reshape(1, 1).clone();
    }
    return {};
  }

  void DrawStatus(cv::Mat& canvas, uint64_t timestamp_us, const FrameSnapshot& snapshot)
  {
    std::ostringstream line;
    line << "ts=" << timestamp_us / 1000ULL << "ms";
    if (snapshot.detector_valid)
    {
      line << " det=" << snapshot.detector.results.size();
    }
    if (snapshot.target_valid)
    {
      line << " tracking=" << (snapshot.target.tracking ? 1 : 0);
    }
    if (runtime_.overlay.candidate_debug && snapshot.candidate_valid)
    {
      line << " cand=" << static_cast<int>(snapshot.candidate.count)
           << " sel=" << static_cast<int>(snapshot.candidate.selected_index);
    }

    cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, 32), cv::Scalar(16, 20, 28),
                  cv::FILLED);
    cv::putText(canvas, line.str(), cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX,
                0.62, cv::Scalar(230, 236, 245), 1, cv::LINE_AA);
  }

  void OpenRecordFiles()
  {
    std::error_code ec;
    std::filesystem::create_directories(output_dir_, ec);
    if (ec)
    {
      XR_LOG_ERROR("VisionPreview failed to create output dir: %s",
                   output_dir_.c_str());
      return;
    }

    detector_file_.open(output_dir_ + "/detector.tsv", std::ios::out);
    metrics_file_.open(output_dir_ + "/metrics.tsv", std::ios::out);
    target_file_.open(output_dir_ + "/target.tsv", std::ios::out);
    ekf_file_.open(output_dir_ + "/ekf_points.tsv", std::ios::out);
    candidate_file_.open(output_dir_ + "/candidate_debug.tsv", std::ios::out);
    candidate_items_file_.open(output_dir_ + "/candidate_items.tsv", std::ios::out);
    if (!detector_file_ || !metrics_file_ || !target_file_ || !ekf_file_ ||
        !candidate_file_ || !candidate_items_file_)
    {
      XR_LOG_ERROR("VisionPreview failed to open record files under: %s",
                   output_dir_.c_str());
      return;
    }

    detector_file_ << "image_timestamp_us\tarmor_index\tnumber\ttype\tcolor\tconfidence"
                   << "\tcenter_x\tcenter_y\tpnp_valid\tpose_x\tpose_y\tpose_z\n";
    metrics_file_ << "image_timestamp_us\tframe_index\tarmor_count\tdecoded_count"
                  << "\tnms_count\tpnp_success_count\tdetector_latency_ms"
                  << "\tpublish_latency_ms\n";
    target_file_ << "image_timestamp_us\ttracking\tid\tarmors_num\tpos_x\tpos_y\tpos_z"
                 << "\tvel_x\tvel_y\tvel_z\tyaw\tv_yaw\tradius_1\tradius_2\tdz\n";
    ekf_file_ << "image_timestamp_us\tpoint_index\tvalid\tx\ty\tz\n";
    candidate_file_ << "image_timestamp_us\tcount\tselected_index\tmatched"
                    << "\tdetection_count\ttracked_armors_num\n";
    candidate_items_file_ << "image_timestamp_us\titem_index\tarmor_index"
                          << "\tface_index\tsame_number\timage_track_id"
                          << "\timage_track_confirmed\tnumber\ttype\tscore"
                          << "\tposition_diff\tyaw_diff\tcenter_x\tcenter_y"
                          << "\tpredicted_yaw\tmeasured_yaw\n";
    record_ready_ = true;
    FlushRecordFiles();
  }

  void WriteRawVideo(const cv::Mat& bgr)
  {
    if (!raw_writer_ready_)
    {
      const std::string path = output_dir_ + "/" + raw_video_name_;
      const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
      raw_writer_ready_ = raw_writer_.open(path, fourcc, runtime_.record_fps,
                                           cv::Size(bgr.cols, bgr.rows), true);
      if (!raw_writer_ready_)
      {
        XR_LOG_ERROR("VisionPreview failed to open raw video: %s", path.c_str());
        return;
      }
    }
    raw_writer_.write(bgr);
  }

  void WriteOverlayVideo(const cv::Mat& bgr)
  {
    if (!overlay_writer_ready_)
    {
      const std::string path = output_dir_ + "/" + overlay_video_name_;
      const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
      overlay_writer_ready_ = overlay_writer_.open(
          path, fourcc, runtime_.record_fps, cv::Size(bgr.cols, bgr.rows), true);
      if (!overlay_writer_ready_)
      {
        XR_LOG_ERROR("VisionPreview failed to open overlay video: %s", path.c_str());
        return;
      }
    }
    overlay_writer_.write(bgr);
  }

  void WriteRecords(const std::vector<DetectorMessage>& detector_records,
                    const std::vector<DetectorMetrics>& metrics_records,
                    const std::vector<TargetMessage>& target_records,
                    const std::vector<EkfPointsMessage>& ekf_records,
                    const std::vector<CandidateDebugMessage>& candidate_records)
  {
    if (!RecordEnabled())
    {
      return;
    }

    for (const auto& detector : detector_records)
    {
      for (std::size_t i = 0; i < detector.results.size(); ++i)
      {
        const auto& armor = detector.results[i];
        detector_file_ << detector.image_timestamp_us << '\t' << i << '\t'
                       << static_cast<int>(armor.number) << '\t'
                       << static_cast<int>(armor.type) << '\t'
                       << static_cast<int>(armor.color) << '\t'
                       << armor.confidence << '\t' << armor.center.x << '\t'
                       << armor.center.y << '\t' << (armor.pnp_valid ? 1 : 0)
                       << '\t' << armor.pose.translation.x() << '\t'
                       << armor.pose.translation.y() << '\t'
                       << armor.pose.translation.z() << '\n';
      }
    }

    for (const auto& metrics : metrics_records)
    {
      metrics_file_ << metrics.image_timestamp_us << '\t' << metrics.frame_index
                    << '\t' << metrics.armor_count << '\t'
                    << metrics.decoded_count << '\t' << metrics.nms_count << '\t'
                    << metrics.pnp_success_count << '\t'
                    << metrics.detector_latency_ms << '\t'
                    << metrics.publish_latency_ms << '\n';
    }

    for (const auto& target : target_records)
    {
      target_file_ << target.image_timestamp_us << '\t'
                   << (target.tracking ? 1 : 0) << '\t'
                   << static_cast<int>(target.id) << '\t' << target.armors_num
                   << '\t' << target.position.x() << '\t' << target.position.y()
                   << '\t' << target.position.z() << '\t' << target.velocity.x()
                   << '\t' << target.velocity.y() << '\t' << target.velocity.z()
                   << '\t' << target.yaw << '\t' << target.v_yaw << '\t'
                   << target.radius_1 << '\t' << target.radius_2 << '\t'
                   << target.dz << '\n';
    }

    for (const auto& ekf : ekf_records)
    {
      ekf_file_ << ekf.image_timestamp_us << "\t0\t" << (ekf.valid[0] ? 1 : 0)
                << '\t' << ekf.center_cam.x() << '\t' << ekf.center_cam.y()
                << '\t' << ekf.center_cam.z() << '\n';
      for (int i = 0; i < 4; ++i)
      {
        ekf_file_ << ekf.image_timestamp_us << '\t' << (i + 1) << '\t'
                  << (ekf.valid[i + 1] ? 1 : 0) << '\t'
                  << ekf.armors_cam[i].x() << '\t' << ekf.armors_cam[i].y()
                  << '\t' << ekf.armors_cam[i].z() << '\n';
      }
    }

    for (const auto& candidate : candidate_records)
    {
      candidate_file_ << candidate.image_timestamp_us << '\t'
                      << static_cast<int>(candidate.count) << '\t'
                      << static_cast<int>(candidate.selected_index) << '\t'
                      << static_cast<int>(candidate.matched) << '\t'
                      << static_cast<int>(candidate.detection_count) << '\t'
                      << static_cast<int>(candidate.tracked_armors_num) << '\n';
      const uint8_t count = std::min<uint8_t>(candidate.count,
                                              CandidateDebugMessage::kMaxItems);
      for (uint8_t i = 0; i < count; ++i)
      {
        const auto& item = candidate.items[i];
        candidate_items_file_ << candidate.image_timestamp_us << '\t'
                              << static_cast<int>(i) << '\t'
                              << static_cast<int>(item.armor_index) << '\t'
                              << static_cast<int>(item.face_index) << '\t'
                              << static_cast<int>(item.same_number) << '\t'
                              << static_cast<int>(item.image_track_id) << '\t'
                              << static_cast<int>(item.image_track_confirmed)
                              << '\t' << static_cast<int>(item.number) << '\t'
                              << static_cast<int>(item.type) << '\t'
                              << item.score << '\t' << item.position_diff
                              << '\t' << item.yaw_diff << '\t' << item.center_x
                              << '\t' << item.center_y << '\t'
                              << item.predicted_yaw << '\t'
                              << item.measured_yaw << '\n';
      }
    }

    FlushRecordFiles();
  }

  void FlushRecordFiles()
  {
    detector_file_.flush();
    metrics_file_.flush();
    target_file_.flush();
    ekf_file_.flush();
    candidate_file_.flush();
    candidate_items_file_.flush();
  }

  bool RecordEnabled() const
  {
    return runtime_.record_raw && record_ready_;
  }

  RuntimeParam runtime_{};
  std::string image_topic_name_;
  std::string output_dir_;
  std::string raw_video_name_;
  std::string overlay_video_name_;
  std::string preview_window_name_;

  std::atomic<bool> running_{false};
  bool realtime_preview_enabled_{false};
  LibXR::Thread image_thread_{};
  LibXR::Thread worker_thread_{};
  std::mutex mutex_;
  std::condition_variable work_cv_;

  std::array<ImageData, image_queue_capacity> image_queue_{};
  std::size_t image_head_{0};
  std::size_t image_count_{0};
  uint32_t image_dropped_{0};
  uint32_t detector_dropped_{0};
  uint32_t tracker_dropped_{0};

  Ring<DetectorMessage, history_capacity> detector_history_{};
  Ring<TargetMessage, history_capacity> target_history_{};
  Ring<EkfPointsMessage, history_capacity> ekf_history_{};
  Ring<CandidateDebugMessage, history_capacity> candidate_history_{};

  Ring<DetectorMessage, record_queue_capacity> detector_record_queue_{};
  Ring<DetectorMetrics, record_queue_capacity> metrics_record_queue_{};
  Ring<TargetMessage, record_queue_capacity> target_record_queue_{};
  Ring<EkfPointsMessage, record_queue_capacity> ekf_record_queue_{};
  Ring<CandidateDebugMessage, record_queue_capacity> candidate_record_queue_{};

  std::ofstream detector_file_{};
  std::ofstream metrics_file_{};
  std::ofstream target_file_{};
  std::ofstream ekf_file_{};
  std::ofstream candidate_file_{};
  std::ofstream candidate_items_file_{};
  cv::VideoWriter raw_writer_{};
  cv::VideoWriter overlay_writer_{};
  bool raw_writer_ready_{false};
  bool overlay_writer_ready_{false};
  bool record_ready_{false};
};
