#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 自瞄网页预览：浏览器拉取原始 Bayer 帧与各层结果，在网页端解码和画图 / Auto-aim web preview: the browser pulls raw Bayer frames and stage results and decodes and draws them itself
depends:
- id: QDU-Robomaster/AutoAimTypes
  ref: same-or-dev
=== END MANIFEST === */
// clang-format on

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "AutoAimTypes.hpp"
#include "PreviewPage.hpp"
#include "PreviewServer.hpp"
#include "PreviewSnapshot.hpp"
#include "logger.hpp"
#include "message.hpp"

/// 预览设置 / Preview settings.
struct PreviewSettings
{
  const char*
      camera_name;    ///< 相机名，决定订阅的 Topic / Camera name, selects the Topics
  uint16_t port;      ///< HTTP 端口，0 为由系统分配 / HTTP port, 0 lets the system choose
  uint32_t fresh_ms;  ///< 某层多久内收到过帧算“新鲜” / Freshness window of a stage
};

/**
 * @brief 网页预览。订阅 `<相机名>_synced` 以及存在的 `_detected`、`_tracked`、`_aimed`，
 *        浏览器每请求一次 `/frame`，就从最深且新鲜的一层复制下一帧发回。
 *        Web preview. Subscribes to `<camera>_synced` and whichever of `_detected`,
 *        `_tracked` and `_aimed` exist; each browser request for `/frame` copies the next
 *        frame of the deepest fresh stage and sends it back.
 *
 * 各层回调平时只记序号与到达时间；只有浏览器在等时才复制约 0.3 MB 的 Bayer 字节，
 * 不持有图像句柄。某层卡住时退回到更浅的层，页面显示各层的序号与距今时间。
 * Stage callbacks normally record only the sequence and arrival time; the ~0.3 MB of
 * Bayer bytes are copied only while a browser is waiting, and no image handle is kept.
 * A stalled stage falls back to a shallower one, and the page shows the sequence and age
 * of every stage.
 */
class VisionPreview
{
 public:
  /// 浏览器请求后等一帧的最长时间 / Longest wait for a frame after a request.
  static constexpr std::chrono::milliseconds FRAME_WAIT{300};

  explicit VisionPreview(const PreviewSettings& settings)
      : camera_name_(settings.camera_name), fresh_(settings.fresh_ms)
  {
    Subscribe<AutoAim::SyncedFrame>(Preview::SYNCED, true);
    Subscribe<AutoAim::DetectedFrame>(Preview::DETECTED, false);
    Subscribe<AutoAim::TrackedFrame>(Preview::TRACKED, false);
    Subscribe<AutoAim::AimedFrame>(Preview::AIMED, false);
    server_ = std::make_unique<PreviewServer>(
        settings.port, [this](std::string_view path) { return Handle(path); });
    XR_LOG_INFO("%s preview: port %u", camera_name_.c_str(),
                static_cast<unsigned>(server_->Port()));
  }

  VisionPreview(const VisionPreview&) = delete;
  VisionPreview& operator=(const VisionPreview&) = delete;

  /// 实际监听的端口 / Actual listening port.
  uint16_t Port() const { return server_->Port(); }

 private:
  using Clock = std::chrono::steady_clock;

  struct Binding
  {
    VisionPreview* self;
    Preview::Stage stage;
  };

  struct StageState
  {
    bool present = false;
    bool seen = false;
    uint64_t sequence = 0;
    Clock::time_point arrival{};
  };

  static uint64_t Sequence(const AutoAim::SyncedFrame& f) { return f.sequence; }
  static uint64_t Sequence(const AutoAim::DetectedFrame& f) { return f.synced.sequence; }
  static uint64_t Sequence(const AutoAim::TrackedFrame& f)
  {
    return Sequence(f.detected);
  }
  static uint64_t Sequence(const AutoAim::AimedFrame& f) { return Sequence(f.tracked); }

  /// 同步层必需，其余层存在才订阅 / The synced stage is required, the others optional.
  template <typename Frame>
  void Subscribe(Preview::Stage stage, bool required)
  {
    const std::string name = StageTopicName(camera_name_, Preview::STAGE_NAMES[stage]);
    if (!required && LibXR::Topic::Find(name.c_str()) == nullptr)
    {
      return;
    }
    bindings_[stage] = {this, stage};
    stages_[stage].present = true;
    auto callback = LibXR::Topic::Callback::Create(
        [](bool, Binding* b, const Frame* frame) { b->self->OnFrame(b->stage, *frame); },
        &bindings_[stage]);
    AutoAim::RequireTopic<const Frame*>(name).RegisterCallback(callback);
  }

  template <typename Frame>
  void OnFrame(Preview::Stage stage, const Frame& frame)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    StageState& s = stages_[stage];
    s.seen = true;
    s.sequence = Sequence(frame);
    s.arrival = Clock::now();
    if (wanted_ == stage)
    {
      Preview::Fill(snapshot_, frame);
      wanted_.reset();
      ready_ = true;
      ready_cv_.notify_all();
    }
  }

  PreviewServer::Response Handle(std::string_view path)
  {
    if (path == "/")
    {
      return {200, "text/html; charset=utf-8", std::string(Preview::PAGE_HTML)};
    }
    if (path != "/frame")
    {
      return {404, "text/plain", "not found\n"};
    }
    std::array<Preview::StageStatus, Preview::STAGE_COUNT> status{};
    std::optional<Preview::Snapshot> snapshot = Capture(status);
    if (!snapshot)
    {
      return {503, "text/plain", "no frame\n"};
    }
    const std::string json = Preview::ToJson(*snapshot, status);
    PreviewServer::Response r{200, "application/octet-stream", {}};
    r.body.reserve(4 + json.size() + snapshot->bayer.size());
    const uint32_t len = static_cast<uint32_t>(json.size());
    for (int i = 0; i < 4; ++i)
    {
      r.body.push_back(static_cast<char>((len >> (8 * i)) & 0xFF));
    }
    r.body += json;
    r.body.append(reinterpret_cast<const char*>(snapshot->bayer.data()),
                  snapshot->bayer.size());
    return r;
  }

  /// 先等最深且新鲜的层，等不到再等同步层 / Wait for the deepest fresh stage, then
  /// for the synced stage.
  std::optional<Preview::Snapshot> Capture(
      std::array<Preview::StageStatus, Preview::STAGE_COUNT>& status)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    Preview::Stage stage = Preview::SYNCED;
    const auto now = Clock::now();
    for (int i = Preview::STAGE_COUNT - 1; i > Preview::SYNCED; --i)
    {
      const StageState& s = stages_[i];
      if (s.present && s.seen && now - s.arrival <= fresh_)
      {
        stage = static_cast<Preview::Stage>(i);
        break;
      }
    }
    bool got = WaitFor(lock, stage);
    if (!got && stage != Preview::SYNCED)
    {
      got = WaitFor(lock, Preview::SYNCED);
    }
    const auto done = Clock::now();
    for (int i = 0; i < Preview::STAGE_COUNT; ++i)
    {
      const StageState& s = stages_[i];
      status[i] = {s.present, s.seen, s.sequence,
                   std::chrono::duration<double, std::milli>(done - s.arrival).count()};
    }
    if (!got)
    {
      return std::nullopt;
    }
    return std::move(snapshot_);
  }

  bool WaitFor(std::unique_lock<std::mutex>& lock, Preview::Stage stage)
  {
    wanted_ = stage;
    ready_ = false;
    const bool got = ready_cv_.wait_for(lock, FRAME_WAIT, [this]() { return ready_; });
    wanted_.reset();
    return got;
  }

  const std::string camera_name_;
  const std::chrono::milliseconds fresh_;
  std::array<Binding, Preview::STAGE_COUNT> bindings_{};
  std::mutex mutex_;
  std::condition_variable ready_cv_;
  std::array<StageState, Preview::STAGE_COUNT> stages_{};
  std::optional<Preview::Stage> wanted_;
  bool ready_ = false;
  Preview::Snapshot snapshot_;
  std::unique_ptr<PreviewServer> server_;
};
