#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

class VisionPreview
{
 public:
  struct RuntimeParam
  {
    // 预览子系统总开关；false 时不启动预览线程，Submit() 直接返回 false。
    bool enabled = false;
    // OpenCV 窗口名；同进程内不同模块应使用不同名字。
    std::string_view preview_window_name = "autoaim_preview";
    // 显示缩放比例，只影响窗口画面，不修改调用方传入的原图。
    double preview_scale = 1.0;
    // cv::waitKey 的事件轮询时间，最小按 1 ms 执行。
    int preview_wait_key_ms = 1;
    // 预览任务队列长度；取值会限制到 [1, 2]，队列满时丢弃旧帧。
    std::size_t queue_capacity = 1;
  };

  using DrawCallback = std::function<void(cv::Mat&)>;

  VisionPreview() = default;

  explicit VisionPreview(RuntimeParam runtime) { Start(runtime); }

  ~VisionPreview() { Stop(); }

  bool Start(RuntimeParam runtime)
  {
    if (Running())
    {
      return true;
    }

    runtime_ = runtime;
    preview_window_name_ = std::string(runtime.preview_window_name);
    dropped_frames_.store(0, std::memory_order_relaxed);
    if (!ShouldRun())
    {
      return false;
    }

    if (!UiAvailable())
    {
      return false;
    }

    queue_capacity_ = runtime.queue_capacity == 0 ? 1U : runtime.queue_capacity;
    queue_capacity_ = std::min<std::size_t>(queue_capacity_, 2U);
    running_.store(true, std::memory_order_release);
    worker_thread_ = std::thread(WorkerThreadMain, this);
    return true;
  }

  bool Running() const { return running_.load(std::memory_order_acquire); }

  uint32_t DroppedFrames() const { return dropped_frames_.load(std::memory_order_relaxed); }

  bool Submit(const cv::Mat& frame, DrawCallback draw)
  {
    if (!Running() || frame.empty())
    {
      return false;
    }

    Job job;
    // 深拷贝发生在调用线程，保证调用方可以立即复用或释放原始图像。
    job.frame = frame.clone();
    job.draw = std::move(draw);
    if (job.frame.empty())
    {
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      while (queued_count_ >= queue_capacity_)
      {
        DropOldestLocked();
      }

      if (queued_count_ == 0)
      {
        job_ = std::move(job);
        queued_count_ = 1;
      }
      else
      {
        next_job_ = std::move(job);
        queued_count_ = 2;
      }
    }
    cv_.notify_one();
    return true;
  }

  void Stop()
  {
    running_.store(false, std::memory_order_release);
    cv_.notify_all();
    if (worker_thread_.joinable() &&
        worker_thread_.get_id() != std::this_thread::get_id())
    {
      worker_thread_.join();
    }
  }

 private:
  struct Job
  {
    cv::Mat frame;
    DrawCallback draw;

    void Reset()
    {
      frame.release();
      draw = nullptr;
    }
  };

  bool ShouldRun() const
  {
    // preview-only 版本只剩窗口输出，enabled 就是唯一运行开关。
    return runtime_.enabled;
  }

  static bool UiAvailable()
  {
    const char* display = std::getenv("DISPLAY");
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    return (display != nullptr && display[0] != '\0') ||
           (wayland_display != nullptr && wayland_display[0] != '\0');
  }

  void DropOldestLocked()
  {
    if (queued_count_ == 0)
    {
      return;
    }

    // 预览永远不反压主链路；队列满时丢掉最旧的未显示帧。
    job_ = std::move(next_job_);
    next_job_.Reset();
    --queued_count_;
    dropped_frames_.fetch_add(1, std::memory_order_relaxed);
  }

  bool Pop(Job& job)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]()
             { return !running_.load(std::memory_order_acquire) || queued_count_ != 0; });
    if (!running_.load(std::memory_order_acquire))
    {
      return false;
    }

    job = std::move(job_);
    job_ = std::move(next_job_);
    next_job_.Reset();
    --queued_count_;
    return true;
  }

  static void WorkerThreadMain(VisionPreview* self)
  {
    while (self->running_.load(std::memory_order_acquire))
    {
      Job job;
      if (!self->Pop(job))
      {
        break;
      }
      self->Process(job);
    }
  }

  void Process(Job& job)
  {
    cv::Mat canvas = std::move(job.frame);
    if (job.draw)
    {
      // overlay 回调只在预览线程执行，避免 UI 绘制阻塞 detector/tracker。
      job.draw(canvas);
    }

    if (runtime_.preview_scale > 0.0 && std::abs(runtime_.preview_scale - 1.0) > 1e-6)
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

  RuntimeParam runtime_{};
  std::string preview_window_name_{"autoaim_preview"};
  std::thread worker_thread_{};
  std::atomic<bool> running_{false};
  std::atomic<uint32_t> dropped_frames_{0};
  std::mutex mutex_{};
  std::condition_variable cv_{};
  std::size_t queue_capacity_{1};
  std::size_t queued_count_{0};
  Job job_{};
  Job next_job_{};
};
