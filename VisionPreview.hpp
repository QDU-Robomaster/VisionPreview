#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 实时视觉预览输出工具
constructor_args:
  runtime:
    enabled: false
    preview_window_name: autoaim_preview
    preview_scale: 1.0
    preview_wait_key_ms: 1
    queue_capacity: 1
    output_mode: window
    web_bind_address: 0.0.0.0
    web_port: 8080
    web_stream_name: ""
    max_fps: 30.0
required_hardware: []
depends: []
=== END MANIFEST === */
// clang-format on

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include "logger.hpp"

/**
 * @brief 异步图像预览工具。
 *
 * 调用方提交一帧 OpenCV 图像和一个绘制回调。`VisionPreview` 在自己的线程里执行绘制，
 * 然后输出到 OpenCV 窗口或内置 HTTP BMP 流。提交线程只负责限频、拷贝图像和入队。
 */
class VisionPreview
{
 public:
  /**
   * @brief 运行时配置。
   */
  struct RuntimeParam
  {
    /// 预览总开关；false 时不启动预览线程，Submit() 直接返回 false。
    bool enabled = false;
    /// OpenCV 窗口名；同进程内不同模块应使用不同名字。
    std::string_view preview_window_name = "autoaim_preview";
    /// 显示缩放比例，只影响窗口画面，不修改调用方传入的原图。
    double preview_scale = 1.0;
    /// cv::waitKey 的事件轮询时间，最小按 1 ms 执行。
    int preview_wait_key_ms = 1;
    /// 预览任务队列长度；取值会限制到 [1, 2]，队列满时丢弃等待帧。
    std::size_t queue_capacity = 1;
    /// 输出模式："window" 使用 OpenCV 窗口；"raw/web/http/bmp" 启动未压缩 BMP 推流。
    std::string_view output_mode = "window";
    /// Web 服务监听地址；实机远程查看通常用 "0.0.0.0"。
    std::string_view web_bind_address = "0.0.0.0";
    /// Web 服务端口；浏览器访问 http://<host>:<port>/。
    uint16_t web_port = 8080;
    /// Web 路由名；为空时用 preview_window_name 生成，例如
    /// /stream/armor_detector_preview。
    std::string_view web_stream_name = "";
    /// 预览最大接受帧率；<= 0 表示不限频。限频在 Submit()
    /// 入口执行，未到间隔时不拷贝图像。
    double max_fps = 30.0;
  };

  /**
   * @brief 在预览线程里执行的绘制回调。
   */
  using DrawCallback = std::function<void(cv::Mat&)>;

  /**
   * @brief 构造未启动的预览对象。
   */
  VisionPreview() = default;

  /**
   * @brief 构造并启动预览对象。
   */
  explicit VisionPreview(RuntimeParam runtime) { Start(runtime); }

  /**
   * @brief 停止预览线程和 web stream。
   */
  ~VisionPreview() { Stop(); }

  /**
   * @brief 按配置启动预览。
   *
   * @return 启动成功返回 true；配置关闭或启动失败返回 false。
   */
  bool Start(RuntimeParam runtime)
  {
    if (Running())
    {
      return true;
    }

    runtime_ = runtime;
    preview_window_name_ = std::string(runtime.preview_window_name);
    output_mode_ = std::string(runtime.output_mode);
    web_bind_address_ = std::string(runtime.web_bind_address);
    web_stream_name_ =
        NormalizeStreamName(runtime.web_stream_name.empty() ? runtime.preview_window_name
                                                            : runtime.web_stream_name);
    dropped_frames_.store(0, std::memory_order_relaxed);
    rate_dropped_frames_.store(0, std::memory_order_relaxed);
    accepted_frames_.store(0, std::memory_order_relaxed);
    ConfigureRateLimit(runtime.max_fps);
    if (!ShouldRun())
    {
      return false;
    }
    if (!WindowMode() && !WebMode())
    {
      XR_LOG_ERROR("VisionPreview disabled: unsupported output_mode=%s name=%s",
                   output_mode_.c_str(), preview_window_name_.c_str());
      return false;
    }

    if (WindowMode() && !UiAvailable())
    {
      XR_LOG_WARN("VisionPreview disabled: display backend unavailable window=%s",
                  preview_window_name_.c_str());
      return false;
    }

    queue_capacity_ = runtime.queue_capacity == 0 ? 1U : runtime.queue_capacity;
    queue_capacity_ = std::min<std::size_t>(queue_capacity_, 2U);
    ClearQueuedJobs();
    session_token_.fetch_add(1, std::memory_order_acq_rel);
    running_.store(true, std::memory_order_release);
    if (WebMode() && !StartWebStream())
    {
      running_.store(false, std::memory_order_release);
      session_token_.fetch_add(1, std::memory_order_acq_rel);
      ClearQueuedJobs();
      XR_LOG_ERROR("VisionPreview failed to start web stream name=%s bind=%s port=%u",
                   web_stream_name_.c_str(), web_bind_address_.c_str(),
                   static_cast<unsigned>(runtime_.web_port));
      return false;
    }
    worker_thread_ = std::thread(WorkerThreadMain, this);
    XR_LOG_INFO("VisionPreview started mode=%s name=%s bind=%s port=%u max_fps=%.2f",
                output_mode_.c_str(), preview_window_name_.c_str(),
                web_bind_address_.c_str(), static_cast<unsigned>(runtime_.web_port),
                runtime_.max_fps);
    return true;
  }

  /**
   * @brief 查询预览线程是否正在运行。
   */
  bool Running() const { return running_.load(std::memory_order_acquire); }

  /**
   * @brief 队列满时丢弃的帧数。
   */
  uint32_t DroppedFrames() const
  {
    return dropped_frames_.load(std::memory_order_relaxed);
  }

  /**
   * @brief 因 max_fps 限频丢弃的帧数。
   */
  uint32_t RateDroppedFrames() const
  {
    return rate_dropped_frames_.load(std::memory_order_relaxed);
  }

  /**
   * @brief 已接受并入队的帧数。
   */
  uint32_t AcceptedFrames() const
  {
    return accepted_frames_.load(std::memory_order_relaxed);
  }

  /**
   * @brief 提交一帧图像和绘制回调。
   *
   * 通过限频后会深拷贝 `frame`，调用方可以立即复用原图。`draw` 在预览线程里执行。
   *
   * @return 已接受入队返回 true；未运行、空图像或被限频丢弃返回 false。
   */
  bool Submit(const cv::Mat& frame, DrawCallback draw)
  {
    if (frame.empty())
    {
      return false;
    }

    const uint64_t session_token = session_token_.load(std::memory_order_acquire);
    if (!Running() || !AcceptByRateLimit(session_token))
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
      if (!running_.load(std::memory_order_acquire) ||
          session_token_.load(std::memory_order_acquire) != session_token)
      {
        return false;
      }
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
      accepted_frames_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_.notify_one();
    return true;
  }

  /**
   * @brief 停止预览线程、注销 web stream，并等待后台线程退出。
   */
  void Stop()
  {
    const bool was_running = running_.exchange(false, std::memory_order_acq_rel);
    session_token_.fetch_add(1, std::memory_order_acq_rel);
    cv_.notify_all();
    if (was_running)
    {
      XR_LOG_INFO(
          "VisionPreview stopping mode=%s name=%s accepted=%u rate_dropped=%u "
          "queue_dropped=%u",
          output_mode_.c_str(), preview_window_name_.c_str(), AcceptedFrames(),
          RateDroppedFrames(), DroppedFrames());
    }
    if (worker_thread_.joinable() &&
        worker_thread_.get_id() != std::this_thread::get_id())
    {
      worker_thread_.join();
    }
    ClearQueuedJobs();
    if (web_server_ && web_stream_)
    {
      web_server_->UnregisterStream(web_stream_);
    }
    web_stream_.reset();
    web_server_.reset();
  }

 private:
  struct Job
  {
    /// 预览线程持有的图像拷贝。
    cv::Mat frame;
    /// 针对这帧图像执行的绘制回调。
    DrawCallback draw;

    /**
     * @brief 清空图像和回调。
     */
    void Reset()
    {
      frame.release();
      draw = nullptr;
    }
  };

  /**
   * @brief 根据总开关判断是否需要启动。
   */
  bool ShouldRun() const
  {
    // enabled 是总开关；具体输出由 output_mode 决定。
    return runtime_.enabled;
  }

  /**
   * @brief 当前配置是否为 web/BMP 输出。
   */
  bool WebMode() const
  {
    return output_mode_ == "raw" || output_mode_ == "bmp" || output_mode_ == "web" ||
           output_mode_ == "http";
  }

  /**
   * @brief 当前配置是否为 OpenCV 窗口输出。
   */
  bool WindowMode() const { return output_mode_ == "window"; }

  /**
   * @brief 根据 max_fps 计算提交间隔。
   */
  void ConfigureRateLimit(double max_fps)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (max_fps > 0.0)
    {
      const auto period_ns = static_cast<int64_t>(std::llround(1.0e9 / max_fps));
      min_submit_interval_ = std::chrono::nanoseconds(std::max<int64_t>(period_ns, 1));
    }
    else
    {
      min_submit_interval_ = std::chrono::steady_clock::duration::zero();
    }
    next_accept_time_ = std::chrono::steady_clock::time_point{};
  }

  /**
   * @brief 判断当前提交是否通过限频。
   */
  bool AcceptByRateLimit(uint64_t session_token)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_.load(std::memory_order_acquire) ||
        session_token_.load(std::memory_order_acquire) != session_token)
    {
      return false;
    }
    if (min_submit_interval_ <= std::chrono::steady_clock::duration::zero())
    {
      return true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now < next_accept_time_)
    {
      rate_dropped_frames_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    next_accept_time_ = now + min_submit_interval_;
    return true;
  }

  /**
   * @brief 判断当前进程是否有可用图形显示后端。
   */
  static bool UiAvailable()
  {
    const char* display = std::getenv("DISPLAY");
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    return (display != nullptr && display[0] != '\0') ||
           (wayland_display != nullptr && wayland_display[0] != '\0');
  }

  /**
   * @brief 丢弃队列中最早进入队列的未处理帧。
   */
  void DropOldestLocked()
  {
    if (queued_count_ == 0)
    {
      return;
    }

    // 预览永远不反压主链路；队列满时丢掉最早进入队列的未显示帧。
    job_ = std::move(next_job_);
    next_job_.Reset();
    --queued_count_;
    dropped_frames_.fetch_add(1, std::memory_order_relaxed);
  }

  /**
   * @brief 释放停止预览会话时仍由队列持有的全部任务。
   */
  void ClearQueuedJobs()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    job_.Reset();
    next_job_.Reset();
    queued_count_ = 0;
  }

  /**
   * @brief 从队列取出下一帧预览任务。
   */
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

  /**
   * @brief 预览线程入口。
   */
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

  /**
   * @brief 执行绘制回调、缩放并输出一帧。
   */
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
      OutputFrame(scaled);
    }
    else
    {
      OutputFrame(canvas);
    }
  }

  /**
   * @brief 按当前输出模式显示或推流一帧。
   */
  void OutputFrame(const cv::Mat& canvas)
  {
    if (WebMode())
    {
      PublishWebFrame(canvas);
      return;
    }

    cv::imshow(preview_window_name_, canvas);
    cv::waitKey(std::max(runtime_.preview_wait_key_ms, 1));
  }

  /**
   * @brief 将一帧编码为 BMP 并发布到 web stream。
   */
  void PublishWebFrame(const cv::Mat& canvas)
  {
    std::vector<uchar> encoded = EncodeBmp(canvas);
    if (encoded.empty())
    {
      return;
    }

    auto stream = web_stream_;
    if (!stream)
    {
      return;
    }
    stream->Publish(std::move(encoded));
  }

  /**
   * @brief 向字节数组写入小端 16 位整数。
   */
  static void WriteLe16(std::vector<uchar>& out, std::size_t offset, uint16_t value)
  {
    out[offset] = static_cast<uchar>(value & 0xFFU);
    out[offset + 1U] = static_cast<uchar>((value >> 8U) & 0xFFU);
  }

  /**
   * @brief 向字节数组写入小端 32 位整数。
   */
  static void WriteLe32(std::vector<uchar>& out, std::size_t offset, uint32_t value)
  {
    out[offset] = static_cast<uchar>(value & 0xFFU);
    out[offset + 1U] = static_cast<uchar>((value >> 8U) & 0xFFU);
    out[offset + 2U] = static_cast<uchar>((value >> 16U) & 0xFFU);
    out[offset + 3U] = static_cast<uchar>((value >> 24U) & 0xFFU);
  }

  /**
   * @brief 将 OpenCV 图像编码为 top-down 24 位 BMP。
   */
  static std::vector<uchar> EncodeBmp(const cv::Mat& canvas)
  {
    if (canvas.empty() || canvas.depth() != CV_8U)
    {
      return {};
    }

    cv::Mat bgr;
    if (canvas.channels() == 3)
    {
      bgr = canvas;
    }
    else if (canvas.channels() == 4)
    {
      cv::cvtColor(canvas, bgr, cv::COLOR_BGRA2BGR);
    }
    else if (canvas.channels() == 1)
    {
      cv::cvtColor(canvas, bgr, cv::COLOR_GRAY2BGR);
    }
    else
    {
      return {};
    }

    const int width = bgr.cols;
    const int height = bgr.rows;
    if (width <= 0 || height <= 0)
    {
      return {};
    }

    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3U;
    const std::size_t stride = (row_bytes + 3U) & ~std::size_t{3U};
    const std::size_t image_size = stride * static_cast<std::size_t>(height);
    const std::size_t header_size = 54U;
    const std::size_t file_size = header_size + image_size;
    if (file_size > 0xFFFFFFFFULL)
    {
      return {};
    }

    std::vector<uchar> out(file_size, 0);
    out[0] = static_cast<uchar>('B');
    out[1] = static_cast<uchar>('M');
    WriteLe32(out, 2, static_cast<uint32_t>(file_size));
    WriteLe32(out, 10, static_cast<uint32_t>(header_size));
    WriteLe32(out, 14, 40U);
    WriteLe32(out, 18, static_cast<uint32_t>(width));
    // 负高度表示 top-down BMP，行顺序与 cv::Mat 保持一致。
    WriteLe32(out, 22, static_cast<uint32_t>(-height));
    WriteLe16(out, 26, 1U);
    WriteLe16(out, 28, 24U);
    WriteLe32(out, 34, static_cast<uint32_t>(image_size));

    uchar* dst = out.data() + header_size;
    for (int y = 0; y < height; ++y)
    {
      const uchar* src = bgr.ptr<uchar>(y);
      std::copy(src, src + row_bytes, dst + static_cast<std::size_t>(y) * stride);
    }
    return out;
  }

  /**
   * @brief 一个 web 预览流的最新帧缓存。
   */
  struct WebStream
  {
    /**
     * @brief 创建指定名称的 stream。
     */
    explicit WebStream(std::string stream_name) : name(std::move(stream_name)) {}

    /**
     * @brief 更新最新帧并唤醒等待中的 HTTP 客户端。
     */
    void Publish(std::vector<uchar> frame)
    {
      const auto byte_count = frame.size();
      const bool first_frame =
          !first_frame_logged.exchange(true, std::memory_order_acq_rel);
      auto snapshot = std::make_shared<const std::vector<uchar>>(std::move(frame));
      {
        std::lock_guard<std::mutex> lock(mutex);
        latest_frame = std::move(snapshot);
        ++frame_seq;
      }
      cv.notify_all();
      if (first_frame)
      {
        XR_LOG_INFO(
            "VisionPreview stream first frame: /stream/%s type=image/bmp bytes=%u",
            name.c_str(), static_cast<unsigned>(byte_count));
      }
    }

    /**
     * @brief 标记 stream 关闭并唤醒客户端。
     */
    void Close()
    {
      active.store(false, std::memory_order_release);
      cv.notify_all();
    }

    /// stream 名称，对应 `/stream/<name>`。
    std::string name;
    /// 保护 latest_frame 和 frame_seq。
    std::mutex mutex;
    /// 新帧通知。
    std::condition_variable cv;
    /// 最近一次发布的不可变 BMP 快照。
    std::shared_ptr<const std::vector<uchar>> latest_frame;
    /// 最新帧序号。
    uint64_t frame_seq{0};
    /// stream 是否仍可向客户端输出。
    std::atomic<bool> active{true};
    /// 首帧日志是否已经输出。
    std::atomic<bool> first_frame_logged{false};
  };

  /**
   * @brief 同进程共享的 HTTP 预览服务器。
   */
  class WebServer : public std::enable_shared_from_this<WebServer>
  {
   public:
    /**
     * @brief 获取或创建指定地址和端口的服务器。
     */
    static std::shared_ptr<WebServer> Acquire(std::string bind_address, uint16_t port)
    {
#if defined(_WIN32)
      (void)bind_address;
      (void)port;
      return nullptr;
#else
      const std::string key = bind_address + ":" + std::to_string(port);
      std::lock_guard<std::mutex> lock(RegistryMutex());
      if (auto existing = Registry()[key].lock())
      {
        XR_LOG_INFO("VisionPreview web server reused bind=%s port=%u",
                    bind_address.c_str(), static_cast<unsigned>(port));
        return existing;
      }

      auto server =
          std::shared_ptr<WebServer>(new WebServer(std::move(bind_address), port));
      if (!server->Start())
      {
        XR_LOG_ERROR("VisionPreview web server start failed bind=%s port=%u",
                     server->bind_address_.c_str(), static_cast<unsigned>(server->port_));
        return nullptr;
      }
      Registry()[key] = server;
      return server;
#endif
    }

    /**
     * @brief 停止服务器和客户端线程。
     */
    ~WebServer() { Stop(); }

    /**
     * @brief 注册一个新的预览流。
     */
    std::shared_ptr<WebStream> RegisterStream(const std::string& name)
    {
      std::lock_guard<std::mutex> lock(streams_mutex_);
      if (streams_.find(name) != streams_.end())
      {
        XR_LOG_ERROR("VisionPreview stream name already exists: %s", name.c_str());
        return nullptr;
      }

      auto stream = std::make_shared<WebStream>(name);
      streams_.emplace(name, stream);
      XR_LOG_PASS("VisionPreview stream registered: /stream/%s", name.c_str());
      return stream;
    }

    /**
     * @brief 注销预览流；最后一个流注销后停止服务器。
     */
    void UnregisterStream(const std::shared_ptr<WebStream>& stream)
    {
      if (!stream)
      {
        return;
      }

      bool empty = false;
      {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        auto it = streams_.find(stream->name);
        if (it != streams_.end() && it->second == stream)
        {
          streams_.erase(it);
          XR_LOG_INFO("VisionPreview stream unregistered: /stream/%s",
                      stream->name.c_str());
        }
        empty = streams_.empty();
      }
      stream->Close();
      if (empty)
      {
        XR_LOG_INFO("VisionPreview web server stopping: no active streams");
        Stop();
      }
    }

   private:
    WebServer(std::string bind_address, uint16_t port)
        : bind_address_(std::move(bind_address)), port_(port)
    {
    }

    /**
     * @brief 全局服务器注册表互斥锁。
     */
    static std::mutex& RegistryMutex()
    {
      static std::mutex mutex;
      return mutex;
    }

    /**
     * @brief 按 bind:port 保存已启动服务器。
     */
    static std::map<std::string, std::weak_ptr<WebServer>>& Registry()
    {
      static std::map<std::string, std::weak_ptr<WebServer>> registry;
      return registry;
    }

    /**
     * @brief 创建 socket、绑定端口并启动 accept 线程。
     */
    bool Start()
    {
#if defined(_WIN32)
      return false;
#else
      int fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0)
      {
        XR_LOG_ERROR("VisionPreview web socket create failed");
        return false;
      }

      int opt = 1;
      (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(port_);
      if (bind_address_.empty() || bind_address_ == "0.0.0.0")
      {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
      }
      else if (::inet_pton(AF_INET, bind_address_.c_str(), &addr.sin_addr) != 1)
      {
        XR_LOG_ERROR("VisionPreview web bind address invalid: %s", bind_address_.c_str());
        ::close(fd);
        return false;
      }

      if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
      {
        XR_LOG_ERROR("VisionPreview web bind failed bind=%s port=%u errno=%d",
                     bind_address_.c_str(), static_cast<unsigned>(port_), errno);
        ::close(fd);
        return false;
      }
      if (::listen(fd, 8) != 0)
      {
        XR_LOG_ERROR("VisionPreview web listen failed bind=%s port=%u errno=%d",
                     bind_address_.c_str(), static_cast<unsigned>(port_), errno);
        ::close(fd);
        return false;
      }

      server_fd_.store(fd, std::memory_order_release);
      running_.store(true, std::memory_order_release);
      server_thread_ = std::thread(ServerThreadMain, this);
      XR_LOG_PASS("VisionPreview web server listening bind=%s port=%u",
                  bind_address_.c_str(), static_cast<unsigned>(port_));
      return true;
#endif
    }

    /**
     * @brief 停止服务器、关闭 socket 并等待客户端线程退出。
     */
    void Stop()
    {
      const bool was_running = running_.exchange(false, std::memory_order_acq_rel);
      if (!was_running)
      {
        return;
      }

      CloseServerSocket();
      NotifyStreamsClosed();
      if (server_thread_.joinable() &&
          server_thread_.get_id() != std::this_thread::get_id())
      {
        server_thread_.join();
      }
      JoinClientThreads();
    }

    /**
     * @brief 关闭监听 socket。
     */
    void CloseServerSocket()
    {
#if !defined(_WIN32)
      const int fd = server_fd_.exchange(-1, std::memory_order_acq_rel);
      if (fd >= 0)
      {
        (void)::shutdown(fd, SHUT_RDWR);
        (void)::close(fd);
      }
#endif
    }

    /**
     * @brief 通知所有 stream 关闭。
     */
    void NotifyStreamsClosed()
    {
      std::lock_guard<std::mutex> lock(streams_mutex_);
      for (auto& item : streams_)
      {
        item.second->Close();
      }
    }

    /**
     * @brief HTTP 服务器线程入口。
     */
    static void ServerThreadMain(WebServer* self) { self->ServerLoop(); }

    /**
     * @brief 接受客户端连接。
     */
    void ServerLoop()
    {
#if !defined(_WIN32)
      while (running_.load(std::memory_order_acquire))
      {
        const int fd = server_fd_.load(std::memory_order_acquire);
        if (fd < 0)
        {
          break;
        }

        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(fd, &read_set);
        timeval timeout{0, 200000};
        const int ready = ::select(fd + 1, &read_set, nullptr, nullptr, &timeout);
        if (ready <= 0)
        {
          continue;
        }

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd =
            ::accept(fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0)
        {
          continue;
        }
        XR_LOG_INFO("VisionPreview web client connected fd=%d", client_fd);
        AddClientThread(client_fd);
      }
#endif
    }

    /**
     * @brief 为新客户端创建处理线程。
     */
    void AddClientThread(int client_fd)
    {
      std::lock_guard<std::mutex> lock(client_threads_mutex_);
      client_threads_.emplace_back(ClientThreadMain, this, client_fd);
    }

    /**
     * @brief 等待所有客户端线程退出。
     */
    void JoinClientThreads()
    {
      std::vector<std::thread> threads;
      {
        std::lock_guard<std::mutex> lock(client_threads_mutex_);
        threads.swap(client_threads_);
      }

      for (auto& thread : threads)
      {
        if (thread.joinable() && thread.get_id() != std::this_thread::get_id())
        {
          thread.join();
        }
      }
    }

    /**
     * @brief 客户端处理线程入口。
     */
    static void ClientThreadMain(WebServer* self, int client_fd)
    {
      self->HandleClient(client_fd);
    }

    /**
     * @brief 解析请求并返回首页、404 或 multipart stream。
     */
    void HandleClient(int client_fd)
    {
#if !defined(_WIN32)
      timeval timeout{1, 0};
      (void)::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      (void)::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

      const std::string request = ReadHttpRequest(client_fd);
      const std::string path = ParseRequestPath(request);
      if (path == "/" || path == "/index.html")
      {
        SendIndexPage(client_fd);
        ::close(client_fd);
        return;
      }

      auto stream = ResolveStream(path);
      if (!stream)
      {
        XR_LOG_WARN("VisionPreview web stream not found path=%s", path.c_str());
        static constexpr std::string_view not_found =
            "HTTP/1.1 404 Not Found\r\n"
            "Connection: close\r\n"
            "Content-Type: text/plain\r\n\r\n"
            "stream not found\n";
        (void)SendAll(client_fd, not_found.data(), not_found.size());
        ::close(client_fd);
        return;
      }

      StreamMultipart(client_fd, stream);
      XR_LOG_INFO("VisionPreview web client disconnected stream=%s",
                  stream->name.c_str());
      ::close(client_fd);
#else
      (void)client_fd;
#endif
    }

    /**
     * @brief 读取 HTTP 请求头。
     */
    static std::string ReadHttpRequest(int client_fd)
    {
      std::string request;
#if !defined(_WIN32)
      char buffer[512];
      while (request.size() < 4096)
      {
        const ssize_t n = ::recv(client_fd, buffer, sizeof(buffer), 0);
        if (n <= 0)
        {
          break;
        }
        request.append(buffer, static_cast<std::size_t>(n));
        if (request.find("\r\n\r\n") != std::string::npos)
        {
          break;
        }
      }
#else
      (void)client_fd;
#endif
      return request;
    }

    /**
     * @brief 从 HTTP 请求行提取 path，忽略 query。
     */
    static std::string ParseRequestPath(const std::string& request)
    {
      std::istringstream stream(request);
      std::string method;
      std::string path;
      stream >> method >> path;
      if (path.empty())
      {
        return "/";
      }
      const auto query_pos = path.find('?');
      if (query_pos != std::string::npos)
      {
        path.resize(query_pos);
      }
      return path;
    }

    /**
     * @brief 根据 URL path 查找 stream。
     */
    std::shared_ptr<WebStream> ResolveStream(const std::string& path)
    {
      std::lock_guard<std::mutex> lock(streams_mutex_);
      if (path == "/stream" && streams_.size() == 1)
      {
        return streams_.begin()->second;
      }

      static constexpr std::string_view prefix = "/stream/";
      if (path.size() < prefix.size() ||
          path.compare(0, prefix.size(), prefix.data(), prefix.size()) != 0)
      {
        return nullptr;
      }

      std::string name = path.substr(prefix.size());
      auto it = streams_.find(name);
      if (it == streams_.end())
      {
        return nullptr;
      }
      return it->second;
    }

    /**
     * @brief 返回列出所有 stream 的简单 HTML 页面。
     */
    bool SendIndexPage(int client_fd)
    {
      std::ostringstream body;
      body << "<!doctype html><html><head><meta charset=\"utf-8\">"
           << "<title>VisionPreview</title>"
           << "<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}"
           << "section{padding:12px}img{display:block;max-width:100%;height:auto}"
           << "h2{font-size:16px;font-weight:500}</style></head><body>";
      {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        if (streams_.empty())
        {
          body << "<section><h2>No active streams</h2></section>";
        }
        for (const auto& item : streams_)
        {
          body << "<section><h2>" << item.first << "</h2><img src=\"/stream/"
               << item.first << "\" alt=\"" << item.first << "\"></section>";
        }
      }
      body << "</body></html>";

      const std::string body_str = body.str();
      std::ostringstream response;
      response << "HTTP/1.1 200 OK\r\n"
               << "Connection: close\r\n"
               << "Cache-Control: no-store\r\n"
               << "Content-Type: text/html; charset=utf-8\r\n"
               << "Content-Length: " << body_str.size() << "\r\n\r\n"
               << body_str;
      const std::string data = response.str();
      return SendAll(client_fd, data.data(), data.size());
    }

    /**
     * @brief 按 multipart/x-mixed-replace 输出 BMP 帧。
     */
    void StreamMultipart(int client_fd, const std::shared_ptr<WebStream>& stream)
    {
      static constexpr std::string_view header =
          "HTTP/1.1 200 OK\r\n"
          "Connection: close\r\n"
          "Cache-Control: no-cache, no-store, must-revalidate\r\n"
          "Pragma: no-cache\r\n"
          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
      if (!SendAll(client_fd, header.data(), header.size()))
      {
        return;
      }

      uint64_t last_seq = 0;
      while (running_.load(std::memory_order_acquire) &&
             stream->active.load(std::memory_order_acquire))
      {
        std::shared_ptr<const std::vector<uchar>> frame;
        {
          std::unique_lock<std::mutex> lock(stream->mutex);
          stream->cv.wait(lock,
                          [this, &stream, last_seq]()
                          {
                            return !running_.load(std::memory_order_acquire) ||
                                   !stream->active.load(std::memory_order_acquire) ||
                                   stream->frame_seq != last_seq;
                          });
          if (!running_.load(std::memory_order_acquire) ||
              !stream->active.load(std::memory_order_acquire))
          {
            break;
          }
          last_seq = stream->frame_seq;
          frame = stream->latest_frame;
        }
        if (!frame || frame->empty())
        {
          continue;
        }

        std::ostringstream part_header;
        part_header << "--frame\r\n"
                    << "Content-Type: image/bmp\r\n"
                    << "Content-Length: " << frame->size() << "\r\n\r\n";
        const std::string part = part_header.str();
        if (!SendAll(client_fd, part.data(), part.size()) ||
            !SendAll(client_fd, reinterpret_cast<const char*>(frame->data()),
                     frame->size()) ||
            !SendAll(client_fd, "\r\n", 2))
        {
          break;
        }
      }
    }

    /**
     * @brief 阻塞发送完整缓冲区。
     */
    static bool SendAll(int fd, const char* data, std::size_t size)
    {
#if !defined(_WIN32)
      std::size_t sent = 0;
      while (sent < size)
      {
#if defined(MSG_NOSIGNAL)
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        const ssize_t n = ::send(fd, data + sent, size - sent, flags);
        if (n <= 0)
        {
          if (errno == EINTR)
          {
            continue;
          }
          return false;
        }
        sent += static_cast<std::size_t>(n);
      }
      return true;
#else
      (void)fd;
      (void)data;
      (void)size;
      return false;
#endif
    }

    /// 监听地址。
    std::string bind_address_;
    /// 监听端口。
    uint16_t port_{8080};
    /// 服务器线程是否运行。
    std::atomic<bool> running_{false};
    /// 监听 socket fd。
    std::atomic<int> server_fd_{-1};
    /// accept 线程。
    std::thread server_thread_;
    /// 保护 streams_。
    std::mutex streams_mutex_;
    /// 已注册 stream。
    std::map<std::string, std::shared_ptr<WebStream>> streams_;
    /// 保护 client_threads_。
    std::mutex client_threads_mutex_;
    /// 当前已创建的客户端线程。
    std::vector<std::thread> client_threads_;
  };

  /**
   * @brief 将 stream 名转换成 URL path 可用的名称。
   */
  static std::string NormalizeStreamName(std::string_view name)
  {
    std::string out;
    out.reserve(name.size());
    for (unsigned char ch : name)
    {
      if (std::isalnum(ch) || ch == '_' || ch == '-')
      {
        out.push_back(static_cast<char>(ch));
      }
      else if (ch == '.' || ch == ' ')
      {
        out.push_back('_');
      }
    }
    if (out.empty())
    {
      out = "stream";
    }
    return out;
  }

  /**
   * @brief 启动或复用 web server，并注册当前 stream。
   */
  bool StartWebStream()
  {
    web_server_ = WebServer::Acquire(web_bind_address_, runtime_.web_port);
    if (!web_server_)
    {
      return false;
    }
    web_stream_ = web_server_->RegisterStream(web_stream_name_);
    if (!web_stream_)
    {
      web_server_.reset();
      return false;
    }
    XR_LOG_INFO("VisionPreview web stream ready url=/stream/%s encoding=bmp",
                web_stream_name_.c_str());
    return true;
  }

  /// 当前运行时配置。
  RuntimeParam runtime_{};
  /// OpenCV 窗口名。
  std::string preview_window_name_{"autoaim_preview"};
  /// 输出模式。
  std::string output_mode_{"window"};
  /// Web 监听地址。
  std::string web_bind_address_{"0.0.0.0"};
  /// Web stream 名。
  std::string web_stream_name_{"autoaim_preview"};
  /// 预览处理线程。
  std::thread worker_thread_{};
  /// 预览线程运行标志。
  std::atomic<bool> running_{false};
  /// Stop/Start 生命周期标识，阻止旧会话中仍在拷图的 Submit 跨会话入队。
  std::atomic<uint64_t> session_token_{0};
  /// 队列满丢弃计数。
  std::atomic<uint32_t> dropped_frames_{0};
  /// 限频丢弃计数。
  std::atomic<uint32_t> rate_dropped_frames_{0};
  /// 接受帧计数。
  std::atomic<uint32_t> accepted_frames_{0};
  /// 保护队列和限频状态。
  std::mutex mutex_{};
  /// 新任务通知。
  std::condition_variable cv_{};
  /// 两次接受帧之间的最小间隔。
  std::chrono::steady_clock::duration min_submit_interval_{};
  /// 下一次允许接受帧的时间点。
  std::chrono::steady_clock::time_point next_accept_time_{};
  /// 队列容量，实际限制为 1 或 2。
  std::size_t queue_capacity_{1};
  /// 当前队列中任务数量。
  std::size_t queued_count_{0};
  /// 队首任务。
  Job job_{};
  /// 第二个等待任务。
  Job next_job_{};
  /// 共享 web server。
  std::shared_ptr<WebServer> web_server_{};
  /// 当前实例注册的 web stream。
  std::shared_ptr<WebStream> web_stream_{};
};
