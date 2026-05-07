#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <cerrno>
#include <cctype>
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
    // 输出模式："window" 使用 OpenCV 窗口；"raw/web/http/bmp" 启动未压缩 BMP 推流。
    std::string_view output_mode = "window";
    // Web 服务监听地址；实机远程查看通常用 "0.0.0.0"。
    std::string_view web_bind_address = "0.0.0.0";
    // Web 服务端口；浏览器访问 http://<host>:<port>/。
    uint16_t web_port = 8080;
    // Web 路由名；为空时用 preview_window_name 生成，例如 /stream/armor_detector_preview。
    std::string_view web_stream_name = "";
    // 预览最大接受帧率；<= 0 表示不限频。限频在 Submit() 入口执行，未到间隔时不拷贝图像。
    double max_fps = 30.0;
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
    output_mode_ = std::string(runtime.output_mode);
    web_bind_address_ = std::string(runtime.web_bind_address);
    web_stream_name_ = NormalizeStreamName(
        runtime.web_stream_name.empty() ? runtime.preview_window_name
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
    running_.store(true, std::memory_order_release);
    if (WebMode() && !StartWebStream())
    {
      running_.store(false, std::memory_order_release);
      XR_LOG_ERROR(
          "VisionPreview failed to start web stream name=%s bind=%s port=%u",
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

  bool Running() const { return running_.load(std::memory_order_acquire); }

  uint32_t DroppedFrames() const { return dropped_frames_.load(std::memory_order_relaxed); }

  uint32_t RateDroppedFrames() const
  {
    return rate_dropped_frames_.load(std::memory_order_relaxed);
  }

  uint32_t AcceptedFrames() const
  {
    return accepted_frames_.load(std::memory_order_relaxed);
  }

  bool Submit(const cv::Mat& frame, DrawCallback draw)
  {
    if (!Running() || frame.empty())
    {
      return false;
    }

    if (!AcceptByRateLimit())
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
    accepted_frames_.fetch_add(1, std::memory_order_relaxed);

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
    const bool was_running = running_.exchange(false, std::memory_order_acq_rel);
    cv_.notify_all();
    if (was_running)
    {
      XR_LOG_INFO(
          "VisionPreview stopping mode=%s name=%s accepted=%u rate_dropped=%u queue_dropped=%u",
          output_mode_.c_str(), preview_window_name_.c_str(),
          AcceptedFrames(), RateDroppedFrames(), DroppedFrames());
    }
    if (worker_thread_.joinable() &&
        worker_thread_.get_id() != std::this_thread::get_id())
    {
      worker_thread_.join();
    }
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
    // enabled 是总开关；具体输出由 output_mode 决定。
    return runtime_.enabled;
  }

  bool WebMode() const
  {
    return output_mode_ == "raw" || output_mode_ == "bmp" ||
           output_mode_ == "web" || output_mode_ == "http";
  }

  bool WindowMode() const { return output_mode_ == "window"; }

  void ConfigureRateLimit(double max_fps)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (max_fps > 0.0)
    {
      const auto period_ns = static_cast<int64_t>(std::llround(1.0e9 / max_fps));
      min_submit_interval_ =
          std::chrono::nanoseconds(std::max<int64_t>(period_ns, 1));
    }
    else
    {
      min_submit_interval_ = std::chrono::steady_clock::duration::zero();
    }
    next_accept_time_ = std::chrono::steady_clock::time_point{};
  }

  bool AcceptByRateLimit()
  {
    if (min_submit_interval_ <= std::chrono::steady_clock::duration::zero())
    {
      return true;
    }

    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    if (now < next_accept_time_)
    {
      rate_dropped_frames_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    next_accept_time_ = now + min_submit_interval_;
    return true;
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
      OutputFrame(scaled);
    }
    else
    {
      OutputFrame(canvas);
    }
  }

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

  static void WriteLe16(std::vector<uchar>& out, std::size_t offset, uint16_t value)
  {
    out[offset] = static_cast<uchar>(value & 0xFFU);
    out[offset + 1U] = static_cast<uchar>((value >> 8U) & 0xFFU);
  }

  static void WriteLe32(std::vector<uchar>& out, std::size_t offset, uint32_t value)
  {
    out[offset] = static_cast<uchar>(value & 0xFFU);
    out[offset + 1U] = static_cast<uchar>((value >> 8U) & 0xFFU);
    out[offset + 2U] = static_cast<uchar>((value >> 16U) & 0xFFU);
    out[offset + 3U] = static_cast<uchar>((value >> 24U) & 0xFFU);
  }

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

  struct WebStream
  {
    explicit WebStream(std::string stream_name) : name(std::move(stream_name)) {}

    void Publish(std::vector<uchar> frame)
    {
      const auto byte_count = frame.size();
      const bool first_frame = !first_frame_logged.exchange(true, std::memory_order_acq_rel);
      {
        std::lock_guard<std::mutex> lock(mutex);
        latest_frame = std::move(frame);
        ++frame_seq;
      }
      cv.notify_all();
      if (first_frame)
      {
        XR_LOG_INFO("VisionPreview stream first frame: /stream/%s type=image/bmp bytes=%u",
                    name.c_str(),
                    static_cast<unsigned>(byte_count));
      }
    }

    void Close()
    {
      active.store(false, std::memory_order_release);
      cv.notify_all();
    }

    std::string name;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<uchar> latest_frame;
    uint64_t frame_seq{0};
    std::atomic<bool> active{true};
    std::atomic<bool> first_frame_logged{false};
  };

  class WebServer : public std::enable_shared_from_this<WebServer>
  {
   public:
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

      auto server = std::shared_ptr<WebServer>(new WebServer(std::move(bind_address), port));
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

    ~WebServer() { Stop(); }

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

    static std::mutex& RegistryMutex()
    {
      static std::mutex mutex;
      return mutex;
    }

    static std::map<std::string, std::weak_ptr<WebServer>>& Registry()
    {
      static std::map<std::string, std::weak_ptr<WebServer>> registry;
      return registry;
    }

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
        XR_LOG_ERROR("VisionPreview web bind address invalid: %s",
                     bind_address_.c_str());
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

    void NotifyStreamsClosed()
    {
      std::lock_guard<std::mutex> lock(streams_mutex_);
      for (auto& item : streams_)
      {
        item.second->Close();
      }
    }

    static void ServerThreadMain(WebServer* self)
    {
      self->ServerLoop();
    }

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

    void AddClientThread(int client_fd)
    {
      std::lock_guard<std::mutex> lock(client_threads_mutex_);
      client_threads_.emplace_back(ClientThreadMain, this, client_fd);
    }

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

    static void ClientThreadMain(WebServer* self, int client_fd)
    {
      self->HandleClient(client_fd);
    }

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
        std::vector<uchar> frame;
        {
          std::unique_lock<std::mutex> lock(stream->mutex);
          stream->cv.wait(lock, [this, &stream, last_seq]()
                          { return !running_.load(std::memory_order_acquire) ||
                                   !stream->active.load(std::memory_order_acquire) ||
                                   stream->frame_seq != last_seq; });
          if (!running_.load(std::memory_order_acquire) ||
              !stream->active.load(std::memory_order_acquire))
          {
            break;
          }
          last_seq = stream->frame_seq;
          frame = stream->latest_frame;
        }
        if (frame.empty())
        {
          continue;
        }

        std::ostringstream part_header;
        part_header << "--frame\r\n"
                    << "Content-Type: image/bmp\r\n"
                    << "Content-Length: " << frame.size() << "\r\n\r\n";
        const std::string part = part_header.str();
        if (!SendAll(client_fd, part.data(), part.size()) ||
            !SendAll(client_fd, reinterpret_cast<const char*>(frame.data()), frame.size()) ||
            !SendAll(client_fd, "\r\n", 2))
        {
          break;
        }
      }
    }

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

    std::string bind_address_;
    uint16_t port_{8080};
    std::atomic<bool> running_{false};
    std::atomic<int> server_fd_{-1};
    std::thread server_thread_;
    std::mutex streams_mutex_;
    std::map<std::string, std::shared_ptr<WebStream>> streams_;
    std::mutex client_threads_mutex_;
    std::vector<std::thread> client_threads_;
  };

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

  RuntimeParam runtime_{};
  std::string preview_window_name_{"autoaim_preview"};
  std::string output_mode_{"window"};
  std::string web_bind_address_{"0.0.0.0"};
  std::string web_stream_name_{"autoaim_preview"};
  std::thread worker_thread_{};
  std::atomic<bool> running_{false};
  std::atomic<uint32_t> dropped_frames_{0};
  std::atomic<uint32_t> rate_dropped_frames_{0};
  std::atomic<uint32_t> accepted_frames_{0};
  std::mutex mutex_{};
  std::condition_variable cv_{};
  std::chrono::steady_clock::duration min_submit_interval_{};
  std::chrono::steady_clock::time_point next_accept_time_{};
  std::size_t queue_capacity_{1};
  std::size_t queued_count_{0};
  Job job_{};
  Job next_job_{};
  std::shared_ptr<WebServer> web_server_{};
  std::shared_ptr<WebStream> web_stream_{};
};
