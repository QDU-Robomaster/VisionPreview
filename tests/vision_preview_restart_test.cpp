#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#endif

#include "VisionPreview.hpp"

struct VisionPreviewTestAccess
{
  static void FailNextWorkerStart(VisionPreview& preview)
  {
    preview.fail_next_worker_start_for_test_.store(true, std::memory_order_release);
  }

  static void FailNextWorkerStartWithBadAlloc(VisionPreview& preview)
  {
    preview.fail_next_worker_start_with_bad_alloc_for_test_.store(
        true, std::memory_order_release);
  }

  static bool HasWebResources(const VisionPreview& preview)
  {
    return preview.web_server_ != nullptr || preview.web_stream_ != nullptr;
  }

  static std::shared_ptr<void> HoldWebServer(const VisionPreview& preview)
  {
    return preview.web_server_;
  }

  static std::size_t ClientThreadCount(VisionPreview& preview)
  {
    auto server = preview.web_server_;
    if (!server)
    {
      return 0;
    }
    std::lock_guard<std::mutex> lock(server->client_threads_mutex_);
    return server->client_threads_.size();
  }

  static bool PublishWebPayload(VisionPreview& preview, std::size_t size)
  {
    auto stream = preview.web_stream_;
    if (!stream)
    {
      return false;
    }
    stream->Publish(std::vector<uchar>(size, 0));
    return true;
  }

  static bool LargeClientSendInProgress(VisionPreview& preview)
  {
    auto server = preview.web_server_;
    if (!server)
    {
      return false;
    }
    std::lock_guard<std::mutex> lock(server->client_threads_mutex_);
    for (const auto& client : server->client_threads_)
    {
      if (client.connection && client.connection->LargeSendInProgress())
      {
        return true;
      }
    }
    return false;
  }
};

namespace
{
using namespace std::chrono_literals;

template <typename Predicate>
bool WaitUntil(Predicate predicate, std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

void Expect(bool condition, const char* message)
{
  if (!condition)
  {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <std::size_t Count, typename Callback>
void RunConcurrently(Callback callback)
{
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> run{false};
  std::array<std::thread, Count> threads;
  for (std::size_t index = 0; index < Count; ++index)
  {
    threads[index] = std::thread(
        [&, index]()
        {
          ready.fetch_add(1, std::memory_order_release);
          while (!run.load(std::memory_order_acquire))
          {
            std::this_thread::yield();
          }
          callback(index);
        });
  }
  Expect(WaitUntil([&]() { return ready.load(std::memory_order_acquire) == Count; }, 2s),
         "concurrent lifecycle callers did not become ready");
  run.store(true, std::memory_order_release);
  for (auto& thread : threads)
  {
    thread.join();
  }
}

#if !defined(_WIN32)
uint16_t FindAvailablePort()
{
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  Expect(fd >= 0, "failed to create port reservation socket");

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  Expect(::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
         "failed to reserve localhost port");

  socklen_t address_length = sizeof(address);
  Expect(::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &address_length) == 0,
         "failed to read reserved localhost port");
  const uint16_t port = ntohs(address.sin_port);
  (void)::close(fd);
  return port;
}

int ConnectClient(uint16_t port, int receive_buffer_size = 0)
{
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0)
  {
    return -1;
  }

  timeval timeout{2, 0};
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  if (receive_buffer_size > 0)
  {
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer_size,
                       sizeof(receive_buffer_size));
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(port);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
  {
    (void)::close(fd);
    return -1;
  }
  return fd;
}

bool SendBytes(int fd, std::string_view data)
{
  std::size_t sent = 0;
  while (sent < data.size())
  {
    const ssize_t result = ::send(fd, data.data() + sent, data.size() - sent, 0);
    if (result < 0 && errno == EINTR)
    {
      continue;
    }
    if (result <= 0)
    {
      return false;
    }
    sent += static_cast<std::size_t>(result);
  }
  return true;
}

bool ReadHttpHeaders(int fd)
{
  std::string response;
  std::array<char, 512> buffer{};
  while (response.size() < 8192U)
  {
    const ssize_t result = ::recv(fd, buffer.data(), buffer.size(), 0);
    if (result < 0 && errno == EINTR)
    {
      continue;
    }
    if (result <= 0)
    {
      return false;
    }
    response.append(buffer.data(), static_cast<std::size_t>(result));
    if (response.find("\r\n\r\n") != std::string::npos)
    {
      return response.rfind("HTTP/1.1 200 OK", 0) == 0;
    }
  }
  return false;
}

void CloseClient(int fd)
{
  if (fd >= 0)
  {
    (void)::shutdown(fd, SHUT_RDWR);
    (void)::close(fd);
  }
}

bool SendIndexRequest(uint16_t port)
{
  const int fd = ConnectClient(port);
  if (fd < 0)
  {
    return false;
  }
  static constexpr std::string_view request =
      "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
  const bool success = SendBytes(fd, request) && ReadHttpHeaders(fd);
  CloseClient(fd);
  return success;
}
#endif

void ExpectBoundedStop(VisionPreview& preview, const char* message)
{
  std::atomic<bool> stopped{false};
  const auto start = std::chrono::steady_clock::now();
  std::thread stopper(
      [&]()
      {
        preview.Stop();
        stopped.store(true, std::memory_order_release);
      });
  Expect(WaitUntil([&]() { return stopped.load(std::memory_order_acquire); }, 2s),
         message);
  stopper.join();
  Expect(std::chrono::steady_clock::now() - start < 2s, message);
}

void TestRestartDiscardsQueuedJob()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview preview;
  const VisionPreview::RuntimeParam runtime{
      .enabled = true,
      .preview_window_name = "restart_test",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 2,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = 0,
      .web_stream_name = "restart_test",
      .max_fps = 0.0,
  };
  Expect(preview.Start(runtime), "initial preview start failed");

  const cv::Mat frame(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
  std::atomic<bool> processing{false};
  std::atomic<bool> release_processing{false};
  std::atomic<bool> stale_job_ran{false};

  Expect(preview.Submit(frame,
                        [&](cv::Mat&)
                        {
                          processing.store(true, std::memory_order_release);
                          while (!release_processing.load(std::memory_order_acquire))
                          {
                            std::this_thread::yield();
                          }
                        }),
         "failed to submit blocking job");
  Expect(WaitUntil([&]() { return processing.load(std::memory_order_acquire); }, 2s),
         "blocking job did not start");
  Expect(preview.Submit(frame, [&](cv::Mat&)
                        { stale_job_ran.store(true, std::memory_order_release); }),
         "failed to queue stale job");

  std::thread stopper([&]() { preview.Stop(); });
  Expect(WaitUntil([&]() { return !preview.Running(); }, 2s),
         "preview did not begin stopping");
  release_processing.store(true, std::memory_order_release);
  stopper.join();
  Expect(!stale_job_ran.load(std::memory_order_acquire), "queued job ran while stopping");

  Expect(preview.Start(runtime), "preview restart failed");
  std::this_thread::sleep_for(100ms);
  Expect(!stale_job_ran.load(std::memory_order_acquire),
         "stale job ran after preview restart");
  preview.Stop();
#endif
}

void TestRestartRejectsInFlightSubmit()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview preview;
  const VisionPreview::RuntimeParam runtime{
      .enabled = true,
      .preview_window_name = "submit_race_test",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 2,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = 0,
      .web_stream_name = "submit_race_test",
      .max_fps = 0.0,
  };

  for (int iteration = 0; iteration < 4; ++iteration)
  {
    Expect(preview.Start(runtime), "preview start for submit race failed");
    const cv::Mat large_frame(4096, 4096, CV_8UC3, cv::Scalar(0, 0, 0));
    std::atomic<bool> submit_started{false};
    std::atomic<bool> restarted{false};
    std::atomic<bool> old_callback_after_restart{false};
    std::thread submitter(
        [&]()
        {
          submit_started.store(true, std::memory_order_release);
          (void)preview.Submit(large_frame,
                               [&](cv::Mat&)
                               {
                                 if (restarted.load(std::memory_order_acquire))
                                 {
                                   old_callback_after_restart.store(
                                       true, std::memory_order_release);
                                 }
                               });
        });
    Expect(
        WaitUntil([&]() { return submit_started.load(std::memory_order_acquire); }, 2s),
        "concurrent submit did not start");
    std::this_thread::sleep_for(1ms);
    preview.Stop();
    submitter.join();

    restarted.store(true, std::memory_order_release);
    Expect(preview.Start(runtime), "preview restart for submit race failed");
    std::this_thread::sleep_for(100ms);
    Expect(!old_callback_after_restart.load(std::memory_order_acquire),
           "in-flight old submit ran after preview restart");
    preview.Stop();
  }
#endif
}

void TestConcurrentLifecycleCalls()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview preview;
  const VisionPreview::RuntimeParam runtime{
      .enabled = true,
      .preview_window_name = "lifecycle_test",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 1,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = 0,
      .web_stream_name = "lifecycle_test",
      .max_fps = 0.0,
  };

  for (int iteration = 0; iteration < 8; ++iteration)
  {
    std::array<bool, 4> started{};
    RunConcurrently<4>([&](std::size_t index)
                       { started[index] = preview.Start(runtime); });
    for (bool result : started)
    {
      Expect(result, "concurrent preview start failed");
    }

    RunConcurrently<4>([&](std::size_t) { preview.Stop(); });
    Expect(!preview.Running(), "concurrent preview stop left it running");
  }

  for (int iteration = 0; iteration < 8; ++iteration)
  {
    RunConcurrently<6>(
        [&](std::size_t index)
        {
          if ((index & 1U) == 0U)
          {
            (void)preview.Start(runtime);
          }
          else
          {
            preview.Stop();
          }
        });
    preview.Stop();
  }
#endif
}

void TestWorkerStartFailureRollsBack()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview preview;
  const VisionPreview::RuntimeParam runtime{
      .enabled = true,
      .preview_window_name = "worker_failure_test",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 1,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = 0,
      .web_stream_name = "worker_failure_test",
      .max_fps = 0.0,
  };

  VisionPreviewTestAccess::FailNextWorkerStart(preview);
  Expect(!preview.Start(runtime), "injected worker start failure was not reported");
  Expect(!preview.Running(), "worker start failure left preview running");
  Expect(!VisionPreviewTestAccess::HasWebResources(preview),
         "worker start failure retained web resources");

  Expect(preview.Start(runtime), "preview did not recover after worker start failure");
  preview.Stop();

  VisionPreviewTestAccess::FailNextWorkerStartWithBadAlloc(preview);
  Expect(!preview.Start(runtime), "injected worker bad_alloc was not reported");
  Expect(!preview.Running(), "worker bad_alloc left preview running");
  Expect(!VisionPreviewTestAccess::HasWebResources(preview),
         "worker bad_alloc retained web resources");

  Expect(preview.Start(runtime), "preview did not recover after worker bad_alloc");
  preview.Stop();
#endif
}

void TestWorkerFailureStopsAndRestarts()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview preview;
  const VisionPreview::RuntimeParam runtime{
      .enabled = true,
      .preview_window_name = "worker_failure_test",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 1,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = 0,
      .web_stream_name = "worker_failure_test",
      .max_fps = 0.0,
  };

  Expect(preview.Start(runtime), "worker failure test start failed");
  const cv::Mat frame(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
  Expect(preview.Submit(frame, [](cv::Mat&) { throw std::runtime_error("injected"); }),
         "worker failure test submit failed");
  Expect(WaitUntil([&]() { return !preview.Running(); }, 2s),
         "worker exception did not stop the preview session");

  Expect(preview.Start(runtime),
         "preview did not restart directly after worker exception");
  preview.Stop();
  Expect(!VisionPreviewTestAccess::HasWebResources(preview),
         "worker exception restart retained web resources");
#endif
}

void TestCompletedClientThreadsAreReaped()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview preview;
  const uint16_t port = FindAvailablePort();
  const VisionPreview::RuntimeParam runtime{
      .enabled = true,
      .preview_window_name = "client_reap_test",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 1,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = port,
      .web_stream_name = "client_reap_test",
      .max_fps = 0.0,
  };
  Expect(preview.Start(runtime), "client reap preview start failed");

  for (int request = 0; request < 32; ++request)
  {
    Expect(SendIndexRequest(port), "web index request failed");
  }
  Expect(
      WaitUntil(
          [&]() { return VisionPreviewTestAccess::ClientThreadCount(preview) == 0; }, 2s),
      "completed web client threads were not reaped");
  preview.Stop();
#endif
}

void TestSlowRequestDoesNotBlockStop()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview preview;
  const uint16_t port = FindAvailablePort();
  const VisionPreview::RuntimeParam runtime{
      .enabled = true,
      .preview_window_name = "slow_request_test",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 1,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = port,
      .web_stream_name = "slow_request_test",
      .max_fps = 0.0,
  };
  Expect(preview.Start(runtime), "slow request preview start failed");

  const int client_fd = ConnectClient(port);
  Expect(client_fd >= 0, "slow request client connect failed");
  Expect(SendBytes(client_fd, "G"), "slow request partial send failed");
  Expect(
      WaitUntil(
          [&]() { return VisionPreviewTestAccess::ClientThreadCount(preview) == 1; }, 2s),
      "slow request client was not accepted");
  ExpectBoundedStop(preview, "slow request client blocked preview stop");
  CloseClient(client_fd);
#endif
}

void TestSlowReaderDoesNotBlockStop()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview preview;
  const uint16_t port = FindAvailablePort();
  const VisionPreview::RuntimeParam runtime{
      .enabled = true,
      .preview_window_name = "slow_reader_test",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 1,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = port,
      .web_stream_name = "slow_reader_test",
      .max_fps = 0.0,
  };
  Expect(preview.Start(runtime), "slow reader preview start failed");

  const int client_fd = ConnectClient(port, 4096);
  Expect(client_fd >= 0, "slow reader client connect failed");
  static constexpr std::string_view request =
      "GET /stream/slow_reader_test HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
  Expect(SendBytes(client_fd, request), "slow reader request failed");
  Expect(ReadHttpHeaders(client_fd), "slow reader stream handshake failed");
  Expect(VisionPreviewTestAccess::PublishWebPayload(preview, 8U * 1024U * 1024U),
         "slow reader payload publish failed");
  Expect(WaitUntil(
             [&]()
             { return VisionPreviewTestAccess::LargeClientSendInProgress(preview); }, 2s),
         "slow reader did not enter the large send path");
  ExpectBoundedStop(preview, "slow reader blocked preview stop");
  CloseClient(client_fd);
#endif
}

void TestRetiredServerCanBeHeldWhilePortIsReacquired()
{
#if defined(_WIN32)
  return;
#else
  VisionPreview first_preview;
  VisionPreview second_preview;
  const uint16_t port = FindAvailablePort();
  const VisionPreview::RuntimeParam first_runtime{
      .enabled = true,
      .preview_window_name = "server_reuse_first",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 1,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = port,
      .web_stream_name = "server_reuse_first",
      .max_fps = 0.0,
  };
  const VisionPreview::RuntimeParam second_runtime{
      .enabled = true,
      .preview_window_name = "server_reuse_second",
      .preview_scale = 1.0,
      .preview_wait_key_ms = 1,
      .queue_capacity = 1,
      .output_mode = "web",
      .web_bind_address = "127.0.0.1",
      .web_port = port,
      .web_stream_name = "server_reuse_second",
      .max_fps = 0.0,
  };

  Expect(first_preview.Start(first_runtime), "first server reuse preview start failed");
  auto held_server = VisionPreviewTestAccess::HoldWebServer(first_preview);
  Expect(held_server != nullptr, "failed to retain first preview web server");
  first_preview.Stop();

  Expect(second_preview.Start(second_runtime),
         "replacement server preview start failed while retired server was retained");
  auto replacement_server = VisionPreviewTestAccess::HoldWebServer(second_preview);
  Expect(replacement_server != nullptr, "failed to retain replacement web server");
  Expect(replacement_server != held_server,
         "replacement preview reused the retired web server");
  Expect(SendIndexRequest(port), "replacement web server accept loop was not running");
  second_preview.Stop();
  replacement_server.reset();
  held_server.reset();
#endif
}
}  // namespace

int main()
{
  TestRestartDiscardsQueuedJob();
  TestRestartRejectsInFlightSubmit();
  TestConcurrentLifecycleCalls();
  TestWorkerStartFailureRollsBack();
  TestWorkerFailureStopsAndRestarts();
  TestCompletedClientThreadsAreReaped();
  TestSlowRequestDoesNotBlockStop();
  TestSlowReaderDoesNotBlockStop();
  TestRetiredServerCanBeHeldWhilePortIsReacquired();
  return EXIT_SUCCESS;
}
