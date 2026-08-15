#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <thread>

#include "VisionPreview.hpp"

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
}  // namespace

int main()
{
  TestRestartDiscardsQueuedJob();
  TestRestartRejectsInFlightSubmit();
  return EXIT_SUCCESS;
}
