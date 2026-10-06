// 快照：原生坐标换到帧坐标、目标与瞄点的投影、JSON；HTTP：页面、帧、最深新鲜层与回退。
//
// Snapshot: native-to-frame mapping, target and aim projection, JSON. HTTP: the page,
// frames, the deepest fresh stage and the fallback.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "VisionPreview.hpp"
#include "libxr.hpp"

namespace
{
void Expect(bool condition, const char* message)
{
  if (!condition)
  {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

bool Near(double a, double b) { return std::abs(a - b) < 1e-6; }

const CameraTypes::CameraCalibration CALIBRATION{1440,  1080,  1000.0,         1000.0,
                                                 720.0, 540.0, {0, 0, 0, 0, 0}};
const CameraTypes::FrameGeometry WIDE{80, 24, 2};

SharedFrame MakeImage(ImagePool& pool, uint32_t counter)
{
  ImagePool::Handle h;
  Expect(pool.Acquire(h) == LibXR::ErrorCode::OK, "image slot");
  h->timestamp_us = LibXR::MicrosecondTimestamp(1000ULL * counter);
  h->geometry = WIDE;
  h->frame_counter = counter;
  h->calibration = &CALIBRATION;
  for (std::size_t i = 0; i < h->data.size(); ++i)
  {
    h->data[i] = static_cast<uint8_t>((i + counter) & 0xFF);
  }
  return SharedFrame(std::move(h));
}

/// 世界系 x 右 y 前 z 上到光学系 x 右 y 下 z 前 / World to optical axes.
void SetWorldToOptical(AutoAim::TrackedFrame& f)
{
  f.output_to_camera_rotation = {1, 0, 0, 0, 0, -1, 0, 1, 0};
  f.output_to_camera_translation = {0, 0, 0};
}

void TestSnapshot()
{
  ImagePool pool(2);
  AutoAim::AimedFrame aimed{};
  AutoAim::SyncedFrame& synced = aimed.tracked.detected.synced;
  synced.sequence = 7;
  synced.image = MakeImage(pool, 3);
  AutoAim::Armor armor{ArmorColor::BLUE, ArmorNumber::THREE, ArmorType::SMALL, 0.9F, {}};
  armor.corners = {AutoAim::Point2f{700.5F, 500.5F},
                   {700.5F, 520.5F},
                   {740.5F, 520.5F},
                   {740.5F, 500.5F}};
  aimed.tracked.detected.armors.push_back(armor);
  SetWorldToOptical(aimed.tracked);
  ArmorTrackerTarget& t = aimed.tracked.target;
  t.tracking = true;
  t.id = ArmorNumber::THREE;
  t.armors_num = 4;
  t.position = Eigen::Vector3d(0.0, 3.0, 0.0);
  t.radius_1 = t.radius_2 = 0.25;
  aimed.aim = {true, true, 0.1F, 0.02F, Eigen::Vector3d(0.0, 2.75, 0.0), 0};

  Preview::Snapshot s;
  Preview::Fill(s, aimed);
  Expect(s.stage == Preview::AIMED && s.sequence == 7 && s.frame_counter == 3, "header");
  Expect(s.bayer.size() == CameraTypes::FRAME_BYTES && s.bayer[10] == 13, "bytes copied");
  // 跳采 2：x = (x_n − 80 + 0.5) / 2 / Decimation 2.
  Expect(s.armors.size() == 1 && Near(s.armors[0].corners[0].x, 310.5) &&
             Near(s.armors[0].corners[0].y, 238.5),
         "armor corners in frame pixels");
  // 中心在光轴上：原生 (720, 540) → 帧 (320.25, 258.25) / Centre on the optical axis.
  Expect(s.target && s.target->centre && Near(s.target->centre->x, 320.25) &&
             Near(s.target->centre->y, 258.25),
         "target centre projected");
  Expect(s.target->plates.size() == 4, "four plates in front of the camera");
  Expect(s.aim && s.aim->point && Near(s.aim->point->x, 320.25), "aim point projected");

  std::array<Preview::StageStatus, Preview::STAGE_COUNT> status{};
  const std::string json = Preview::ToJson(s, status);
  Expect(json.find("\"stage\":\"aimed\"") != std::string::npos, "json stage");
  Expect(json.find("\"number\":\"three\"") != std::string::npos, "json armor number");
  Expect(json.find("\"fire\":true") != std::string::npos, "json aim");
  Expect(json.find("\"centre\":[320.2,258.2]") != std::string::npos ||
             json.find("\"centre\":[320.3,258.3]") != std::string::npos,
         "json target centre");
}

struct HttpReply
{
  int status = 0;
  std::string body;
};

HttpReply Get(uint16_t port, const char* path)
{
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  Expect(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "connect");
  const std::string request = std::string("GET ") + path + " HTTP/1.1\r\nHost: t\r\n\r\n";
  Expect(::send(fd, request.data(), request.size(), 0) ==
             static_cast<ssize_t>(request.size()),
         "send");
  std::string reply;
  char buf[65536];
  ssize_t n = 0;
  while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0)
  {
    reply.append(buf, static_cast<std::size_t>(n));
  }
  ::close(fd);
  HttpReply r;
  r.status = std::atoi(reply.c_str() + 9);
  const std::size_t head = reply.find("\r\n\r\n");
  Expect(head != std::string::npos, "http header");
  r.body = reply.substr(head + 4);
  return r;
}

std::string FrameJson(const HttpReply& r)
{
  Expect(r.status == 200 && r.body.size() > 4, "frame reply");
  uint32_t len = 0;
  for (int i = 0; i < 4; ++i)
  {
    len |= static_cast<uint32_t>(static_cast<uint8_t>(r.body[i])) << (8 * i);
  }
  Expect(r.body.size() == 4 + len + CameraTypes::FRAME_BYTES,
         "body = length + json + bayer");
  return r.body.substr(4, len);
}

void TestServer()
{
  LibXR::Topic synced_topic =
      LibXR::Topic::CreateTopic<const AutoAim::SyncedFrame*>("pv_synced");
  LibXR::Topic detected_topic =
      LibXR::Topic::CreateTopic<const AutoAim::DetectedFrame*>("pv_detected");
  // 进程结束前不析构：Topic 回调没有注销 / Kept to process end: callbacks stay.
  auto* preview = new VisionPreview({"pv", 0, 100});
  const uint16_t port = preview->Port();
  Expect(port != 0, "listening");

  std::atomic<bool> run{true};
  std::atomic<bool> with_detection{true};
  std::thread producer(
      [&]()
      {
        ImagePool pool(4);
        for (uint32_t i = 1; run.load(); ++i)
        {
          AutoAim::DetectedFrame frame{};
          frame.synced.sequence = i;
          frame.synced.image = MakeImage(pool, i);
          const AutoAim::SyncedFrame* s = &frame.synced;
          synced_topic.Publish(s);
          if (with_detection.load())
          {
            const AutoAim::DetectedFrame* d = &frame;
            detected_topic.Publish(d);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  const HttpReply page = Get(port, "/");
  Expect(page.status == 200 && page.body.find("<canvas") != std::string::npos, "page");
  Expect(Get(port, "/nothing").status == 404, "404");

  const std::string deep = FrameJson(Get(port, "/frame"));
  std::printf("with detections: %.80s...\n", deep.c_str());
  Expect(deep.find("\"stage\":\"detected\"") != std::string::npos, "deepest fresh stage");

  with_detection.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const std::string shallow = FrameJson(Get(port, "/frame"));
  std::printf("detector stalled: %.80s...\n", shallow.c_str());
  Expect(shallow.find("\"stage\":\"synced\"") != std::string::npos,
         "falls back to synced");
  Expect(shallow.find("\"name\":\"detected\",\"present\":true,\"seen\":true") !=
             std::string::npos,
         "stalled stage still listed");

  run.store(false);
  producer.join();
}
}  // namespace

int main()
{
  LibXR::PlatformInit();
  TestSnapshot();
  TestServer();
  std::puts("vision_preview_test passed");
  std::fflush(stdout);
  std::_Exit(EXIT_SUCCESS);
}
