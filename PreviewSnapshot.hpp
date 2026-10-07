#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "AutoAimTypes.hpp"

/**
 * @brief 预览的一帧：原始 Bayer 字节加上画图用的叠加数据，坐标均为帧像素。
 *        One preview frame: the raw Bayer bytes plus the overlay to draw, all in frame
 *        pixels.
 *
 * 从某一层的帧复制而来，不持有图像句柄。
 * Copied from a stage frame; holds no image handle.
 */
namespace Preview
{
/// 层序号，越大越深 / Stage index, deeper is larger.
enum Stage : int
{
  SYNCED = 0,
  DETECTED = 1,
  TRACKED = 2,
  AIMED = 3,
  STAGE_COUNT = 4,
};

inline constexpr std::array<std::string_view, STAGE_COUNT> STAGE_NAMES = {
    AutoAim::STAGE_SYNCED, AutoAim::STAGE_DETECTED, AutoAim::STAGE_TRACKED,
    AutoAim::STAGE_AIMED};

/// 帧像素中的一点 / A point in frame pixels.
struct Pixel
{
  double x;
  double y;
};

struct ArmorOverlay
{
  ArmorColor color;
  ArmorNumber number;
  float confidence;
  std::array<Pixel, 4> corners;  ///< 左上、左下、右下、右上 / TL, BL, BR, TR
};

/// 跟踪目标：中心与各板的四角（相机后方的点不画）/ Tracked target: the centre and
/// the corners of every plate (points behind the camera are left out).
struct TargetOverlay
{
  ArmorNumber id;
  std::optional<Pixel> centre;
  std::vector<std::array<Pixel, 4>> plates;  ///< 只含整板在相机前方的 / Fully in front
  int face;                                  ///< 跟踪器的当前板 / Tracker's face
};

struct AimOverlay
{
  bool control;
  bool fire;
  float yaw;
  float pitch;
  int plate;
  std::optional<Pixel> point;
};

struct Snapshot
{
  Stage stage = SYNCED;
  uint64_t sequence = 0;
  uint64_t timestamp_us = 0;
  uint32_t frame_counter = 0;
  CameraTypes::FrameGeometry geometry{};
  std::array<float, 4> rotation_wxyz{1, 0, 0, 0};
  std::vector<ArmorOverlay> armors;
  std::optional<TargetOverlay> target;
  std::optional<AimOverlay> aim;
  std::vector<uint8_t> bayer;  ///< FRAME_BYTES 字节 BayerRG8 / BayerRG8 bytes
};

/// 世界点经相机投影到帧像素；在相机后方时为空 / Project a world point to frame
/// pixels; empty behind the camera.
inline std::optional<Pixel> ProjectToFrame(const Eigen::Vector3d& world,
                                           const AutoAim::TrackedFrame& tracked,
                                           const CameraTypes::CameraCalibration& cal,
                                           const CameraTypes::FrameGeometry& geometry)
{
  const auto& r = tracked.output_to_camera_rotation;
  const auto& t = tracked.output_to_camera_translation;
  double pc[3];
  for (int i = 0; i < 3; ++i)
  {
    pc[i] =
        r[3 * i] * world.x() + r[3 * i + 1] * world.y() + r[3 * i + 2] * world.z() + t[i];
  }
  if (pc[2] <= 1e-3)
  {
    return std::nullopt;
  }
  const double x = pc[0] / pc[2];
  const double y = pc[1] / pc[2];
  const auto& d = cal.distortion;
  const double r2 = x * x + y * y;
  const double radial = 1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2;
  const double xd = x * radial + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x);
  const double yd = y * radial + d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y;
  const auto frame =
      CameraTypes::NativeToFrame(geometry, {cal.fx * xd + cal.cx, cal.fy * yd + cal.cy});
  return Pixel{frame.x, frame.y};
}

/// 同步层：图像与 IMU / Synced stage: image and IMU.
inline void Fill(Snapshot& s, const AutoAim::SyncedFrame& f)
{
  const ImageFrame& image = *f.image;
  s.stage = SYNCED;
  s.sequence = f.sequence;
  s.timestamp_us = static_cast<uint64_t>(image.timestamp_us);
  s.frame_counter = image.frame_counter;
  s.geometry = image.geometry;
  s.rotation_wxyz = f.imu.rotation_wxyz;
  s.bayer.assign(image.data.begin(), image.data.end());
  s.armors.clear();
  s.target.reset();
  s.aim.reset();
}

/// 检测层：再加检测框 / Detected stage: adds the detections.
inline void Fill(Snapshot& s, const AutoAim::DetectedFrame& f)
{
  Fill(s, f.synced);
  s.stage = DETECTED;
  for (const AutoAim::Armor& a : f.armors)
  {
    ArmorOverlay o{a.color, a.number, a.confidence, {}};
    for (int k = 0; k < 4; ++k)
    {
      const auto p =
          CameraTypes::NativeToFrame(s.geometry, {a.corners[k].x, a.corners[k].y});
      o.corners[k] = {p.x, p.y};
    }
    s.armors.push_back(o);
  }
}

/// 跟踪层：再加目标的中心与各板 / Tracked stage: adds the target centre and plates.
inline void Fill(Snapshot& s, const AutoAim::TrackedFrame& f)
{
  Fill(s, f.detected);
  s.stage = TRACKED;
  const ArmorTrackerTarget& t = f.target;
  const CameraTypes::CameraCalibration* cal = f.detected.synced.image->calibration;
  if (!t.tracking || cal == nullptr || t.armors_num <= 0)
  {
    return;
  }
  TargetOverlay o{
      t.id, ProjectToFrame(t.position, f, *cal, s.geometry), {}, t.tracked_face_index};
  const bool large = t.id == ArmorNumber::ONE || t.id == ArmorNumber::BASE;
  const double hw =
      large ? AutoAim::LARGE_ARMOR_HALF_WIDTH : AutoAim::SMALL_ARMOR_HALF_WIDTH;
  const double hh = AutoAim::ARMOR_HALF_HEIGHT;
  // 相机在世界系中的位置 −Rᵀt，只画朝向相机的板（背面与侧面被车身挡住）。
  // Camera position in the world, −Rᵀt; only plates facing the camera are drawn (the
  // back and the sides are hidden by the body).
  const auto& rot = f.output_to_camera_rotation;
  const auto& tr = f.output_to_camera_translation;
  Eigen::Vector3d camera = Eigen::Vector3d::Zero();
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      camera(j) -= rot[3 * i + j] * tr[i];
    }
  }
  for (int i = 0; i < t.armors_num; ++i)
  {
    const double a = t.yaw + i * 2.0 * M_PI / t.armors_num;
    const bool odd = t.armors_num == 4 && i % 2 == 1;
    const double r = odd ? t.radius_2 : t.radius_1;
    const Eigen::Vector3d centre(t.position.x() + r * std::sin(a),
                                 t.position.y() - r * std::cos(a),
                                 t.position.z() + (odd ? t.dz : 0.0));
    const Eigen::Vector3d outward(std::sin(a), -std::cos(a), 0.0);
    if (outward.dot(camera - centre) <= 0.0)
    {
      continue;
    }
    const Eigen::Vector3d side(std::cos(a), std::sin(a), 0.0);  // 板的横向 / Lateral
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    const Eigen::Vector3d corners[4] = {
        centre - hw * side + hh * up, centre - hw * side - hh * up,
        centre + hw * side - hh * up, centre + hw * side + hh * up};
    std::array<Pixel, 4> plate{};
    bool visible = true;
    for (int k = 0; k < 4 && visible; ++k)
    {
      const auto p = ProjectToFrame(corners[k], f, *cal, s.geometry);
      visible = p.has_value();
      if (visible)
      {
        plate[k] = *p;
      }
    }
    if (visible)
    {
      o.plates.push_back(plate);
    }
  }
  s.target = o;
}

/// 瞄准层：再加瞄准结果 / Aimed stage: adds the aim result.
inline void Fill(Snapshot& s, const AutoAim::AimedFrame& f)
{
  Fill(s, f.tracked);
  s.stage = AIMED;
  const AutoAim::AimResult& a = f.aim;
  AimOverlay o{a.control, a.fire, a.yaw, a.pitch, a.plate, std::nullopt};
  const CameraTypes::CameraCalibration* cal =
      f.tracked.detected.synced.image->calibration;
  if (a.plate >= 0 && cal != nullptr)
  {
    o.point = ProjectToFrame(a.aim_point, f.tracked, *cal, s.geometry);
  }
  s.aim = o;
}

/// 各层最新一帧的序号与距今时间 / Latest sequence and age of every stage.
struct StageStatus
{
  bool present;  ///< 该层 Topic 是否存在 / Whether the stage Topic exists
  bool seen;     ///< 是否收到过 / Whether any frame arrived
  uint64_t sequence;
  double age_ms;
};

namespace Detail
{
inline void AppendPixel(std::string& out, const Pixel& p)
{
  char buf[48];
  std::snprintf(buf, sizeof(buf), "[%.1f,%.1f]", p.x, p.y);
  out += buf;
}

inline void AppendQuad(std::string& out, const std::array<Pixel, 4>& q)
{
  out += '[';
  for (int k = 0; k < 4; ++k)
  {
    if (k > 0)
    {
      out += ',';
    }
    AppendPixel(out, q[k]);
  }
  out += ']';
}
}  // namespace Detail

/// 叠加数据与各层状态写成 JSON / The overlay and the stage status as JSON.
inline std::string ToJson(const Snapshot& s,
                          const std::array<StageStatus, STAGE_COUNT>& stages)
{
  using Detail::AppendPixel;
  using Detail::AppendQuad;
  std::string out;
  out.reserve(1024);
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "{\"stage\":\"%.*s\",\"sequence\":%llu,\"timestamp_us\":%llu,"
                "\"frame_counter\":%u,\"width\":%u,\"height\":%u,"
                "\"geometry\":{\"roi_x\":%u,\"roi_y\":%u,\"decimation\":%u},"
                "\"quaternion\":[%.5f,%.5f,%.5f,%.5f],",
                static_cast<int>(STAGE_NAMES[s.stage].size()),
                STAGE_NAMES[s.stage].data(), static_cast<unsigned long long>(s.sequence),
                static_cast<unsigned long long>(s.timestamp_us), s.frame_counter,
                CameraTypes::FRAME_WIDTH, CameraTypes::FRAME_HEIGHT, s.geometry.roi_x,
                s.geometry.roi_y, static_cast<unsigned>(s.geometry.decimation),
                s.rotation_wxyz[0], s.rotation_wxyz[1], s.rotation_wxyz[2],
                s.rotation_wxyz[3]);
  out += buf;

  out += "\"stages\":[";
  for (int i = 0; i < STAGE_COUNT; ++i)
  {
    const StageStatus& st = stages[i];
    std::snprintf(buf, sizeof(buf),
                  "%s{\"name\":\"%.*s\",\"present\":%s,\"seen\":%s,\"sequence\":%llu,"
                  "\"age_ms\":%.1f}",
                  i > 0 ? "," : "", static_cast<int>(STAGE_NAMES[i].size()),
                  STAGE_NAMES[i].data(), st.present ? "true" : "false",
                  st.seen ? "true" : "false",
                  static_cast<unsigned long long>(st.sequence), st.age_ms);
    out += buf;
  }
  out += "],\"armors\":[";
  for (std::size_t i = 0; i < s.armors.size(); ++i)
  {
    const ArmorOverlay& a = s.armors[i];
    const auto n = static_cast<std::size_t>(a.number);
    const std::string_view name =
        n < ARMOR_NUMBER_NAMES.size() ? ARMOR_NUMBER_NAMES[n] : "?";
    std::snprintf(buf, sizeof(buf),
                  "%s{\"color\":\"%s\",\"number\":\"%.*s\",\"conf\":%.2f,\"corners\":",
                  i > 0 ? "," : "",
                  a.color == ArmorColor::RED      ? "red"
                  : a.color == ArmorColor::BLUE   ? "blue"
                  : a.color == ArmorColor::PURPLE ? "purple"
                  : a.color == ArmorColor::OFF    ? "off"
                                                  : "unknown",
                  static_cast<int>(name.size()), name.data(), a.confidence);
    out += buf;
    AppendQuad(out, a.corners);
    out += '}';
  }
  out += "],\"target\":";
  if (s.target)
  {
    const TargetOverlay& t = *s.target;
    const auto n = static_cast<std::size_t>(t.id);
    const std::string_view name =
        n < ARMOR_NUMBER_NAMES.size() ? ARMOR_NUMBER_NAMES[n] : "?";
    std::snprintf(buf, sizeof(buf), "{\"number\":\"%.*s\",\"face\":%d,\"centre\":",
                  static_cast<int>(name.size()), name.data(), t.face);
    out += buf;
    if (t.centre)
    {
      AppendPixel(out, *t.centre);
    }
    else
    {
      out += "null";
    }
    out += ",\"plates\":[";
    for (std::size_t i = 0; i < t.plates.size(); ++i)
    {
      if (i > 0)
      {
        out += ',';
      }
      AppendQuad(out, t.plates[i]);
    }
    out += "]}";
  }
  else
  {
    out += "null";
  }
  out += ",\"aim\":";
  if (s.aim)
  {
    const AimOverlay& a = *s.aim;
    std::snprintf(buf, sizeof(buf),
                  "{\"control\":%s,\"fire\":%s,\"yaw\":%.4f,\"pitch\":%.4f,\"plate\":%d,"
                  "\"point\":",
                  a.control ? "true" : "false", a.fire ? "true" : "false", a.yaw, a.pitch,
                  a.plate);
    out += buf;
    if (a.point)
    {
      AppendPixel(out, *a.point);
    }
    else
    {
      out += "null";
    }
    out += '}';
  }
  else
  {
    out += "null";
  }
  out += '}';
  return out;
}
}  // namespace Preview
