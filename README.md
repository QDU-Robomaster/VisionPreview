# VisionPreview

`VisionPreview` 是视觉链路的统一预览与落盘模块。算法模块只发布 topic，不再直接创建窗口、绘制 overlay 或写视频。

## 数据输入

- 图像：订阅 `CameraFrameSync` 暴露的共享图像 topic。
- 检测：订阅 `armor_detector/armors_result` 和 `armor_detector/metrics`。
- 跟踪：订阅 `tracker/target`、`tracker/ekf_points`、`tracker/candidate_debug`。
- 弹道：如果编译时存在 `Aimer.hpp`，额外订阅 `aimer/trajectory`。

所有 overlay 都按 `image_timestamp_us` 对齐。图像队列只保留最新帧，预览线程处理不过来时丢旧图，不反压 detector/tracker。
弹道 overlay 使用同帧 `tracker/target` 和 `tracker/ekf_points` 估计 `world -> camera`，只负责投影 Aimer 发布的模型弹道，不在预览模块里重新解算弹道。`record_overlay` 会直接输出带 overlay 的视频文件，不需要桌面录屏。

## 配置开关

- `enabled`：总开关。
- `record_raw`：写原始视频和 topic 数据 TSV。
- `record_overlay`：直接写带 overlay 的视频文件。
- `realtime_preview`：打开实时窗口。
- `overlay.detector`：绘制 detector 框、角点和置信度。
- `overlay.tracker`：绘制 tracker EKF 中心和装甲板点。
- `overlay.aimer_trajectory`：绘制 Aimer 的模型弹道曲线和命中点。
- `overlay.candidate_debug`：显示候选统计。

关闭 `record_raw`、`record_overlay` 和 `realtime_preview` 时，模块不会启动线程或注册回调。
