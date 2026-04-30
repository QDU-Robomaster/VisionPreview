# VisionPreview

`VisionPreview` 是视觉链路的统一预览与落盘模块。算法模块只发布 topic，不再直接创建窗口、绘制 overlay 或写视频。

## 数据输入

- 图像：订阅 `CameraFrameSync` 暴露的共享图像 topic。
- 检测：订阅 `armor_detector/armors_result` 和 `armor_detector/metrics`。
- 跟踪：订阅 `tracker/target`、`tracker/ekf_points`、`tracker/candidate_debug`。

所有 overlay 都按 `image_timestamp_us` 对齐。图像队列只保留最新帧，预览线程处理不过来时丢旧图，不反压 detector/tracker。

## 配置开关

- `enabled`：总开关。
- `record_raw`：写原始视频和 topic 数据 TSV。
- `realtime_preview`：打开实时窗口。
- `overlay.detector`：绘制 detector 框、角点和置信度。
- `overlay.tracker`：绘制 tracker EKF 中心和装甲板点。
- `overlay.candidate_debug`：显示候选统计。

关闭 `record_raw` 和 `realtime_preview` 时，模块不会启动线程或注册回调。
