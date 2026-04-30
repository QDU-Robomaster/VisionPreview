# VisionPreview

`VisionPreview` 是视觉链路的统一预览与落盘模块。算法模块只发布 topic，不再直接创建窗口、绘制 overlay 或写视频。

## 数据输入

- 图像：订阅 `CameraFrameSync` 暴露的共享图像 topic。
- 检测：订阅 `armor_detector/armors_result` 和 `armor_detector/metrics`。
- 跟踪：订阅 `tracker/target`、`tracker/ekf_points`、`tracker/candidate_debug`。

所有 overlay 都按 `image_timestamp_us` 对齐。图像队列只保留最新帧，预览线程处理不过来时丢旧图，不反压 detector/tracker。

## 配置开关

- `enabled`：总开关。
- `record_raw`：写原始视频、overlay 视频和 topic 数据 TSV。
- `realtime_preview`：打开实时窗口。
- `overlay.detector`：绘制 detector 框、角点和置信度。
- `overlay.tracker`：绘制 tracker 中心和 tracker 选中观测对应的 EKF 面。
- `overlay.candidate_debug`：显示候选统计。
- `overlay.model_faces`：额外绘制 EKF 模型补全的全部装甲板点，默认关闭。

关闭 `record_raw` 和 `realtime_preview` 时，模块不会启动线程或注册回调。

`record_raw` 打开后默认写：

- `raw.avi`：原始图像。
- `overlay.avi`：带 detector、tracker 匹配面和状态信息的预览视频。
- `detector.tsv`、`metrics.tsv`、`target.tsv`、`ekf_points.tsv`、
  `candidate_debug.tsv`、`candidate_items.tsv`：按 topic 落盘的数据。

## 颜色约定

- detector 普通框：按装甲板颜色绘制。
- `M`：tracker 当前匹配的 detector 观测，品红色。
- `TC`：tracker 目标中心，绿色。
- `EF`：tracker 匹配面的 EKF 投影，黄色。
- `model*`：EKF 模型补全的其它装甲板点，灰色，仅 `overlay.model_faces=true` 时显示。
