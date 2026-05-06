# VisionPreview

`VisionPreview` 是只负责实时预览的轻量工具。

它不订阅 topic，不做录像，不写 TSV，也不关心 detector/tracker 的消息类型。需要实时预览的模块直接组合它，在自己的线程里调用 `Submit(frame, draw_callback)`。

`enabled` 是唯一运行开关：`false` 时不启动线程，`Submit()` 直接返回 `false`；`true` 时启动 OpenCV 窗口预览。`Submit()` 会在调用线程里深拷贝输入图像，然后立即返回。预览线程拿到这份拷贝后执行 `draw_callback(cv::Mat&)` 绘制 overlay，再调用 `imshow/waitKey` 显示。预览线程处理不过来时丢旧帧，不反压 detector、tracker 或相机同步线程。

配置项：

- `enabled`：实时预览总开关。
- `preview_window_name`：OpenCV 窗口名。
- `preview_scale`：显示缩放比例，只影响窗口画面。
- `preview_wait_key_ms`：OpenCV 窗口事件轮询时间，单位 ms。
- `queue_capacity`：预览任务队列长度，队列满时丢弃旧帧。

最小用法：

```cpp
VisionPreview preview({
    .enabled = true,
    .preview_window_name = "detector_preview",
    .preview_scale = 0.5,
});

preview.Submit(frame, [result](cv::Mat& canvas) {
  // 在预览线程里绘制 overlay。
});
```
