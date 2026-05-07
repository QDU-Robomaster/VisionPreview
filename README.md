# VisionPreview

`VisionPreview` 是只负责实时预览的轻量工具。

它不订阅 topic，不做录像，不写 TSV，也不关心 detector/tracker 的消息类型。需要实时预览的模块直接组合它，在自己的线程里调用 `Submit(frame, draw_callback)`。

`enabled` 是总开关：`false` 时不启动线程，`Submit()` 直接返回 `false`；`true` 时根据 `output_mode` 输出预览。`output_mode: "window"` 使用 OpenCV 窗口；`output_mode: "raw"`、`"web"`、`"http"` 或 `"bmp"` 启动未压缩 BMP 推流，不依赖显示屏或 `DISPLAY`。`Submit()` 会在调用线程里深拷贝输入图像，然后立即返回。预览线程拿到这份拷贝后执行 `draw_callback(cv::Mat&)` 绘制 overlay，再显示或推流。预览线程处理不过来时丢旧帧，不反压 detector、tracker 或相机同步线程。

多个模块可以同时开 web 预览。同一个进程内相同 `web_bind_address:web_port` 会复用同一个 HTTP server，每个 `VisionPreview` 实例注册一个独立 stream。浏览器打开根路径会看到所有 stream；单独取流路径是 `/stream/<web_stream_name>`。关键节点会通过 `XR_LOG_*` 打印：server 启动/复用、stream 注册/注销、客户端连接/断开、404 和首帧发布。

配置项：

- `enabled`：实时预览总开关。
- `preview_window_name`：OpenCV 窗口名。
- `preview_scale`：显示缩放比例，只影响窗口画面。
- `preview_wait_key_ms`：OpenCV 窗口事件轮询时间，单位 ms。
- `queue_capacity`：预览任务队列长度，队列满时丢弃旧帧。
- `output_mode`：输出模式；`"window"` 为 OpenCV 窗口，`"raw"` / `"web"` / `"http"` / `"bmp"` 为未压缩 BMP 推流。
- `web_bind_address`：web 监听地址，默认 `0.0.0.0`。
- `web_port`：web 监听端口，默认 `8080`。
- `web_stream_name`：web stream 名；为空时由 `preview_window_name` 生成。

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

Headless 未压缩推流用法：

```cpp
VisionPreview preview({
    .enabled = true,
    .preview_window_name = "detector_preview",
    .preview_scale = 0.5,
    .queue_capacity = 1,
    .output_mode = "raw",
    .web_bind_address = "0.0.0.0",
    .web_port = 8080,
    .web_stream_name = "armor_detector",
});
```

浏览器打开 `http://<host>:8080/` 查看所有画面；直接取流地址是 `http://<host>:8080/stream/armor_detector`。如果同端口只有一个 stream，也兼容 `/stream`。
