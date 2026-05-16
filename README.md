# VisionPreview

`VisionPreview` 用于实时查看视觉模块输出的画面。调用方提交一帧 OpenCV 图像和绘制
函数，预览线程在图像拷贝上绘制内容，然后输出到 OpenCV 窗口或浏览器。

这个模块不订阅 topic，不保存文件，也不规定绘制内容。Detector、Tracker、Aimer、
VisionCapture 等模块按自己的结果绘制框、文字、轨迹或标定板角点。

## 工作方式

调用 `Start()` 后，`Submit(frame, draw)` 会先检查 `max_fps`。通过限频后，模块在调用线程
深拷贝图像并放入队列，随后立即返回。预览线程取出图像，执行 `draw(cv::Mat&)`，再显示或
推流。

队列长度最多为 `2`。当预览线程来不及处理时，模块丢弃等待帧，不阻塞相机、检测或跟踪线程。

## 输出模式

`output_mode: "window"` 使用 OpenCV 窗口。运行环境必须有 `DISPLAY` 或 `WAYLAND_DISPLAY`。

`output_mode: "raw"`、`"bmp"`、`"web"`、`"http"` 启动内置 HTTP 服务。输出格式是未压缩
BMP，浏览器访问：

```text
http://<host>:<web_port>/
http://<host>:<web_port>/stream/<web_stream_name>
```

同一进程内多个 `VisionPreview` 可以共用同一个 `web_bind_address:web_port`。每个实例注册
自己的 stream。根路径会列出当前所有 stream；只有一个 stream 时也可以访问 `/stream`。

## 配置

- `enabled`：预览总开关。为 `false` 时不启动线程，`Submit()` 返回 `false`。
- `preview_window_name`：OpenCV 窗口名，也是默认 stream 名来源。
- `preview_scale`：输出缩放比例，只影响预览图。
- `preview_wait_key_ms`：窗口模式下 `cv::waitKey()` 的等待时间，单位 ms。
- `queue_capacity`：预览队列长度，实际限制为 `1` 或 `2`。
- `output_mode`：`"window"` 或 `"raw"` / `"bmp"` / `"web"` / `"http"`。
- `web_bind_address`：HTTP 监听地址，实机远程查看通常用 `0.0.0.0`。
- `web_port`：HTTP 监听端口。
- `web_stream_name`：stream 名。为空时由 `preview_window_name` 生成。
- `max_fps`：预览最大接受帧率。小于等于 `0` 表示不限频。

## 示例

窗口预览：

```cpp
VisionPreview preview({
    .enabled = true,
    .preview_window_name = "detector_preview",
    .preview_scale = 0.5,
    .output_mode = "window",
});

preview.Submit(frame, [result](cv::Mat& image) {
  // 在 image 上绘制框、文字或轨迹。
});
```

浏览器预览：

```cpp
VisionPreview preview({
    .enabled = true,
    .preview_window_name = "detector_preview",
    .preview_scale = 0.5,
    .queue_capacity = 1,
    .output_mode = "web",
    .web_bind_address = "0.0.0.0",
    .web_port = 8080,
    .web_stream_name = "armor_detector",
    .max_fps = 30.0,
});
```

打开 `http://<host>:8080/stream/armor_detector` 查看画面。

## 计数

- `AcceptedFrames()`：通过限频并进入队列的帧数。
- `RateDroppedFrames()`：被 `max_fps` 丢弃的帧数。
- `DroppedFrames()`：队列满时丢弃的等待帧数。

这些计数用于判断预览是否拖慢或跟不上视觉链路。
