# VisionPreview

实时视觉预览库：异步绘制并输出到 OpenCV 窗口或浏览器 / Real-time vision preview library that draws asynchronously and outputs to an OpenCV window or a browser

## 1. 模块作用 / Purpose

VisionPreview 用于实时查看视觉模块输出的画面。调用方提交一帧 OpenCV 图像和绘制函数，预览线程在图像拷贝上绘制内容，再输出到 OpenCV 窗口或浏览器。ArmorDetector、ArmorTracker、Aimer、VisionCapture 等模块按各自的结果绘制框、文字、轨迹或标定板角点。

调用 `Start()`（或使用带 `RuntimeParam` 的构造函数）后，`Submit(frame, draw)` 先检查 `max_fps`。通过限频后，模块在调用线程深拷贝图像并放入队列，随即返回。预览线程取出图像，执行 `draw(cv::Mat&)`，按 `preview_scale` 缩放，再显示或推流。

队列长度至多为 2。预览线程来不及处理时，模块丢弃最早的等待帧，相机、检测或跟踪线程继续运行。`Stop()` 释放仍在等待的任务并等待预览线程退出；同一实例再次 `Start()` 时处理新会话的帧。预览线程抛出异常时，模块记录错误并停止本次会话，之后可以重新 `Start()`。

VisionPreview shows the output of the vision Modules in real time. The caller submits an OpenCV image and a draw function, the preview thread draws on a copy of the image, and the result is output to an OpenCV window or a browser. Modules such as ArmorDetector, ArmorTracker, Aimer and VisionCapture draw boxes, text, trajectories or board corners according to their own results.

After `Start()` is called (or the constructor with `RuntimeParam` is used), `Submit(frame, draw)` first checks `max_fps`. When the rate limit is passed, the Module deep-copies the image in the calling thread, puts it into the queue and returns. The preview thread takes the image, runs `draw(cv::Mat&)`, scales it by `preview_scale` and then displays or streams it.

The queue length is at most 2. When the preview thread cannot keep up, the Module drops the oldest waiting frame and the camera, detection or tracking threads continue to run. `Stop()` releases the tasks still waiting and waits for the preview thread to exit; when the same instance is started again, it processes the frames of the new session. When the preview thread throws an exception, the Module logs the error and stops the session, after which `Start()` can be called again.

## 2. 输出模式 / Output Modes

`output_mode: "window"` 使用 OpenCV 窗口，运行环境需要 `DISPLAY` 或 `WAYLAND_DISPLAY`，缺少时预览不启动。

`output_mode: "raw"`、`"bmp"`、`"web"`、`"http"` 启动内置 HTTP 服务（非 Windows 平台）。每帧编码为未压缩 24 位 BMP，以 `multipart/x-mixed-replace` 推送，浏览器访问：

```text
http://<host>:<web_port>/
http://<host>:<web_port>/stream/<web_stream_name>
```

同一进程内的多个 `VisionPreview` 共用同一个 `web_bind_address:web_port`。每个实例注册自己的 stream，stream 名各不相同。根路径列出当前所有 stream；只有一个 stream 时也可访问 `/stream`。每个服务同时服务的客户端最多 16 个；最后一个 stream 注销后服务关闭。

计数：

- `AcceptedFrames()`：通过限频并进入队列的帧数。
- `RateDroppedFrames()`：被 `max_fps` 丢弃的帧数。
- `DroppedFrames()`：队列满时丢弃的等待帧数。

这些计数用于判断预览是否跟得上视觉链路，`Stop()` 时打印到日志。

`output_mode: "window"` uses an OpenCV window and requires `DISPLAY` or `WAYLAND_DISPLAY` in the environment; the preview does not start when both are missing.

`output_mode: "raw"`, `"bmp"`, `"web"` and `"http"` start the built-in HTTP service (non-Windows platforms). Each frame is encoded as an uncompressed 24-bit BMP and pushed as `multipart/x-mixed-replace`; the browser opens:

```text
http://<host>:<web_port>/
http://<host>:<web_port>/stream/<web_stream_name>
```

Several `VisionPreview` instances in one process share one `web_bind_address:web_port`. Each instance registers its own stream, and the stream names differ. The root path lists all current streams; `/stream` is also available when there is only one stream. Each service serves at most 16 clients at the same time; the service shuts down after the last stream is unregistered.

Counters:

- `AcceptedFrames()`: frames that passed the rate limit and entered the queue.
- `RateDroppedFrames()`: frames dropped by `max_fps`.
- `DroppedFrames()`: waiting frames dropped because the queue was full.

The counters show whether the preview keeps up with the vision chain and are printed to the log at `Stop()`.

## 3. 构造接口 / Constructor

```cpp
class VisionPreview {
 public:
  struct RuntimeParam { ... };
  using DrawCallback = std::function<void(cv::Mat&)>;

  VisionPreview();                               // 未启动 / not started
  explicit VisionPreview(RuntimeParam runtime);  // 构造并 Start(runtime) / construct and Start(runtime)

  bool Start(RuntimeParam runtime);
  void Stop();
  bool Running() const;
  bool Submit(const cv::Mat& frame, DrawCallback draw);

  uint32_t AcceptedFrames() const;
  uint32_t RateDroppedFrames() const;
  uint32_t DroppedFrames() const;
};
```

`Start()` 在配置关闭或启动失败时返回 `false`。`Submit()` 在未运行、空图像、被限频或会话已切换时返回 `false`，成功入队时返回 `true`。

依赖：无。

配置参数（`RuntimeParam`）：

- `enabled`：预览总开关，默认 `false`；为 `false` 时不启动线程，`Submit()` 返回 `false`。
- `preview_window_name`：OpenCV 窗口名，也是默认 stream 名的来源，默认 `"autoaim_preview"`。
- `preview_scale`：输出缩放比例，作用于预览图，默认 `1.0`。
- `preview_wait_key_ms`：窗口模式下 `cv::waitKey()` 的等待时间，单位 ms，最小按 1 执行，默认 `1`。
- `queue_capacity`：预览队列长度，取值限制在 1 到 2，默认 `1`。
- `output_mode`：`"window"`，或 `"raw"` / `"bmp"` / `"web"` / `"http"`，默认 `"window"`。
- `web_bind_address`：HTTP 监听地址，远程查看时使用 `"0.0.0.0"`，默认 `"0.0.0.0"`。
- `web_port`：HTTP 监听端口，默认 `8080`。
- `web_stream_name`：stream 名，默认为空，此时由 `preview_window_name` 生成（保留字母、数字、`_`、`-`，`.` 和空格转为 `_`）。
- `max_fps`：预览接受的最大帧率，默认 `30.0`；小于等于 0 表示不限频。

`Start()` returns `false` when the configuration is disabled or the start fails. `Submit()` returns `false` when the preview is not running, the image is empty, the rate limit drops the frame or the session has changed, and `true` when the frame was queued.

Dependencies: none.

Configuration parameters (`RuntimeParam`):

- `enabled`: master switch of the preview, default `false`; with `false` no thread is started and `Submit()` returns `false`.
- `preview_window_name`: OpenCV window name and the source of the default stream name, default `"autoaim_preview"`.
- `preview_scale`: output scale applied to the preview image, default `1.0`.
- `preview_wait_key_ms`: wait time of `cv::waitKey()` in window mode in ms, at least 1 is used, default `1`.
- `queue_capacity`: preview queue length, limited to 1 to 2, default `1`.
- `output_mode`: `"window"`, or `"raw"` / `"bmp"` / `"web"` / `"http"`, default `"window"`.
- `web_bind_address`: HTTP listen address, `"0.0.0.0"` serves remote viewing, default `"0.0.0.0"`.
- `web_port`: HTTP listen port, default `8080`.
- `web_stream_name`: stream name, default empty, in which case it is generated from `preview_window_name` (letters, digits, `_` and `-` are kept, `.` and spaces become `_`).
- `max_fps`: maximum accepted frame rate of the preview, default `30.0`; a value less than or equal to 0 disables the rate limit.

## 4. Topic

无 / None

## 5. 配置示例 / Configuration Example

VisionPreview 是库（`standalone: false`），由其他模块在 `depends` 中声明并包含 `VisionPreview.hpp`，实例由使用它的模块创建。这些模块的配置带有 `RuntimeParam` 字段，在 BSP 的 `User/xrobot.yaml` 中按 YAML map 填写，字符串字段写成 C++ 字符串字面量。以下为 `QDU-Robomaster/VisionCapture` 的 `preview_in`：

VisionPreview is a library (`standalone: false`) that other Modules declare in `depends` and use by including `VisionPreview.hpp`; the instance is created by the Module that uses it. The configuration of these Modules has a `RuntimeParam` field, filled in as a YAML map in the BSP `User/xrobot.yaml`, with string fields written as C++ string literals. The following is the `preview_in` of `QDU-Robomaster/VisionCapture`:

```yaml
preview_in:
  enabled: true
  preview_window_name: "vision_capture"
  preview_scale: 0.5
  preview_wait_key_ms: 1
  queue_capacity: 1
  output_mode: "web"
  web_bind_address: "0.0.0.0"
  web_port: 8080
  web_stream_name: "vision_capture"
  max_fps: 30.0
```

浏览器访问 `http://<host>:8080/stream/vision_capture` 查看画面。在 C++ 中直接使用：

The browser opens `http://<host>:8080/stream/vision_capture` to view the image. Direct use in C++:

```cpp
VisionPreview preview({
    .enabled = true,
    .preview_window_name = "detector_preview",
    .preview_scale = 0.5,
    .output_mode = "window",
});

preview.Submit(frame, [result](cv::Mat& image) {
  // 在 image 上绘制框、文字或轨迹 / draw boxes, text or trajectories on image
});
```

## 6. 依赖与硬件 / Dependencies and Hardware

依赖：LibXR（日志），OpenCV 4（`core`、`imgproc`、`highgui`），由 `CMakeLists.txt` 通过 `find_package(OpenCV 4 REQUIRED ...)` 引入。

硬件：窗口模式使用的显示后端（`DISPLAY` 或 `WAYLAND_DISPLAY`），或 Web 模式使用的网络接口。

测试：在启用 `BUILD_TESTING` 的 BSP 构建中，模块加入 `vision_preview_restart_test`（重启、并发生命周期、worker 失败回滚、HTTP 客户端回收与慢客户端停止），用 `ctest` 运行。

Dependencies: LibXR (logging) and OpenCV 4 (`core`, `imgproc`, `highgui`), brought in by `CMakeLists.txt` through `find_package(OpenCV 4 REQUIRED ...)`.

Hardware: the display backend used by the window mode (`DISPLAY` or `WAYLAND_DISPLAY`), or the network interface used by the Web mode.

Tests: in a BSP build with `BUILD_TESTING` enabled, the Module adds `vision_preview_restart_test` (restart, concurrent lifecycle, worker failure rollback, HTTP client recycling and slow client stop), run with `ctest`.
