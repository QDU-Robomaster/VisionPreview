# VisionPreview

`VisionPreview` 用于实时查看视觉模块输出的画面。调用方提交一帧 OpenCV 图像和绘制
函数，预览线程在图像拷贝上绘制内容，然后输出到 OpenCV 窗口或浏览器。

这个模块不订阅 topic，不保存文件，也不规定绘制内容。ArmorDetector、ArmorTracker、
Aimer、VisionCapture 等模块按自己的结果绘制框、文字、轨迹或标定板角点。

## 工作方式

调用 `Start()`（或带 `RuntimeParam` 的构造函数）后，`Submit(frame, draw)` 会先检查
`max_fps`。通过限频后，模块在调用线程深拷贝图像并放入队列，随后立即返回。预览线程
取出图像，执行 `draw(cv::Mat&)`，按 `preview_scale` 缩放，再显示或推流。

队列长度最多为 `2`。当预览线程来不及处理时，模块丢弃最早的等待帧，不阻塞相机、
检测或跟踪线程。`Stop()` 会释放仍在等待的任务并等待预览线程退出；同一实例再次
`Start()` 时不会处理上一次会话的旧帧。预览线程抛出异常时会记录错误并停止本次会话，
之后可以重新 `Start()`。

## 输出模式

`output_mode: "window"` 使用 OpenCV 窗口。运行环境必须有 `DISPLAY` 或 `WAYLAND_DISPLAY`，
否则预览不启动。

`output_mode: "raw"`、`"bmp"`、`"web"`、`"http"` 启动内置 HTTP 服务（仅非 Windows
平台）。每帧编码为未压缩 24 位 BMP，以 `multipart/x-mixed-replace` 推送，浏览器访问：

```text
http://<host>:<web_port>/
http://<host>:<web_port>/stream/<web_stream_name>
```

同一进程内多个 `VisionPreview` 可以共用同一个 `web_bind_address:web_port`。每个实例注册
自己的 stream，stream 名不能重复。根路径会列出当前所有 stream；只有一个 stream 时也可以
访问 `/stream`。每个服务最多同时服务 16 个客户端；最后一个 stream 注销后服务关闭。

## 依赖

本仓库是库（`standalone: false`），不会被单独实例化；视觉模块在自己的 `depends` 中声明
`QDU-Robomaster/VisionPreview` 并包含 `VisionPreview.hpp`。

外部依赖：OpenCV 4（`core`、`imgproc`、`highgui`），由 `CMakeLists.txt` 通过
`find_package(OpenCV 4 REQUIRED ...)` 引入。

## 公共接口

```cpp
class VisionPreview {
 public:
  struct RuntimeParam { ... };
  using DrawCallback = std::function<void(cv::Mat&)>;

  VisionPreview();                               // 未启动
  explicit VisionPreview(RuntimeParam runtime);  // 构造并 Start(runtime)

  bool Start(RuntimeParam runtime);
  void Stop();
  bool Running() const;
  bool Submit(const cv::Mat& frame, DrawCallback draw);

  uint32_t AcceptedFrames() const;
  uint32_t RateDroppedFrames() const;
  uint32_t DroppedFrames() const;
};
```

`Start()` 在配置关闭或启动失败时返回 `false`。`Submit()` 在未运行、空图像、被限频或
会话已切换时返回 `false`；成功入队返回 `true`。

`RuntimeParam` 字段：

- `enabled`：预览总开关，默认 `false`。为 `false` 时不启动线程，`Submit()` 返回 `false`。
- `preview_window_name`：OpenCV 窗口名，也是默认 stream 名来源，默认 `"autoaim_preview"`。
- `preview_scale`：输出缩放比例，只影响预览图，默认 `1.0`。
- `preview_wait_key_ms`：窗口模式下 `cv::waitKey()` 的等待时间，单位 ms，最小按 1 执行，
  默认 `1`。
- `queue_capacity`：预览队列长度，实际限制为 `1` 或 `2`，默认 `1`。
- `output_mode`：`"window"` 或 `"raw"` / `"bmp"` / `"web"` / `"http"`，默认 `"window"`。
- `web_bind_address`：HTTP 监听地址，实机远程查看通常用 `"0.0.0.0"`（默认）。
- `web_port`：HTTP 监听端口，默认 `8080`。
- `web_stream_name`：stream 名，默认为空，此时由 `preview_window_name` 生成（只保留字母、
  数字、`_`、`-`，`.` 和空格转为 `_`）。
- `max_fps`：预览最大接受帧率，默认 `30.0`；小于等于 `0` 表示不限频。

## 计数

- `AcceptedFrames()`：通过限频并进入队列的帧数。
- `RateDroppedFrames()`：被 `max_fps` 丢弃的帧数。
- `DroppedFrames()`：队列满时丢弃的等待帧数。

这些计数用于判断预览是否拖慢或跟不上视觉链路；`Stop()` 时会打印到日志。

## 使用

依赖本库的模块会通过 `depends` 自动引入它，一般无需手动添加；需要单独引入时：

```sh
xrobot module add QDU-Robomaster/VisionPreview
xrobot setup
```

本库不创建实例。使用它的模块在自己的配置里带一个 `preview` 字段（类型为
`VisionPreview::RuntimeParam`），在 BSP 的 `User/xrobot.yaml` 中按 YAML map 填写，
字符串字段写成 C++ 字符串字面量：

```yaml
preview:
  enabled: true
  preview_window_name: '"armor_detector_preview"'
  preview_scale: 0.5
  preview_wait_key_ms: 1
  queue_capacity: 1
  output_mode: '"web"'
  web_bind_address: '"0.0.0.0"'
  web_port: 8080
  web_stream_name: '"armor_detector"'
  max_fps: 30.0
```

打开 `http://<host>:8080/stream/armor_detector` 查看画面。

在 C++ 中直接使用：

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

`xrobot module show .`（在本仓库中）或 `xrobot module show Modules/QDU-Robomaster/VisionPreview`
（在 BSP 中）打印清单。

## 测试

在打开 `BUILD_TESTING` 的 BSP 构建中，本模块会加入 `vision_preview_restart_test`
（重启、并发生命周期、worker 失败回滚、HTTP 客户端回收与慢客户端停止），用 `ctest` 运行。
