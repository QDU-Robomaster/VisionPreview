# VisionPreview

自瞄网页预览：浏览器拉取原始 Bayer 帧与各层结果，在网页端解码和画图 / Auto-aim web preview where the browser pulls raw Bayer frames and stage results and decodes and draws them itself

## 1. 模块作用 / Purpose

VisionPreview 在车上开一个 HTTP 端口。浏览器打开 `http://<主机>:<端口>/` 后，页面循环请求 `/frame`；每次请求，模块从自瞄链路上最深且新鲜的一层复制下一帧，连同这一层及之前各层的结果发回。服务端不去马赛克、不编码图像、不画图，这些都在浏览器里做。

VisionPreview opens an HTTP port on the robot. A browser opening `http://<host>:<port>/` gets a page that requests `/frame` in a loop; for each request the Module copies the next frame of the deepest fresh stage of the auto-aim chain and sends it back with the results of that stage and the ones before it. The server does no demosaicing, image encoding or drawing; the browser does all of that.

页面上画的内容：

- 检测框（按颜色画成红、蓝、紫、灭灯为灰）与编号、置信度；
- 跟踪目标的各块板和中心（绿色）；
- 瞄点：不开火为橙色十字，开火时为红色并加圆圈；
- 右侧文字：显示的是哪一层、帧序号与帧计数、帧几何，以及每一层最新的序号和距今时间（超过 100 ms 标红），检测数、目标编号与当前板、瞄准结果。

The page draws:

- detections (red, blue, purple, or grey for lights off, by colour) with number and confidence;
- the plates and centre of the tracked target (green);
- the aim point: an orange cross when not firing, red with a circle when firing;
- text on the right: which stage is shown, the sequence and frame counter, the frame geometry, the latest sequence and age of every stage (red above 100 ms), the number of detections, the target number and face, and the aim result.

## 2. 取哪一层 / Which Stage Is Shown

模块订阅 `<相机名>_synced`，以及已存在的 `<相机名>_detected`、`_tracked`、`_aimed`。各层回调平时只记录序号和到达时间。浏览器请求时，模块选出 `fresh_ms` 内收到过帧的最深一层，等它的下一帧（最多 300 ms）并复制下来；等不到就退回同步层再等一次。某一层卡住时画面仍然更新，页面上能看到是哪一层停了。

The Module subscribes to `<camera>_synced` and to whichever of `<camera>_detected`, `_tracked` and `_aimed` exist. Stage callbacks normally record only the sequence and arrival time. On a request the Module picks the deepest stage that received a frame within `fresh_ms`, waits for its next frame (at most 300 ms) and copies it; otherwise it falls back to the synced stage and waits once more. When a stage stalls the picture keeps updating and the page shows which stage stopped.

复制只在浏览器等待时发生（每帧约 0.3 MB），模块不持有图像句柄，不占相机的图像槽。没人打开页面时不复制。

Copies happen only while a browser waits (about 0.3 MB per frame); the Module keeps no image handle and holds no camera image slot. Nothing is copied when no page is open.

## 3. `/frame` 格式 / `/frame` Format

响应体依次为：4 字节小端的 JSON 长度、JSON、640×512 BayerRG8 原始字节。JSON 中的坐标都是帧像素（检测角点由原生像素按帧几何换算，目标与瞄点用帧携带的标定和 `TrackedFrame` 的世界到相机变换投影）：

The body is the JSON length as 4 little-endian bytes, the JSON, then the 640×512 BayerRG8 bytes. All coordinates in the JSON are frame pixels (detection corners are mapped from native pixels by the frame geometry; the target and aim point are projected with the calibration carried by the frame and the world-to-camera transform of `TrackedFrame`):

```json
{"stage":"aimed","sequence":580,"timestamp_us":11600000,"frame_counter":580,
 "width":640,"height":512,"geometry":{"roi_x":80,"roi_y":24,"decimation":2},
 "quaternion":[1,0,0,0],
 "stages":[{"name":"synced","present":true,"seen":true,"sequence":580,"age_ms":0.4}, …],
 "armors":[{"color":"blue","number":"three","conf":0.93,"corners":[[x,y],…]}],
 "target":{"number":"three","face":0,"centre":[x,y],"plates":[[[x,y],…],…]},
 "aim":{"control":true,"fire":false,"yaw":0.1,"pitch":0.02,"plate":0,"point":[x,y]}}
```

没有到达的层对应字段为 `null` 或空数组。页面把每个 2×2 Bayer 单元拼成一个像素（320×256）再放大显示。

Fields of stages not reached are `null` or empty. The page turns every 2×2 Bayer cell into one pixel (320×256) and scales it up.

## 4. 配置示例 / Configuration Example

```yaml
modules:
  - module: QDU-Robomaster/VisionPreview
    id: preview
    args:
      - settings:
          camera_name: "gimbal"
          port: 8080
          fresh_ms: 100
```

预览实例列在同一相机的 CFS（或回放相机）、检测器、跟踪器与 Aimer 之后，构造时只订阅已存在的层。`port` 为 0 时由系统分配，实际端口写在启动日志里。

The preview instance is listed after the CFS (or replay camera), detector, tracker and Aimer of the same camera, and subscribes only to the stages that exist at construction. With `port` 0 the system chooses the port, which the startup log reports.

## 5. 测试 / Tests

`tests/preview_test.cpp` 检查：原生角点换到帧像素、目标中心与瞄点的投影、JSON 字段；通过本机 HTTP 取页面、404、`/frame` 的长度与内容，检测层持续发布时显示检测层，检测层停止 200 ms 后退回同步层并仍列出检测层的状态。

`tests/preview_test.cpp` checks the mapping of native corners to frame pixels, the projection of the target centre and aim point, and the JSON fields; over local HTTP it fetches the page, a 404, and `/frame` with its length and content, sees the detected stage while detections are published, and the fallback to the synced stage 200 ms after detections stop, with the detected stage still listed.

## 6. 依赖 / Dependencies

AutoAimTypes（含 CameraBase）、LibXR、Linux 套接字。

AutoAimTypes (with CameraBase), LibXR, Linux sockets.
