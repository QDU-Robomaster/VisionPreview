#pragma once

#include <string_view>

/**
 * @brief 预览网页：循环请求 /frame，在浏览器里把 Bayer 拼成彩色图并画叠加。
 *        Preview page: requests /frame in a loop, turns the Bayer bytes into colour and
 *        draws the overlay in the browser.
 *
 * /frame 的响应体：4 字节小端 JSON 长度、JSON、640×512 BayerRG8 字节。
 * Body of /frame: the JSON length as 4 little-endian bytes, the JSON, then the 640×512
 * BayerRG8 bytes.
 */
namespace Preview
{
inline constexpr std::string_view PAGE_HTML = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><title>auto-aim preview</title>
<style>
body{margin:0;background:#111;color:#ddd;font:13px monospace}
#wrap{display:flex;flex-wrap:wrap;gap:8px;padding:8px}
canvas{background:#000;max-width:100%;image-rendering:pixelated}
#info{white-space:pre;min-width:260px}
.stale{color:#e66}
</style></head><body>
<div id="wrap"><canvas id="c" width="640" height="512"></canvas><div id="info">connecting…</div></div>
<script>
const W = 640, H = 512;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const half = document.createElement('canvas');
half.width = W / 2; half.height = H / 2;
const hctx = half.getContext('2d');
const img = hctx.createImageData(W / 2, H / 2);

function demosaic(b) {
  // RGGB：每个 2×2 单元拼成一个像素 / every 2x2 cell becomes one pixel
  const d = img.data;
  let o = 0;
  for (let y = 0; y < H; y += 2) {
    const r0 = y * W, r1 = r0 + W;
    for (let x = 0; x < W; x += 2) {
      d[o] = b[r0 + x];
      d[o + 1] = (b[r0 + x + 1] + b[r1 + x]) >> 1;
      d[o + 2] = b[r1 + x + 1];
      d[o + 3] = 255;
      o += 4;
    }
  }
  hctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(half, 0, 0, W, H);
}

function quad(q, colour, width) {
  ctx.strokeStyle = colour; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(q[0][0], q[0][1]);
  for (let k = 1; k < 4; ++k) ctx.lineTo(q[k][0], q[k][1]);
  ctx.closePath(); ctx.stroke();
}

function cross(p, colour, r) {
  ctx.strokeStyle = colour; ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(p[0] - r, p[1]); ctx.lineTo(p[0] + r, p[1]);
  ctx.moveTo(p[0], p[1] - r); ctx.lineTo(p[0], p[1] + r);
  ctx.stroke();
}

function draw(j) {
  ctx.font = '12px monospace';
  for (const a of j.armors) {
    const c = {red: '#f44', blue: '#4af', purple: '#c6f', off: '#888'}[a.color] || '#ccc';
    quad(a.corners, c, 2);
    ctx.fillStyle = c;
    ctx.fillText(a.number + ' ' + a.conf.toFixed(2), a.corners[0][0], a.corners[0][1] - 4);
  }
  if (j.target) {
    j.target.plates.forEach((q) => quad(q, '#4f4', 1));
    if (j.target.centre) cross(j.target.centre, '#4f4', 6);
  }
  if (j.aim && j.aim.point) {
    const c = j.aim.fire ? '#f22' : '#fa0';
    cross(j.aim.point, c, 10);
    if (j.aim.fire) {
      ctx.strokeStyle = c; ctx.beginPath();
      ctx.arc(j.aim.point[0], j.aim.point[1], 14, 0, 2 * Math.PI); ctx.stroke();
    }
  }
}

function describe(j) {
  const lines = [];
  lines.push('shown: ' + j.stage + ' #' + j.sequence + '  counter ' + j.frame_counter);
  lines.push('geometry: roi ' + j.geometry.roi_x + ',' + j.geometry.roi_y +
             ' decimation ' + j.geometry.decimation);
  lines.push('');
  for (const s of j.stages) {
    if (!s.present) { lines.push(s.name.padEnd(9) + ' (no topic)'); continue; }
    const age = s.seen ? s.age_ms.toFixed(0) + ' ms ago' : 'never';
    const mark = !s.seen || s.age_ms > 100 ? '<span class="stale">' : '<span>';
    lines.push(mark + s.name.padEnd(9) + ' #' + s.sequence + '  ' + age + '</span>');
  }
  lines.push('');
  lines.push('armors: ' + j.armors.length);
  if (j.target) lines.push('target: ' + j.target.number + ' face ' + j.target.face);
  if (j.aim) {
    lines.push('aim: control ' + j.aim.control + ' fire ' + j.aim.fire +
               ' plate ' + j.aim.plate);
    lines.push('     yaw ' + j.aim.yaw.toFixed(3) + ' pitch ' + j.aim.pitch.toFixed(3));
  }
  info.innerHTML = lines.join('\n');
}

async function loop() {
  while (true) {
    try {
      const r = await fetch('/frame', {cache: 'no-store'});
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const buf = await r.arrayBuffer();
      const len = new DataView(buf).getUint32(0, true);
      const j = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, len)));
      demosaic(new Uint8Array(buf, 4 + len, W * H));
      draw(j);
      describe(j);
    } catch (e) {
      info.textContent = 'no frame: ' + e.message;
      await new Promise((ok) => setTimeout(ok, 500));
    }
  }
}
loop();
</script></body></html>
)HTML";
}  // namespace Preview
