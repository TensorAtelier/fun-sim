// Fourier epicycles: a chain of rotating phasors (complex DFT terms, largest first) whose
// tip retraces a closed path — a preset, a freehand drawing, or edges pulled from an image.
import { createLayout, section, slider, select, checkbox, buttons, readout, createCanvas, loop, h } from '../ui.js';

const BASE_PERIOD = 12; // seconds per full cycle at 1x speed
const TAU = Math.PI * 2;

// ---------------------------------------------------------------- presets (math coords, y up)
function polygon(pts) { return pts.map(([x, y]) => ({ x, y })); }
function parametric(fn, n = 600) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(fn((i / n) * TAU));
  return out;
}
const PRESETS = {
  heart: {
    label: 'Heart',
    make: () => parametric((t) => ({
      x: 16 * Math.sin(t) ** 3,
      y: 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t),
    })),
  },
  star: {
    label: 'Five-point star',
    make: () => {
      const pts = [];
      for (let i = 0; i < 10; i++) {
        const r = i % 2 ? 0.4 : 1;
        const a = Math.PI / 2 + (i * Math.PI) / 5;
        pts.push({ x: r * Math.cos(a), y: r * Math.sin(a) });
      }
      return pts;
    },
  },
  pi: {
    label: 'π',
    make: () => polygon([
      [-1, 0.46], [-0.93, 0.66], [-0.78, 0.8], [1, 0.8], [1, 0.56], [0.52, 0.56],
      [0.52, -0.46], [0.57, -0.62], [0.68, -0.68], [0.8, -0.62], [0.9, -0.52], [0.96, -0.62],
      [0.86, -0.8], [0.68, -0.9], [0.46, -0.88], [0.32, -0.74], [0.27, -0.5], [0.27, 0.56],
      [-0.28, 0.56], [-0.33, -0.1], [-0.42, -0.52], [-0.56, -0.82], [-0.72, -0.9],
      [-0.84, -0.8], [-0.8, -0.62], [-0.64, -0.4], [-0.56, 0.0], [-0.52, 0.56], [-0.76, 0.56], [-0.9, 0.42],
    ]),
  },
  trefoil: {
    label: 'Trefoil',
    make: () => parametric((t) => ({ x: Math.sin(t) + 2 * Math.sin(2 * t), y: Math.cos(t) - 2 * Math.cos(2 * t) })),
  },
  square: {
    label: 'Square (Gibbs ringing)',
    make: () => polygon([[-1, 1], [1, 1], [1, -1], [-1, -1]]),
  },
};

// ---------------------------------------------------------------- geometry + DFT
// Center on bbox and scale so the longest side spans 2 units.
function normalize(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of pts) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const s = 2 / Math.max(x1 - x0, y1 - y0, 1e-9);
  return pts.map((p) => ({ x: (p.x - cx) * s, y: (p.y - cy) * s }));
}

// Uniform arc-length resampling of the closed polyline to n points.
function resample(pts, n) {
  const P = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const q = P[P.length - 1];
    if (Math.hypot(pts[i].x - q.x, pts[i].y - q.y) > 1e-9) P.push(pts[i]);
  }
  const m = P.length;
  if (m < 2) return Array.from({ length: n }, () => ({ ...P[0] }));
  const cum = new Float64Array(m + 1);
  for (let i = 0; i < m; i++) {
    const a = P[i], b = P[(i + 1) % m];
    cum[i + 1] = cum[i] + Math.hypot(b.x - a.x, b.y - a.y);
  }
  const L = cum[m];
  const out = new Array(n);
  let seg = 0;
  for (let i = 0; i < n; i++) {
    const s = (i / n) * L;
    while (seg < m - 1 && cum[seg + 1] < s) seg++;
    const a = P[seg], b = P[(seg + 1) % m];
    const len = cum[seg + 1] - cum[seg];
    const u = len > 0 ? (s - cum[seg]) / len : 0;
    out[i] = { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u };
  }
  return out;
}

// Complex DFT X_k = 1/N Σ z_n e^{-2πikn/N}. Returns terms with DC first, rest by amplitude.
function dft(pts) {
  const N = pts.length;
  const cosT = new Float64Array(N), sinT = new Float64Array(N);
  for (let m = 0; m < N; m++) { cosT[m] = Math.cos((TAU * m) / N); sinT[m] = Math.sin((TAU * m) / N); }
  const terms = [];
  for (let k = 0; k < N; k++) {
    let re = 0, im = 0, idx = 0;
    for (let n = 0; n < N; n++) {
      const x = pts[n].x, y = pts[n].y, c = cosT[idx], s = sinT[idx];
      re += x * c + y * s;
      im += y * c - x * s;
      idx += k; if (idx >= N) idx -= N;
    }
    re /= N; im /= N;
    const f = k <= N / 2 ? k : k - N;
    terms.push({ f, re, im, amp: Math.hypot(re, im), phase: Math.atan2(im, re) });
  }
  const dc = terms.find((t) => t.f === 0);
  const rest = terms.filter((t) => t.f !== 0).sort((a, b) => b.amp - a.amp);
  return [dc, ...rest];
}

// Reconstruct the curve from the first `count` terms at M uniform samples.
function reconstruct(terms, count, M) {
  const cx = new Float64Array(M), cy = new Float64Array(M);
  for (let i = 0; i < count; i++) {
    const t = terms[i];
    const cr = Math.cos((TAU * t.f) / M), sr = Math.sin((TAU * t.f) / M);
    let pr = t.re, pi = t.im;
    for (let j = 0; j < M; j++) {
      cx[j] += pr; cy[j] += pi;
      const nr = pr * cr - pi * sr; pi = pr * sr + pi * cr; pr = nr;
    }
  }
  return { cx, cy, M };
}

// ---------------------------------------------------------------- image processing
function gaussianBlur(src, w, h, sigma) {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++) a += k[i + r] * src[y * w + Math.min(w - 1, Math.max(0, x + i))];
    tmp[y * w + x] = a;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++) a += k[i + r] * tmp[Math.min(h - 1, Math.max(0, y + i)) * w + x];
    out[y * w + x] = a;
  }
  return out;
}

// Canny-like: Sobel → non-maximum suppression → hysteresis. Returns Uint8 edge mask.
function cannyEdges(g, w, h, thr) {
  const mag = new Float32Array(w * h), dir = new Uint8Array(w * h);
  let max = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = -g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1] + g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1];
    const gy = -g[i - w - 1] - 2 * g[i - w] - g[i - w + 1] + g[i + w - 1] + 2 * g[i + w] + g[i + w + 1];
    const m = Math.hypot(gx, gy);
    mag[i] = m; if (m > max) max = m;
    let a = (Math.atan2(gy, gx) * 180) / Math.PI; if (a < 0) a += 180;
    dir[i] = a < 22.5 || a >= 157.5 ? 0 : a < 67.5 ? 1 : a < 112.5 ? 2 : 3;
  }
  if (max === 0) return new Uint8Array(w * h);
  const nms = new Float32Array(w * h);
  const off = [[1, 0], [1, 1], [0, 1], [-1, 1]];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, m = mag[i];
    const [dx, dy] = off[dir[i]];
    if (m >= mag[i + dy * w + dx] && m >= mag[i - dy * w - dx]) nms[i] = m / max;
  }
  const hi = thr, lo = thr * 0.45;
  const mask = new Uint8Array(w * h), stack = [];
  for (let i = 0; i < w * h; i++) if (nms[i] >= hi) { mask[i] = 1; stack.push(i); }
  while (stack.length) {
    const i = stack.pop(), x = i % w, y = (i / w) | 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const j = ny * w + nx;
      if (!mask[j] && nms[j] >= lo) { mask[j] = 1; stack.push(j); }
    }
  }
  return mask;
}

// Keep one pixel per s×s cell, growing s until at most `target` points remain.
function subsample(mask, w, h, target) {
  let pts = [];
  for (let s = 1; s < 32; s++) {
    pts = [];
    const seen = new Set();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      const key = ((y / s) | 0) * 100000 + ((x / s) | 0);
      if (seen.has(key)) continue;
      seen.add(key); pts.push({ x, y });
    }
    if (pts.length <= target) break;
  }
  return pts;
}

// Greedy nearest-neighbour tour using a uniform spatial grid.
function greedyTour(pts, w, h) {
  const n = pts.length;
  if (n < 2) return pts.slice();
  const cs = 8;
  const gw = Math.ceil(w / cs) + 1, gh = Math.ceil(h / cs) + 1;
  const cells = Array.from({ length: gw * gh }, () => []);
  const where = new Int32Array(n);
  pts.forEach((p, i) => {
    const c = cells[((p.y / cs) | 0) * gw + ((p.x / cs) | 0)];
    where[i] = c.length; c.push(i);
  });
  const remove = (i) => {
    const p = pts[i], c = cells[((p.y / cs) | 0) * gw + ((p.x / cs) | 0)];
    const k = where[i], last = c.pop();
    if (last !== i) { c[k] = last; where[last] = k; }
  };
  // start at the top-left-most point
  let cur = 0;
  for (let i = 1; i < n; i++) if (pts[i].x + pts[i].y < pts[cur].x + pts[cur].y) cur = i;
  const order = [pts[cur]];
  remove(cur);
  const maxR = Math.max(gw, gh);
  for (let step = 1; step < n; step++) {
    const p = pts[cur], gx = (p.x / cs) | 0, gy = (p.y / cs) | 0;
    let best = -1, bestD = Infinity;
    for (let r = 0; r <= maxR; r++) {
      for (let yy = gy - r; yy <= gy + r; yy++) {
        if (yy < 0 || yy >= gh) continue;
        const edgeRow = yy === gy - r || yy === gy + r;
        for (let xx = gx - r; xx <= gx + r; xx += edgeRow ? 1 : 2 * r || 1) {
          if (xx < 0 || xx >= gw) continue;
          for (const j of cells[yy * gw + xx]) {
            const d = (pts[j].x - p.x) ** 2 + (pts[j].y - p.y) ** 2;
            if (d < bestD) { bestD = d; best = j; }
          }
        }
      }
      if (best >= 0 && bestD <= (r * cs) ** 2) break;
    }
    if (best < 0) break;
    remove(best); order.push(pts[best]); cur = best;
  }
  return order;
}

function otsu(g) {
  const hist = new Float64Array(256);
  for (const v of g) hist[Math.min(255, Math.max(0, (v * 255) | 0))]++;
  const total = g.length;
  let sumAll = 0; for (let i = 0; i < 256; i++) sumAll += i * hist[i];
  let wB = 0, sumB = 0, best = 0, thr = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = total - wB; if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const v = wB * wF * (mB - mF) ** 2;
    if (v > best) { best = v; thr = t; }
  }
  return thr / 255;
}

// Threshold → pick foreground (the class touching the border least) → largest 4-connected
// region → Moore-neighbour boundary trace. Returns ordered contour points.
function silhouette(g, w, h, cut) {
  let fg = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) fg[i] = g[i] < cut ? 1 : 0;
  let border = 0, borderFg = 0;
  for (let x = 0; x < w; x++) { border += 2; borderFg += fg[x] + fg[(h - 1) * w + x]; }
  for (let y = 0; y < h; y++) { border += 2; borderFg += fg[y * w] + fg[y * w + w - 1]; }
  if (borderFg > border / 2) for (let i = 0; i < w * h; i++) fg[i] ^= 1;
  // largest component
  const label = new Int32Array(w * h);
  let bestLabel = 0, bestSize = 0, next = 1;
  const stack = [];
  for (let s = 0; s < w * h; s++) {
    if (!fg[s] || label[s]) continue;
    let size = 0; label[s] = next; stack.push(s);
    while (stack.length) {
      const i = stack.pop(); size++;
      const x = i % w, y = (i / w) | 0;
      if (x > 0 && fg[i - 1] && !label[i - 1]) { label[i - 1] = next; stack.push(i - 1); }
      if (x < w - 1 && fg[i + 1] && !label[i + 1]) { label[i + 1] = next; stack.push(i + 1); }
      if (y > 0 && fg[i - w] && !label[i - w]) { label[i - w] = next; stack.push(i - w); }
      if (y < h - 1 && fg[i + w] && !label[i + w]) { label[i + w] = next; stack.push(i + w); }
    }
    if (size > bestSize) { bestSize = size; bestLabel = next; }
    next++;
  }
  if (bestSize < 20) return { contour: [], mask: fg };
  const inside = (x, y) => x >= 0 && y >= 0 && x < w && y < h && label[y * w + x] === bestLabel;
  const D = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  const dirOf = (dx, dy) => D.findIndex(([a, b]) => a === dx && b === dy);
  let sx = 0, sy = 0;
  outer: for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (inside(x, y)) { sx = x; sy = y; break outer; }
  const contour = [{ x: sx, y: sy }];
  let px = sx, py = sy, back = 4; // came from the west
  const startBack = back;
  const limit = w * h * 4;
  for (let steps = 0; steps < limit; steps++) {
    let found = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (back + k) % 8;
      if (inside(px + D[d][0], py + D[d][1])) { found = d; break; }
    }
    if (found < 0) break; // isolated pixel
    const prev = (found + 7) % 8;
    const bx = px + D[prev][0], by = py + D[prev][1];
    px += D[found][0]; py += D[found][1];
    back = dirOf(bx - px, by - py);
    if (px === sx && py === sy && back === startBack) break;
    if (px === sx && py === sy && contour.length > 2) break;
    contour.push({ x: px, y: py });
  }
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) mask[i] = label[i] === bestLabel ? 1 : 0;
  return { contour, mask };
}

// ---------------------------------------------------------------- module
export default {
  id: 'fourier',
  title: 'Fourier',
  glyph: '∿',
  tag: 'math',
  blurb: 'Hundreds of spinning circles, each a term of a Fourier series, conspire to draw a heart, your doodle, or the edges of any photo.',

  mount(root) {
    const css = getComputedStyle(document.documentElement);
    const cv = (name, fb) => (css.getPropertyValue(name).trim() || fb);
    const C = {
      accent: cv('--accent', '#f5b544'), cyan: cv('--accent-2', '#5ec8e5'), muted: cv('--muted', '#8a93a6'),
      line: cv('--line', '#242c3b'), bg: cv('--bg', '#0b0e14'), text: cv('--text', '#e6e9ef'),
    };
    const MONO = "'IBM Plex Mono', ui-monospace, monospace";

    const { stage, panel } = createLayout(root, {
      title: 'Fourier Epicycles',
      desc: 'Any closed curve is a sum of circles turning at whole-number speeds. Sort the DFT terms by size, chain them tip to tail, and the last tip retraces the shape. Add circles to watch the approximation sharpen.',
    });

    // ---- state
    const S = {
      mode: 'play',          // 'play' | 'draw'
      source: 'Heart',
      raw: null,             // normalized path (y down)
      terms: [],
      N: 512,
      K: 60,                 // circles (non-DC terms)
      speed: 1,
      playing: true,
      showCircles: true,
      showOriginal: true,
      t: 0,                  // phase in cycles
      trailStart: 0,
      curve: null,
      resampled: [],
      totalEnergy: 1,
      drawPts: [],
      drawing: false,
    };
    const view = { w: 0, h: 0, scale: 1, ox: 0, oy: 0 };
    let pendingRebuild = 0, pendingImage = 0;
    const objectURLs = [];

    // ---- canvas + overlays
    const cnv = createCanvas(stage, () => fit());
    const { canvas, ctx } = cnv;
    canvas.style.touchAction = 'none';
    const hud = h('div', { class: 'hud' });
    stage.append(hud);
    const hint = h('div', {
      style: 'position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);pointer-events:none;text-align:center;'
        + `font-family:${MONO};color:${C.muted};font-size:13px;letter-spacing:.06em;line-height:1.8;display:none`,
    });
    hint.innerHTML = `<div style="font-size:30px;color:${C.accent};opacity:.8">✎</div>DRAW ONE CLOSED STROKE<br><span style="opacity:.6">release to transform</span>`;
    stage.append(hint);

    // ---- panel: source
    const src = section(panel, 'Source');
    const presetSel = select(src, {
      label: 'Preset',
      options: Object.entries(PRESETS).map(([k, p]) => [k, p.label]),
      value: 'heart',
      onChange: (k) => loadPreset(k),
    });
    const fileInput = h('input', { type: 'file', accept: 'image/*', style: 'display:none' });
    src.append(fileInput);
    const [drawBtn] = buttons(src, [
      { label: '✎ Draw your own', onClick: () => enterDraw() },
      { label: '⇪ Upload image', primary: true, onClick: () => fileInput.click() },
    ]);
    src.append(h('div', { class: 'hint' }, 'Tip: you can also drop an image onto the stage.'));
    fileInput.addEventListener('change', () => {
      const f = fileInput.files?.[0];
      if (f) loadImageFile(f);
      fileInput.value = '';
    });

    // ---- panel: image
    const imgSec = section(panel, 'Image → path');
    const img = { w: 0, h: 0, gray: null, name: '' };
    const imgMode = select(imgSec, {
      label: 'Extraction',
      options: [['edges', 'Edges (Canny-like + greedy tour)'], ['silhouette', 'Silhouette contour']],
      value: 'edges',
      onChange: () => { syncImgControls(); scheduleImage(); },
    });
    const edgeThr = slider(imgSec, {
      label: 'Edge threshold', min: 0.03, max: 0.6, step: 0.01, value: 0.18,
      format: (v) => v.toFixed(2), onInput: () => scheduleImage(),
    });
    const lumaCut = slider(imgSec, {
      label: 'Luma cutoff', min: 0.02, max: 0.98, step: 0.01, value: 0.5,
      format: (v) => v.toFixed(2), onInput: () => scheduleImage(),
    });
    const blurS = slider(imgSec, {
      label: 'Blur σ', min: 0.5, max: 4, step: 0.1, value: 1.4,
      format: (v) => v.toFixed(1), onInput: () => scheduleImage(),
    });
    const preview = h('canvas', {
      width: 300, height: 180,
      style: `width:100%;display:block;border:1px solid ${C.line};border-radius:8px;background:${C.bg};margin-bottom:6px`,
    });
    const pctx = preview.getContext('2d');
    const imgInfo = h('div', { class: 'hint' }, 'No image loaded — upload one to trace its edges.');
    imgSec.append(preview, imgInfo);
    function syncImgControls() {
      const edges = imgMode.value === 'edges';
      edgeThr.input.closest('.ctl').style.display = edges ? '' : 'none';
      lumaCut.input.closest('.ctl').style.display = edges ? 'none' : '';
    }
    syncImgControls();
    drawPreviewEmpty();

    // ---- panel: transform
    const tf = section(panel, 'Transform');
    const nSl = slider(tf, {
      label: 'Sample points N', min: 128, max: 2048, step: 64, value: S.N,
      onInput: (v) => { S.N = v; scheduleRebuild(); },
    });
    const kSl = slider(tf, {
      label: 'Circles', min: 1, max: S.N - 1, step: 1, value: S.K,
      onInput: (v) => { S.K = v; updateCurve(); },
    });

    // ---- panel: playback
    const pb = section(panel, 'Playback');
    slider(pb, {
      label: 'Speed', min: 0.1, max: 4, step: 0.05, value: S.speed,
      format: (v) => v.toFixed(2) + '×', onInput: (v) => { S.speed = v; },
    });
    checkbox(pb, { label: 'Show circles', checked: true, onChange: (v) => { S.showCircles = v; } });
    checkbox(pb, { label: 'Show original path', checked: true, onChange: (v) => { S.showOriginal = v; } });
    const [playBtn] = buttons(pb, [
      { label: '❚❚ Pause', onClick: () => setPlaying(!S.playing) },
      { label: 'Clear trail', onClick: () => resetTrail() },
    ]);

    const stats = section(panel, 'Readout');
    const ro = readout(stats, ['Source', 'Circles used', 'N points', 'Energy captured', 'Largest radius']);

    function setPlaying(p) {
      S.playing = p;
      playBtn.textContent = p ? '❚❚ Pause' : '▶ Play';
    }

    // ---- core pipeline
    function setPath(pts, sourceName) {
      if (!pts || pts.length < 3) return;
      S.raw = normalize(pts);
      S.source = sourceName;
      S.mode = 'play';
      hint.style.display = 'none';
      drawBtn.classList.remove('primary');
      rebuild();
    }

    function rebuild() {
      if (!S.raw) return;
      S.resampled = resample(S.raw, S.N);
      S.terms = dft(S.resampled);
      S.totalEnergy = S.terms.slice(1).reduce((a, t) => a + t.amp * t.amp, 0) || 1;
      kSl.input.max = S.N - 1;
      if (S.K > S.N - 1) S.K = S.N - 1;
      kSl.set(S.K);
      S.t = 0;
      fit();
      updateCurve();
    }

    function scheduleRebuild() {
      cancelAnimationFrame(pendingRebuild);
      pendingRebuild = requestAnimationFrame(rebuild);
    }

    function updateCurve() {
      if (!S.terms.length) return;
      const count = Math.min(S.K + 1, S.terms.length);
      const M = Math.max(720, Math.min(2048, S.N * 2));
      S.curve = reconstruct(S.terms, count, M);
      S.trailStart = S.t;
      updateReadout();
    }

    function resetTrail() { S.trailStart = S.t; }

    function updateReadout() {
      let e = 0;
      const count = Math.min(S.K + 1, S.terms.length);
      for (let i = 1; i < count; i++) e += S.terms[i].amp ** 2;
      ro.set('Source', S.source);
      ro.set('Circles used', `${count - 1} / ${S.terms.length - 1}`);
      ro.set('N points', String(S.N));
      ro.set('Energy captured', ((100 * e) / S.totalEnergy).toFixed(e / S.totalEnergy > 0.999 ? 3 : 1) + '%');
      ro.set('Largest radius', S.terms[1] ? (S.terms[1].amp * view.scale).toFixed(0) + ' px' : '—');
      hud.textContent = `${S.source.toUpperCase()} · ${count - 1} circles · N=${S.N}`;
    }

    function fit() {
      view.w = cnv.size.w; view.h = cnv.size.h;
      if (!S.resampled.length) return;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const p of S.resampled) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
      const pad = Math.max(36, Math.min(view.w, view.h) * 0.1);
      const bw = Math.max(x1 - x0, 1e-6), bh = Math.max(y1 - y0, 1e-6);
      view.scale = Math.max(1, Math.min((view.w - 2 * pad) / bw, (view.h - 2 * pad) / bh));
      view.ox = view.w / 2 - ((x0 + x1) / 2) * view.scale;
      view.oy = view.h / 2 - ((y0 + y1) / 2) * view.scale;
      if (S.terms.length) updateReadout();
    }

    function loadPreset(key) {
      const p = PRESETS[key];
      setPath(p.make().map((q) => ({ x: q.x, y: -q.y })), p.label);
      presetSel.value = key;
    }

    // ---- drawing
    function enterDraw() {
      S.mode = 'draw';
      S.drawPts = [];
      S.drawing = false;
      hint.style.display = '';
      drawBtn.classList.add('primary');
    }
    const localPt = (e) => {
      const r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    canvas.addEventListener('pointerdown', (e) => {
      if (S.mode !== 'draw') return;
      canvas.setPointerCapture(e.pointerId);
      S.drawing = true;
      S.drawPts = [localPt(e)];
      hint.style.display = 'none';
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!S.drawing) return;
      const p = localPt(e), q = S.drawPts[S.drawPts.length - 1];
      if (Math.hypot(p.x - q.x, p.y - q.y) > 1.5) S.drawPts.push(p);
    });
    const endDraw = () => {
      if (!S.drawing) return;
      S.drawing = false;
      let len = 0;
      for (let i = 1; i < S.drawPts.length; i++) len += Math.hypot(S.drawPts[i].x - S.drawPts[i - 1].x, S.drawPts[i].y - S.drawPts[i - 1].y);
      if (S.drawPts.length < 8 || len < 40) { S.drawPts = []; hint.style.display = ''; return; }
      const pts = S.drawPts;
      S.drawPts = [];
      setPath(pts, 'Your drawing');
    };
    canvas.addEventListener('pointerup', endDraw);
    canvas.addEventListener('pointercancel', endDraw);

    // ---- image upload
    const onDragOver = (e) => { e.preventDefault(); };
    const onDrop = (e) => {
      e.preventDefault();
      const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/'));
      if (f) loadImageFile(f);
    };
    stage.addEventListener('dragover', onDragOver);
    stage.addEventListener('drop', onDrop);

    function loadImageFile(file) {
      const url = URL.createObjectURL(file);
      objectURLs.push(url);
      const im = new Image();
      im.onload = () => {
        const sc = Math.min(1, 300 / Math.max(im.naturalWidth, im.naturalHeight));
        const w = Math.max(8, Math.round(im.naturalWidth * sc)), hh = Math.max(8, Math.round(im.naturalHeight * sc));
        const off = document.createElement('canvas');
        off.width = w; off.height = hh;
        const o = off.getContext('2d', { willReadFrequently: true });
        o.fillStyle = '#fff'; o.fillRect(0, 0, w, hh);
        o.drawImage(im, 0, 0, w, hh);
        const d = o.getImageData(0, 0, w, hh).data;
        const gray = new Float32Array(w * hh);
        for (let i = 0; i < w * hh; i++) gray[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
        Object.assign(img, { w, h: hh, gray, name: file.name.replace(/\.[^.]+$/, '').slice(0, 18) || 'Image' });
        lumaCut.set(otsu(gray));
        processImage();
      };
      im.onerror = () => { imgInfo.textContent = 'Could not read that file as an image.'; };
      im.src = url;
    }

    function scheduleImage() {
      if (!img.gray) return;
      cancelAnimationFrame(pendingImage);
      pendingImage = requestAnimationFrame(processImage);
    }

    function processImage() {
      if (!img.gray) return;
      const { w, h: hh } = img;
      const blurred = gaussianBlur(img.gray, w, hh, blurS.value);
      let mask, path;
      if (imgMode.value === 'edges') {
        mask = cannyEdges(blurred, w, hh, edgeThr.value);
        const pts = subsample(mask, w, hh, 2500);
        path = greedyTour(pts, w, hh);
        const nEdge = mask.reduce((a, b) => a + b, 0);
        imgInfo.textContent = `${w}×${hh}px · ${nEdge} edge px → ${path.length} tour points`;
      } else {
        const r = silhouette(blurred, w, hh, lumaCut.value);
        mask = r.mask; path = r.contour;
        imgInfo.textContent = `${w}×${hh}px · contour of ${path.length} px`;
      }
      drawPreview(mask, path, w, hh);
      if (path.length < 8) {
        imgInfo.textContent += ' — too few points; adjust the threshold.';
        return;
      }
      setPath(path, img.name);
    }

    function drawPreviewEmpty() {
      pctx.fillStyle = C.bg; pctx.fillRect(0, 0, preview.width, preview.height);
      pctx.fillStyle = C.muted; pctx.globalAlpha = 0.5;
      pctx.font = `12px ${MONO}`; pctx.textAlign = 'center';
      pctx.fillText('edge preview', preview.width / 2, preview.height / 2 + 4);
      pctx.globalAlpha = 1;
    }

    function drawPreview(mask, path, w, hh) {
      preview.width = w; preview.height = hh;
      const id = pctx.createImageData(w, hh);
      const edges = imgMode.value === 'edges';
      for (let i = 0; i < w * hh; i++) {
        const on = mask[i];
        id.data[i * 4] = on ? (edges ? 94 : 30) : 11;
        id.data[i * 4 + 1] = on ? (edges ? 200 : 60) : 14;
        id.data[i * 4 + 2] = on ? (edges ? 229 : 72) : 20;
        id.data[i * 4 + 3] = 255;
      }
      pctx.putImageData(id, 0, 0);
      if (path.length > 1) {
        pctx.strokeStyle = C.accent; pctx.globalAlpha = edges ? 0.35 : 0.9; pctx.lineWidth = 1;
        pctx.beginPath(); pctx.moveTo(path[0].x + 0.5, path[0].y + 0.5);
        for (const p of path) pctx.lineTo(p.x + 0.5, p.y + 0.5);
        pctx.closePath(); pctx.stroke(); pctx.globalAlpha = 1;
      }
    }

    // ---- render
    const X = (x) => view.ox + x * view.scale;
    const Y = (y) => view.oy + y * view.scale;

    function drawBackdrop() {
      ctx.clearRect(0, 0, view.w, view.h);
      // subtle dotted grid
      ctx.fillStyle = C.line;
      const g = 32;
      const cx = view.w / 2, cy = view.h / 2;
      for (let x = cx % g; x < view.w; x += g) for (let y = cy % g; y < view.h; y += g) ctx.fillRect(x - 0.5, y - 0.5, 1, 1);
      ctx.strokeStyle = C.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, cy + 0.5); ctx.lineTo(view.w, cy + 0.5); ctx.moveTo(cx + 0.5, 0); ctx.lineTo(cx + 0.5, view.h); ctx.stroke();
      ctx.font = `10px ${MONO}`; ctx.fillStyle = C.muted; ctx.globalAlpha = 0.6;
      ctx.fillText('Re', view.w - 22, cy - 6); ctx.fillText('Im', cx + 6, 14);
      ctx.globalAlpha = 1;
    }

    function strokeCurve(from, count, width, color, alpha) {
      const { cx, cy, M } = S.curve;
      if (count < 2) return;
      ctx.beginPath();
      ctx.moveTo(X(cx[from % M]), Y(cy[from % M]));
      for (let j = 1; j < count; j++) { const k = (from + j) % M; ctx.lineTo(X(cx[k]), Y(cy[k])); }
      ctx.lineWidth = width; ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.stroke();
      ctx.globalAlpha = 1;
    }

    function render() {
      drawBackdrop();
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';

      if (S.mode === 'draw') {
        const pts = S.drawPts;
        if (pts.length > 1) {
          for (const [w, a] of [[7, 0.12], [2, 0.95]]) {
            ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
            for (const p of pts) ctx.lineTo(p.x, p.y);
            ctx.lineWidth = w; ctx.strokeStyle = C.accent; ctx.globalAlpha = a; ctx.stroke();
          }
          ctx.globalAlpha = 0.4; ctx.setLineDash([4, 5]); ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(pts[pts.length - 1].x, pts[pts.length - 1].y); ctx.lineTo(pts[0].x, pts[0].y); ctx.stroke();
          ctx.setLineDash([]); ctx.globalAlpha = 1;
        }
        return;
      }
      if (!S.curve) return;

      // original path, faint
      if (S.showOriginal && S.resampled.length) {
        ctx.beginPath();
        S.resampled.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
        ctx.closePath();
        ctx.setLineDash([2, 4]); ctx.lineWidth = 1; ctx.strokeStyle = C.muted; ctx.globalAlpha = 0.35; ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      }

      // trail: first period grows from trailStart; afterwards the full shape is redrawn
      const { M } = S.curve;
      const progress = S.t - S.trailStart;
      const tipIdx = Math.floor((((S.t % 1) + 1) % 1) * M);
      let from, count;
      if (progress >= 1) { from = tipIdx + 1; count = M; }
      else {
        from = Math.floor((((S.trailStart % 1) + 1) % 1) * M);
        count = Math.max(0, Math.floor(progress * M)) + 1;
      }
      strokeCurve(from, count, 6, C.accent, 0.1);
      strokeCurve(from, count, 3, C.accent, 0.14);
      strokeCurve(from, count, 1.6, C.accent, 0.9);
      // bright comet head behind the tip
      const tail = Math.min(count, Math.floor(M * 0.06));
      strokeCurve(tipIdx - tail + M + 1, tail, 2.4, '#ffe2a8', 0.9);

      // epicycles
      const count2 = Math.min(S.K + 1, S.terms.length);
      const ang = TAU * S.t;
      let x = S.terms[0].re, y = S.terms[0].im;
      const pts = [[x, y]];
      for (let i = 1; i < count2; i++) {
        const tm = S.terms[i];
        const a = tm.f * ang + tm.phase;
        x += tm.amp * Math.cos(a); y += tm.amp * Math.sin(a);
        pts.push([x, y]);
      }
      if (S.showCircles) {
        ctx.strokeStyle = C.cyan;
        ctx.lineWidth = 1;
        const maxR = S.terms[1] ? S.terms[1].amp * view.scale : 1;
        for (let i = 1; i < count2; i++) {
          const r = S.terms[i].amp * view.scale;
          if (r < 0.6) break; // sorted by amplitude — the rest are smaller still
          ctx.globalAlpha = 0.1 + 0.22 * Math.sqrt(r / maxR);
          ctx.beginPath(); ctx.arc(X(pts[i - 1][0]), Y(pts[i - 1][1]), r, 0, TAU); ctx.stroke();
        }
        ctx.globalAlpha = 0.7; ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.moveTo(X(pts[0][0]), Y(pts[0][1]));
        for (let i = 1; i < pts.length; i++) {
          if (S.terms[i].amp * view.scale < 0.3) break;
          ctx.lineTo(X(pts[i][0]), Y(pts[i][1]));
        }
        ctx.stroke();
        // pivot at origin of the chain
        ctx.globalAlpha = 0.9; ctx.fillStyle = C.cyan;
        ctx.beginPath(); ctx.arc(X(pts[0][0]), Y(pts[0][1]), 2.5, 0, TAU); ctx.fill();
        ctx.globalAlpha = 1;
      }
      // tip
      const tx = X(x), ty = Y(y);
      const glow = ctx.createRadialGradient(tx, ty, 0, tx, ty, 14);
      glow.addColorStop(0, 'rgba(245,181,68,0.55)'); glow.addColorStop(1, 'rgba(245,181,68,0)');
      ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(tx, ty, 14, 0, TAU); ctx.fill();
      ctx.fillStyle = '#fff4dc'; ctx.beginPath(); ctx.arc(tx, ty, 2.8, 0, TAU); ctx.fill();
    }

    const stop = loop((dt) => {
      if (S.playing && S.mode === 'play' && S.curve) {
        S.t += (dt * S.speed) / BASE_PERIOD;
        // keep numbers bounded while preserving the "trail complete" state
        if (S.t > 1000) { const k = Math.floor(S.t) - 2; S.t -= k; S.trailStart -= k; }
      }
      render();
    });

    loadPreset('heart');

    return () => {
      stop();
      cnv.destroy();
      cancelAnimationFrame(pendingRebuild);
      cancelAnimationFrame(pendingImage);
      stage.removeEventListener('dragover', onDragOver);
      stage.removeEventListener('drop', onDrop);
      objectURLs.forEach((u) => URL.revokeObjectURL(u));
    };
  },
};
