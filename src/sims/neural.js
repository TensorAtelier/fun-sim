// Neural-net playground: a small multilayer perceptron trains live on 2D data. The stage shows
// the network (every neuron as a thumbnail of its activation over the plane), the output
// decision surface with the data on top, and the train/test loss curves.
import { createLayout, section, slider, select, checkbox, buttons, readout, createCanvas, loop, h } from '../ui.js';
import { MLP, Trainer, ACTIVATIONS, FEATURES, DATASETS, DOMAIN, featureList, featurize, makeDataset, splitData } from './neural/net.js';
import { PRESETS, presetConfig } from './neural/presets.js';

const C = {
  bg: '#0b0e14', bg2: '#121722', panel: '#151b27', line: '#242c3b', text: '#e6e9ef',
  muted: '#8a93a6', amber: '#f5b544', cyan: '#5ec8e5', good: '#7bd88f', danger: '#ef6b6b',
};
const MONO = '"IBM Plex Mono", ui-monospace, monospace';
const font = (px, weight = 400) => `${weight} ${px}px ${MONO}`;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const hexRgb = (hex) => { const n = parseInt(hex.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; };
const rgba = (hex, a) => { const [r, g, b] = hexRgb(hex); return `rgba(${r},${g},${b},${a})`; };
const fmtLoss = (v) => (Number.isFinite(v) ? (v >= 100 ? v.toExponential(1) : v.toFixed(3)) : '—');
const fmtPct = (v) => (Number.isFinite(v) ? (v * 100).toFixed(1) + ' %' : '—');
const fmtLr = (v) => (v < 0.001 ? v.toExponential(0) : v < 0.1 ? +v.toPrecision(2) + '' : +v.toPrecision(2) + '');
const SPEEDS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
const MAIN_RES = 60, THUMB_RES = 32;

// Diverging amber <-> cyan colormap over t in [-1, 1] as a 256-entry RGB lookup table.
// `strength` < 1 keeps the ends muted (for the main surface, so the points stand out).
function makeLut(strength, neutral) {
  const lut = new Uint8ClampedArray(256 * 3);
  const A = hexRgb(C.amber), B = hexRgb(C.cyan), N = hexRgb(neutral);
  for (let i = 0; i < 256; i++) {
    const t = i / 127.5 - 1;
    const m = Math.pow(Math.abs(t), 0.85) * strength;
    const E = t >= 0 ? A : B;
    for (let k = 0; k < 3; k++) lut[i * 3 + k] = N[k] + (E[k] - N[k]) * m;
  }
  return lut;
}
const LUT_MAIN = makeLut(0.62, '#131824');
const LUT_THUMB = makeLut(1, '#171d2a');

function roundRect(ctx, x, y, w, hh, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + hh, r);
  ctx.arcTo(x + w, y + hh, x, y + hh, r);
  ctx.arcTo(x, y + hh, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export default {
  id: 'neural',
  title: 'Neural Net',
  glyph: '⊛',
  tag: 'machine learning',
  blurb: 'Watch a small neural network learn to separate spirals, rings and XOR in real time — every neuron drawn as a map of what it responds to.',

  mount(root) {
    const { stage, panel } = createLayout(root, {
      title: 'Neural network',
      desc: 'A multilayer perceptron trained by backpropagation, live. Each square is one neuron, coloured by its activation across the plane; the big map is the output — the network’s belief that a point is amber.',
    });

    // ------------------------------------------------------------ state
    const S = { ...presetConfig(PRESETS[0]), speed: 6, showTest: true, discretize: false, logLoss: true, paint: 1 };
    let pts = [];
    let feats = featureList(S.features);
    let net = null, trainer = null, weightSeed = 1;
    let trX, trY, teX, teY, nTr = 0, nTe = 0;
    let epoch = 0, playing = true, diverged = false;
    const M = { trL: NaN, teL: NaN, trA: NaN, teA: NaN };
    let hist = { e: [], tr: [], te: [], every: 1 };
    let hover = null;        // hovered neuron {l, j}
    let pointer = null;      // pointer position over the heatmap in domain coords
    const ripples = [];
    let frame = 0, mainDirty = true, thumbDirty = true, flow = 0;

    function rebuildData() {
      const tr = featurize(pts.filter((p) => !p.test), feats);
      const te = featurize(pts.filter((p) => p.test), feats);
      trX = tr.X; trY = tr.Y; nTr = tr.Y.length;
      teX = te.X; teY = te.Y; nTe = te.Y.length;
      if (trainer) trainer.setData(trX, trY, nTr);
    }
    function regenerate(seed) {
      if (seed != null) S.seed = seed;
      pts = splitData(makeDataset(S.dataset, S.points, S.noise, S.seed), S.testFrac, S.seed);
      rebuildData();
      rebuildNet();
    }
    function rebuildNet() {
      net = new MLP([feats.length, ...S.hidden, 1], S.activation, weightSeed);
      trainer = new Trainer(net, trX, trY, nTr, weightSeed * 31 + 7);
      epoch = 0;
      diverged = false;
      hover = null;
      mainGrid = null; // shape changed: force a same-frame re-evaluation
      hist = { e: [], tr: [], te: [], every: 1 };
      evalMetrics();
      record();
      mainDirty = thumbDirty = true;
      updateReadout();
    }
    function evalMetrics() {
      const a = net.evaluate(trX, trY, nTr), b = net.evaluate(teX, teY, nTe);
      M.trL = a.loss; M.trA = a.acc; M.teL = b.loss; M.teA = b.acc;
    }
    function record() {
      if (epoch % hist.every) return;
      hist.e.push(epoch); hist.tr.push(M.trL); hist.te.push(M.teL);
      if (hist.e.length > 2400) {
        for (const k of ['e', 'tr', 'te']) hist[k] = hist[k].filter((_, i) => i % 2 === 0);
        hist.every *= 2;
      }
    }
    const opts = () => ({ lr: S.lr, batch: S.batch, l2: S.l2, optimizer: S.optimizer });
    function afterEpoch() {
      epoch++;
      evalMetrics();
      if (!net.isFinite() || !Number.isFinite(M.trL)) {
        diverged = true;
        setPlaying(false);
        return false;
      }
      record();
      return true;
    }
    function trainFrame() {
      if (!nTr || diverged) return;
      const t0 = performance.now(), n = SPEEDS[S.speed - 1], o = opts();
      for (let s = 0; s < n; s++) {
        if (trainer.step(o) && !afterEpoch()) break;
        if (performance.now() - t0 > 9) break;
      }
      mainDirty = true;
    }
    function stepEpoch() {
      if (!nTr || diverged) return;
      const o = opts();
      while (!trainer.step(o));
      afterEpoch();
      mainDirty = thumbDirty = true;
      updateReadout();
    }

    // ------------------------------------------------------------ controls
    const guide = section(panel, 'Guided tour');
    const presetSel = select(guide, {
      value: 'custom',
      options: PRESETS.map((p) => [p.id, p.name]),
      onChange: (id) => applyPreset(PRESETS.find((p) => p.id === id)),
    });
    const hintEl = h('p', { class: 'hint', style: `margin:0;padding:8px 10px;border-left:2px solid ${C.amber};background:rgba(245,181,68,0.06);border-radius:0 6px 6px 0` }, PRESETS[0].hint);
    guide.append(hintEl);

    const train = section(panel, 'Training');
    const [playBtn] = buttons(train, [
      { label: 'Pause', primary: true, onClick: () => setPlaying(!playing) },
      { label: 'Step', onClick: () => stepEpoch() },
      { label: 'Reset weights', onClick: () => { weightSeed++; rebuildNet(); } },
    ]);
    function setPlaying(v) {
      playing = v && !diverged;
      playBtn.textContent = playing ? 'Pause' : 'Play';
    }
    const ro = readout(train, ['Epoch', 'Train loss', 'Test loss', 'Train accuracy', 'Test accuracy', 'Parameters']);
    function updateReadout() {
      ro.set('Epoch', epoch.toLocaleString('en-US'));
      ro.set('Train loss', fmtLoss(M.trL));
      ro.set('Test loss', fmtLoss(M.teL));
      ro.set('Train accuracy', fmtPct(M.trA));
      ro.set('Test accuracy', fmtPct(M.teA));
      ro.set('Parameters', net ? String(net.paramCount) : '—');
    }
    train.append(h('div', { style: 'height:10px' }));
    slider(train, {
      label: 'Training speed', min: 1, max: SPEEDS.length, step: 1, value: S.speed,
      format: (v) => `${SPEEDS[v - 1]} batch${SPEEDS[v - 1] > 1 ? 'es' : ''}/frame`,
      onInput: (v) => { S.speed = v; },
    });

    const dataSec = section(panel, 'Data');
    const dsSel = select(dataSec, {
      label: 'Dataset', value: S.dataset, options: DATASETS,
      onChange: (v) => { S.dataset = v; toCustom(); regenerate(); },
    });
    const noiseSl = slider(dataSec, {
      label: 'Noise', min: 0, max: 0.5, step: 0.01, value: S.noise, format: (v) => `${Math.round(v * 100)} %`,
      onInput: (v) => { S.noise = v; regenerate(); },
    });
    const ptsSl = slider(dataSec, {
      label: 'Points', min: 20, max: 600, step: 10, value: S.points,
      onInput: (v) => { S.points = v; regenerate(); },
    });
    const testSl = slider(dataSec, {
      label: 'Held-out test share', min: 0.1, max: 0.9, step: 0.05, value: S.testFrac, format: (v) => `${Math.round(v * 100)} %`,
      onInput: (v) => { S.testFrac = v; regenerate(); },
    });
    buttons(dataSec, [
      { label: 'Regenerate', onClick: () => regenerate((Math.random() * 1e9) | 0) },
      { label: 'Clear points', onClick: () => { pts = []; toCustom(); rebuildData(); rebuildNet(); } },
    ]);
    select(dataSec, {
      label: 'Click adds', value: '1', options: [['1', 'Amber (A) — shift/right-click: cyan'], ['0', 'Cyan (B) — shift/right-click: amber']],
      onChange: (v) => { S.paint = +v; },
    });
    checkbox(dataSec, { label: 'Show test points (hollow)', checked: S.showTest, onChange: (v) => { S.showTest = v; } });
    checkbox(dataSec, { label: 'Discretize output', checked: S.discretize, onChange: (v) => { S.discretize = v; mainDirty = true; } });

    const featSec = section(panel, 'Input features');
    const featGrid = h('div', { style: 'display:grid;grid-template-columns:1fr 1fr;column-gap:8px' });
    featSec.append(featGrid);
    const featBoxes = {};
    for (const F of FEATURES) {
      featBoxes[F.id] = checkbox(featGrid, {
        label: F.label, checked: S.features.includes(F.id),
        onChange: (v) => {
          const next = FEATURES.filter((G) => (G.id === F.id ? v : S.features.includes(G.id))).map((G) => G.id);
          if (!next.length) { featBoxes[F.id].checked = true; return; }
          S.features = next;
          feats = featureList(S.features);
          rebuildData();
          rebuildNet();
        },
      });
    }

    const netSec = section(panel, 'Network');
    const archEl = h('div', {});
    netSec.append(archEl);
    const smallBtn = (label, onClick, disabled) => {
      const b = h('button', { class: 'btn', style: 'padding:2px 0;width:26px;font-family:var(--mono)' }, label);
      b.disabled = !!disabled;
      b.addEventListener('click', onClick);
      return b;
    };
    function renderArch() {
      const rows = [];
      const n = S.hidden.length;
      rows.push(h('div', { class: 'ctl-row', style: 'align-items:center;margin-bottom:8px' },
        h('span', {}, 'Hidden layers'),
        h('span', { style: 'display:flex;gap:6px;align-items:center' },
          smallBtn('–', () => { S.hidden.pop(); archChanged(); }, n === 0),
          h('span', { class: 'ctl-val', style: 'min-width:14px;text-align:center' }, String(n)),
          smallBtn('+', () => { S.hidden.push(S.hidden[n - 1] || 4); archChanged(); }, n >= 5))));
      S.hidden.forEach((k, i) => {
        const dots = h('span', { style: 'display:flex;gap:2px;margin-right:4px' });
        for (let d = 0; d < 8; d++) dots.append(h('span', { style: `width:5px;height:5px;border-radius:1px;background:${d < k ? C.amber : C.line}` }));
        rows.push(h('div', { class: 'ctl-row', style: 'align-items:center;margin-bottom:6px' },
          h('span', { style: 'color:var(--muted);font-size:12px' }, `Layer ${i + 1}`),
          h('span', { style: 'display:flex;gap:6px;align-items:center' },
            dots,
            smallBtn('–', () => { S.hidden[i]--; archChanged(); }, k <= 1),
            h('span', { class: 'ctl-val', style: 'min-width:14px;text-align:center' }, String(k)),
            smallBtn('+', () => { S.hidden[i]++; archChanged(); }, k >= 8))));
      });
      archEl.replaceChildren(...rows);
    }
    function archChanged() { renderArch(); rebuildNet(); }
    const actSel = select(netSec, {
      label: 'Activation', value: S.activation,
      options: Object.keys(ACTIVATIONS).map((a) => [a, a === 'relu' ? 'ReLU' : a]),
      onChange: (v) => { S.activation = v; rebuildNet(); },
    });
    netSec.append(h('p', { class: 'hint', style: 'margin:0' }, 'Output: one sigmoid unit, binary cross-entropy loss. Weights start Xavier (He for ReLU).'));

    const optSec = section(panel, 'Optimizer');
    const optSel = select(optSec, {
      label: 'Method', value: S.optimizer, options: [['sgd', 'SGD'], ['momentum', 'Momentum (0.9)'], ['adam', 'Adam']],
      onChange: (v) => { S.optimizer = v; net.resetOptimizer(); },
    });
    const lrSl = slider(optSec, {
      label: 'Learning rate', min: -4, max: Math.log10(3), step: 0.01, value: Math.log10(S.lr),
      format: (v) => fmtLr(10 ** v),
      onInput: (v) => { S.lr = +(10 ** v).toPrecision(2); },
    });
    const batchSl = slider(optSec, {
      label: 'Batch size', min: 1, max: 64, step: 1, value: S.batch,
      onInput: (v) => { S.batch = v; },
    });
    const L2S = [0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1];
    const l2Sel = select(optSec, {
      label: 'L2 regularisation', value: '0', options: L2S.map((v) => [String(v), v ? String(v) : 'none']),
      onChange: (v) => { S.l2 = +v; },
    });
    checkbox(optSec, { label: 'Log-scale loss chart', checked: S.logLoss, onChange: (v) => { S.logLoss = v; } });

    function toCustom() { if (presetSel.value !== 'custom') presetSel.value = 'custom'; }
    function applyPreset(p) {
      hintEl.textContent = p.hint;
      if (p.id === 'custom') return;
      Object.assign(S, presetConfig(p));
      feats = featureList(S.features);
      dsSel.value = S.dataset;
      noiseSl.set(S.noise); ptsSl.set(S.points); testSl.set(S.testFrac);
      for (const F of FEATURES) featBoxes[F.id].checked = S.features.includes(F.id);
      actSel.value = S.activation; optSel.value = S.optimizer;
      lrSl.set(Math.log10(S.lr)); batchSl.set(S.batch); l2Sel.value = String(S.l2);
      renderArch();
      weightSeed = 1;
      regenerate();
      setPlaying(true);
    }

    // ------------------------------------------------------------ grid evaluation
    // acts[l] holds layer l's activations over a res x res grid: acts[l][j * res² + cell].
    function evalGrid(res, prev) {
      const N = res * res, sizes = net.sizes;
      const key = sizes.join(',') + '@' + res;
      const G = prev && prev.key === key ? prev : { key, res, acts: sizes.map((s) => new Float32Array(s * N)) };
      const x = new Float64Array(feats.length);
      for (let r = 0; r < res; r++) {
        const y = DOMAIN - ((r + 0.5) / res) * 2 * DOMAIN;
        for (let c = 0; c < res; c++) {
          const xx = -DOMAIN + ((c + 0.5) / res) * 2 * DOMAIN;
          for (let k = 0; k < feats.length; k++) x[k] = feats[k].f(xx, y);
          net.forward(x);
          const cell = r * res + c;
          for (let l = 0; l < sizes.length; l++) {
            const a = net.a[l], arr = G.acts[l];
            for (let j = 0; j < a.length; j++) arr[j * N + cell] = a[j];
          }
        }
      }
      return G;
    }
    // Map a neuron's activation to [-1, 1] for the colormap.
    function nodeScale(G, l, j) {
      const N = G.res * G.res, arr = G.acts[l];
      if (l === net.L) return (v) => 2 * v - 1;
      if (l > 0 && S.activation === 'tanh') return (v) => v;
      if (l > 0 && S.activation === 'sigmoid') return (v) => 2 * v - 1;
      let m = 1e-9;
      for (let k = j * N; k < (j + 1) * N; k++) m = Math.max(m, Math.abs(arr[k]));
      return (v) => v / m;
    }
    function paintNode(G, l, j, img, lut, discrete) {
      const N = G.res * G.res, arr = G.acts[l], sc = nodeScale(G, l, j), d = img.data;
      for (let k = 0; k < N; k++) {
        let t = sc(arr[j * N + k]);
        if (!Number.isFinite(t)) t = 0;
        if (discrete) t = t >= 0 ? 0.85 : -0.85;
        const i = Math.round((clamp(t, -1, 1) + 1) * 127.5) * 3;
        d[k * 4] = lut[i]; d[k * 4 + 1] = lut[i + 1]; d[k * 4 + 2] = lut[i + 2]; d[k * 4 + 3] = 255;
      }
    }
    const mkCanvas = (n) => { const c = document.createElement('canvas'); c.width = n; c.height = n; return c; };
    const mainCv = mkCanvas(MAIN_RES), mainCtx = mainCv.getContext('2d');
    const mainImg = mainCtx.createImageData(MAIN_RES, MAIN_RES);
    let mainGrid = null, thumbGrid = null;
    let thumbs = []; // thumbs[l][j] -> canvas
    const thumbImg = new ImageData(THUMB_RES, THUMB_RES);

    function refreshMain() {
      mainGrid = evalGrid(MAIN_RES, mainGrid);
      paintMain();
    }
    function paintMain() {
      const hv = hover && hover.l <= net.L && hover.j < net.sizes[hover.l] ? hover : null;
      const l = hv ? hv.l : net.L, j = hv ? hv.j : 0;
      paintNode(mainGrid, l, j, mainImg, LUT_MAIN, S.discretize && l === net.L);
      mainCtx.putImageData(mainImg, 0, 0);
    }
    function refreshThumbs() {
      thumbGrid = evalGrid(THUMB_RES, thumbGrid);
      thumbs = net.sizes.map((n, l) => Array.from({ length: n }, (_, j) => {
        const c = (thumbs[l] && thumbs[l][j]) || mkCanvas(THUMB_RES);
        paintNode(thumbGrid, l, j, thumbImg, LUT_THUMB, false);
        c.getContext('2d').putImageData(thumbImg, 0, 0);
        return c;
      }));
    }

    // ------------------------------------------------------------ layout
    const R = { heat: null, chart: null, net: null };
    function computeLayout(w, hh) {
      const pad = 14;
      if (w >= hh * 1.15 && w > 520) {
        const chartH = clamp(hh * 0.24, 96, 190);
        const s = Math.max(80, Math.min(hh - chartH - 3 * pad - 16, w * 0.54));
        R.heat = { x: w - pad - s, y: pad, s };
        R.chart = { x: R.heat.x - 22, y: pad + s + pad + 16, w: s + 22, h: hh - (pad + s + pad + 16) - pad };
        R.net = { x: pad, y: pad, w: R.heat.x - 22 - 3 * pad, h: hh - 2 * pad };
      } else {
        const netH = clamp(hh * 0.34, 100, 280);
        R.net = { x: pad, y: pad, w: w - 2 * pad, h: netH };
        const top = pad + netH + pad, avail = hh - top - pad - 16;
        let s = Math.min(avail, (w - 2 * pad - 22) * 0.6);
        const chartW = w - 2 * pad - 22 - s - pad;
        if (chartW >= 110) {
          R.heat = { x: pad + 22, y: top, s: Math.max(60, s) };
          R.chart = { x: R.heat.x + s + pad, y: top, w: chartW, h: s + 16 };
        } else {
          s = Math.max(60, Math.min(avail - 110, w - 2 * pad - 22));
          R.heat = { x: pad + 22 + (w - 2 * pad - 22 - s) / 2, y: top, s };
          R.chart = { x: pad, y: top + s + 16 + pad, w: w - 2 * pad, h: hh - (top + s + 16 + pad) - pad };
        }
      }
    }

    // Node positions in the network diagram.
    let nodes = [], nodeIdx = [];
    function layoutNet() {
      const box = R.net, sizes = net.sizes, ncol = sizes.length;
      const top = box.y + 22, bottom = box.y + box.h - 6;
      const left = box.x + 46, right = box.x + box.w - 8;
      const maxN = Math.max(...sizes);
      const colGap = ncol > 1 ? (right - left) / (ncol - 1) : 0;
      let ns = Math.min(64, (bottom - top) / (maxN + (maxN - 1) * 0.3), colGap ? colGap * 0.46 : 64);
      ns = Math.max(8, ns);
      nodes = [];
      sizes.forEach((n, l) => {
        const s = l === net.L ? Math.min(ns * 1.3, bottom - top) : ns;
        const gap = ns * 0.3;
        const total = n * s + (n - 1) * gap;
        const cx = ncol > 1 ? left + ns / 2 + (l * (right - left - ns)) / (ncol - 1) : (left + right) / 2;
        const y0 = (top + bottom) / 2 - total / 2;
        nodeIdx[l] = [];
        for (let j = 0; j < n; j++) nodes.push(nodeIdx[l][j] = { l, j, x: cx - s / 2, y: y0 + j * (s + gap), s });
      });
      nodeIdx.length = sizes.length;
      return ns;
    }
    const nodeAt = (l, j) => nodeIdx[l] && nodeIdx[l][j];

    // ------------------------------------------------------------ drawing
    const cv = createCanvas(stage, (w, hh) => { computeLayout(w, hh); mainDirty = true; });
    const ctx = cv.ctx;

    function drawNet() {
      const box = R.net;
      if (box.w < 60 || box.h < 40) return;
      const ns = layoutNet();
      // column headers, just above the tallest column
      const headY = Math.max(box.y + 4, Math.min(...nodes.map((n) => n.y)) - 26);
      ctx.font = font(10, 500);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      net.sizes.forEach((n, l) => {
        const nd = nodeAt(l, 0);
        const cx = nd.x + nd.s / 2;
        const label = l === 0 ? 'INPUT' : l === net.L ? 'OUTPUT' : `L${l} · ${n}`;
        ctx.fillStyle = l === 0 || l === net.L ? C.muted : rgba(C.text, 0.75);
        ctx.fillText(label, cx, headY);
      });
      // edges
      const maxW = 5.5;
      flow = (flow + (playing ? 0.6 : 0)) % 1000;
      for (let l = 0; l < net.L; l++) {
        const W = net.W[l], nIn = net.sizes[l], nOut = net.sizes[l + 1];
        for (let j = 0; j < nOut; j++) {
          const b = nodeAt(l + 1, j);
          const bx = b.x, by = b.y + b.s / 2;
          for (let i = 0; i < nIn; i++) {
            const a = nodeAt(l, i);
            const ax = a.x + a.s, ay = a.y + a.s / 2;
            const w = W[j * nIn + i];
            if (!Number.isFinite(w)) continue;
            const mag = 1 - Math.exp(-Math.abs(w) / 1.2);
            const hot = hover && ((hover.l === l && hover.j === i) || (hover.l === l + 1 && hover.j === j));
            const dim = hover && !hot ? 0.35 : 1;
            const col = w >= 0 ? C.amber : C.cyan;
            const mx = (ax + bx) / 2;
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.bezierCurveTo(mx, ay, mx, by, bx, by);
            ctx.strokeStyle = rgba(col, (0.14 + 0.7 * mag) * dim);
            ctx.lineWidth = 0.4 + maxW * mag;
            ctx.stroke();
            if (playing && mag > 0.12) {
              ctx.setLineDash([2, 10]);
              ctx.lineDashOffset = -flow * 20;
              ctx.strokeStyle = rgba('#ffffff', 0.18 * mag * dim);
              ctx.lineWidth = Math.max(1, 0.4 + maxW * mag * 0.5);
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        }
      }
      // nodes
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      for (const nd of nodes) {
        const img = thumbs[nd.l] && thumbs[nd.l][nd.j];
        const isHover = hover && hover.l === nd.l && hover.j === nd.j;
        ctx.save();
        roundRect(ctx, nd.x, nd.y, nd.s, nd.s, Math.min(4, nd.s * 0.15));
        ctx.fillStyle = C.panel;
        ctx.fill();
        ctx.clip();
        if (img) ctx.drawImage(img, nd.x, nd.y, nd.s, nd.s);
        ctx.restore();
        roundRect(ctx, nd.x + 0.5, nd.y + 0.5, nd.s - 1, nd.s - 1, Math.min(4, nd.s * 0.15));
        ctx.strokeStyle = isHover ? C.text : nd.l === net.L ? rgba(C.text, 0.45) : 'rgba(58,68,88,0.9)';
        ctx.lineWidth = isHover ? 2 : 1;
        ctx.stroke();
        // bias tick: a small bar under the node, length ∝ |b|, colour by sign
        if (nd.l > 0 && nd.s >= 14) {
          const b = net.b[nd.l - 1][nd.j];
          if (Number.isFinite(b)) {
            const len = (nd.s - 4) * (1 - Math.exp(-Math.abs(b)));
            ctx.fillStyle = rgba(b >= 0 ? C.amber : C.cyan, 0.85);
            ctx.fillRect(nd.x + 2, nd.y + nd.s + 2, Math.max(1, len), 2);
          }
        }
        if (nd.l === 0) {
          ctx.font = font(clamp(ns * 0.36, 9, 12));
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = isHover ? C.text : C.muted;
          ctx.fillText(feats[nd.j].label, nd.x - 6, nd.y + nd.s / 2);
        }
      }
      // hovered node caption
      if (hover) {
        const nd = nodeAt(hover.l, hover.j);
        if (nd) {
          const lines = [hover.l === 0 ? `input ${feats[hover.j].label}` : hover.l === net.L ? 'output P(amber)' : `layer ${hover.l} · neuron ${hover.j + 1}`];
          if (hover.l > 0) lines.push(`bias ${net.b[hover.l - 1][hover.j] >= 0 ? '+' : '−'}${Math.abs(net.b[hover.l - 1][hover.j]).toFixed(3)}`);
          ctx.font = font(11);
          const tw = Math.max(...lines.map((s) => ctx.measureText(s).width)) + 14;
          let tx = nd.x + nd.s / 2 - tw / 2, ty = nd.y + nd.s + 8;
          tx = clamp(tx, box.x, box.x + box.w - tw);
          if (ty + 40 > box.y + box.h) ty = nd.y - 8 - lines.length * 15 - 6;
          roundRect(ctx, tx, ty, tw, lines.length * 15 + 8, 5);
          ctx.fillStyle = 'rgba(11,14,20,0.92)';
          ctx.fill();
          ctx.strokeStyle = C.line;
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          lines.forEach((s, k) => { ctx.fillStyle = k ? C.muted : C.text; ctx.fillText(s, tx + 7, ty + 5 + k * 15); });
        }
      }
    }

    const toPx = (v) => R.heat.x + ((v + DOMAIN) / (2 * DOMAIN)) * R.heat.s;
    const toPy = (v) => R.heat.y + ((DOMAIN - v) / (2 * DOMAIN)) * R.heat.s;

    function drawContour() {
      const G = mainGrid, res = G.res, arr = G.acts[net.L];
      const { x: hx, y: hy, s } = R.heat;
      const cell = s / res;
      const px = (c) => hx + (c + 0.5) * cell, py = (r) => hy + (r + 0.5) * cell;
      ctx.beginPath();
      for (let r = 0; r < res - 1; r++) {
        for (let c = 0; c < res - 1; c++) {
          const v0 = arr[r * res + c] - 0.5, v1 = arr[r * res + c + 1] - 0.5;
          const v2 = arr[(r + 1) * res + c + 1] - 0.5, v3 = arr[(r + 1) * res + c] - 0.5;
          const code = (v0 > 0 ? 8 : 0) | (v1 > 0 ? 4 : 0) | (v2 > 0 ? 2 : 0) | (v3 > 0 ? 1 : 0);
          if (code === 0 || code === 15) continue;
          // edge crossing points: top (0-1), right (1-2), bottom (3-2), left (0-3)
          const e = [
            () => [px(c + v0 / (v0 - v1)), py(r)],
            () => [px(c + 1), py(r + v1 / (v1 - v2))],
            () => [px(c + v3 / (v3 - v2)), py(r + 1)],
            () => [px(c), py(r + v0 / (v0 - v3))],
          ];
          const seg = (a, b) => { const p = e[a](), q = e[b](); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); };
          switch (code) {
            case 1: case 14: seg(3, 2); break;
            case 2: case 13: seg(2, 1); break;
            case 3: case 12: seg(3, 1); break;
            case 4: case 11: seg(0, 1); break;
            case 6: case 9: seg(0, 2); break;
            case 7: case 8: seg(3, 0); break;
            case 5: seg(3, 0); seg(2, 1); break;
            case 10: seg(0, 1); seg(3, 2); break;
          }
        }
      }
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'rgba(11,14,20,0.55)';
      ctx.lineWidth = 3.5;
      ctx.stroke();
      ctx.strokeStyle = 'rgba(236,239,245,0.9)';
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }

    function drawHeat() {
      const { x: hx, y: hy, s } = R.heat;
      const showingOutput = !hover;
      // surface
      ctx.save();
      roundRect(ctx, hx, hy, s, s, 6);
      ctx.clip();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(mainCv, hx, hy, s, s);
      // axes through the origin + light grid
      ctx.strokeStyle = 'rgba(230,233,239,0.07)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let v = -6; v <= 6; v += 3) {
        ctx.moveTo(Math.round(toPx(v)) + 0.5, hy); ctx.lineTo(Math.round(toPx(v)) + 0.5, hy + s);
        ctx.moveTo(hx, Math.round(toPy(v)) + 0.5); ctx.lineTo(hx + s, Math.round(toPy(v)) + 0.5);
      }
      ctx.stroke();
      if (showingOutput) drawContour();
      // points
      const pr = clamp(s / 120, 2.4, 4.6);
      for (const pass of [0, 1]) {
        for (const p of pts) {
          if (p.test !== !!pass) continue;
          if (p.test && !S.showTest) continue;
          const x = toPx(p.x), y = toPy(p.y), col = p.label ? C.amber : C.cyan;
          ctx.beginPath();
          ctx.arc(x, y, pr, 0, Math.PI * 2);
          if (!p.test) {
            ctx.fillStyle = col;
            ctx.fill();
            ctx.strokeStyle = 'rgba(11,14,20,0.85)';
            ctx.lineWidth = 1.2;
            ctx.stroke();
          } else {
            ctx.fillStyle = 'rgba(11,14,20,0.55)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(11,14,20,0.85)';
            ctx.lineWidth = 3.2;
            ctx.stroke();
            ctx.strokeStyle = col;
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
      }
      // click ripples
      for (const rp of ripples) {
        ctx.beginPath();
        ctx.arc(toPx(rp.x), toPy(rp.y), pr + rp.t * 22, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(rp.label ? C.amber : C.cyan, 0.8 * (1 - rp.t));
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.restore();
      roundRect(ctx, hx + 0.5, hy + 0.5, s - 1, s - 1, 6);
      ctx.strokeStyle = hover ? rgba(C.text, 0.5) : C.line;
      ctx.lineWidth = 1;
      ctx.stroke();

      // ticks
      ctx.font = font(10);
      ctx.fillStyle = C.muted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (let v = -6; v <= 6; v += 3) ctx.fillText(String(v).replace('-', '−'), toPx(v), hy + s + 4);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let v = -6; v <= 6; v += 3) ctx.fillText(String(v).replace('-', '−'), hx - 5, toPy(v));

      // caption chip (top-left)
      let cap;
      if (hover) cap = hover.l === 0 ? `input ${feats[hover.j].label}` : hover.l === net.L ? 'output' : `layer ${hover.l} · neuron ${hover.j + 1} (${S.activation})`;
      else cap = S.discretize ? 'output · class' : 'output · P(amber)';
      if (pointer && !hover && s > 160) {
        const x = new Float64Array(feats.length);
        for (let k = 0; k < feats.length; k++) x[k] = feats[k].f(pointer.x, pointer.y);
        const p = net.forward(x);
        cap += `   (${pointer.x.toFixed(1)}, ${pointer.y.toFixed(1)}) → ${Number.isFinite(p) ? p.toFixed(2) : '—'}`;
      }
      chip(cap, hx + 8, hy + 8, hover ? C.text : C.muted);
      if (diverged) {
        ctx.font = font(12, 500);
        const msg = 'diverged — weights overflowed. Lower the rate, then reset.';
        const tw = ctx.measureText(msg).width;
        if (tw + 24 < s) {
          roundRect(ctx, hx + s / 2 - tw / 2 - 12, hy + s / 2 - 16, tw + 24, 32, 6);
          ctx.fillStyle = 'rgba(11,14,20,0.9)';
          ctx.fill();
          ctx.strokeStyle = C.danger;
          ctx.stroke();
          ctx.fillStyle = C.danger;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(msg, hx + s / 2, hy + s / 2);
        }
      }
      if (!pts.length) {
        ctx.font = font(12);
        ctx.fillStyle = rgba(C.text, 0.75);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('click to add amber · shift-click for cyan', hx + s / 2, hy + s / 2 + (diverged ? 30 : 0));
      }
    }
    function chip(text, x, y, color) {
      ctx.font = font(11);
      const tw = ctx.measureText(text).width;
      roundRect(ctx, x, y, tw + 14, 20, 5);
      ctx.fillStyle = 'rgba(11,14,20,0.72)';
      ctx.fill();
      ctx.fillStyle = color;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x + 7, y + 10.5);
    }

    function drawChart() {
      const r = R.chart;
      if (!r || r.w < 80 || r.h < 50) return;
      roundRect(ctx, r.x, r.y, r.w, r.h, 8);
      ctx.fillStyle = 'rgba(18,23,34,0.62)';
      ctx.fill();
      ctx.strokeStyle = C.line;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.font = font(10, 500);
      ctx.fillStyle = C.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('LOSS', r.x + 10, r.y + 8);
      // legend with current values (one line if it fits beside the title, else stacked)
      ctx.font = font(11);
      ctx.textAlign = 'right';
      const legend = [[C.good, `test ${fmtLoss(M.teL)}`], [C.text, `train ${fmtLoss(M.trL)}`]];
      const oneLine = legend.reduce((a, [, t]) => a + ctx.measureText(t).width + 22, 0) + 50 < r.w;
      if (oneLine) {
        let lx = r.x + r.w - 10;
        for (const [col, txt] of legend) {
          ctx.fillStyle = col;
          ctx.fillText(txt, lx, r.y + 7);
          const tw = ctx.measureText(txt).width;
          ctx.fillRect(lx - tw - 12, r.y + 13, 8, 2);
          lx -= tw + 22;
        }
      } else {
        ctx.textAlign = 'left';
        legend.slice().reverse().forEach(([col, txt], k) => {
          const y = r.y + 24 + k * 14;
          ctx.fillStyle = col;
          ctx.fillRect(r.x + 10, y + 6, 8, 2);
          ctx.fillText(txt, r.x + 22, y);
        });
      }
      const top = oneLine ? 28 : 58;
      const P = { x: r.x + 38, y: r.y + top, w: r.w - 48, h: r.h - top - 20 };
      if (P.w < 30 || P.h < 16) return;
      const n = hist.e.length;
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) {
        for (const v of [hist.tr[i], hist.te[i]]) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
      if (!Number.isFinite(lo)) { lo = 0.01; hi = 1; }
      const log = S.logLoss;
      let y0, y1;
      if (log) {
        y0 = Math.floor(Math.log10(Math.max(lo, 1e-6)));
        y1 = Math.ceil(Math.log10(Math.max(hi, 1e-6) * 1.0001));
        if (y1 <= y0) y1 = y0 + 1;
      } else { y0 = 0; y1 = Math.max(hi * 1.08, 1e-3); }
      const yv = (v) => (log ? Math.log10(Math.max(v, 1e-6)) : v);
      const Y = (v) => P.y + P.h - ((clamp(yv(v), y0, y1) - y0) / (y1 - y0)) * P.h;
      const eMax = Math.max(10, epoch);
      const X = (e) => P.x + (e / eMax) * P.w;
      // y grid + labels
      ctx.font = font(9);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.strokeStyle = 'rgba(36,44,59,0.8)';
      ctx.beginPath();
      const yTicks = [];
      if (log) { const stepD = Math.ceil((y1 - y0) / 4); for (let d = y0; d <= y1; d += stepD) yTicks.push([10 ** d, d >= 0 ? String(10 ** d) : `1e${d}`]); }
      else for (let k = 0; k <= 3; k++) { const v = (y1 * k) / 3; yTicks.push([v, v.toFixed(v < 1 ? 2 : 1)]); }
      for (const [v, lab] of yTicks) {
        const yy = Math.round(Y(v)) + 0.5;
        ctx.moveTo(P.x, yy); ctx.lineTo(P.x + P.w, yy);
        ctx.fillStyle = C.muted;
        ctx.fillText(lab, P.x - 5, yy);
      }
      ctx.stroke();
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('0', P.x, P.y + P.h + 4);
      ctx.textAlign = 'right';
      ctx.fillText(`epoch ${epoch.toLocaleString('en-US')}`, P.x + P.w, P.y + P.h + 4);
      if (n < 1) return;
      const line = (arr, col, width) => {
        ctx.beginPath();
        const stride = Math.max(1, Math.floor(n / (P.w * 1.5)));
        let started = false;
        for (let i = 0; i < n; i += stride) {
          const v = arr[i];
          if (!Number.isFinite(v)) { started = false; continue; }
          const x = X(hist.e[i]), y = Y(v);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        if (Number.isFinite(arr[n - 1])) ctx.lineTo(X(hist.e[n - 1]), Y(arr[n - 1]));
        ctx.strokeStyle = col;
        ctx.lineWidth = width;
        ctx.lineJoin = 'round';
        ctx.stroke();
      };
      if (nTe) line(hist.te, C.good, 1.6);
      line(hist.tr, C.text, 1.6);
      // current-value dots
      for (const [arr, col] of [[hist.tr, C.text], [hist.te, C.good]]) {
        const v = arr[n - 1];
        if (!Number.isFinite(v)) continue;
        ctx.beginPath();
        ctx.arc(X(hist.e[n - 1]), Y(v), 2.6, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
      }
    }

    function draw() {
      const { w, h: H } = cv.size;
      ctx.clearRect(0, 0, w, H);
      if (w < 60 || H < 60 || !R.heat) return;
      drawNet();
      drawHeat();
      drawChart();
    }

    // ------------------------------------------------------------ pointer
    function toDomain(e) {
      const rect = cv.canvas.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      const { x: hx, y: hy, s } = R.heat || { x: 0, y: 0, s: 0 };
      const inside = px >= hx && px <= hx + s && py >= hy && py <= hy + s;
      return { px, py, inside, x: ((px - hx) / s) * 2 * DOMAIN - DOMAIN, y: DOMAIN - ((py - hy) / s) * 2 * DOMAIN };
    }
    function onMove(e) {
      const d = toDomain(e);
      const hit = nodes.find((n) => d.px >= n.x && d.px <= n.x + n.s && d.py >= n.y && d.py <= n.y + n.s);
      const nh = hit ? { l: hit.l, j: hit.j } : null;
      if ((nh && (!hover || nh.l !== hover.l || nh.j !== hover.j)) || (!nh && hover)) {
        hover = nh;
        if (mainGrid) paintMain();
      }
      pointer = d.inside ? { x: d.x, y: d.y } : null;
      cv.canvas.style.cursor = hit ? 'pointer' : d.inside ? 'crosshair' : 'default';
    }
    function onLeave() {
      pointer = null;
      if (hover) { hover = null; if (mainGrid) paintMain(); }
    }
    function onDown(e) {
      const d = toDomain(e);
      if (!d.inside) return;
      e.preventDefault();
      const other = e.button === 2 || e.shiftKey;
      const label = other ? 1 - S.paint : S.paint;
      pts.push({ x: clamp(d.x, -5.95, 5.95), y: clamp(d.y, -5.95, 5.95), label, test: false });
      ripples.push({ x: d.x, y: d.y, label, t: 0 });
      rebuildData();
      evalMetrics();
      mainDirty = true;
      updateReadout();
    }
    const onContext = (e) => e.preventDefault();
    cv.canvas.addEventListener('pointermove', onMove);
    cv.canvas.addEventListener('pointerleave', onLeave);
    cv.canvas.addEventListener('pointerdown', onDown);
    cv.canvas.addEventListener('contextmenu', onContext);

    // ------------------------------------------------------------ go
    renderArch();
    regenerate();
    setPlaying(true);

    let roTimer = 0;
    const stop = loop((dt) => {
      frame++;
      if (playing) trainFrame();
      if (mainDirty && (frame % 2 === 0 || !playing)) { refreshMain(); mainDirty = false; }
      if (thumbDirty || (playing && frame % 4 === 0)) { refreshThumbs(); thumbDirty = false; }
      for (let i = ripples.length - 1; i >= 0; i--) { ripples[i].t += dt * 1.6; if (ripples[i].t >= 1) ripples.splice(i, 1); }
      if (!mainGrid) refreshMain();
      draw();
      roTimer += dt;
      if (roTimer > 0.1) { roTimer = 0; updateReadout(); }
    });

    return () => {
      stop();
      cv.destroy();
      cv.canvas.removeEventListener('pointermove', onMove);
      cv.canvas.removeEventListener('pointerleave', onLeave);
      cv.canvas.removeEventListener('pointerdown', onDown);
      cv.canvas.removeEventListener('contextmenu', onContext);
    };
  },
};
