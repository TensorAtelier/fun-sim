// Stats Lab — a handful of live statistical experiments sharing one stage:
//   1. Central Limit Theorem   2. Monte Carlo π   3. Random walks (1D / 2D)   4. Gambler's ruin
// Each experiment is a factory (controls, env) -> { resize, frame, reset, pointer?, destroy? }.
import { createLayout, h, section, slider, select, checkbox, buttons, readout, createCanvas, loop } from '../ui.js';

const C = {
  bg: '#0b0e14', bg2: '#121722', panel: '#151b27', line: '#242c3b', text: '#e6e9ef',
  muted: '#8a93a6', accent: '#f5b544', accent2: '#5ec8e5', good: '#7bd88f', danger: '#ef6b6b',
};
const MONO = '"IBM Plex Mono", ui-monospace, monospace';
const font = (px, weight = 400) => `${weight} ${px}px ${MONO}`;

// ---------------------------------------------------------------- helpers
function hexRgb(hex) { const n = parseInt(hex.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function rgba(hex, a) { const [r, g, b] = hexRgb(hex); return `rgba(${r},${g},${b},${a})`; }
function mix(h1, h2, t, a = 1) {
  const A = hexRgb(h1), B = hexRgb(h2);
  return `rgba(${Math.round(A[0] + (B[0] - A[0]) * t)},${Math.round(A[1] + (B[1] - A[1]) * t)},${Math.round(A[2] + (B[2] - A[2]) * t)},${a})`;
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const fmtN = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const fmtSigned = (v, d = 4) => (Number.isFinite(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d) : '—');
function gauss() { let u = 0; while (u === 0) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }
const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

function niceStep(range, count) {
  const raw = range / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}
function ticks(lo, hi, count = 5) {
  const s = niceStep(hi - lo, count);
  const out = [];
  for (let v = Math.ceil(lo / s - 1e-9) * s; v <= hi + s * 1e-9; v += s) out.push(Math.abs(v) < s * 1e-9 ? 0 : v);
  return { vals: out, step: s };
}
const tickFmt = (v, step) => v.toFixed(Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));

function rr(ctx, x, y, w, hh, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + hh, r);
  ctx.arcTo(x + w, y + hh, x, y + hh, r);
  ctx.arcTo(x, y + hh, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
// A framed card with an uppercase title (left) and optional note (right).
function card(ctx, r, title, note, noteColor = C.muted) {
  ctx.save();
  rr(ctx, r.x, r.y, r.w, r.h, 10);
  ctx.fillStyle = 'rgba(18,23,34,0.62)';
  ctx.fill();
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.textBaseline = 'top';
  if (title) {
    ctx.font = font(10, 500);
    ctx.fillStyle = C.muted;
    ctx.textAlign = 'left';
    ctx.fillText(title.toUpperCase(), r.x + 12, r.y + 10);
  }
  if (note) {
    ctx.font = font(11);
    ctx.fillStyle = noteColor;
    ctx.textAlign = 'right';
    ctx.fillText(note, r.x + r.w - 12, r.y + 9);
  }
  ctx.restore();
}
function inset(r, l, t, rt, b) { return { x: r.x + l, y: r.y + t, w: Math.max(10, r.w - l - rt), h: Math.max(10, r.h - t - b) }; }

// Grid lines + tick labels. sx/sy map value -> px. xt/yt from ticks().
function axes(ctx, r, sx, sy, xt, yt, { xf, yf, xLabel, yLabel } = {}) {
  ctx.save();
  ctx.lineWidth = 1;
  ctx.font = font(10);
  ctx.fillStyle = C.muted;
  if (yt) {
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of yt.vals) {
      const y = Math.round(sy(v)) + 0.5;
      if (y < r.y - 1 || y > r.y + r.h + 1) continue;
      ctx.strokeStyle = rgba(C.line, 0.8);
      ctx.beginPath(); ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y); ctx.stroke();
      ctx.fillText(yf ? yf(v) : tickFmt(v, yt.step), r.x - 6, y);
    }
  }
  if (xt) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const v of xt.vals) {
      const x = Math.round(sx(v)) + 0.5;
      if (x < r.x - 1 || x > r.x + r.w + 1) continue;
      ctx.strokeStyle = C.line;
      ctx.beginPath(); ctx.moveTo(x, r.y + r.h); ctx.lineTo(x, r.y + r.h + 4); ctx.stroke();
      ctx.fillText(xf ? xf(v) : tickFmt(v, xt.step), x, r.y + r.h + 6);
    }
  }
  ctx.strokeStyle = '#34405a';
  ctx.beginPath(); ctx.moveTo(r.x, r.y + r.h + 0.5); ctx.lineTo(r.x + r.w, r.y + r.h + 0.5); ctx.stroke();
  if (xLabel) { ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.muted, 0.8); ctx.fillText(xLabel, r.x + r.w, r.y + r.h - 4); }
  if (yLabel) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.muted, 0.8); ctx.fillText(yLabel, r.x + 4, r.y + 2); }
  ctx.restore();
}

function dashedV(ctx, x, y0, y1, color, dash = [4, 4]) {
  ctx.save(); ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, y0); ctx.lineTo(Math.round(x) + 0.5, y1); ctx.stroke(); ctx.restore();
}
function dashedH(ctx, y, x0, x1, color, dash = [4, 4]) {
  ctx.save(); ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); ctx.stroke(); ctx.restore();
}
function label(ctx, text, x, y, color, align = 'left', base = 'middle', size = 10, weight = 400) {
  ctx.font = font(size, weight); ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(text, x, y);
}
function glowDot(ctx, x, y, r, color, blur = 10) {
  ctx.save(); ctx.shadowColor = color; ctx.shadowBlur = blur; ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.restore();
}

// Offscreen layer at device resolution, drawn in CSS px.
function makeLayer() {
  const cv = document.createElement('canvas');
  const ctx = cv.getContext('2d');
  return {
    cv, ctx, w: 0, h: 0,
    resize(w, hh, dpr) {
      this.w = Math.max(1, Math.round(w)); this.h = Math.max(1, Math.round(hh));
      cv.width = Math.max(1, Math.round(this.w * dpr)); cv.height = Math.max(1, Math.round(this.h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    },
    clear() { ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height); ctx.restore(); },
  };
}

// Speed slider over a discrete list of rates.
function speedSlider(parent, { label: lbl = 'Speed', values, value, unit = '/frame', onInput }) {
  let idx = 0;
  values.forEach((v, i) => { if (Math.abs(v - value) < Math.abs(values[idx] - value)) idx = i; });
  const f = (v) => (v >= 1000 ? v / 1000 + 'k' : String(v));
  const s = slider(parent, { label: lbl, min: 0, max: values.length - 1, step: 1, value: idx, format: (i) => f(values[i]) + unit, onInput: (i) => onInput?.(values[i]) });
  return { get value() { return values[s.value]; } };
}

// Throttled readout updates (~8 Hz).
function throttled(fn, every = 0.12) {
  let acc = every;
  return (dt, force) => { acc += dt; if (acc >= every || force) { acc = 0; fn(); } };
}

// ================================================================= 1. CLT
const CLT_B = 160;
function gridDist({ lo = 0, hi = 1, pdf, cdf, heights }) {
  const B = CLT_B, w = (hi - lo) / B;
  const mass = new Float64Array(B);
  for (let i = 0; i < B; i++) {
    if (heights) mass[i] = Math.max(0, heights[i]);
    else if (cdf) mass[i] = cdf(lo + (i + 1) * w) - cdf(lo + i * w);
    else mass[i] = pdf(lo + (i + 0.5) * w) * w;
  }
  let tot = mass.reduce((a, b) => a + b, 0);
  if (!(tot > 0)) { mass.fill(1); tot = B; }
  const cum = new Float64Array(B + 1);
  let mu = 0, m2 = 0, maxD = 0;
  for (let i = 0; i < B; i++) {
    mass[i] /= tot;
    cum[i + 1] = cum[i] + mass[i];
    const c = lo + (i + 0.5) * w;
    mu += mass[i] * c;
    m2 += mass[i] * (c * c + (w * w) / 12);
    maxD = Math.max(maxD, mass[i] / w);
  }
  return {
    discrete: false, lo, hi, w, mass, mu, var: m2 - mu * mu, maxD, sumH: heights ? tot : 0,
    density: (i) => mass[i] / w,
    sample() {
      const u = Math.random() * cum[B];
      let a = 0, b = B - 1;
      while (a < b) { const m = (a + b + 1) >> 1; if (cum[m] <= u) a = m; else b = m - 1; }
      return lo + (a + Math.random()) * w;
    },
  };
}
const DICE = { discrete: true, lo: 0.5, hi: 6.5, lattice: 1, mu: 3.5, var: 35 / 12, sample: () => 1 + Math.floor(Math.random() * 6) };
const CLT_DISTS = {
  uniform: { label: 'Uniform', make: () => gridDist({ pdf: () => 1 }) },
  exponential: { label: 'Exponential', make: () => gridDist({ lo: 0, hi: 5, cdf: (x) => 1 - Math.exp(-x) }) },
  bimodal: { label: 'Bimodal', make: () => gridDist({ pdf: (x) => 0.55 * npdf(x, 0.27, 0.08) + 0.45 * npdf(x, 0.74, 0.065) }) },
  dice: { label: 'Dice 1–6', make: () => DICE },
  arcsine: { label: 'U-shaped', make: () => gridDist({ cdf: (x) => (2 / Math.PI) * Math.asin(Math.sqrt(clamp(x, 0, 1))) }) },
  custom: { label: 'Hand-drawn' },
};

function createCLT(ctl, env) {
  let distKey = 'uniform', n = 5, speed = 0.5, zoom = true, showCurve = true;
  let dist = CLT_DISTS.uniform.make();
  let custom = null;
  let W = 0, H = 0;
  // accumulated means
  let N = 0, sum = 0, sumsq = 0, counts = new Float64Array(1), bins = null, yMaxB = 1, acc = 0;
  let flights = [], pulses = [];
  let drawing = false, lastPaint = null, hoverTop = false;

  const s1 = section(ctl, 'Source distribution');
  const distSel = select(s1, {
    options: [['uniform', 'Uniform'], ['exponential', 'Exponential (λ = 1)'], ['bimodal', 'Bimodal mixture'], ['dice', 'Dice (discrete 1–6)'], ['arcsine', 'U-shaped (arcsine)'], ['custom', 'Draw your own ✎']],
    value: distKey,
    onChange: (v) => { if (v === 'custom' && !custom) seedCustom(); distKey = v; rebuild(); },
  });
  s1.append(h('p', { class: 'hint' }, 'Drag on the top chart to sketch any distribution — even a lopsided one. The means still end up bell-shaped.'));

  const s2 = section(ctl, 'Sampling');
  slider(s2, { label: 'Sample size n', min: 1, max: 100, value: n, onInput: (v) => { n = v; resetMeans(); } });
  const sp = speedSlider(s2, { label: 'Means per frame', values: [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], value: speed, onInput: (v) => { speed = v; } });
  speed = sp.value;
  checkbox(s2, { label: 'Zoom bottom chart to fit', checked: zoom, onChange: (v) => { zoom = v; resetMeans(); } });
  checkbox(s2, { label: 'Overlay predicted normal N(μ, σ²/n)', checked: showCurve, onChange: (v) => { showCurve = v; } });

  const s3 = section(ctl, 'Sample means');
  const ro = readout(s3, ['Means drawn', 'Source μ', 'Source σ', 'Mean of x̄', 'SD of x̄', 'σ / √n', 'Skew of x̄']);
  s3.append(h('p', { class: 'hint' }, 'Slow speeds show individual samples dropping out of the source and merging into one mean.'));

  function seedCustom() {
    custom = new Float64Array(CLT_B);
    if (!dist.discrete && dist.mass) {
      let mx = 0;
      for (let i = 0; i < CLT_B; i++) mx = Math.max(mx, dist.mass[i]);
      for (let i = 0; i < CLT_B; i++) custom[i] = (dist.mass[i] / mx) * 0.8;
    } else {
      for (let i = 0; i < CLT_B; i++) { const x = (i + 0.5) / CLT_B; custom[i] = 0.12 + 0.6 * Math.exp(-((x - 0.2) ** 2) / 0.004) + 0.35 * Math.exp(-((x - 0.7) ** 2) / 0.02); }
    }
  }
  function rebuild() {
    dist = distKey === 'custom' ? gridDist({ heights: custom }) : CLT_DISTS[distKey].make();
    resetMeans();
  }
  function resetMeans() {
    N = 0; sum = 0; sumsq = 0; acc = 0; flights = []; pulses = [];
    sum3 = 0;
    const sd = Math.sqrt(dist.var / n);
    let blo = dist.lo, bhi = dist.hi;
    if (zoom) { blo = Math.max(dist.lo, dist.mu - 4.2 * sd); bhi = Math.min(dist.hi, dist.mu + 4.2 * sd); }
    let bw = (bhi - blo) / 60, off = blo;
    if (dist.discrete) { const L = dist.lattice / n; bw = Math.max(1, Math.ceil(bw / L - 1e-9)) * L; off = -L / 2; }
    const m0 = Math.floor((blo - off) / bw + 1e-9), m1 = Math.floor((bhi - off) / bw - 1e-9);
    const nb = Math.max(1, m1 - m0 + 1);
    bins = { off, bw, m0, nb, xlo: off + m0 * bw, xhi: off + (m0 + nb) * bw };
    counts = new Float64Array(nb);
    yMaxB = (1 / (sd * Math.sqrt(2 * Math.PI))) * 1.15;
    updRO(0, true);
  }
  let sum3 = 0;
  function commit(m) {
    N++; sum += m; sumsq += m * m; sum3 += m * m * m;
    const b = Math.floor((m - bins.off) / bins.bw + 1e-9) - bins.m0;
    if (b >= 0 && b < bins.nb) counts[b]++;
  }
  function binOf(m) { return Math.floor((m - bins.off) / bins.bw + 1e-9) - bins.m0; }

  const updRO = throttled(() => {
    const mean = N ? sum / N : NaN;
    const v = N > 1 ? Math.max(0, (sumsq - N * mean * mean) / (N - 1)) : NaN;
    const sd = Math.sqrt(v);
    const m3 = N > 2 ? sum3 / N - 3 * mean * (sumsq / N) + 2 * mean ** 3 : NaN;
    const pv = N > 2 ? sumsq / N - mean * mean : NaN;
    ro.set('Means drawn', fmtInt(N));
    ro.set('Source μ', fmtN(dist.mu, 4));
    ro.set('Source σ', fmtN(Math.sqrt(dist.var), 4));
    ro.set('Mean of x̄', fmtN(mean, 4));
    ro.set('SD of x̄', fmtN(sd, 4));
    ro.set('σ / √n', fmtN(Math.sqrt(dist.var / n), 4));
    ro.set('Skew of x̄', pv > 0 ? fmtSigned(m3 / pv ** 1.5, 3) : '—');
  });

  // ---- layout
  function lay() {
    const pad = 14, gap = clamp(H * 0.09, 30, 70);
    const avail = H - 2 * pad - gap;
    const topH = Math.round(avail * 0.4);
    const top = { x: pad, y: pad, w: W - 2 * pad, h: topH };
    const bot = { x: pad, y: pad + topH + gap, w: W - 2 * pad, h: avail - topH };
    return { top, bot, ti: inset(top, 48, 32, 16, 24), bi: inset(bot, 48, 32, 16, 24) };
  }
  const topScale = () => {
    if (dist.discrete) return (1 / 6) / 0.72;
    if (distKey === 'custom') return 1 / (dist.sumH * dist.w);
    return dist.maxD / 0.85;
  };
  function srcHeightAt(v) { // density at v
    if (dist.discrete) return 1 / 6;
    const i = clamp(Math.floor((v - dist.lo) / dist.w), 0, CLT_B - 1);
    return dist.density(i);
  }

  function spawn() {
    const xs = new Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { xs[i] = dist.sample(); s += xs[i]; }
    const mean = s / n;
    const animateAll = speed <= 0.2;
    if (animateAll || (speed <= 2 && flights.length < 3)) {
      const shown = xs.slice(0, Math.min(n, 28));
      const D = speed <= 0.1 ? 1.9 : 1.35;
      flights.push({
        mean, t: 0, D, done: false,
        parts: shown.map((v, i) => ({ v, f: Math.random(), jit: (Math.random() - 0.5) * 0.5, delay: (i / shown.length) * 0.3 * D })),
      });
    } else commit(mean);
  }

  function frame(dt, ctx, w, hh, running) {
    W = w; H = hh;
    if (running) {
      acc += speed;
      let guard = 0;
      while (acc >= 1 && guard++ < 2000) { acc -= 1; spawn(); }
      for (const f of flights) {
        f.t += dt;
        if (!f.done && f.t >= f.D * 0.88) { f.done = true; commit(f.mean); pulses.push({ m: f.mean, t: 0 }); }
      }
      flights = flights.filter((f) => f.t < f.D);
      for (const p of pulses) p.t += dt;
      pulses = pulses.filter((p) => p.t < 0.6);
    }
    updRO(dt);
    const L = lay();
    drawTop(ctx, L);
    drawBottom(ctx, L, dt);
    drawFlights(ctx, L);
  }

  function drawTop(ctx, L) {
    const { top, ti } = L;
    const title = `Source · ${CLT_DISTS[distKey].label}`;
    card(ctx, top, title, `μ = ${dist.mu.toFixed(3)}   σ = ${Math.sqrt(dist.var).toFixed(3)}`);
    const ys = topScale();
    const sx = (v) => ti.x + ((v - dist.lo) / (dist.hi - dist.lo)) * ti.w;
    const sy = (d) => ti.y + ti.h - (d / ys) * ti.h;
    axes(ctx, ti, sx, sy, ticks(dist.lo, dist.hi, dist.discrete ? 6 : 6), ticks(0, ys, 3), { yLabel: dist.discrete ? 'P(x)' : 'density' });
    ctx.save();
    ctx.beginPath(); ctx.rect(ti.x, ti.y - 2, ti.w, ti.h + 2); ctx.clip();
    if (dist.discrete) {
      for (let k = 1; k <= 6; k++) {
        const x0 = sx(k - 0.3), x1 = sx(k + 0.3), y = sy(1 / 6);
        const g = ctx.createLinearGradient(0, y, 0, ti.y + ti.h);
        g.addColorStop(0, rgba(C.accent2, 0.55)); g.addColorStop(1, rgba(C.accent2, 0.08));
        ctx.fillStyle = g; ctx.fillRect(x0, y, x1 - x0, ti.y + ti.h - y);
        ctx.fillStyle = C.accent2; ctx.fillRect(x0, y, x1 - x0, 2);
      }
    } else {
      const g = ctx.createLinearGradient(0, ti.y, 0, ti.y + ti.h);
      g.addColorStop(0, rgba(C.accent2, 0.42)); g.addColorStop(1, rgba(C.accent2, 0.04));
      ctx.beginPath();
      ctx.moveTo(sx(dist.lo), ti.y + ti.h);
      for (let i = 0; i < CLT_B; i++) {
        const y = sy(dist.density(i));
        ctx.lineTo(sx(dist.lo + i * dist.w), y);
        ctx.lineTo(sx(dist.lo + (i + 1) * dist.w), y);
      }
      ctx.lineTo(sx(dist.hi), ti.y + ti.h);
      ctx.closePath();
      ctx.fillStyle = g; ctx.fill();
      ctx.beginPath();
      for (let i = 0; i < CLT_B; i++) {
        const y = sy(dist.density(i));
        const xm = sx(dist.lo + (i + 0.5) * dist.w);
        if (i === 0) ctx.moveTo(sx(dist.lo), y);
        ctx.lineTo(xm, y);
      }
      ctx.lineTo(sx(dist.hi), sy(dist.density(CLT_B - 1)));
      ctx.strokeStyle = C.accent2; ctx.lineWidth = 1.6; ctx.lineJoin = 'round';
      ctx.shadowColor = rgba(C.accent2, 0.6); ctx.shadowBlur = 8; ctx.stroke();
    }
    ctx.restore();
    // μ marker and ±σ bracket
    const xm = sx(dist.mu);
    dashedV(ctx, xm, ti.y, ti.y + ti.h, rgba(C.accent, 0.8));
    label(ctx, 'μ', xm + 4, ti.y + 6, C.accent, 'left', 'top', 11, 600);
    const sdv = Math.sqrt(dist.var);
    const yb = ti.y + ti.h - 8;
    ctx.save(); ctx.strokeStyle = rgba(C.accent, 0.55); ctx.lineWidth = 1;
    const a = sx(Math.max(dist.lo, dist.mu - sdv)), b = sx(Math.min(dist.hi, dist.mu + sdv));
    ctx.beginPath(); ctx.moveTo(a, yb - 3); ctx.lineTo(a, yb + 3); ctx.moveTo(a, yb); ctx.lineTo(b, yb); ctx.moveTo(b, yb - 3); ctx.lineTo(b, yb + 3); ctx.stroke();
    ctx.restore();
    label(ctx, '±σ', b + 4, yb, rgba(C.accent, 0.8), 'left', 'middle', 9);
    // draw hint
    const hint = distKey === 'custom' ? (drawing ? 'drawing…' : '✎ drag to reshape') : '✎ drag to draw your own';
    label(ctx, hint, ti.x + ti.w - 4, ti.y + 6, hoverTop || drawing ? C.accent : rgba(C.muted, 0.75), 'right', 'top', 10);
  }

  function bottomMaps(bi) {
    const sx = (v) => bi.x + ((v - bins.xlo) / (bins.xhi - bins.xlo)) * bi.w;
    const sy = (d) => bi.y + bi.h - (d / yMaxB) * bi.h;
    return { sx, sy };
  }

  function drawBottom(ctx, L, dt) {
    const { bot, bi } = L;
    const sd = Math.sqrt(dist.var / n);
    card(ctx, bot, `Sample means x̄ · n = ${n}`, `${fmtInt(N)} means`, C.accent);
    // y scale easing
    let maxD = 0;
    for (let i = 0; i < bins.nb; i++) maxD = Math.max(maxD, counts[i]);
    maxD = N ? maxD / (N * bins.bw) : 0;
    const peak = 1 / (sd * Math.sqrt(2 * Math.PI));
    const target = clamp(maxD, peak, peak * (N < 200 ? 1.1 : 1.6)) * 1.15; // early noisy bars may clip
    yMaxB += (target - yMaxB) * Math.min(1, dt * 6);
    const { sx, sy } = bottomMaps(bi);
    axes(ctx, bi, sx, sy, ticks(bins.xlo, bins.xhi, 7), ticks(0, yMaxB, 3), { yLabel: 'density' });
    ctx.save();
    ctx.beginPath(); ctx.rect(bi.x, bi.y - 4, bi.w, bi.h + 4); ctx.clip();
    if (N > 0) {
      const g = ctx.createLinearGradient(0, bi.y, 0, bi.y + bi.h);
      g.addColorStop(0, rgba(C.accent, 0.9)); g.addColorStop(1, rgba(C.accent, 0.28));
      ctx.fillStyle = g;
      const bwPx = (bins.bw / (bins.xhi - bins.xlo)) * bi.w;
      const gapPx = bwPx > 5 ? 1 : 0;
      for (let i = 0; i < bins.nb; i++) {
        if (!counts[i]) continue;
        const d = counts[i] / (N * bins.bw);
        const x = sx(bins.xlo + i * bins.bw), y = sy(d);
        ctx.fillRect(x + gapPx * 0.5, y, bwPx - gapPx, bi.y + bi.h - y);
      }
    }
    if (showCurve) {
      ctx.beginPath();
      const steps = 220;
      for (let k = 0; k <= steps; k++) {
        const v = bins.xlo + ((bins.xhi - bins.xlo) * k) / steps;
        const y = sy(npdf(v, dist.mu, sd));
        if (k === 0) ctx.moveTo(sx(v), y); else ctx.lineTo(sx(v), y);
      }
      ctx.strokeStyle = C.accent2; ctx.lineWidth = 2;
      ctx.shadowColor = rgba(C.accent2, 0.7); ctx.shadowBlur = 10; ctx.stroke();
    }
    ctx.restore();
    // pulses where means landed
    for (const p of pulses) {
      const b = binOf(p.m);
      const d = b >= 0 && b < bins.nb && N ? counts[b] / (N * bins.bw) : 0;
      const x = sx(p.m), y = sy(d);
      const k = p.t / 0.6;
      ctx.save(); ctx.strokeStyle = rgba(C.accent, 1 - k); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, 3 + k * 14, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
    }
    // μ line
    const xm = sx(dist.mu);
    if (xm >= bi.x && xm <= bi.x + bi.w) {
      dashedV(ctx, xm, bi.y, bi.y + bi.h, rgba(C.text, 0.35));
      label(ctx, 'μ', xm + 4, bi.y + 6, rgba(C.text, 0.7), 'left', 'top', 11, 600);
    }
    // legend
    const lx = bi.x + bi.w - 6;
    let ly = bi.y + 6;
    if (showCurve) {
      label(ctx, `N(μ, σ²/n)  σ/√n = ${sd.toFixed(3)}`, lx - 22, ly + 5, C.accent2, 'right', 'middle', 10);
      ctx.save(); ctx.strokeStyle = C.accent2; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(lx - 16, ly + 5); ctx.lineTo(lx, ly + 5); ctx.stroke(); ctx.restore();
      ly += 16;
    }
    label(ctx, 'histogram of x̄', lx - 22, ly + 5, C.accent, 'right', 'middle', 10);
    ctx.fillStyle = rgba(C.accent, 0.8); ctx.fillRect(lx - 16, ly + 1, 16, 8);
    if (zoom && (bins.xlo > dist.lo + 1e-9 || bins.xhi < dist.hi - 1e-9)) {
      label(ctx, 'zoomed to μ ± 4.2σ/√n', bi.x + 6, bi.y + bi.h - 6, rgba(C.muted, 0.7), 'left', 'bottom', 9);
    }
  }

  function drawFlights(ctx, L) {
    if (!flights.length) return;
    const { ti, bi } = L;
    const ys = topScale();
    const txs = (v) => ti.x + ((v - dist.lo) / (dist.hi - dist.lo)) * ti.w;
    const tys = (d) => ti.y + ti.h - (d / ys) * ti.h;
    const { sx, sy } = bottomMaps(bi);
    ctx.save();
    for (const f of flights) {
      const b = binOf(f.mean);
      const d = b >= 0 && b < bins.nb && N ? counts[b] / (N * bins.bw) : 0;
      const tx = clamp(sx(f.mean), bi.x, bi.x + bi.w), ty = Math.max(bi.y, sy(d) - 3);
      const mStart = 0.42 * f.D, mEnd = 0.88 * f.D;
      const mk = clamp((f.t - mStart) / (mEnd - mStart), 0, 1);
      const e = ease(mk);
      // mean marker on the source axis while samples are gathered
      if (f.t < mEnd) {
        const xm = txs(clamp(f.mean, dist.lo, dist.hi));
        const a = clamp(f.t / (0.3 * f.D), 0, 1) * (1 - e);
        ctx.fillStyle = rgba(C.accent, a);
        ctx.beginPath(); ctx.moveTo(xm, ti.y + ti.h + 1); ctx.lineTo(xm - 5, ti.y + ti.h + 9); ctx.lineTo(xm + 5, ti.y + ti.h + 9); ctx.closePath(); ctx.fill();
        if (a > 0.2) label(ctx, 'x̄', xm, ti.y + ti.h + 11, rgba(C.accent, a), 'center', 'top', 10, 600);
      }
      for (const p of f.parts) {
        if (f.t < p.delay) continue;
        const appear = clamp((f.t - p.delay) / 0.18, 0, 1);
        const vx = dist.discrete ? p.v + p.jit : p.v;
        const x0 = txs(vx);
        const y0 = lerp(ti.y + ti.h - 2, tys(srcHeightAt(p.v)) + 2, p.f);
        const x = lerp(x0, tx, e);
        const y = lerp(y0, ty, e) - Math.sin(e * Math.PI) * 24;
        const r = lerp(2.6, 1.4, e) * (0.4 + 0.6 * appear);
        ctx.globalAlpha = appear * (1 - 0.5 * e);
        ctx.fillStyle = mix(C.accent2, C.accent, e);
        ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 6;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (mk > 0.6) glowDot(ctx, tx, ty, 2 + 2.5 * (mk - 0.6) / 0.4, C.accent, 12);
    }
    ctx.restore();
  }

  function pointer(type, x, y) {
    if (!W) return;
    const { ti } = lay();
    const inTop = x >= ti.x - 4 && x <= ti.x + ti.w + 4 && y >= ti.y - 20 && y <= ti.y + ti.h + 6;
    if (type === 'down' && inTop) {
      if (distKey !== 'custom') { seedCustom(); distKey = 'custom'; distSel.value = 'custom'; }
      drawing = true; lastPaint = null; paint(x, y, ti);
      return true;
    }
    if (type === 'move') {
      hoverTop = inTop;
      env.canvas.style.cursor = inTop || drawing ? 'crosshair' : '';
      if (drawing) paint(x, y, ti);
    }
    if (type === 'up' || type === 'leave') { drawing = false; lastPaint = null; if (type === 'leave') hoverTop = false; }
    return false;
  }
  function paint(x, y, ti) {
    const i1 = clamp(Math.floor(((x - ti.x) / ti.w) * CLT_B), 0, CLT_B - 1);
    const f1 = clamp((ti.y + ti.h - y) / ti.h, 0, 1);
    const [i0, f0] = lastPaint || [i1, f1];
    const lo = Math.min(i0, i1), hi = Math.max(i0, i1);
    for (let i = lo; i <= hi; i++) {
      const t = hi === lo ? 1 : (i - i0) / (i1 - i0);
      custom[i] = lerp(f0, f1, clamp(t, 0, 1));
    }
    // soft brush so single clicks leave a visible bump
    for (let d = 1; d <= 3; d++) for (const j of [lo - d, hi + d]) if (j >= 0 && j < CLT_B) custom[j] = lerp(custom[j], f1, 0.45 / d);
    lastPaint = [i1, f1];
    rebuild();
  }

  resetMeans();
  return {
    resize() {},
    frame,
    reset: resetMeans,
    pointer,
    destroy() { env.canvas.style.cursor = ''; },
  };
}

// ================================================================= 2. Monte Carlo π
const PI_SE = 4 * Math.sqrt((Math.PI / 4) * (1 - Math.PI / 4)); // ≈ 1.642 — SD of the estimator × √N
function createPi(ctl, env) {
  let mode = 'quarter', speed = 50, show2 = true;
  let N = 0, inside = 0, hist = [], nextRec = 1;
  const BUF = 60000;
  const bx = new Float32Array(BUF), by = new Float32Array(BUF);
  const layer = makeLayer();
  let W = 0, H = 0, sq = null, plot = null;

  const s1 = section(ctl, 'Setup');
  select(s1, { label: 'Target shape', options: [['quarter', 'Quarter circle in unit square'], ['full', 'Full circle in square']], value: mode, onChange: (v) => { mode = v; reset(); } });
  const sp = speedSlider(s1, { label: 'Darts per frame', values: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], value: speed, onInput: (v) => { speed = v; } });
  speed = sp.value;
  checkbox(s1, { label: 'Show ±2σ band too', checked: show2, onChange: (v) => { show2 = v; } });
  s1.append(h('p', { class: 'hint' }, 'The fraction of darts inside the circle → π/4, so π ≈ 4 · inside / N. The error shrinks like 1.64/√N: every extra digit costs 100× more darts.'));
  const s2 = section(ctl, 'Estimate');
  const ro = readout(s2, ['Darts N', 'Inside', 'π estimate', 'Error', 'Error / σ', 'Correct decimals']);

  function reset() {
    N = 0; inside = 0; hist = []; nextRec = 1; layer.clear();
    updRO(0, true);
  }
  const updRO = throttled(() => {
    const est = N ? (4 * inside) / N : NaN;
    const err = est - Math.PI;
    ro.set('Darts N', fmtInt(N));
    ro.set('Inside', N ? `${fmtInt(inside)} (${((100 * inside) / N).toFixed(2)}%)` : '—');
    ro.set('π estimate', N ? est.toFixed(6) : '—');
    ro.set('Error', N ? fmtSigned(err, 6) : '—');
    ro.set('Error / σ', N ? fmtSigned(err / (PI_SE / Math.sqrt(N)), 2) + 'σ' : '—');
    ro.set('Correct decimals', N ? String(Math.max(0, Math.floor(-Math.log10(Math.abs(err) + 1e-15)))) : '—');
  });

  function lay(w, hh) {
    const pad = 14;
    if (w >= hh * 1.15) {
      const s = Math.floor(Math.min(hh - 2 * pad - 30, (w - 3 * pad) * 0.5));
      const sqr = { x: pad + 8, y: Math.round((hh - s) / 2) + 8, s };
      const px = sqr.x + s + pad + 8;
      return { sq: sqr, plot: { x: px, y: pad, w: w - px - pad, h: hh - 2 * pad } };
    }
    const s = Math.floor(Math.min(w - 2 * pad - 16, (hh - 3 * pad) * 0.56 - 24));
    const sqr = { x: Math.round((w - s) / 2), y: pad + 24, s };
    const py = sqr.y + s + pad;
    return { sq: sqr, plot: { x: pad, y: py, w: w - 2 * pad, h: hh - py - pad } };
  }
  function toPx(u, v) {
    const s = sq.s;
    return mode === 'quarter' ? [u * s, (1 - v) * s] : [((u + 1) / 2) * s, ((1 - v) / 2) * s];
  }
  function drawPoints(from, to) { // indices into logical sequence [from, to)
    const ctx = layer.ctx, ps = Math.max(1.2, sq.s / 320);
    for (const inFlag of [true, false]) {
      ctx.beginPath();
      for (let k = from; k < to; k++) {
        const j = k % BUF;
        const u = bx[j], v = by[j];
        if ((u * u + v * v <= 1) !== inFlag) continue;
        const [x, y] = toPx(u, v);
        ctx.rect(x - ps / 2, y - ps / 2, ps, ps);
      }
      ctx.fillStyle = inFlag ? rgba(C.accent, 0.8) : rgba(C.accent2, 0.6);
      ctx.fill();
    }
  }
  function resize(w, hh, dpr) {
    W = w; H = hh;
    ({ sq, plot } = lay(w, hh));
    layer.resize(sq.s, sq.s, dpr);
    drawPoints(Math.max(0, N - BUF), N);
  }

  function frame(dt, ctx, w, hh, running) {
    if (w !== W || hh !== H || !sq) resize(w, hh, env.size.dpr);
    if (running) {
      const start = N;
      for (let i = 0; i < speed; i++) {
        const u = mode === 'quarter' ? Math.random() : Math.random() * 2 - 1;
        const v = mode === 'quarter' ? Math.random() : Math.random() * 2 - 1;
        const j = N % BUF; bx[j] = u; by[j] = v;
        if (u * u + v * v <= 1) inside++;
        N++;
        if (N >= nextRec) { hist.push([N, (4 * inside) / N]); nextRec = Math.max(N + 1, Math.ceil(N * 1.015)); }
      }
      drawPoints(start, N);
    }
    updRO(dt);
    drawSquare(ctx, running);
    drawPlot(ctx);
  }

  function drawSquare(ctx, running) {
    const { x, y, s } = sq;
    ctx.save();
    ctx.fillStyle = 'rgba(18,23,34,0.8)';
    ctx.fillRect(x, y, s, s);
    ctx.drawImage(layer.cv, x, y, s, s);
    ctx.strokeStyle = '#3a4660'; ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
    ctx.strokeStyle = rgba(C.text, 0.85); ctx.lineWidth = 1.5;
    ctx.shadowColor = rgba(C.accent, 0.8); ctx.shadowBlur = 8;
    ctx.beginPath();
    if (mode === 'quarter') ctx.arc(x, y + s, s, -Math.PI / 2, 0);
    else ctx.arc(x + s / 2, y + s / 2, s / 2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    // newest darts
    if (running && N) {
      const k = Math.min(6, N);
      for (let i = N - k; i < N; i++) {
        const j = i % BUF; const [px, py] = toPx(bx[j], by[j]);
        glowDot(ctx, x + px, y + py, 2.6, bx[j] * bx[j] + by[j] * by[j] <= 1 ? C.accent : C.accent2, 12);
      }
    }
    label(ctx, mode === 'quarter' ? 'x² + y² ≤ 1   (x, y) ∈ [0,1]²' : 'x² + y² ≤ 1   (x, y) ∈ [−1,1]²', x, y - 8, C.muted, 'left', 'bottom', 10);
    // legend
    const ly = y + s + 14;
    if (ly < H - 4) {
      ctx.fillStyle = C.accent; ctx.fillRect(x, ly - 4, 8, 8);
      label(ctx, 'inside', x + 13, ly, C.muted, 'left', 'middle', 10);
      ctx.fillStyle = C.accent2; ctx.fillRect(x + 72, ly - 4, 8, 8);
      label(ctx, 'outside', x + 85, ly, C.muted, 'left', 'middle', 10);
    }
  }

  function drawPlot(ctx) {
    const r = plot;
    const est = N ? (4 * inside) / N : NaN;
    card(ctx, r, 'Convergence · estimate vs log₁₀ N', N ? `N = ${fmtInt(N)}` : '');
    // headline
    label(ctx, N ? `π ≈ ${est.toFixed(6)}` : 'π ≈ …', r.x + 14, r.y + 44, C.accent, 'left', 'middle', clamp(r.w / 16, 16, 30), 600);
    if (N) {
      const err = est - Math.PI;
      label(ctx, `error ${fmtSigned(err, 6)}   ·   1σ = ${(PI_SE / Math.sqrt(N)).toFixed(6)}`, r.x + 14, r.y + 44 + clamp(r.w / 22, 14, 22), Math.abs(err) < PI_SE / Math.sqrt(N) ? C.good : C.muted, 'left', 'middle', 10);
    }
    const pi = inset(r, 50, 92, 18, 26);
    const xmax = Math.max(3, Math.log10(Math.max(1, N)) * 1.06 + 0.1);
    const yr = 0.7;
    const sx = (lx) => pi.x + (lx / xmax) * pi.w;
    const sy = (v) => pi.y + pi.h / 2 - ((v - Math.PI) / yr) * (pi.h / 2);
    axes(ctx, pi, sx, sy, ticks(0, xmax, 6), ticks(Math.PI - yr, Math.PI + yr, 5), { xf: (v) => (v === Math.round(v) ? (v <= 3 ? String(10 ** v) : `1e${v}`) : v.toFixed(1)), yLabel: 'estimate', xLabel: 'log₁₀ N' });
    ctx.save();
    ctx.beginPath(); ctx.rect(pi.x, pi.y, pi.w, pi.h); ctx.clip();
    const band = (k, alpha) => {
      ctx.beginPath();
      const steps = 120;
      for (let i = 0; i <= steps; i++) { const lx = (xmax * i) / steps; ctx.lineTo(sx(lx), sy(Math.PI + (k * PI_SE) / Math.sqrt(10 ** lx))); }
      for (let i = steps; i >= 0; i--) { const lx = (xmax * i) / steps; ctx.lineTo(sx(lx), sy(Math.PI - (k * PI_SE) / Math.sqrt(10 ** lx))); }
      ctx.closePath(); ctx.fillStyle = rgba(C.accent2, alpha); ctx.fill();
    };
    if (show2) band(2, 0.08);
    band(1, 0.16);
    dashedH(ctx, sy(Math.PI), pi.x, pi.x + pi.w, rgba(C.accent2, 0.9), [5, 4]);
    if (hist.length) {
      ctx.beginPath();
      hist.forEach(([n, e], i) => { const X = sx(Math.log10(n)), Y = sy(e); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); });
      ctx.lineTo(sx(Math.log10(N)), sy(est));
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1.6; ctx.lineJoin = 'round';
      ctx.shadowColor = rgba(C.accent, 0.6); ctx.shadowBlur = 6;
      ctx.stroke();
    }
    ctx.restore();
    if (N) glowDot(ctx, sx(Math.log10(N)), clamp(sy(est), pi.y, pi.y + pi.h), 3.5, C.accent, 14);
    label(ctx, 'π', pi.x + pi.w - 4, sy(Math.PI) - 8, C.accent2, 'right', 'middle', 12, 600);
    label(ctx, show2 ? '±1σ, ±2σ  (σ = 1.64/√N)' : '±1σ  (σ = 1.64/√N)', pi.x + 6, pi.y + pi.h - 6, rgba(C.accent2, 0.8), 'left', 'bottom', 9);
  }

  reset();
  return { resize, frame, reset };
}

// ================================================================= 3. Random walks
const WALK_COLORS = ['#5ec8e5', '#7fc6d6', '#a2c3b4', '#c9bf8f', '#e3ba68', '#f5b544'];
function createWalk(ctl, env) {
  let mode = '1d';
  let inner = null;
  const s0 = section(ctl, 'Walk type');
  select(s0, { options: [['1d', 'Many 1D walkers (±1 steps)'], ['2d', 'One 2D Brownian path']], value: mode, onChange: (v) => { mode = v; build(); } });
  const box = h('div');
  ctl.append(box);
  let size = null;
  function build() {
    box.replaceChildren();
    inner = mode === '1d' ? walk1D(box, env) : walk2D(box, env);
    if (size) inner.resize(...size);
  }
  build();
  return {
    resize(w, hh, dpr) { size = [w, hh, dpr]; inner.resize(w, hh, dpr); },
    frame(...a) { inner.frame(...a); },
    reset() { inner.reset(); },
  };
}

function walk1D(ctl, env) {
  let Wk = 150, T = 600, speed = 2, auto = true;
  let pos, hist, t = 0, acc = 0, hold = 0;
  const layer = makeLayer();
  let W = 0, H = 0, L = null, lastDpr = 1;

  const s1 = section(ctl, 'Walkers');
  slider(s1, { label: 'Number of walkers', min: 10, max: 400, step: 10, value: Wk, onInput: (v) => { Wk = v; reset(); } });
  slider(s1, { label: 'Steps per run', min: 100, max: 2000, step: 50, value: T, onInput: (v) => { T = v; reset(); } });
  const sp = speedSlider(s1, { label: 'Steps per frame', values: [0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20], value: speed, onInput: (v) => { speed = v; } });
  speed = sp.value;
  checkbox(s1, { label: 'Start a new run when finished', checked: auto, onChange: (v) => { auto = v; } });
  s1.append(h('p', { class: 'hint' }, 'Each walker flips a coin every step. Nobody drifts on average, but the spread grows like √t — the envelope — and the endpoints pile up into a Gaussian.'));
  const s2 = section(ctl, 'Spread');
  const ro = readout(s2, ['Step t', 'Mean position', 'RMS position', '√t (theory)', 'Within ±√t', 'Within ±2√t']);

  function reset() {
    pos = new Int32Array(Wk);
    hist = new Int16Array(Wk * (T + 1));
    t = 0; acc = 0; hold = 0;
    layer.clear();
    updRO(0, true);
  }
  const updRO = throttled(() => {
    let s = 0, s2 = 0, in1 = 0, in2 = 0;
    const r = Math.sqrt(t);
    for (let i = 0; i < Wk; i++) { const p = pos[i]; s += p; s2 += p * p; if (Math.abs(p) <= r) in1++; if (Math.abs(p) <= 2 * r) in2++; }
    ro.set('Step t', `${fmtInt(t)} / ${fmtInt(T)}`);
    ro.set('Mean position', fmtSigned(s / Wk, 2));
    ro.set('RMS position', fmtN(Math.sqrt(s2 / Wk), 2));
    ro.set('√t (theory)', fmtN(r, 2));
    ro.set('Within ±√t', t ? `${((100 * in1) / Wk).toFixed(0)}%  (≈68%)` : '—');
    ro.set('Within ±2√t', t ? `${((100 * in2) / Wk).toFixed(0)}%  (≈95%)` : '—');
  });

  function lay(w, hh) {
    const pad = 14;
    const histW = clamp(w * 0.2, 80, 190);
    const main = { x: pad, y: pad, w: w - 3 * pad - histW, h: hh - 2 * pad };
    const side = { x: main.x + main.w + pad, y: pad, w: histW, h: hh - 2 * pad };
    const pi = inset(main, 46, 32, 14, 24);
    return { main, side, pi, si: { x: side.x + 10, y: pi.y, w: side.w - 20, h: pi.h } };
  }
  const yr = () => 3.3 * Math.sqrt(T);
  const X = (tt) => (tt / T) * L.pi.w;
  const Y = (p) => L.pi.h / 2 - (p / yr()) * (L.pi.h / 2);

  function drawSteps(t0, t1) {
    if (t1 <= t0) return;
    const ctx = layer.ctx, stride = T + 1;
    const alpha = clamp(10 / Wk + 0.12, 0.12, 0.6);
    ctx.lineWidth = Wk > 200 ? 0.8 : 1;
    ctx.lineJoin = 'round';
    for (let b = 0; b < WALK_COLORS.length; b++) {
      ctx.beginPath();
      for (let i = b; i < Wk; i += WALK_COLORS.length) {
        const base = i * stride;
        ctx.moveTo(X(t0), Y(hist[base + t0]));
        for (let k = t0 + 1; k <= t1; k++) ctx.lineTo(X(k), Y(hist[base + k]));
      }
      ctx.strokeStyle = rgba(WALK_COLORS[b], alpha);
      ctx.stroke();
    }
  }
  function resize(w, hh, dpr) {
    W = w; H = hh; lastDpr = dpr;
    L = lay(w, hh);
    layer.resize(L.pi.w, L.pi.h, dpr);
    drawSteps(0, t);
  }
  function step() {
    t++;
    const stride = T + 1;
    for (let i = 0; i < Wk; i++) { pos[i] += Math.random() < 0.5 ? -1 : 1; hist[i * stride + t] = pos[i]; }
  }

  function frame(dt, ctx, w, hh, running) {
    if (!L || w !== W || hh !== H) resize(w, hh, env.size.dpr || lastDpr);
    if (running) {
      if (t >= T) {
        hold += dt;
        if (auto && hold > 2.5) reset();
      } else {
        const t0 = t;
        acc += speed;
        while (acc >= 1 && t < T) { acc -= 1; step(); }
        drawSteps(t0, t);
      }
    }
    updRO(dt);
    const { main, side, pi, si } = L;
    card(ctx, main, 'Position vs time', t >= T ? 'run complete' : `t = ${fmtInt(t)}`, t >= T ? C.good : C.muted);
    const sxT = (tt) => pi.x + X(tt);
    const syP = (p) => pi.y + Y(p);
    axes(ctx, pi, sxT, syP, ticks(0, T, 6), ticks(-yr(), yr(), 6), { xLabel: 'time t', yLabel: 'position' });
    ctx.drawImage(layer.cv, pi.x, pi.y, pi.w, pi.h);
    // envelopes
    ctx.save();
    ctx.beginPath(); ctx.rect(pi.x, pi.y, pi.w, pi.h); ctx.clip();
    for (const [k, a] of [[1, 0.95], [2, 0.5]]) {
      ctx.strokeStyle = rgba(C.text, a); ctx.lineWidth = k === 1 ? 1.5 : 1;
      ctx.setLineDash(k === 1 ? [] : [5, 5]);
      for (const sgn of [1, -1]) {
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) { const tt = (T * i) / 120; const yy = syP(sgn * k * Math.sqrt(tt)); if (i === 0) ctx.moveTo(sxT(tt), yy); else ctx.lineTo(sxT(tt), yy); }
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    dashedH(ctx, syP(0), pi.x, pi.x + pi.w, rgba(C.muted, 0.5), [2, 4]);
    ctx.restore();
    label(ctx, '+√t', pi.x + pi.w - 4, syP(Math.sqrt(T)) - 8, C.text, 'right', 'middle', 10);
    label(ctx, '+2√t', pi.x + pi.w - 4, syP(2 * Math.sqrt(T)) - 8, rgba(C.text, 0.6), 'right', 'middle', 10);
    // walker heads
    if (t > 0 && t < T) {
      ctx.fillStyle = rgba(C.accent, 0.9);
      const x = sxT(t);
      for (let i = 0; i < Wk; i++) { ctx.fillRect(x - 1, syP(pos[i]) - 1, 2, 2); }
    }
    // histogram of current positions on the right edge
    card(ctx, side, 'Positions', '');
    if (t > 0) {
      const sd = Math.sqrt(t);
      const par = t % 2;
      const bw = Math.max(2, 2 * Math.round((6 * sd) / 26 / 2));
      const half = bw / 2;
      const cnt = new Map();
      let mx = 1;
      for (let i = 0; i < Wk; i++) { const u = (pos[i] - par) / 2; const b = Math.floor(u / half); const c = (cnt.get(b) || 0) + 1; cnt.set(b, c); if (c > mx) mx = c; }
      // expected count per bin
      const expPeak = Wk * bw * npdf(0, 0, sd);
      const scale = si.w / Math.max(mx, expPeak) * 0.95;
      ctx.save(); ctx.beginPath(); ctx.rect(si.x, si.y, si.w, si.h); ctx.clip();
      for (const [b, c] of cnt) {
        const lo = par + 2 * b * half - 1, hiV = lo + bw; // value interval covered
        const y0 = syP(hiV), y1 = syP(lo);
        const g = ctx.createLinearGradient(si.x, 0, si.x + si.w, 0);
        g.addColorStop(0, rgba(C.accent, 0.85)); g.addColorStop(1, rgba(C.accent, 0.35));
        ctx.fillStyle = g;
        ctx.fillRect(si.x, y0 + 0.5, c * scale, Math.max(1, y1 - y0 - 1));
      }
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const p = -yr() + (2 * yr() * i) / 100;
        const xx = si.x + Wk * bw * npdf(p, 0, sd) * scale;
        if (i === 0) ctx.moveTo(xx, syP(p)); else ctx.lineTo(xx, syP(p));
      }
      ctx.strokeStyle = C.accent2; ctx.lineWidth = 1.8; ctx.shadowColor = rgba(C.accent2, 0.6); ctx.shadowBlur = 8; ctx.stroke();
      ctx.restore();
      label(ctx, 'N(0, t)', si.x + si.w, si.y + 2, C.accent2, 'right', 'top', 10);
    }
  }

  reset();
  return { resize, frame, reset };
}

function walk2D(ctl, env) {
  const MAXP = 60000;
  let speed = 20, tail = 20000;
  const xs = new Float64Array(MAXP), ys = new Float64Array(MAXP);
  let steps = 0, cx = 0, cy = 0, maxR = 0, acc = 0;
  let view = null;

  const s1 = section(ctl, 'Brownian path');
  const sp = speedSlider(s1, { label: 'Steps per frame', values: [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], value: speed, onInput: (v) => { speed = v; } });
  speed = sp.value;
  slider(s1, { label: 'Visible tail', min: 1000, max: MAXP, step: 1000, value: tail, format: (v) => fmtInt(v) + ' steps', onInput: (v) => { tail = v; } });
  s1.append(h('p', { class: 'hint' }, 'Gaussian steps of unit variance. The dashed ring has radius √N — the root-mean-square distance from the start after N steps. Older path fades to cyan.'));
  const s2 = section(ctl, 'Distance');
  const ro = readout(s2, ['Steps N', 'Distance |r|', '√N (RMS theory)', '|r| / √N', 'Max distance']);

  function reset() {
    steps = 0; cx = 0; cy = 0; maxR = 0; acc = 0; xs[0] = 0; ys[0] = 0; view = null;
    updRO(0, true);
  }
  const updRO = throttled(() => {
    const r = Math.hypot(cx, cy);
    ro.set('Steps N', fmtInt(steps));
    ro.set('Distance |r|', fmtN(r, 2));
    ro.set('√N (RMS theory)', fmtN(Math.sqrt(steps), 2));
    ro.set('|r| / √N', steps ? fmtN(r / Math.sqrt(steps), 3) : '—');
    ro.set('Max distance', fmtN(maxR, 2));
  });

  function frame(dt, ctx, w, hh, running) {
    if (running) {
      acc += speed;
      while (acc >= 1) {
        acc -= 1;
        cx += gauss() * Math.SQRT1_2; cy += gauss() * Math.SQRT1_2;
        steps++;
        const j = steps % MAXP; xs[j] = cx; ys[j] = cy;
        maxR = Math.max(maxR, Math.hypot(cx, cy));
      }
    }
    updRO(dt);
    const pad = 14;
    const r = { x: pad, y: pad, w: w - 2 * pad, h: hh - 2 * pad };
    card(ctx, r, '2D random walk', `N = ${fmtInt(steps)}`);
    const pr = inset(r, 20, 34, 20, 20);
    const count = Math.min(steps + 1, tail);
    const first = steps + 1 - count;
    // bbox of visible window (plus origin)
    let x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    for (let k = first; k <= steps; k++) { const j = k % MAXP; const a = xs[j], b = ys[j]; if (a < x0) x0 = a; if (a > x1) x1 = a; if (b < y0) y0 = b; if (b > y1) y1 = b; }
    const span = Math.max(x1 - x0, y1 - y0, 12) * 1.15;
    const target = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, span };
    if (!view) view = { ...target };
    const k = Math.min(1, dt * 3);
    view.cx += (target.cx - view.cx) * k; view.cy += (target.cy - view.cy) * k;
    view.span += (Math.max(target.span, Math.max(x1 - x0, y1 - y0) * 1.02) - view.span) * k;
    const sc = Math.min(pr.w, pr.h) / view.span;
    const ox = pr.x + pr.w / 2, oy = pr.y + pr.h / 2;
    const SX = (v) => ox + (v - view.cx) * sc, SY = (v) => oy - (v - view.cy) * sc;

    ctx.save();
    ctx.beginPath(); ctx.rect(r.x + 1, r.y + 26, r.w - 2, r.h - 27); ctx.clip();
    // faint grid in walk units
    const gs = niceStep(view.span, 8);
    ctx.strokeStyle = rgba(C.line, 0.55); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = Math.floor((view.cx - view.span) / gs) * gs; g <= view.cx + view.span; g += gs) { const X = Math.round(SX(g)) + 0.5; ctx.moveTo(X, r.y); ctx.lineTo(X, r.y + r.h); }
    for (let g = Math.floor((view.cy - view.span) / gs) * gs; g <= view.cy + view.span; g += gs) { const Y = Math.round(SY(g)) + 0.5; ctx.moveTo(r.x, Y); ctx.lineTo(r.x + r.w, Y); }
    ctx.stroke();
    // RMS ring
    if (steps > 0) {
      ctx.setLineDash([5, 5]); ctx.strokeStyle = rgba(C.text, 0.35);
      ctx.beginPath(); ctx.arc(SX(0), SY(0), Math.sqrt(steps) * sc, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
      label(ctx, '√N', SX(0) + Math.sqrt(steps) * sc * 0.707 + 4, SY(0) - Math.sqrt(steps) * sc * 0.707 - 4, rgba(C.text, 0.55), 'left', 'bottom', 10);
    }
    // path in age chunks
    const CH = 48;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    for (let c = 0; c < CH; c++) {
      const a = first + Math.floor((count * c) / CH), b = first + Math.floor((count * (c + 1)) / CH);
      if (b <= a) continue;
      const age = (c + 1) / CH;
      ctx.beginPath();
      let j = a % MAXP;
      ctx.moveTo(SX(xs[j]), SY(ys[j]));
      for (let q = a + 1; q <= Math.min(b, steps); q++) { j = q % MAXP; ctx.lineTo(SX(xs[j]), SY(ys[j])); }
      ctx.strokeStyle = mix(C.accent2, C.accent, age ** 1.5, 0.06 + 0.9 * age ** 2);
      ctx.lineWidth = 0.7 + 1.1 * age;
      ctx.stroke();
    }
    // origin + head
    const Ox = SX(0), Oy = SY(0);
    ctx.strokeStyle = C.good; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(Ox - 6, Oy); ctx.lineTo(Ox + 6, Oy); ctx.moveTo(Ox, Oy - 6); ctx.lineTo(Ox, Oy + 6); ctx.stroke();
    label(ctx, 'start', Ox + 8, Oy + 8, C.good, 'left', 'top', 10);
    ctx.restore();
    glowDot(ctx, SX(cx), SY(cy), 4, C.accent, 16);
    label(ctx, `grid = ${gs} units`, r.x + 12, r.y + r.h - 8, rgba(C.muted, 0.8), 'left', 'bottom', 9);
  }

  reset();
  return { resize() {}, frame, reset };
}

// ================================================================= 4. Gambler's ruin
function ruinProb(k, N, p) {
  if (Math.abs(p - 0.5) < 1e-9) return 1 - k / N;
  const r = (1 - p) / p;
  return (r ** k - r ** N) / (1 - r ** N);
}
function expDuration(k, N, p) {
  if (Math.abs(p - 0.5) < 1e-9) return k * (N - k);
  const q = 1 - p, r = q / p;
  return k / (q - p) - (N / (q - p)) * ((1 - r ** k) / (1 - r ** N));
}

function createRuin(ctl, env) {
  let k = 10, Nt = 20, p = 0.49, Wk = 50, speed = 2;
  let pos, tt, jit, acc = 0;
  let stats = new Map(); // k -> {g, r, dur}
  let flashes = [];
  const layer = makeLayer();
  let W = 0, H = 0, L = null, Tmax = 100;

  const s1 = section(ctl, 'The game');
  const kS = slider(s1, { label: 'Starting bankroll k', min: 1, max: Nt - 1, value: k, format: (v) => `$${v}`, onInput: (v) => { k = v; respawn(); } });
  slider(s1, { label: 'Goal N', min: 4, max: 100, value: Nt, format: (v) => `$${v}`, onInput: (v) => {
    Nt = v; kS.input.max = Nt - 1; if (k > Nt - 1) { k = Nt - 1; } kS.set(k); resetStats();
  } });
  slider(s1, { label: 'Win probability p', min: 0.3, max: 0.7, step: 0.01, value: p, format: (v) => v.toFixed(2), onInput: (v) => { p = v; resetStats(); } });
  s1.append(h('p', { class: 'hint' }, 'Bet $1 per round, win with probability p, stop at $0 (ruin) or $N. Even a tiny house edge (p = 0.49) makes ruin very likely when the goal is far away. Move k to add more points to the ruin curve.'));
  const s2 = section(ctl, 'Simulation');
  slider(s2, { label: 'Simultaneous gamblers', min: 1, max: 200, value: Wk, onInput: (v) => { Wk = v; respawn(); } });
  const sp = speedSlider(s2, { label: 'Rounds per frame', values: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200], value: speed, onInput: (v) => { speed = v; } });
  speed = sp.value;
  const s3 = section(ctl, 'Outcomes at k');
  const ro = readout(s3, ['Games finished', 'Ruined', 'P(ruin) empirical', 'P(ruin) theory', 'Mean duration', 'E[duration]']);

  function respawn() {
    pos = new Int32Array(Wk).fill(k);
    tt = new Int32Array(Wk);
    jit = Float32Array.from({ length: Wk }, () => Math.random() - 0.5);
    acc = 0; flashes = [];
    Tmax = niceCeil(Math.max(20, 2.2 * expDuration(k, Nt, p)));
    layer.clear();
    updRO(0, true);
  }
  function resetStats() { stats = new Map(); respawn(); }
  function niceCeil(v) { const s = niceStep(v, 4); return Math.ceil(v / s) * s; }
  const cur = () => stats.get(k) || { g: 0, r: 0, dur: 0 };
  const updRO = throttled(() => {
    const s = cur();
    ro.set('Games finished', fmtInt(s.g));
    ro.set('Ruined', s.g ? `${fmtInt(s.r)}` : '—');
    ro.set('P(ruin) empirical', s.g ? `${(s.r / s.g).toFixed(3)} ± ${(1.96 * Math.sqrt(Math.max(1e-9, (s.r / s.g) * (1 - s.r / s.g)) / s.g)).toFixed(3)}` : '—');
    ro.set('P(ruin) theory', ruinProb(k, Nt, p).toFixed(3));
    ro.set('Mean duration', s.g ? fmtN(s.dur / s.g, 1) : '—');
    ro.set('E[duration]', fmtN(expDuration(k, Nt, p), 1));
  });

  function lay(w, hh) {
    const pad = 14;
    if (w >= 640) {
      const colW = clamp(w * 0.3, 220, 340);
      const main = { x: pad, y: pad, w: w - 3 * pad - colW, h: hh - 2 * pad };
      const tallyH = 118;
      const tally = { x: main.x + main.w + pad, y: pad, w: colW, h: tallyH };
      const curve = { x: tally.x, y: pad + tallyH + pad, w: colW, h: hh - 3 * pad - tallyH };
      return { main, tally, curve, mi: inset(main, 50, 32, 16, 24), ci: inset(curve, 40, 32, 14, 24) };
    }
    const mh = Math.round((hh - 3 * pad) * 0.6);
    const main = { x: pad, y: pad, w: w - 2 * pad, h: mh };
    const curve = { x: pad, y: pad * 2 + mh, w: w - 2 * pad, h: hh - 3 * pad - mh };
    return { main, tally: null, curve, mi: inset(main, 50, 32, 16, 24), ci: inset(curve, 40, 32, 14, 24) };
  }
  const PX = (t) => (t / Tmax) * L.mi.w;
  const PY = (b, i) => L.mi.h - (b / Nt) * L.mi.h + jit[i] * Math.min(3, (L.mi.h / Nt) * 0.6);

  function resize(w, hh, dpr) {
    W = w; H = hh; L = lay(w, hh);
    layer.resize(L.mi.w, L.mi.h, dpr);
  }

  function frame(dt, ctx, w, hh, running) {
    if (!L || w !== W || hh !== H) resize(w, hh, env.size.dpr);
    if (running) {
      const lc = layer.ctx;
      // fade old trails
      lc.save(); lc.globalCompositeOperation = 'destination-out';
      lc.fillStyle = `rgba(0,0,0,${clamp(0.035 * dt * 60, 0, 1)})`; lc.fillRect(0, 0, layer.w, layer.h); lc.restore();
      acc += speed;
      const nSteps = Math.floor(acc); acc -= nSteps;
      if (nSteps > 0) {
        lc.lineWidth = 1.1; lc.lineJoin = 'round';
        lc.strokeStyle = rgba(C.accent2, clamp(14 / Wk + 0.25, 0.25, 0.8));
        lc.beginPath();
        const ends = [];
        for (let i = 0; i < Wk; i++) {
          lc.moveTo(PX(tt[i]), PY(pos[i], i));
          for (let s = 0; s < nSteps; s++) {
            pos[i] += Math.random() < p ? 1 : -1; tt[i]++;
            lc.lineTo(PX(tt[i]), PY(pos[i], i));
            if (pos[i] <= 0 || pos[i] >= Nt) {
              const ruined = pos[i] <= 0;
              const st = stats.get(k) || { g: 0, r: 0, dur: 0 };
              st.g++; if (ruined) st.r++; st.dur += tt[i];
              stats.set(k, st);
              ends.push([PX(tt[i]), PY(pos[i], i), ruined]);
              if (flashes.length < 40) flashes.push({ x: PX(tt[i]), y: PY(pos[i], i), ruined, t: 0 });
              pos[i] = k; tt[i] = 0;
              lc.moveTo(PX(0), PY(k, i));
            }
          }
        }
        lc.stroke();
        for (const [x, y, ruined] of ends) { lc.fillStyle = ruined ? C.danger : C.good; lc.fillRect(x - 1.5, y - 1.5, 3, 3); }
      }
      for (const f of flashes) f.t += dt;
      flashes = flashes.filter((f) => f.t < 0.5);
    }
    updRO(dt);
    drawMain(ctx);
    if (L.tally) drawTally(ctx);
    drawCurve(ctx);
  }

  function drawMain(ctx) {
    const { main, mi } = L;
    card(ctx, main, `Bankroll paths · p = ${p.toFixed(2)}`, `start $${k} → goal $${Nt}`);
    const sx = (t) => mi.x + (t / Tmax) * mi.w;
    const sy = (b) => mi.y + mi.h - (b / Nt) * mi.h;
    axes(ctx, mi, sx, sy, ticks(0, Tmax, 6), ticks(0, Nt, 5), { xLabel: 'rounds played', yf: (v) => '$' + v });
    // absorbing barriers
    ctx.save();
    ctx.fillStyle = rgba(C.good, 0.08); ctx.fillRect(mi.x, sy(Nt) - 6, mi.w, 6);
    ctx.fillStyle = rgba(C.danger, 0.08); ctx.fillRect(mi.x, sy(0), mi.w, 6);
    ctx.strokeStyle = C.good; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(mi.x, sy(Nt)); ctx.lineTo(mi.x + mi.w, sy(Nt)); ctx.stroke();
    ctx.strokeStyle = C.danger; ctx.beginPath(); ctx.moveTo(mi.x, sy(0)); ctx.lineTo(mi.x + mi.w, sy(0)); ctx.stroke();
    ctx.restore();
    dashedH(ctx, sy(k), mi.x, mi.x + mi.w, rgba(C.accent, 0.6), [3, 5]);
    ctx.drawImage(layer.cv, mi.x, mi.y, mi.w, mi.h);
    // heads
    ctx.fillStyle = C.accent;
    for (let i = 0; i < Wk; i++) { const x = mi.x + PX(tt[i]); if (x <= mi.x + mi.w) ctx.fillRect(x - 1.5, mi.y + PY(pos[i], i) - 1.5, 3, 3); }
    for (const f of flashes) {
      const a = 1 - f.t / 0.5;
      ctx.strokeStyle = rgba(f.ruined ? C.danger : C.good, a); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(mi.x + f.x, mi.y + f.y, 2 + 10 * (1 - a), 0, Math.PI * 2); ctx.stroke();
    }
    label(ctx, 'goal', mi.x + mi.w - 4, sy(Nt) + 4, C.good, 'right', 'top', 10, 600);
    label(ctx, 'ruin', mi.x + mi.w - 4, sy(0) - 4, C.danger, 'right', 'bottom', 10, 600);
    label(ctx, `k = $${k}`, mi.x + mi.w - 4, sy(k) - 4, C.accent, 'right', 'bottom', 10);
    label(ctx, `E[duration] ≈ ${expDuration(k, Nt, p).toFixed(0)} rounds`, mi.x + 6, mi.y + 4, rgba(C.muted, 0.9), 'left', 'top', 10);
  }

  function drawTally(ctx) {
    const r = L.tally;
    const s = cur();
    card(ctx, r, 'Outcomes', `${fmtInt(s.g)} games`);
    const th = ruinProb(k, Nt, p);
    const rows = [['ruined', s.g ? s.r / s.g : 0, th, C.danger], ['reached goal', s.g ? 1 - s.r / s.g : 0, 1 - th, C.good]];
    const bx = r.x + 14, bw = r.w - 28;
    rows.forEach(([name, emp, theo, col], i) => {
      const y = r.y + 36 + i * 38;
      label(ctx, name, bx, y, C.muted, 'left', 'top', 10);
      label(ctx, s.g ? `${(emp * 100).toFixed(1)}%` : '—', bx + bw, y, col, 'right', 'top', 11, 600);
      const by = y + 16;
      ctx.fillStyle = rgba(C.line, 0.9); ctx.fillRect(bx, by, bw, 8);
      ctx.fillStyle = rgba(col, 0.85); ctx.fillRect(bx, by, bw * emp, 8);
      const tx = bx + bw * theo;
      ctx.fillStyle = C.text; ctx.fillRect(Math.round(tx) - 1, by - 3, 2, 14);
    });
    label(ctx, '│ = theory', r.x + r.w - 14, r.y + r.h - 8, rgba(C.muted, 0.8), 'right', 'bottom', 9);
  }

  function drawCurve(ctx) {
    const { curve, ci } = L;
    card(ctx, curve, 'P(ruin) vs starting k', L.tally ? '' : (() => { const s = cur(); return s.g ? `emp ${(s.r / s.g).toFixed(3)} · theory ${ruinProb(k, Nt, p).toFixed(3)}` : ''; })());
    const sx = (kk) => ci.x + (kk / Nt) * ci.w;
    const sy = (v) => ci.y + ci.h - v * ci.h;
    axes(ctx, ci, sx, sy, ticks(0, Nt, 5), ticks(0, 1, 4), { xf: (v) => '$' + v });
    // fair-game reference
    ctx.save();
    ctx.setLineDash([3, 4]); ctx.strokeStyle = rgba(C.muted, 0.5); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sx(0), sy(1)); ctx.lineTo(sx(Nt), sy(0)); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const kk = (Nt * i) / 200; const v = ruinProb(kk, Nt, p); if (i === 0) ctx.moveTo(sx(kk), sy(v)); else ctx.lineTo(sx(kk), sy(v)); }
    ctx.strokeStyle = C.accent2; ctx.lineWidth = 2; ctx.shadowColor = rgba(C.accent2, 0.6); ctx.shadowBlur = 8; ctx.stroke();
    ctx.restore();
    for (const [kk, s] of stats) {
      if (!s.g) continue;
      const e = s.r / s.g;
      const ci95 = 1.96 * Math.sqrt(Math.max(1e-9, e * (1 - e)) / s.g);
      const x = sx(kk);
      const isCur = kk === k;
      ctx.strokeStyle = rgba(C.accent, isCur ? 0.9 : 0.45); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, sy(Math.min(1, e + ci95))); ctx.lineTo(x, sy(Math.max(0, e - ci95))); ctx.stroke();
      if (isCur) glowDot(ctx, x, sy(e), 4, C.accent, 12);
      else { ctx.fillStyle = rgba(C.accent, 0.6); ctx.beginPath(); ctx.arc(x, sy(e), 2.5, 0, Math.PI * 2); ctx.fill(); }
    }
    const lx = ci.x + ci.w - 4;
    label(ctx, 'theory', lx, ci.y + 4, C.accent2, 'right', 'top', 10);
    label(ctx, 'empirical ± 95%', lx, ci.y + 18, C.accent, 'right', 'top', 10);
    label(ctx, 'fair game', lx, ci.y + 32, rgba(C.muted, 0.8), 'right', 'top', 10);
  }

  respawn();
  return { resize, frame, reset: resetStats };
}

// ================================================================= module
const EXPERIMENTS = {
  clt: { name: 'Central Limit Theorem', make: createCLT, blurb: 'Average n draws from any distribution and the averages form a bell curve with spread σ/√n.' },
  pi: { name: 'Monte Carlo π', make: createPi, blurb: 'Throw random darts at a square; the share landing in the circle estimates π.' },
  walk: { name: 'Random walks', make: createWalk, blurb: 'Coin-flip walkers spread like √t; a 2D walk traces a Brownian path.' },
  ruin: { name: "Gambler's ruin", make: createRuin, blurb: 'Bet $1 until broke or at the goal — compare the ruin rate with the exact formula.' },
};

export default {
  id: 'clt',
  title: 'Stats Lab',
  glyph: 'σ',
  tag: 'statistics',
  blurb: 'Watch chance turn into law: the central limit theorem, Monte Carlo π, random walks and the gambler’s ruin, sampled live.',
  mount(root) {
    const { stage, panel } = createLayout(root, {
      title: 'Stats Lab',
      desc: 'Four live experiments where randomness, repeated enough, turns into predictable shapes. Pick one and watch the samples pile up against the theory.',
    });
    let kind = 'clt', exp = null, paused = false;
    const sec = section(panel, 'Experiment');
    select(sec, { options: Object.entries(EXPERIMENTS).map(([k, v]) => [k, v.name]), value: kind, onChange: (v) => build(v) });
    const blurb = h('p', { class: 'hint' });
    sec.append(blurb);
    const [pauseBtn] = buttons(sec, [
      { label: 'Pause', onClick: () => { paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; pauseBtn.classList.toggle('primary', paused); } },
      { label: 'Reset', onClick: () => exp?.reset() },
    ]);
    const sub = h('div');
    panel.append(sub);

    const cv = createCanvas(stage, (w, hh) => exp?.resize(w, hh, cv.size.dpr));
    const env = { canvas: cv.canvas, get size() { return cv.size; } };

    function build(k) {
      exp?.destroy?.();
      sub.replaceChildren();
      kind = k;
      blurb.textContent = EXPERIMENTS[k].blurb;
      exp = EXPERIMENTS[k].make(sub, env);
      if (cv.size.w > 0) exp.resize(cv.size.w, cv.size.h, cv.size.dpr);
    }
    build(kind);

    const stop = loop((dt) => {
      const { ctx, size } = cv;
      ctx.clearRect(0, 0, size.w, size.h);
      if (size.w < 40 || size.h < 40) return;
      exp.frame(paused ? 0 : dt, ctx, size.w, size.h, !paused);
    });

    // pointer routing (CLT uses it to draw the source distribution)
    const pt = (e) => { const r = cv.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const onDown = (e) => { if (exp?.pointer?.('down', ...pt(e))) { cv.canvas.setPointerCapture?.(e.pointerId); e.preventDefault(); } };
    const onMove = (e) => exp?.pointer?.('move', ...pt(e));
    const onUp = (e) => exp?.pointer?.('up', ...pt(e));
    const onLeave = (e) => exp?.pointer?.('leave', ...pt(e));
    cv.canvas.style.touchAction = 'none';
    cv.canvas.addEventListener('pointerdown', onDown);
    cv.canvas.addEventListener('pointermove', onMove);
    cv.canvas.addEventListener('pointerup', onUp);
    cv.canvas.addEventListener('pointercancel', onUp);
    cv.canvas.addEventListener('pointerleave', onLeave);

    return () => {
      stop();
      cv.destroy();
      exp?.destroy?.();
      cv.canvas.removeEventListener('pointerdown', onDown);
      cv.canvas.removeEventListener('pointermove', onMove);
      cv.canvas.removeEventListener('pointerup', onUp);
      cv.canvas.removeEventListener('pointercancel', onUp);
      cv.canvas.removeEventListener('pointerleave', onLeave);
    };
  },
};
