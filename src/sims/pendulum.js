// Double pendulum & chaos: an ensemble of K double pendulums whose starting angles differ
// by i·ε. Exact Lagrangian equations of motion, fixed-step RK4. They start in lock-step and
// then fly apart — the spread grows roughly like e^{λt} until it saturates.
import { createLayout, section, slider, select, checkbox, buttons, readout, createCanvas, loop, h } from '../ui.js';

const TAU = Math.PI * 2;
const wrap = (a) => a - TAU * Math.floor((a + Math.PI) / TAU);
const H = 1 / 2000;          // integrator step (s of sim time)
const REC_DT = 1 / 60;       // trail / phase-space sample interval (sim s)
const MONO = "'IBM Plex Mono', ui-monospace, monospace";
const C = {
  bg: '#0b0e14', accent: '#f5b544', accent2: '#5ec8e5', muted: '#8a93a6', line: '#242c3b',
  good: '#7bd88f', danger: '#ef6b6b', text: '#e6e9ef',
};
// Ensemble gradient: amber → rose → violet → cyan → green.
const STOPS = ['#f5b544', '#ef6b6b', '#b98cf0', '#5ec8e5', '#7bd88f'].map((hex) => [
  parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16),
]);
function gradient(u) {
  const x = Math.min(0.9999, Math.max(0, u)) * (STOPS.length - 1);
  const k = Math.floor(x), f = x - k, a = STOPS[k], b = STOPS[k + 1];
  const c = a.map((v, j) => Math.round(v + (b[j] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---- physics -------------------------------------------------------------------------
// θ measured from the downward vertical. Standard Lagrangian result for two point masses
// on massless rods, plus optional viscous damping −c·ω on each joint.
function accel(P, t1, w1, t2, w2, out) {
  const { L1, L2, m1, m2, g, c } = P;
  const d = t1 - t2, sd = Math.sin(d), cd = Math.cos(d);
  const D = 2 * m1 + m2 - m2 * Math.cos(2 * d);
  out[0] = (-g * (2 * m1 + m2) * Math.sin(t1) - m2 * g * Math.sin(t1 - 2 * t2)
    - 2 * sd * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * cd)) / (L1 * D) - c * w1;
  out[1] = (2 * sd * (w1 * w1 * L1 * (m1 + m2) + g * (m1 + m2) * Math.cos(t1)
    + w2 * w2 * L2 * m2 * cd)) / (L2 * D) - c * w2;
}

const K4 = [0, 0];
function rk4(P, s, i, h) {
  const t1 = s.t1[i], w1 = s.w1[i], t2 = s.t2[i], w2 = s.w2[i];
  accel(P, t1, w1, t2, w2, K4); const a1 = K4[0], b1 = K4[1];
  const t1b = t1 + 0.5 * h * w1, w1b = w1 + 0.5 * h * a1, t2b = t2 + 0.5 * h * w2, w2b = w2 + 0.5 * h * b1;
  accel(P, t1b, w1b, t2b, w2b, K4); const a2 = K4[0], b2 = K4[1];
  const t1c = t1 + 0.5 * h * w1b, w1c = w1 + 0.5 * h * a2, t2c = t2 + 0.5 * h * w2b, w2c = w2 + 0.5 * h * b2;
  accel(P, t1c, w1c, t2c, w2c, K4); const a3 = K4[0], b3 = K4[1];
  const t1d = t1 + h * w1c, w1d = w1 + h * a3, t2d = t2 + h * w2c, w2d = w2 + h * b3;
  accel(P, t1d, w1d, t2d, w2d, K4); const a4 = K4[0], b4 = K4[1];
  s.t1[i] = t1 + (h / 6) * (w1 + 2 * w1b + 2 * w1c + w1d);
  s.w1[i] = w1 + (h / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
  s.t2[i] = t2 + (h / 6) * (w2 + 2 * w2b + 2 * w2c + w2d);
  s.w2[i] = w2 + (h / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
}

// Total mechanical energy, potential zeroed at the hanging-rest position (so E ≥ 0).
function energy(P, t1, w1, t2, w2) {
  const { L1, L2, m1, m2, g } = P;
  const T = 0.5 * m1 * L1 * L1 * w1 * w1
    + 0.5 * m2 * (L1 * L1 * w1 * w1 + L2 * L2 * w2 * w2 + 2 * L1 * L2 * w1 * w2 * Math.cos(t1 - t2));
  const V = (m1 + m2) * g * L1 * (1 - Math.cos(t1)) + m2 * g * L2 * (1 - Math.cos(t2));
  return T + V;
}

const PRESETS = {
  chaotic: { label: 'Chaotic', L1: 1, L2: 1, m1: 1, m2: 1, g: 9.81, c: 0, K: 40, epsExp: -6, a1: 1.9, a2: 2.4 },
  periodic: { label: 'Near-periodic small swing', L1: 1, L2: 1, m1: 1, m2: 1, g: 9.81, c: 0, K: 30, epsExp: -3, a1: 0.2, a2: 0.2 * Math.SQRT2 },
  inverted: { label: 'Upside-down', L1: 1, L2: 1, m1: 1, m2: 1, g: 9.81, c: 0, K: 50, epsExp: -5, a1: Math.PI - 0.001, a2: Math.PI + 0.002 },
  single: { label: 'Single pendulum', L1: 1.6, L2: 0.05, m1: 1, m2: 0.01, g: 9.81, c: 0, K: 20, epsExp: -2.5, a1: 1.5, a2: 1.5 },
};

const fmtExp = (v) => {
  if (!isFinite(v)) return '—';
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  if (e >= -2 && e < 4) return v.toFixed(Math.max(0, 3 - Math.max(0, e)));
  return (v / 10 ** e).toFixed(2) + 'e' + e;
};

export default {
  id: 'pendulum',
  title: 'Pendulum',
  glyph: '∞',
  tag: 'chaos',
  blurb: 'A swarm of double pendulums, started a billionth of a radian apart, march in lock-step — then scatter. Sensitive dependence, made visible.',

  mount(root) {
    const { stage, panel } = createLayout(root, {
      title: 'Double pendulum',
      desc: 'Two rods, two masses, exact equations of motion. Launch many copies whose angles differ by a tiny ε and watch the difference grow exponentially until they share nothing but their energy.',
    });

    // ---- state ------------------------------------------------------------------------
    const P = { L1: 1, L2: 1, m1: 1, m2: 1, g: 9.81, c: 0 };
    let base = { a1: 1.9, a2: 2.4 };
    let K = 40, epsExp = -6, trailSec = 2.5, speed = 1, inset = 'phase', showSpark = true;
    let playing = true, simT = 0, acc = 0, nextRec = 0, nextHist = 0, histDt = 1 / 20;
    let E0 = 0, dragging = 0;
    let S = null, colors = [];
    // trails: ring buffer of world-space tip positions, shared head for all pendulums
    let cap = 0, tx = null, ty = null, head = 0, tlen = 0;
    // histories (sim-time sampled): time, energy drift fraction of #1, log10 θ2-spread
    let histT = [], histE = [], histS = [];
    let lambda = NaN, satT = Infinity;

    const PH = 420; // phase-space accumulation canvas (drawn scaled into the inset)
    const phase = document.createElement('canvas');
    phase.width = phase.height = PH;
    const pctx = phase.getContext('2d');

    function allocTrails() {
      cap = Math.max(2, Math.ceil(trailSec / REC_DT) + 1);
      tx = new Float32Array(K * cap); ty = new Float32Array(K * cap);
      head = 0; tlen = 0;
    }

    function setAll(a1, a2, withEps) {
      const eps = withEps ? 10 ** epsExp : 0;
      for (let i = 0; i < K; i++) {
        S.t1[i] = a1 + i * eps; S.t2[i] = a2 + i * eps; S.w1[i] = 0; S.w2[i] = 0;
      }
    }

    function init() {
      S = { t1: new Float64Array(K), w1: new Float64Array(K), t2: new Float64Array(K), w2: new Float64Array(K) };
      colors = Array.from({ length: K }, (_, i) => gradient(K === 1 ? 0 : i / (K - 1)));
      setAll(base.a1, base.a2, true);
      simT = 0; acc = 0; nextRec = 0; nextHist = 0; histDt = 1 / 20;
      histT = []; histE = []; histS = []; lambda = NaN; satT = Infinity;
      E0 = energy(P, S.t1[0], S.w1[0], S.t2[0], S.w2[0]);
      allocTrails();
      pctx.clearRect(0, 0, PH, PH);
      record(); sampleHist();
      nextRec = REC_DT; nextHist = histDt;
    }

    function spread() {
      if (K < 2) return NaN;
      let s = 0; const r = S.t2[0];
      for (let i = 1; i < K; i++) { const d = wrap(S.t2[i] - r); s += d * d; }
      return Math.sqrt(s / (K - 1));
    }

    function record() {
      const { L1, L2 } = P;
      for (let i = 0; i < K; i++) {
        const x = L1 * Math.sin(S.t1[i]) + L2 * Math.sin(S.t2[i]);
        const y = L1 * Math.cos(S.t1[i]) + L2 * Math.cos(S.t2[i]);
        tx[i * cap + head] = x; ty[i * cap + head] = y;
      }
      head = (head + 1) % cap; tlen = Math.min(cap, tlen + 1);
      // phase-space dots
      pctx.globalAlpha = K > 20 ? 0.18 : 0.35;
      for (let i = 0; i < K; i++) {
        pctx.fillStyle = colors[i];
        const px = (wrap(S.t1[i]) + Math.PI) / TAU * PH, py = (Math.PI - wrap(S.t2[i])) / TAU * PH;
        pctx.fillRect(px - 0.8, py - 0.8, 1.6, 1.6);
      }
      pctx.globalAlpha = 1;
    }

    function sampleHist() {
      const E = energy(P, S.t1[0], S.w1[0], S.t2[0], S.w2[0]);
      const sp = spread();
      histT.push(simT);
      histE.push(E0 > 1e-12 ? (E - E0) / E0 : 0);
      histS.push(isFinite(sp) ? Math.log10(Math.max(sp, 1e-16)) : NaN);
      if (isFinite(sp) && sp > 0.1 && satT === Infinity) satT = simT;
      if (histT.length > 1600) {
        const keep = (_, j) => j % 2 === 0;
        histT = histT.filter(keep); histE = histE.filter(keep); histS = histS.filter(keep);
        histDt *= 2;
      }
      estimateLambda();
    }

    // Least-squares slope of ln(spread) over the growth window: from one decade above the
    // starting spread until the spread first exceeds 0.1 rad.
    function estimateLambda() {
      if (K < 2 || histS.length < 3) { lambda = NaN; return; }
      let s0 = NaN;
      for (const v of histS) if (isFinite(v) && v > -15.9) { s0 = v; break; }
      if (!isFinite(s0)) { lambda = NaN; return; }
      let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, tmin = Infinity, tmax = -Infinity;
      for (let j = 0; j < histT.length; j++) {
        const t = histT[j], y = histS[j];
        if (t > satT || !isFinite(y) || y < s0 + 1 || y > -1) continue;
        const yl = y * Math.LN10;
        n++; sx += t; sy += yl; sxx += t * t; sxy += t * yl;
        tmin = Math.min(tmin, t); tmax = Math.max(tmax, t);
      }
      if (n < 8 || tmax - tmin < 0.8) { lambda = NaN; return; }
      lambda = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    }

    function physicsChanged() {
      // Lengths / masses / g changed live: re-reference the energy to the new Hamiltonian.
      E0 = energy(P, S.t1[0], S.w1[0], S.t2[0], S.w2[0]);
      histE = histE.map(() => NaN);
    }

    // ---- controls ---------------------------------------------------------------------
    const secRun = section(panel, 'Run');
    const [btnPlay] = buttons(secRun, [
      { label: 'Pause', primary: true, onClick: () => setPlaying(!playing) },
      { label: 'Reset', onClick: () => { init(); setPlaying(true); } },
    ]);
    function setPlaying(p) { playing = p; btnPlay.textContent = p ? 'Pause' : 'Play'; }
    const presetSel = select(secRun, {
      label: 'Preset',
      options: Object.entries(PRESETS).map(([k, v]) => [k, v.label]),
      value: 'chaotic',
      onChange: (k) => applyPreset(k),
    });
    const sSpeed = slider(secRun, { label: 'Speed', min: 0.1, max: 3, step: 0.05, value: speed, format: (v) => v.toFixed(2) + '×', onInput: (v) => { speed = v; } });

    const secEns = section(panel, 'Ensemble');
    const sK = slider(secEns, { label: 'Pendulums', min: 1, max: 100, step: 1, value: K, onInput: (v) => { K = v; init(); } });
    const sEps = slider(secEns, { label: 'Offset ε (rad)', min: -9, max: -2, step: 0.25, value: epsExp, format: (v) => '10^' + v.toFixed(2).replace(/\.?0+$/, ''), onInput: (v) => { epsExp = v; init(); } });
    const sTrail = slider(secEns, { label: 'Trail length', min: 0, max: 8, step: 0.25, value: trailSec, format: (v) => v.toFixed(2) + ' s', onInput: (v) => { trailSec = v; allocTrails(); } });

    const secPhys = section(panel, 'Physics');
    const sL1 = slider(secPhys, { label: 'Length L₁', min: 0.05, max: 2, step: 0.05, value: P.L1, format: (v) => v.toFixed(2) + ' m', onInput: (v) => { P.L1 = v; physicsChanged(); } });
    const sL2 = slider(secPhys, { label: 'Length L₂', min: 0.05, max: 2, step: 0.05, value: P.L2, format: (v) => v.toFixed(2) + ' m', onInput: (v) => { P.L2 = v; physicsChanged(); } });
    const sM1 = slider(secPhys, { label: 'Mass m₁', min: 0.1, max: 5, step: 0.05, value: P.m1, format: (v) => v.toFixed(2) + ' kg', onInput: (v) => { P.m1 = v; physicsChanged(); } });
    const sM2 = slider(secPhys, { label: 'Mass m₂', min: 0.01, max: 5, step: 0.01, value: P.m2, format: (v) => v.toFixed(2) + ' kg', onInput: (v) => { P.m2 = v; physicsChanged(); } });
    const sG = slider(secPhys, { label: 'Gravity g', min: 0, max: 25, step: 0.01, value: P.g, format: (v) => v.toFixed(2) + ' m/s²', onInput: (v) => { P.g = v; physicsChanged(); } });
    const sC = slider(secPhys, { label: 'Damping', min: 0, max: 0.5, step: 0.005, value: P.c, format: (v) => v.toFixed(3) + ' /s', onInput: (v) => { P.c = v; physicsChanged(); } });

    const secView = section(panel, 'View');
    select(secView, {
      label: 'Inset plot',
      options: [['phase', 'Phase space θ₁ vs θ₂'], ['energy', 'Energy drift (#1)'], ['none', 'Hidden']],
      value: inset,
      onChange: (v) => { inset = v; },
    });
    checkbox(secView, { label: 'Divergence sparkline', checked: showSpark, onChange: (v) => { showSpark = v; } });

    const secOut = section(panel, 'Readout');
    const ro = readout(secOut, ['Time', 'Energy #1', 'Drift', 'θ₂ spread', 'λ est.']);
    secOut.append(h('p', { class: 'hint' },
      'Drag either bob of the lead pendulum to pose it; release to relaunch the ensemble. Space toggles play. ',
      'λ is fitted to the exponential phase of the spread (≈ largest Lyapunov exponent).'));

    function applyPreset(k) {
      const p = PRESETS[k];
      Object.assign(P, { L1: p.L1, L2: p.L2, m1: p.m1, m2: p.m2, g: p.g, c: p.c });
      K = p.K; epsExp = p.epsExp; base = { a1: p.a1, a2: p.a2 };
      sL1.set(P.L1); sL2.set(P.L2); sM1.set(P.m1); sM2.set(P.m2); sG.set(P.g); sC.set(P.c);
      sK.set(K); sEps.set(epsExp);
      presetSel.value = k;
      init(); setPlaying(true);
    }

    // ---- canvas + interaction --------------------------------------------------------
    const cv = createCanvas(stage);
    const { canvas, ctx, size } = cv;
    canvas.style.touchAction = 'none';

    function geom() {
      const w = size.w, hh = size.h;
      const scale = Math.min(w, hh) * 0.44 / Math.max(P.L1 + P.L2, 2);
      return { cx: w / 2, cy: hh * 0.5, scale };
    }
    function bobs(i) {
      const { cx, cy, scale } = geom();
      const x1 = cx + P.L1 * scale * Math.sin(S.t1[i]), y1 = cy + P.L1 * scale * Math.cos(S.t1[i]);
      const x2 = x1 + P.L2 * scale * Math.sin(S.t2[i]), y2 = y1 + P.L2 * scale * Math.cos(S.t2[i]);
      return { cx, cy, x1, y1, x2, y2 };
    }
    const bobR = (m) => 4 + 5 * Math.cbrt(m);
    function pick(e) {
      const r = canvas.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const b = bobs(0);
      const d1 = Math.hypot(px - b.x1, py - b.y1), d2 = Math.hypot(px - b.x2, py - b.y2);
      const hit = 24;
      let which = 0;
      if (d2 < hit && d2 <= d1) which = 2; else if (d1 < hit) which = 1;
      return { px, py, which, b };
    }
    function onDown(e) {
      const p = pick(e);
      if (!p.which) return;
      dragging = p.which;
      canvas.setPointerCapture?.(e.pointerId);
      canvas.style.cursor = 'grabbing';
      base = { a1: S.t1[0], a2: S.t2[0] };
      pose(p);
      e.preventDefault();
    }
    function pose(p) {
      const b = p.b;
      if (dragging === 1) base.a1 = Math.atan2(p.px - b.cx, p.py - b.cy);
      else base.a2 = Math.atan2(p.px - b.x1, p.py - b.y1);
      setAll(base.a1, base.a2, false);
      tlen = 0; head = 0;
    }
    function onMove(e) {
      const p = pick(e);
      if (dragging) { pose(p); return; }
      canvas.style.cursor = p.which ? 'grab' : 'default';
    }
    function onUp() {
      if (!dragging) return;
      dragging = 0;
      canvas.style.cursor = 'grab';
      init(); setPlaying(true);
    }
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    function onKey(e) {
      if (e.code !== 'Space') return;
      const tag = (e.target && e.target.tagName) || '';
      if (/INPUT|SELECT|TEXTAREA|BUTTON/.test(tag)) return;
      e.preventDefault();
      setPlaying(!playing);
    }
    window.addEventListener('keydown', onKey);

    // ---- drawing ------------------------------------------------------------------------
    function box(x, y, w, hh, title) {
      ctx.save();
      ctx.fillStyle = 'rgba(12,16,24,0.86)';
      ctx.strokeStyle = C.line; ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x + 0.5, y + 0.5, w, hh, 8); else ctx.rect(x + 0.5, y + 0.5, w, hh);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.muted; ctx.font = `10px ${MONO}`; ctx.textBaseline = 'alphabetic';
      ctx.textAlign = 'left';
      ctx.fillText(title.toUpperCase(), x + 10, y + 16);
      ctx.restore();
    }

    function drawScene() {
      const w = size.w, hh = size.h;
      const { cx, cy, scale } = geom();
      const reach = (P.L1 + P.L2) * scale;

      // guides
      ctx.save();
      ctx.strokeStyle = 'rgba(138,147,166,0.13)'; ctx.lineWidth = 1; ctx.setLineDash([2, 6]);
      ctx.beginPath(); ctx.arc(cx, cy, reach, 0, TAU); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, cy, P.L1 * scale, 0, TAU); ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = 'rgba(138,147,166,0.08)';
      ctx.beginPath(); ctx.moveTo(cx - reach - 16, cy); ctx.lineTo(cx + reach + 16, cy);
      ctx.moveTo(cx, cy - reach - 16); ctx.lineTo(cx, cy + reach + 16); ctx.stroke();
      ctx.restore();

      // trails (oldest → newest in bands of rising opacity)
      if (tlen > 1) {
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        const nb = Math.min(10, tlen - 1);
        const aBase = K === 1 ? 0.95 : Math.min(0.9, 2.4 / Math.sqrt(K));
        const start = (head - tlen + cap) % cap;
        for (let i = K - 1; i >= 0; i--) {
          ctx.strokeStyle = colors[i];
          const off = i * cap;
          for (let b = 0; b < nb; b++) {
            const k0 = Math.floor(b * (tlen - 1) / nb), k1 = Math.floor((b + 1) * (tlen - 1) / nb);
            if (k1 <= k0) continue;
            const f = (b + 1) / nb;
            ctx.globalAlpha = aBase * f * f;
            ctx.lineWidth = (i === 0 ? 1.2 : 0.7) + 1.3 * f;
            ctx.beginPath();
            for (let k = k0; k <= k1; k++) {
              const j = off + (start + k) % cap;
              const x = cx + tx[j] * scale, y = cy + ty[j] * scale;
              if (k === k0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      // rods + bobs (lead pendulum on top)
      const r1 = bobR(P.m1), r2 = bobR(P.m2);
      ctx.save();
      for (let i = K - 1; i >= 1; i--) {
        const b = bobs(i);
        ctx.globalAlpha = Math.max(0.12, 0.5 / Math.sqrt(K));
        ctx.strokeStyle = colors[i]; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(b.cx, b.cy); ctx.lineTo(b.x1, b.y1); ctx.lineTo(b.x2, b.y2); ctx.stroke();
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = colors[i];
        ctx.beginPath(); ctx.arc(b.x2, b.y2, r2 * 0.7, 0, TAU); ctx.fill();
        ctx.globalAlpha = 0.35;
        ctx.beginPath(); ctx.arc(b.x1, b.y1, r1 * 0.6, 0, TAU); ctx.fill();
      }
      ctx.restore();

      const b = bobs(0);
      ctx.save();
      ctx.strokeStyle = 'rgba(230,233,239,0.9)'; ctx.lineWidth = 2; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(b.cx, b.cy); ctx.lineTo(b.x1, b.y1); ctx.lineTo(b.x2, b.y2); ctx.stroke();
      ctx.shadowColor = colors[0]; ctx.shadowBlur = 18;
      ctx.fillStyle = C.bg; ctx.strokeStyle = colors[0]; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(b.x1, b.y1, r1, 0, TAU); ctx.fill(); ctx.stroke();
      ctx.fillStyle = colors[0];
      ctx.beginPath(); ctx.arc(b.x2, b.y2, r2, 0, TAU); ctx.fill();
      ctx.shadowBlur = 0;
      // pivot
      ctx.fillStyle = C.bg; ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(b.cx, b.cy, 4.5, 0, TAU); ctx.fill(); ctx.stroke();
      if (dragging) {
        const [x, y] = dragging === 1 ? [b.x1, b.y1] : [b.x2, b.y2];
        ctx.strokeStyle = C.accent2; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(x, y, (dragging === 1 ? r1 : r2) + 8, 0, TAU); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.accent2; ctx.font = `11px ${MONO}`; ctx.textAlign = 'left';
        ctx.fillText(`θ₁ ${(wrap(base.a1) * 180 / Math.PI).toFixed(1)}°  θ₂ ${(wrap(base.a2) * 180 / Math.PI).toFixed(1)}°`, x + 16, y - 12);
      }
      ctx.restore();

      // corner status
      ctx.save();
      ctx.font = `11px ${MONO}`; ctx.textAlign = 'right'; ctx.fillStyle = C.muted;
      ctx.fillText(`t = ${simT.toFixed(2)} s`, w - 14, 22);
      ctx.fillText(`${K} × ε=10^${epsExp}`, w - 14, 38);
      if (!playing && !dragging) { ctx.fillStyle = C.accent; ctx.fillText('❚❚ paused', w - 14, 54); }
      ctx.textAlign = 'left'; ctx.fillStyle = 'rgba(138,147,166,0.7)';
      ctx.fillText('drag a bob to pose', 14, hh - 14);
      ctx.restore();
    }

    function drawSpark() {
      if (!showSpark || K < 2) return;
      const x = 14, y = 14, w = Math.min(230, size.w * 0.42), hh = 92;
      box(x, y, w, hh, 'θ₂ spread · log₁₀');
      const px = x + 34, py = y + 26, pw = w - 44, ph = hh - 38;
      const ymin = Math.min(-12, Math.floor(epsExp - 1)), ymax = 1;
      const Y = (v) => py + ph - (v - ymin) / (ymax - ymin) * ph;
      ctx.save();
      ctx.font = `9px ${MONO}`; ctx.fillStyle = C.muted; ctx.textAlign = 'right';
      for (const v of [0, -4, -8, -12]) {
        if (v < ymin) continue;
        ctx.strokeStyle = 'rgba(36,44,59,0.9)';
        ctx.beginPath(); ctx.moveTo(px, Y(v)); ctx.lineTo(px + pw, Y(v)); ctx.stroke();
        ctx.fillText(String(v), px - 5, Y(v) + 3);
      }
      const tMax = Math.max(simT, 5);
      const X = (t) => px + t / tMax * pw;
      let started = false;
      ctx.beginPath();
      for (let j = 0; j < histT.length; j++) {
        const v = histS[j];
        if (!isFinite(v) || v < ymin) { started = false; continue; }
        const xx = X(histT[j]), yy = Y(Math.min(v, ymax));
        if (!started) { ctx.moveTo(xx, yy); started = true; } else ctx.lineTo(xx, yy);
      }
      ctx.strokeStyle = C.accent2; ctx.lineWidth = 1.5; ctx.stroke();
      // fitted exponential
      if (isFinite(lambda)) {
        ctx.textAlign = 'right'; ctx.fillStyle = C.accent; ctx.font = `10px ${MONO}`;
        ctx.fillText(`λ≈${lambda.toFixed(2)}/s`, x + w - 8, y + 16);
      }
      ctx.restore();
    }

    function drawInset() {
      if (inset === 'none') return;
      const w = size.w, hh = size.h;
      const iw = Math.max(170, Math.min(290, w * 0.34));
      const ih = inset === 'phase' ? iw + 20 : iw * 0.66;
      const x = w - iw - 14, y = hh - ih - 14;
      if (inset === 'phase') {
        box(x, y, iw, ih, 'phase space');
        const pw = iw - 40, px = x + 28, py = y + 26;
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(px, py, pw, pw);
        ctx.strokeStyle = 'rgba(36,44,59,1)'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (const f of [0.25, 0.5, 0.75]) {
          ctx.moveTo(px + f * pw, py); ctx.lineTo(px + f * pw, py + pw);
          ctx.moveTo(px, py + f * pw); ctx.lineTo(px + pw, py + f * pw);
        }
        ctx.stroke();
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(phase, px, py, pw, pw);
        // live markers
        for (let i = K - 1; i >= 0; i--) {
          const mx = px + (wrap(S.t1[i]) + Math.PI) / TAU * pw, my = py + (Math.PI - wrap(S.t2[i])) / TAU * pw;
          ctx.fillStyle = colors[i];
          ctx.beginPath(); ctx.arc(mx, my, i === 0 ? 3 : 1.8, 0, TAU); ctx.fill();
        }
        ctx.font = `9px ${MONO}`; ctx.fillStyle = C.muted;
        ctx.textAlign = 'center';
        ctx.fillText('−π', px, py + pw + 11); ctx.fillText('θ₁', px + pw / 2, py + pw + 11); ctx.fillText('π', px + pw, py + pw + 11);
        ctx.textAlign = 'right';
        ctx.fillText('π', px - 4, py + 7); ctx.fillText('θ₂', px - 4, py + pw / 2 + 3); ctx.fillText('−π', px - 4, py + pw);
        ctx.restore();
      } else {
        box(x, y, iw, ih, 'energy drift · #1');
        const px = x + 12, py = y + 28, pw = iw - 24, ph = ih - 44;
        let m = 0;
        for (const v of histE) if (isFinite(v)) m = Math.max(m, Math.abs(v));
        m = Math.max(m * 1.2, 1e-12);
        const tMax = Math.max(simT, 5);
        const X = (t) => px + t / tMax * pw, Y = (v) => py + ph / 2 - v / m * ph / 2;
        ctx.save();
        ctx.strokeStyle = C.line; ctx.setLineDash([2, 4]);
        ctx.beginPath(); ctx.moveTo(px, Y(0)); ctx.lineTo(px + pw, Y(0)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        let started = false;
        for (let j = 0; j < histT.length; j++) {
          const v = histE[j];
          if (!isFinite(v)) { started = false; continue; }
          if (!started) { ctx.moveTo(X(histT[j]), Y(v)); started = true; } else ctx.lineTo(X(histT[j]), Y(v));
        }
        ctx.strokeStyle = P.c > 0 ? C.danger : C.good; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.font = `9px ${MONO}`; ctx.fillStyle = C.muted; ctx.textAlign = 'left';
        ctx.fillText(`+${fmtExp(m * 100)}%`, px + 2, py + 8);
        ctx.fillText(`−${fmtExp(m * 100)}%`, px + 2, py + ph);
        ctx.textAlign = 'right';
        ctx.fillText(`${tMax.toFixed(0)} s`, px + pw, py + ph + 12);
        ctx.fillText(P.c > 0 ? 'damped: energy decays' : 'RK4, h = 0.5 ms', px + pw, y + 16);
        ctx.restore();
      }
    }

    let roTimer = 0;
    function updateReadout(dt) {
      roTimer -= dt;
      if (roTimer > 0) return;
      roTimer = 0.12;
      const E = energy(P, S.t1[0], S.w1[0], S.t2[0], S.w2[0]);
      const drift = E0 > 1e-12 ? (E - E0) / E0 * 100 : 0;
      const sp = spread();
      ro.set('Time', simT.toFixed(2) + ' s');
      ro.set('Energy #1', E.toFixed(4) + ' J');
      ro.set('Drift', (drift >= 0 ? '+' : '') + fmtExp(drift) + ' %');
      ro.set('θ₂ spread', isFinite(sp) ? fmtExp(sp) + ' rad' : '—');
      ro.set('λ est.', isFinite(lambda) ? lambda.toFixed(2) + ' /s' : '—');
    }

    // ---- main loop ------------------------------------------------------------------------
    init();
    const stop = loop((dt) => {
      if (playing && !dragging) {
        acc += dt * speed;
        let steps = 0;
        while (acc >= H && steps < 6000) {
          for (let i = 0; i < K; i++) rk4(P, S, i, H);
          simT += H; acc -= H; steps++;
          if (simT >= nextRec) { record(); nextRec += REC_DT; }
          if (simT >= nextHist) { sampleHist(); nextHist += histDt; }
        }
        if (steps >= 6000) acc = 0;
      }
      if (size.w < 2) return;
      ctx.clearRect(0, 0, size.w, size.h);
      drawScene();
      drawSpark();
      drawInset();
      updateReadout(dt);
    });

    return () => {
      stop();
      cv.destroy();
      window.removeEventListener('keydown', onKey);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
    };
  },
};
