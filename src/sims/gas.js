// Gas & collisions: a 2D hard-disk gas. Exact elastic collisions along the contact normal
// (mass-weighted), positional overlap resolution, uniform-grid broad phase, adaptive fixed
// substeps. Live speed histogram vs 2D Maxwell–Boltzmann, wall-impulse pressure vs PA = NkT,
// equipartition between species, barometric atmosphere, a draggable piston, a divider with a
// gap (and an optional Maxwell's demon) and a Brownian tracer.
import { createLayout, section, slider, select, checkbox, buttons, readout, createCanvas, loop, h } from '../ui.js';

const C = {
  bg: '#0b0e14', panel: '#151b27', line: '#242c3b', text: '#e6e9ef', muted: '#8a93a6',
  amber: '#f5b544', cyan: '#5ec8e5', good: '#7bd88f', danger: '#ef6b6b',
};
const MONO = "'IBM Plex Mono', ui-monospace, monospace";
const MAXN = 2001;            // 2000 gas particles + 1 Brownian tracer
const T_UNIT = 100;           // kT = T · T_UNIT  (px²/s² for unit mass)
const KT_REF = 100 * T_UNIT;  // colour-map reference temperature (T = 100)
const DIV_HT = 2;             // divider half-thickness (px)
const HB = 36;                // speed-histogram bins
const DB = 22;                // density-profile bins
const SERIES_MAX = 260;

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const rgb = (hx) => [1, 3, 5].map((k) => parseInt(hx.slice(k, k + 2), 16));
const STOPS = [[0, '#2c6a8f'], [0.28, '#5ec8e5'], [0.5, '#cfe6dc'], [0.72, '#f5b544'], [1, '#ef6b6b']]
  .map(([t, c]) => [t, rgb(c)]);
function cmap(t, a = 1) {
  t = clamp(t, 0, 1);
  let k = 0;
  while (k < STOPS.length - 2 && t > STOPS[k + 1][0]) k++;
  const [t0, c0] = STOPS[k], [t1, c1] = STOPS[k + 1];
  const f = (t - t0) / (t1 - t0);
  const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * f));
  return a === 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}
const NB = 32;
const LUT = Array.from({ length: NB }, (_, k) => cmap(k / (NB - 1)));

function gauss() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const BASE = {
  N: 600, radius: 3, T: 100, g: 0, two: false, ratio: 6, divider: false, gapOpen: true, gap: 0.3,
  demon: false, brownian: false, colorBy: 'speed', init: 'mb',
};
const PRESETS = {
  relax: { label: 'Relax to Maxwell–Boltzmann', N: 900, radius: 3.5, init: 'equal' },
  species: { label: 'Two species (equipartition)', N: 500, two: true, ratio: 8, colorBy: 'species', init: 'equal' },
  diffusion: { label: 'Diffusion (mixing)', N: 900, radius: 2.5, divider: true, gap: 0.35, colorBy: 'tag' },
  hotcold: { label: 'Hot | cold (heat flow)', N: 800, radius: 2.5, divider: true, gap: 0.2, init: 'hotcold' },
  demon: { label: "Maxwell's demon", N: 700, radius: 2.5, divider: true, gap: 0.16, demon: true },
  atmosphere: { label: 'Atmosphere (gravity)', N: 1000, radius: 2.5, g: 60 },
  brownian: { label: 'Brownian motion', N: 1400, radius: 2, brownian: true },
};

export default {
  id: 'gas',
  title: 'Gas & Collisions',
  glyph: '⁂',
  tag: 'physics',
  blurb: 'Thousands of elastic disks start at one speed and relax into the Maxwell–Boltzmann distribution. Compress them with a piston, add gravity, mix two gases, or set a demon loose.',

  mount(root) {
    const { stage, panel } = createLayout(root, {
      title: 'Gas & Collisions',
      desc: 'A 2D hard-disk gas with exact elastic collisions. Everything thermodynamic here (temperature, pressure, the speed distribution, entropy) comes out of billiard-ball mechanics.',
    });

    // ---------------------------------------------------------------- state
    const P = { ...BASE, ...PRESETS.relax, preset: 'relax', speed: 1, trail: true };
    const X = new Float64Array(MAXN), Y = new Float64Array(MAXN);
    const VX = new Float64Array(MAXN), VY = new Float64Array(MAXN);
    const R = new Float64Array(MAXN), M = new Float64Array(MAXN), IM = new Float64Array(MAXN);
    const SP = new Uint8Array(MAXN);   // 0 light, 1 heavy, 2 Brownian tracer
    const TAG = new Uint8Array(MAXN);  // starting side (0 left, 1 right)
    const BIN = new Uint8Array(MAXN);
    let n = 0, bigIdx = -1;

    const box = { x0: 0, y0: 0, x1: 1, y1: 1 };
    let layout = null;
    let divX = 0.5;
    let pistonX = 1, pistonTarget = 1, pistonU = 0, dragging = false, pistonTouched = false;
    let running = true, simTime = 0, ready = false;
    let v0Start = 0; // single starting speed (equal-speed init), for the histogram marker

    // measurements
    let kTnow = KT_REF, kTs = KT_REF, vmaxLast = 0;
    const st = { ke0: 0, c0: 0, ke1: 0, c1: 0, keL: 0, cL: 0, keR: 0, cR: 0, phi: 0 };
    let wallImp = 0, impTime = 0, Pmeas = NaN;
    let mixS = NaN, mbDist = NaN;
    const hist = [new Float64Array(HB), new Float64Array(HB)];
    let histFresh = true, histYMax = 1;
    const dens = [new Float64Array(DB), new Float64Array(DB)];
    let densFresh = true;
    let series = [], seriesMode = '', seriesClock = 0;
    let trail = [];

    // grid
    let cs = 6, gw = 1, gh = 1;
    let cellStart = new Int32Array(2), cellFill = new Int32Array(1);
    const cellIdx = new Int32Array(MAXN);

    // ---------------------------------------------------------------- setup helpers
    const heavyR = () => P.radius * Math.min(2.2, Math.pow(P.ratio, 0.25));
    const bigR = () => Math.max(13, P.radius * 6);
    const gapTop = () => (box.y0 + box.y1) / 2 - (P.gap * (box.y1 - box.y0)) / 2;
    const gapBot = () => (box.y0 + box.y1) / 2 + (P.gap * (box.y1 - box.y0)) / 2;

    function particleArea() {
      let a = 0;
      for (let i = 0; i < n; i++) a += Math.PI * R[i] * R[i];
      return a;
    }
    function pistonMin() {
      const H = box.y1 - box.y0;
      let m = box.x0 + Math.max(40, 0.12 * (box.x1 - box.x0), particleArea() / (0.62 * H));
      if (P.divider) m = Math.max(m, divX + 30);
      return Math.min(m, box.x1);
    }

    function lattice(count, rx0, ry0, rx1, ry1, r, avoid) {
      const W = Math.max(1, rx1 - rx0), H = Math.max(1, ry1 - ry0);
      let want = Math.ceil(count * 1.1) + 4;
      for (let tries = 0; tries < 12; tries++) {
        const cols = Math.max(1, Math.ceil(Math.sqrt((want * W) / H)));
        const rows = Math.max(1, Math.ceil(want / cols));
        const dx = W / cols, dy = H / rows;
        const jx = Math.max(0, (dx - 2 * r) / 2) * 0.9, jy = Math.max(0, (dy - 2 * r) / 2) * 0.9;
        const sites = [];
        for (let a = 0; a < rows; a++) for (let b = 0; b < cols; b++) {
          const x = rx0 + (b + 0.5) * dx + (Math.random() * 2 - 1) * jx;
          const y = ry0 + (a + 0.5) * dy + (Math.random() * 2 - 1) * jy;
          if (avoid && Math.hypot(x - avoid.x, y - avoid.y) < avoid.r) continue;
          sites.push([x, y]);
        }
        if (sites.length >= count) return shuffle(sites).slice(0, count);
        want = Math.ceil(want * 1.3);
      }
      return Array.from({ length: count }, () => [rx0 + Math.random() * W, ry0 + Math.random() * H]);
    }

    function setMass(i) {
      const s = SP[i];
      if (s === 0) { M[i] = 1; R[i] = P.radius; }
      else if (s === 1) { M[i] = P.ratio; R[i] = heavyR(); }
      IM[i] = 1 / M[i];
    }

    function reset() {
      if (!layout) return;
      pistonX = pistonTarget = box.x1; pistonU = 0;
      divX = (box.x0 + box.x1) / 2;
      n = 0; bigIdx = -1;
      const kT = P.T * T_UNIT;
      const N = P.N;
      let avoid = null;
      if (P.brownian) {
        bigIdx = 0; n = 1;
        const Rb = bigR();
        X[0] = (box.x0 + box.x1) / 2; Y[0] = (box.y0 + box.y1) / 2; VX[0] = VY[0] = 0;
        R[0] = Rb; M[0] = (Rb / P.radius) ** 2; IM[0] = 1 / M[0]; SP[0] = 2; TAG[0] = 0;
        avoid = { x: X[0], y: Y[0], r: Rb + P.radius + 1 };
      }
      const pad = P.radius * 2.2 + 1;
      let sites;
      if (P.divider) {
        const nl = Math.floor(N / 2);
        sites = [
          ...lattice(nl, box.x0 + 1, box.y0 + 1, divX - DIV_HT - 1, box.y1 - 1, pad / 2),
          ...lattice(N - nl, divX + DIV_HT + 1, box.y0 + 1, box.x1 - 1, box.y1 - 1, pad / 2),
        ];
      } else {
        sites = lattice(N, box.x0 + 1, box.y0 + 1, box.x1 - 1, box.y1 - 1, pad / 2, avoid);
      }
      const heavyCount = P.two ? Math.round(N / 2) : 0;
      const order = shuffle(Array.from({ length: N }, (_, k) => k));
      for (let k = 0; k < N; k++) {
        const i = n++;
        SP[i] = order[k] < heavyCount ? 1 : 0;
        setMass(i);
        const [sx, sy] = sites[k];
        X[i] = clamp(sx, box.x0 + R[i], box.x1 - R[i]);
        Y[i] = clamp(sy, box.y0 + R[i], box.y1 - R[i]);
        TAG[i] = X[i] < divX ? 0 : 1;
        if (P.g > 0 && !P.divider) {
          // start already in the barometric profile so potential energy isn't dumped into heat
          const Hh = box.y1 - box.y0 - 2 * R[i], beta = (M[i] * P.g) / kT;
          const hgt = -Math.log(1 - Math.random() * (1 - Math.exp(-beta * Hh))) / beta;
          Y[i] = box.y1 - R[i] - hgt;
        }
      }
      // velocities
      let mMean = 0, cnt = 0;
      for (let i = 0; i < n; i++) if (SP[i] !== 2) { mMean += M[i]; cnt++; }
      mMean /= Math.max(1, cnt);
      v0Start = P.init === 'equal' ? Math.sqrt((2 * kT) / mMean) : 0;
      for (let i = 0; i < n; i++) {
        if (SP[i] === 2) continue;
        if (P.init === 'equal') {
          const a = Math.random() * Math.PI * 2;
          VX[i] = v0Start * Math.cos(a); VY[i] = v0Start * Math.sin(a);
        } else {
          const kTi = P.init === 'hotcold' ? (TAG[i] === 0 ? kT * 2.2 : kT * 0.35) : kT;
          const s = Math.sqrt(kTi / M[i]);
          VX[i] = gauss() * s; VY[i] = gauss() * s;
        }
      }
      simTime = 0; wallImp = 0; impTime = 0; Pmeas = NaN; mixS = NaN; mbDist = NaN;
      histFresh = true; densFresh = true; series = []; seriesClock = 0; trail = [];
      computeStats();
      kTs = kTnow;
    }

    function applyRadii() {
      for (let i = 0; i < n; i++) if (SP[i] !== 2) setMass(i);
      if (bigIdx >= 0) R[bigIdx] = Math.max(R[bigIdx], P.radius * 3);
      pistonTarget = Math.max(pistonTarget, pistonMin());
    }

    function scaleVel(f) {
      for (let i = 0; i < n; i++) { VX[i] *= f; VY[i] *= f; }
      computeStats();
    }
    function rescaleTo(kT) {
      computeStats();
      if (kTnow > 0) scaleVel(Math.sqrt(kT / kTnow));
    }

    // ---------------------------------------------------------------- physics
    function buildGrid() {
      let maxR = 0;
      for (let i = 0; i < n; i++) if (SP[i] !== 2 && R[i] > maxR) maxR = R[i];
      cs = Math.max(2 * maxR, 3);
      gw = Math.max(1, Math.ceil((box.x1 - box.x0) / cs));
      gh = Math.max(1, Math.ceil((box.y1 - box.y0) / cs));
      const nc = gw * gh;
      if (cellStart.length < nc + 1) { cellStart = new Int32Array(nc + 1); cellFill = new Int32Array(nc); }
      cellStart.fill(0, 0, nc + 1);
      for (let i = 0; i < n; i++) {
        if (SP[i] === 2) continue;
        const c = cellOf(i);
        cellStart[c + 1]++;
      }
      for (let c = 0; c < nc; c++) cellStart[c + 1] += cellStart[c];
      cellFill.set(cellStart.subarray(0, nc));
      for (let i = 0; i < n; i++) {
        if (SP[i] === 2) continue;
        cellIdx[cellFill[cellOf(i)]++] = i;
      }
    }
    function cellOf(i) {
      const cx = clamp(((X[i] - box.x0) / cs) | 0, 0, gw - 1);
      const cy = clamp(((Y[i] - box.y0) / cs) | 0, 0, gh - 1);
      return cy * gw + cx;
    }

    function pair(i, j) {
      const dx = X[j] - X[i], dy = Y[j] - Y[i];
      const rs = R[i] + R[j];
      const d2 = dx * dx + dy * dy;
      if (d2 >= rs * rs) return;
      let d = Math.sqrt(d2), nx, ny;
      if (d < 1e-9) { const a = Math.random() * 6.2832; nx = Math.cos(a); ny = Math.sin(a); d = 0; }
      else { nx = dx / d; ny = dy / d; }
      const imi = IM[i], imj = IM[j], w = imi + imj;
      // positional correction, split by inverse mass
      const ov = rs - d;
      X[i] -= nx * ov * (imi / w); Y[i] -= ny * ov * (imi / w);
      X[j] += nx * ov * (imj / w); Y[j] += ny * ov * (imj / w);
      // elastic impulse along the contact normal
      const vn = (VX[j] - VX[i]) * nx + (VY[j] - VY[i]) * ny;
      if (vn < 0) {
        const J = (-2 * vn) / w;
        VX[i] -= J * imi * nx; VY[i] -= J * imi * ny;
        VX[j] += J * imj * nx; VY[j] += J * imj * ny;
      }
    }

    function collideAll() {
      for (let cy = 0; cy < gh; cy++) {
        for (let cx = 0; cx < gw; cx++) {
          const c = cy * gw + cx;
          const s = cellStart[c], e = cellStart[c + 1];
          if (s === e) continue;
          const right = cx + 1 < gw, left = cx > 0, down = cy + 1 < gh;
          for (let a = s; a < e; a++) {
            const i = cellIdx[a];
            for (let b = a + 1; b < e; b++) pair(i, cellIdx[b]);
            if (right) for (let b = cellStart[c + 1], be = cellStart[c + 2]; b < be; b++) pair(i, cellIdx[b]);
            if (down) {
              const d = c + gw;
              if (left) for (let b = cellStart[d - 1], be = cellStart[d]; b < be; b++) pair(i, cellIdx[b]);
              for (let b = cellStart[d], be = cellStart[d + 1]; b < be; b++) pair(i, cellIdx[b]);
              if (right) for (let b = cellStart[d + 1], be = cellStart[d + 2]; b < be; b++) pair(i, cellIdx[b]);
            }
          }
        }
      }
      if (bigIdx >= 0) {
        const b = bigIdx, reach = R[b] + cs;
        const cx0 = clamp(((X[b] - reach - box.x0) / cs) | 0, 0, gw - 1);
        const cx1 = clamp(((X[b] + reach - box.x0) / cs) | 0, 0, gw - 1);
        const cy0 = clamp(((Y[b] - reach - box.y0) / cs) | 0, 0, gh - 1);
        const cy1 = clamp(((Y[b] + reach - box.y0) / cs) | 0, 0, gh - 1);
        for (let cy = cy0; cy <= cy1; cy++) for (let cx = cx0; cx <= cx1; cx++) {
          const c = cy * gw + cx;
          for (let a = cellStart[c], e = cellStart[c + 1]; a < e; a++) pair(b, cellIdx[a]);
        }
      }
    }

    function integrate(dt, u) {
      const g = P.g, x0 = box.x0, y0 = box.y0, y1 = box.y1, px = pistonX;
      let imp = 0;
      for (let i = 0; i < n; i++) {
        const r = R[i], m = M[i];
        let vx = VX[i], vy = VY[i] + g * dt;
        let x = X[i] + vx * dt, y = Y[i] + vy * dt;
        if (x - r < x0) { x = x0 + r; if (vx < 0) { imp -= 2 * m * vx; vx = -vx; } }
        if (x + r > px) { x = px - r; if (vx > u) { imp += 2 * m * (vx - u); vx = 2 * u - vx; } }
        if (y - r < y0) { y = y0 + r; if (vy < 0) { imp -= 2 * m * vy; vy = -vy; } }
        if (y + r > y1) { y = y1 - r; if (vy > 0) { imp += 2 * m * vy; vy = -vy; } }
        X[i] = x; Y[i] = y; VX[i] = vx; VY[i] = vy;
      }
      wallImp += imp;
    }

    function segCollide(i, a, b) {
      const y = Y[i];
      const cy = y < a ? a : y > b ? b : y;
      const ddx = X[i] - divX, ddy = y - cy;
      const rr = R[i] + DIV_HT;
      const d2 = ddx * ddx + ddy * ddy;
      if (d2 >= rr * rr) return;
      let d = Math.sqrt(d2), nx, ny;
      if (d < 1e-9) { nx = VX[i] > 0 ? -1 : 1; ny = 0; d = 0; } else { nx = ddx / d; ny = ddy / d; }
      X[i] += nx * (rr - d); Y[i] += ny * (rr - d);
      const vn = VX[i] * nx + VY[i] * ny;
      if (vn < 0) { VX[i] -= 2 * vn * nx; VY[i] -= 2 * vn * ny; }
    }
    function dividerPass() {
      if (!P.divider) return;
      const gt = gapTop(), gb = gapBot(), thr = 2 * kTnow;
      for (let i = 0; i < n; i++) {
        const reach = R[i] + DIV_HT, ddx = X[i] - divX;
        if (ddx > reach || ddx < -reach) continue;
        let solid = !P.gapOpen;
        if (!solid && P.demon) {
          // The demon lets fast molecules pass left→right and slow ones right→left.
          const fast = M[i] * (VX[i] * VX[i] + VY[i] * VY[i]) > thr;
          solid = ddx < 0 ? !fast : fast;
        }
        if (solid) segCollide(i, box.y0 - 50, box.y1 + 50);
        else { segCollide(i, box.y0 - 50, gt); segCollide(i, gb, box.y1 + 50); }
      }
    }

    function step(frameDt) {
      const dt = frameDt * P.speed;
      if (dt <= 0) return;
      pistonTarget = clamp(pistonTarget, pistonMin(), box.x1);
      const u = clamp((pistonTarget - pistonX) / dt, -1200, 1200);
      pistonU = u;
      let rmin = Infinity;
      for (let i = 0; i < n; i++) if (R[i] < rmin) rmin = R[i];
      rmin = Math.max(1, rmin);
      const sub = clamp(Math.ceil(((vmaxLast + Math.abs(u) + Math.abs(P.g) * dt) * dt) / (0.35 * rmin)), 2, 40);
      const hdt = dt / sub;
      for (let s = 0; s < sub; s++) {
        pistonX += u * hdt;
        integrate(hdt, u);
        dividerPass();
        buildGrid();
        collideAll();
      }
      lastSub = sub;
      simTime += dt;
      impTime += dt;
      if (impTime >= 0.3) {
        const perim = 2 * ((pistonX - box.x0) + (box.y1 - box.y0));
        const p = wallImp / (perim * impTime);
        Pmeas = Number.isFinite(Pmeas) ? Pmeas + (p - Pmeas) * 0.25 : p;
        wallImp = 0; impTime = 0;
      }
    }
    let lastSub = 0;

    // Paused piston drag: move the wall and just shove particles out of the way.
    function nudgePistonPaused() {
      pistonTarget = clamp(pistonTarget, pistonMin(), box.x1);
      pistonX = pistonTarget; pistonU = 0;
      for (let i = 0; i < n; i++) if (X[i] + R[i] > pistonX) X[i] = pistonX - R[i];
    }

    // ---------------------------------------------------------------- measurements
    function computeStats() {
      let ke = 0, vmax = 0, area = 0;
      st.ke0 = st.c0 = st.ke1 = st.c1 = st.keL = st.cL = st.keR = st.cR = 0;
      for (let i = 0; i < n; i++) {
        const v2 = VX[i] * VX[i] + VY[i] * VY[i];
        const k = 0.5 * M[i] * v2;
        ke += k;
        area += Math.PI * R[i] * R[i];
        if (v2 > vmax) vmax = v2;
        if (SP[i] === 0) { st.ke0 += k; st.c0++; } else if (SP[i] === 1) { st.ke1 += k; st.c1++; }
        if (X[i] < divX) { st.keL += k; st.cL++; } else { st.keR += k; st.cR++; }
      }
      kTnow = n ? ke / n : 0;   // 2D: ⟨KE⟩ = kT (two quadratic degrees of freedom)
      vmaxLast = Math.sqrt(vmax);
      st.phi = area / Math.max(1, (pistonX - box.x0) * (box.y1 - box.y0));
    }

    const mbCdf = (v, m, kT) => 1 - Math.exp((-m * v * v) / (2 * kT));
    function histRange() { return 4.2 * Math.sqrt(Math.max(kTs, 1)); }

    function updateHist() {
      const vr = histRange(), dv = vr / HB;
      const cnt = [new Float64Array(HB), new Float64Array(HB)], tot = [0, 0];
      for (let i = 0; i < n; i++) {
        const s = SP[i];
        if (s === 2) continue;
        const v = Math.sqrt(VX[i] * VX[i] + VY[i] * VY[i]);
        tot[s]++;
        const k = (v / dv) | 0;
        if (k < HB) cnt[s][k]++;
      }
      const a = histFresh ? 1 : clamp(0.04 + 25 / Math.max(1, n), 0.04, 0.3);
      for (let s = 0; s < 2; s++) {
        if (!tot[s]) { hist[s].fill(0); continue; }
        for (let k = 0; k < HB; k++) hist[s][k] += (cnt[s][k] / tot[s] - hist[s][k]) * a;
      }
      histFresh = false;
      // total-variation distance of the light-species histogram from Maxwell–Boltzmann
      let tv = 0;
      for (let k = 0; k < HB; k++) {
        const pth = mbCdf((k + 1) * dv, 1, kTnow) - mbCdf(k * dv, 1, kTnow);
        tv += Math.abs(hist[0][k] - pth);
      }
      tv += 1 - mbCdf(vr, 1, kTnow);
      mbDist = 0.5 * tv;
    }

    function updateDensity() {
      if (P.g <= 0) { densFresh = true; return; }
      const H = box.y1 - box.y0;
      const cnt = [new Float64Array(DB), new Float64Array(DB)], tot = [0, 0];
      for (let i = 0; i < n; i++) {
        const s = SP[i];
        if (s === 2) continue;
        const k = clamp(((box.y1 - Y[i]) / H * DB) | 0, 0, DB - 1);
        cnt[s][k]++; tot[s]++;
      }
      const a = densFresh ? 1 : 0.05;
      for (let s = 0; s < 2; s++) {
        for (let k = 0; k < DB; k++) dens[s][k] += ((tot[s] ? cnt[s][k] / tot[s] : 0) - dens[s][k]) * a;
      }
      densFresh = false;
    }

    function mixingEntropy() {
      const GX = 12, GY = 8, W = pistonX - box.x0, H = box.y1 - box.y0;
      const a = new Float64Array(GX * GY), b = new Float64Array(GX * GY);
      let na = 0, nb = 0;
      for (let i = 0; i < n; i++) {
        const c = clamp(((Y[i] - box.y0) / H * GY) | 0, 0, GY - 1) * GX + clamp(((X[i] - box.x0) / W * GX) | 0, 0, GX - 1);
        if (TAG[i] === 0) { a[c]++; na++; } else { b[c]++; nb++; }
      }
      const xlx = (x) => (x > 0 ? x * Math.log(x) : 0);
      let S = 0;
      for (let c = 0; c < GX * GY; c++) {
        const t = a[c] + b[c];
        if (t) S -= t * (xlx(a[c] / t) + xlx(b[c] / t));
      }
      const f = na / Math.max(1, na + nb);
      const Smax = (na + nb) * -(xlx(f) + xlx(1 - f));
      return Smax > 0 ? S / Smax : 0;
    }

    function currentSeriesMode() {
      if (P.divider && P.colorBy === 'tag') return 'mix';
      if (P.divider) return 'lr';
      if (P.two) return 'ke';
      return 'p';
    }
    function sampleSeries() {
      const mode = currentSeriesMode();
      if (mode !== seriesMode) { series = []; seriesMode = mode; }
      let s;
      if (mode === 'mix') { mixS = mixingEntropy(); s = [mixS]; }
      else if (mode === 'lr') s = [st.cL ? st.keL / st.cL / T_UNIT : NaN, st.cR ? st.keR / st.cR / T_UNIT : NaN];
      else if (mode === 'ke') s = [st.c0 ? st.ke0 / st.c0 / T_UNIT : NaN, st.c1 ? st.ke1 / st.c1 / T_UNIT : NaN];
      else {
        if (!Number.isFinite(Pmeas)) return;
        const A = (pistonX - box.x0) * (box.y1 - box.y0);
        s = [Pmeas, (n * kTnow) / A];
      }
      series.push(s);
      if (series.length > SERIES_MAX) series.shift();
    }

    // ---------------------------------------------------------------- layout & resize
    const cv = createCanvas(stage, (w, H) => {
      const M0 = 20;
      const old = ready ? { ...box } : null;
      const wide = w >= 760;
      let nb;
      if (wide) {
        const sw = clamp(w * 0.27, 230, 320);
        nb = { x0: M0, y0: M0 + 22, x1: w - M0 * 2 - sw, y1: H - M0 - 10 };
        layout = { wide, side: { x: w - M0 - sw, y: M0, w: sw, h: H - 2 * M0 } };
      } else {
        const bh = clamp(H * 0.3, 120, 170);
        nb = { x0: M0, y0: M0 + 22, x1: w - M0, y1: H - M0 - bh - 14 };
        layout = { wide, side: { x: M0, y: H - M0 - bh, w: w - 2 * M0, h: bh } };
      }
      nb.x1 = Math.max(nb.x1, nb.x0 + 80);
      nb.y1 = Math.max(nb.y1, nb.y0 + 80);
      if (old) {
        const sx = (nb.x1 - nb.x0) / (old.x1 - old.x0), sy = (nb.y1 - nb.y0) / (old.y1 - old.y0);
        const mapX = (x) => nb.x0 + (x - old.x0) * sx, mapY = (y) => nb.y0 + (y - old.y0) * sy;
        for (let i = 0; i < n; i++) { X[i] = mapX(X[i]); Y[i] = mapY(Y[i]); }
        trail = trail.map(([x, y]) => [mapX(x), mapY(y)]);
        pistonX = mapX(pistonX); pistonTarget = mapX(pistonTarget);
        Object.assign(box, nb);
        divX = (box.x0 + box.x1) / 2;
      } else {
        Object.assign(box, nb);
        ready = true;
        reset();
      }
      if (!running) draw();
    });
    const { canvas, ctx, size } = cv;

    // ---------------------------------------------------------------- piston dragging
    const nearPiston = (mx, my) => Math.abs(mx - (pistonX + 4)) < 16 && my > box.y0 - 6 && my < box.y1 + 6;
    const onDown = (e) => {
      if (!nearPiston(e.offsetX, e.offsetY)) return;
      dragging = true; pistonTouched = true;
      canvas.setPointerCapture?.(e.pointerId);
      pistonTarget = e.offsetX - 4;
      e.preventDefault();
    };
    const onMove = (e) => {
      if (dragging) {
        pistonTarget = e.offsetX - 4;
        if (!running) { nudgePistonPaused(); draw(); }
      }
      canvas.style.cursor = dragging || nearPiston(e.offsetX, e.offsetY) ? 'ew-resize' : 'default';
    };
    const onUp = (e) => {
      dragging = false;
      canvas.releasePointerCapture?.(e.pointerId);
    };
    canvas.style.touchAction = 'none';
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);

    // ---------------------------------------------------------------- drawing
    function rr(x, y, w, H, r) {
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y, w, H, r); else ctx.rect(x, y, w, H);
    }
    function panelFrame(p, title, legend) {
      rr(p.x + 0.5, p.y + 0.5, p.w - 1, p.h - 1, 8);
      ctx.fillStyle = 'rgba(21,27,39,0.82)'; ctx.fill();
      ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke();
      ctx.font = `500 10px ${MONO}`; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
      ctx.fillStyle = C.muted;
      ctx.fillText(title.toUpperCase(), p.x + 10, p.y + 9);
      if (legend) {
        let lx = p.x + p.w - 10;
        ctx.textAlign = 'right';
        for (let k = legend.length - 1; k >= 0; k--) {
          const [txt, col, dash] = legend[k];
          ctx.fillStyle = C.text; ctx.fillText(txt, lx, p.y + 9);
          lx -= ctx.measureText(txt).width + 5;
          ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash ? [3, 2] : []);
          ctx.beginPath(); ctx.moveTo(lx - 10, p.y + 15); ctx.lineTo(lx, p.y + 15); ctx.stroke();
          ctx.setLineDash([]);
          lx -= 22;
        }
      }
    }

    function colorIndex(i) {
      if (P.colorBy === 'species') return SP[i] === 1 ? -2 : -1;
      if (P.colorBy === 'tag') return TAG[i] === 0 ? -2 : -1;
      const v = Math.sqrt(VX[i] * VX[i] + VY[i] * VY[i]);
      return clamp(Math.round((v / (3.2 * Math.sqrt(KT_REF))) * (NB - 1)), 0, NB - 1);
    }

    function drawBox() {
      const { x0, y0, x1, y1 } = box;
      // chamber
      rr(x0, y0, x1 - x0, y1 - y0, 3);
      ctx.fillStyle = 'rgba(13,17,26,0.9)'; ctx.fill();
      // faint grid
      ctx.strokeStyle = 'rgba(36,44,59,0.45)'; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = x0 + 40; x < x1; x += 40) { ctx.moveTo(Math.round(x) + 0.5, y0); ctx.lineTo(Math.round(x) + 0.5, y1); }
      for (let y = y1 - 40; y > y0; y -= 40) { ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); }
      ctx.stroke();
      // gravity shading
      if (P.g > 0) {
        const gr = ctx.createLinearGradient(0, y0, 0, y1);
        gr.addColorStop(0, 'rgba(94,200,229,0)');
        gr.addColorStop(1, `rgba(94,200,229,${Math.min(0.09, P.g / 2500)})`);
        ctx.fillStyle = gr; ctx.fillRect(x0, y0, pistonX - x0, y1 - y0);
      }
      // compressed-away region behind the piston
      if (pistonX < x1 - 1) {
        ctx.save();
        ctx.beginPath(); ctx.rect(pistonX, y0, x1 - pistonX, y1 - y0); ctx.clip();
        ctx.fillStyle = 'rgba(11,14,20,0.85)'; ctx.fillRect(pistonX, y0, x1 - pistonX, y1 - y0);
        ctx.strokeStyle = 'rgba(138,147,166,0.12)'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let k = -(y1 - y0); k < x1 - pistonX; k += 9) { ctx.moveTo(pistonX + k, y1); ctx.lineTo(pistonX + k + (y1 - y0), y0); }
        ctx.stroke();
        ctx.restore();
      }
      rr(x0 - 0.5, y0 - 0.5, x1 - x0 + 1, y1 - y0 + 1, 3);
      ctx.strokeStyle = '#34405a'; ctx.lineWidth = 1.5; ctx.stroke();
    }

    function drawDivider() {
      if (!P.divider) return;
      const gt = gapTop(), gb = gapBot();
      ctx.fillStyle = '#4a5670';
      ctx.fillRect(divX - DIV_HT, box.y0, DIV_HT * 2, gt - box.y0);
      ctx.fillRect(divX - DIV_HT, gb, DIV_HT * 2, box.y1 - gb);
      if (!P.gapOpen) {
        ctx.fillStyle = 'rgba(138,147,166,0.55)';
        ctx.fillRect(divX - DIV_HT, gt, DIV_HT * 2, gb - gt);
      } else if (P.demon) {
        ctx.strokeStyle = 'rgba(245,181,68,0.55)'; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(divX, gt); ctx.lineTo(divX, gb); ctx.stroke(); ctx.setLineDash([]);
        // the demon's eye
        const ey = gt - 12;
        ctx.fillStyle = C.amber;
        ctx.beginPath(); ctx.ellipse(divX, ey, 7, 4.5, 0, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = C.bg; ctx.beginPath(); ctx.arc(divX + Math.sin(simTime * 3) * 2.5, ey, 2, 0, Math.PI * 2); ctx.fill();
        ctx.font = `10px ${MONO}`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillStyle = C.amber; ctx.fillText('demon', divX, ey - 7);
      }
      ctx.fillStyle = '#8a93a6';
      ctx.beginPath(); ctx.arc(divX, gt, DIV_HT + 0.5, 0, Math.PI * 2); ctx.arc(divX, gb, DIV_HT + 0.5, 0, Math.PI * 2); ctx.fill();
    }

    function drawParticles() {
      for (let i = 0; i < n; i++) BIN[i] = SP[i] === 2 ? 255 : 100 + colorIndex(i);
      const groups = [[98, C.amber], [99, C.cyan]];
      for (let k = 0; k < NB; k++) groups.push([100 + k, LUT[k]]);
      for (const [b, col] of groups) {
        ctx.beginPath();
        let any = false;
        for (let i = 0; i < n; i++) {
          if (BIN[i] !== b) continue;
          any = true;
          const r = R[i];
          ctx.moveTo(X[i] + r, Y[i]);
          ctx.arc(X[i], Y[i], r, 0, Math.PI * 2);
        }
        if (any) { ctx.fillStyle = col; ctx.fill(); }
      }
      // heavy species get a subtle rim
      if (P.two && P.colorBy === 'species') {
        ctx.strokeStyle = 'rgba(11,14,20,0.6)'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < n; i++) if (SP[i] === 1) { ctx.moveTo(X[i] + R[i], Y[i]); ctx.arc(X[i], Y[i], R[i], 0, Math.PI * 2); }
        ctx.stroke();
      }
    }

    function drawBrownian() {
      if (bigIdx < 0) return;
      if (P.trail && trail.length > 1) {
        const L = trail.length, chunk = 40;
        ctx.lineWidth = 1.5; ctx.lineJoin = 'round';
        for (let s = 0; s < L - 1; s += chunk) {
          const a = 0.12 + 0.8 * (s / L);
          ctx.strokeStyle = `rgba(245,181,68,${a.toFixed(3)})`;
          ctx.beginPath(); ctx.moveTo(trail[s][0], trail[s][1]);
          for (let k = s + 1; k <= Math.min(L - 1, s + chunk); k++) ctx.lineTo(trail[k][0], trail[k][1]);
          ctx.stroke();
        }
        ctx.fillStyle = C.amber;
        ctx.beginPath(); ctx.arc(trail[0][0], trail[0][1], 3, 0, Math.PI * 2); ctx.fill();
      }
      const b = bigIdx, r = R[b];
      ctx.save();
      ctx.shadowColor = 'rgba(245,181,68,0.55)'; ctx.shadowBlur = 18;
      const gr = ctx.createRadialGradient(X[b] - r * 0.35, Y[b] - r * 0.35, r * 0.1, X[b], Y[b], r);
      gr.addColorStop(0, '#fff4dc'); gr.addColorStop(0.55, '#f5c46a'); gr.addColorStop(1, '#b97a1c');
      ctx.fillStyle = gr;
      ctx.beginPath(); ctx.arc(X[b], Y[b], r, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    function drawPiston() {
      const { y0, y1 } = box;
      const x = pistonX;
      const active = dragging;
      ctx.fillStyle = active ? '#ffd07a' : C.amber;
      ctx.fillRect(x, y0, 5, y1 - y0);
      const hy = (y0 + y1) / 2, hh = Math.min(56, (y1 - y0) * 0.3);
      rr(x - 3, hy - hh / 2, 12, hh, 4);
      ctx.fill();
      ctx.strokeStyle = 'rgba(11,14,20,0.55)'; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = -1; k <= 1; k++) { ctx.moveTo(x - 0.5 + 3 * (k + 1) + 0.5, hy - hh / 2 + 8); ctx.lineTo(x - 0.5 + 3 * (k + 1) + 0.5, hy + hh / 2 - 8); }
      ctx.stroke();
      if (!pistonTouched) {
        const pulse = 0.55 + 0.45 * Math.sin(performance.now() / 350);
        ctx.font = `11px ${MONO}`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillStyle = `rgba(245,181,68,${pulse.toFixed(3)})`;
        ctx.fillText('◀ drag piston', x - 10, hy);
      }
    }

    function drawHud() {
      ctx.font = `11px ${MONO}`; ctx.textBaseline = 'bottom';
      ctx.textAlign = 'left'; ctx.fillStyle = C.muted;
      const T = kTnow / T_UNIT;
      const parts = [`N ${n}`, `T ${T.toFixed(1)}`, `φ ${(st.phi * 100).toFixed(1)}%`];
      if (Number.isFinite(Pmeas)) parts.push(`P ${Pmeas.toFixed(2)}`);
      ctx.fillText(parts.join('  ·  '), box.x0, box.y0 - 7);
      ctx.textAlign = 'right';
      ctx.fillText(`t ${simTime.toFixed(1)} s${running ? '' : '  ❚❚ paused'}`, box.x1, box.y0 - 7);
      if (P.g > 0) {
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = 'rgba(94,200,229,0.8)';
        ctx.fillText(`g ↓ ${P.g}`, box.x0 + 8, box.y1 - 6);
      }
    }

    function drawHist(p) {
      const two = P.two;
      const legend = two
        ? [['light', C.cyan], ['heavy', C.amber]]
        : [['sim', LUT[18]], ['M–B', C.text]];
      panelFrame(p, 'Speed distribution', legend);
      const L = p.x + 12, Rt = p.x + p.w - 10, T = p.y + 30, B = p.y + p.h - 24;
      const W = Rt - L, H = B - T;
      if (W < 20 || H < 20) return;
      const vr = histRange(), dv = vr / HB, bw = W / HB;
      const kT = kTnow;
      const pdf = (v, m) => ((m * v) / kT) * Math.exp((-m * v * v) / (2 * kT)) * dv; // prob per bin
      // y scale
      let ym = 0;
      for (let s = 0; s < (two ? 2 : 1); s++) for (let k = 0; k < HB; k++) ym = Math.max(ym, hist[s][k]);
      const mRange = two ? [1, P.ratio] : [1];
      for (const m of mRange) ym = Math.max(ym, pdf(Math.sqrt(kT / m), m));
      ym = Math.min(ym, 1);
      histYMax += ((ym * 1.12) - histYMax) * (histYMax < ym * 1.12 ? 0.5 : 0.04);
      const sy = H / Math.max(histYMax, 1e-6);
      // bars
      const species = two ? [0, 1] : [0];
      for (const s of species) {
        for (let k = 0; k < HB; k++) {
          const v = hist[s][k];
          if (v <= 0) continue;
          const bh = Math.min(H, v * sy);
          if (two) ctx.fillStyle = s === 0 ? 'rgba(94,200,229,0.45)' : 'rgba(245,181,68,0.45)';
          else ctx.fillStyle = cmap(((k + 0.5) * dv) / (3.2 * Math.sqrt(KT_REF)), 0.85);
          ctx.fillRect(L + k * bw + 0.5, B - bh, Math.max(1, bw - 1), bh);
        }
      }
      // theory curves
      const curves = two ? [[1, '#9be3f3'], [P.ratio, '#ffd07a']] : [[1, C.text]];
      for (const [m, col] of curves) {
        ctx.beginPath();
        for (let px = 0; px <= W; px += 2) {
          const v = (px / W) * vr;
          const y = B - Math.min(H + 4, pdf(v, m) * sy);
          if (px === 0) ctx.moveTo(L + px, y); else ctx.lineTo(L + px, y);
        }
        ctx.strokeStyle = col; ctx.lineWidth = 1.75; ctx.stroke();
      }
      // start-speed marker
      if (v0Start > 0 && v0Start < vr && simTime < 8) {
        const x = L + (v0Start / vr) * W;
        ctx.globalAlpha = clamp(1 - simTime / 8, 0, 1);
        ctx.strokeStyle = C.danger; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, B); ctx.stroke(); ctx.setLineDash([]);
        ctx.font = `10px ${MONO}`; ctx.fillStyle = C.danger; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText('start: all v = v₀', x + 4, T);
        ctx.globalAlpha = 1;
      }
      // axis
      ctx.strokeStyle = C.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L, B + 0.5); ctx.lineTo(Rt, B + 0.5); ctx.stroke();
      const vp = Math.sqrt(kT);                  // most probable speed (light species)
      const xp = L + (vp / vr) * W;
      ctx.beginPath(); ctx.moveTo(xp, B); ctx.lineTo(xp, B + 4); ctx.stroke();
      ctx.font = `10px ${MONO}`; ctx.fillStyle = C.muted; ctx.textBaseline = 'top';
      ctx.textAlign = 'center'; ctx.fillText('v_p', xp, B + 5);
      ctx.textAlign = 'left'; ctx.fillText('0', L, B + 5);
      ctx.textAlign = 'right'; ctx.fillText(`${Math.round(vr)} px/s`, Rt, B + 5);
      if (Number.isFinite(mbDist) && !two) {
        ctx.textAlign = 'right'; ctx.textBaseline = 'top';
        ctx.fillStyle = mbDist < 0.1 ? C.good : C.amber;
        ctx.fillText(`distance from M–B ${(mbDist * 100).toFixed(0)}%`, Rt, T - 2);
      }
    }

    function drawDensity(p) {
      const two = P.two;
      panelFrame(p, 'Density vs height', [['sim', C.cyan], ['e^(−mgh/kT)', C.amber]]);
      const L = p.x + 12, Rt = p.x + p.w - 12, T = p.y + 30, B = p.y + p.h - 22;
      const W = Rt - L, H = B - T;
      if (W < 20 || H < 20) return;
      const Hb = box.y1 - box.y0, bh = H / DB;
      const theory = (m) => {
        const beta = (m * P.g) / Math.max(kTnow, 1e-6);
        const out = new Float64Array(DB);
        const Z = 1 - Math.exp(-beta * Hb);
        for (let k = 0; k < DB; k++) {
          const ha = (k / DB) * Hb, hb = ((k + 1) / DB) * Hb;
          out[k] = Z > 1e-9 ? (Math.exp(-beta * ha) - Math.exp(-beta * hb)) / Z : 1 / DB;
        }
        return out;
      };
      const th = [theory(1), two ? theory(P.ratio) : null];
      let xm = 0;
      for (let s = 0; s < (two ? 2 : 1); s++) for (let k = 0; k < DB; k++) xm = Math.max(xm, dens[s][k], th[s][k]);
      const sx = W / (xm * 1.1 || 1);
      for (let s = 0; s < (two ? 2 : 1); s++) {
        ctx.fillStyle = s === 0 ? 'rgba(94,200,229,0.5)' : 'rgba(245,181,68,0.4)';
        for (let k = 0; k < DB; k++) {
          const y = B - (k + 1) * bh;
          ctx.fillRect(L, y + 0.5, dens[s][k] * sx, Math.max(1, bh - 1));
        }
        ctx.beginPath();
        for (let k = 0; k < DB; k++) {
          const y = B - (k + 0.5) * bh, x = L + th[s][k] * sx;
          if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = s === 0 ? C.amber : '#ffd07a'; ctx.lineWidth = 1.75; ctx.stroke();
      }
      ctx.strokeStyle = C.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L - 0.5, T); ctx.lineTo(L - 0.5, B); ctx.stroke();
      ctx.font = `10px ${MONO}`; ctx.fillStyle = C.muted; ctx.textBaseline = 'top'; ctx.textAlign = 'right';
      const Hs = kTnow / Math.max(P.g, 1e-6);
      ctx.fillText(`scale height kT/mg = ${Math.round(Hs)} px`, Rt, B + 5);
      ctx.textAlign = 'left'; ctx.fillText('floor', L, B + 5);
    }

    function drawSeries(p) {
      const mode = currentSeriesMode();
      const cfg = {
        p: ['Pressure', [['measured', C.amber], ['NkT/A', C.cyan, true]]],
        lr: ['Temperature by side', [['left', C.amber], ['right', C.cyan]]],
        ke: ['⟨KE⟩ per species', [['light', C.cyan], ['heavy', C.amber]]],
        mix: ['Mixing entropy S/Smax', [['S', C.good]]],
      }[mode];
      panelFrame(p, cfg[0], cfg[1]);
      const L = p.x + 12, Rt = p.x + p.w - 12, T = p.y + 32, B = p.y + p.h - 20;
      const W = Rt - L, H = B - T;
      if (W < 20 || H < 20) return;
      ctx.strokeStyle = C.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L, B + 0.5); ctx.lineTo(Rt, B + 0.5); ctx.stroke();
      if (series.length < 2) {
        ctx.font = `10px ${MONO}`; ctx.fillStyle = C.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('collecting…', L + W / 2, T + H / 2);
        return;
      }
      let lo = Infinity, hi = -Infinity;
      for (const s of series) for (const v of s) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      if (mode === 'mix') { lo = 0; hi = 1; }
      else { const pad = (hi - lo) * 0.15 + Math.abs(hi) * 0.05 + 1e-6; lo = Math.max(0, lo - pad); hi += pad; }
      const colors = mode === 'mix' ? [C.good] : mode === 'ke' ? [C.cyan, C.amber] : [C.amber, C.cyan];
      const dashes = mode === 'p' ? [[], [4, 3]] : [[], []];
      const nS = series[0].length;
      for (let c = 0; c < nS; c++) {
        ctx.beginPath();
        let started = false;
        for (let k = 0; k < series.length; k++) {
          const v = series[k][c];
          if (!Number.isFinite(v)) continue;
          const x = L + (k / (SERIES_MAX - 1)) * W, y = B - ((v - lo) / (hi - lo)) * H;
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = colors[c]; ctx.lineWidth = 1.75; ctx.setLineDash(dashes[c]); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.font = `10px ${MONO}`; ctx.fillStyle = C.muted; ctx.textBaseline = 'top';
      ctx.textAlign = 'left'; ctx.fillText(`${(SERIES_MAX / 8).toFixed(0)} s window`, L, B + 4);
      ctx.textAlign = 'right';
      const last = series[series.length - 1];
      ctx.fillStyle = colors[0];
      ctx.fillText(last.map((v) => (Number.isFinite(v) ? v.toFixed(mode === 'mix' ? 2 : 1) : '—')).join('  /  '), Rt, B + 4);
      ctx.textBaseline = 'top'; ctx.fillStyle = C.muted;
      ctx.fillText(hi.toFixed(mode === 'mix' ? 1 : 1), Rt, T - 2);
    }

    function draw() {
      if (!ready) return;
      const { w, h: Hc } = size;
      ctx.clearRect(0, 0, w, Hc);
      drawBox();
      ctx.save();
      ctx.beginPath(); ctx.rect(box.x0, box.y0, pistonX - box.x0, box.y1 - box.y0); ctx.clip();
      drawBrownian();
      drawParticles();
      ctx.restore();
      drawDivider();
      drawPiston();
      drawHud();
      // side panels
      const s = layout.side, gap = 12;
      const showDens = P.g > 0;
      if (layout.wide) {
        const hHist = showDens ? s.h * 0.4 : s.h * 0.56;
        const hDen = showDens ? s.h * 0.3 : 0;
        drawHist({ x: s.x, y: s.y, w: s.w, h: hHist - gap / 2 });
        if (showDens) drawDensity({ x: s.x, y: s.y + hHist + gap / 2, w: s.w, h: hDen - gap });
        drawSeries({ x: s.x, y: s.y + hHist + hDen + gap / 2, w: s.w, h: s.h - hHist - hDen - gap / 2 });
      } else {
        const w1 = s.w * 0.58;
        drawHist({ x: s.x, y: s.y, w: w1 - gap / 2, h: s.h });
        const p2 = { x: s.x + w1 + gap / 2, y: s.y, w: s.w - w1 - gap / 2, h: s.h };
        if (showDens) drawDensity(p2); else drawSeries(p2);
      }
    }

    // ---------------------------------------------------------------- controls
    const sc = section(panel, 'Scenario');
    const presetSel = select(sc, {
      label: 'Preset',
      options: Object.entries(PRESETS).map(([k, v]) => [k, v.label]),
      value: 'relax',
      onChange: (k) => applyPreset(k),
    });
    const [playBtn] = buttons(sc, [
      { label: 'Pause', primary: true, onClick: () => { running = !running; playBtn.textContent = running ? 'Pause' : 'Play'; } },
      { label: 'Reset', onClick: () => reset() },
    ]);
    const hintEl = h('p', { class: 'hint' });
    sc.append(hintEl);

    const sg = section(panel, 'Gas');
    const sN = slider(sg, { label: 'Particles N', min: 10, max: 2000, step: 10, value: P.N, onInput: (v) => { P.N = v; reset(); } });
    const sRad = slider(sg, { label: 'Radius', min: 1.5, max: 8, step: 0.5, value: P.radius, format: (v) => `${v} px`, onInput: (v) => { P.radius = v; applyRadii(); } });
    const sT = slider(sg, { label: 'Temperature', min: 10, max: 400, step: 5, value: P.T, onInput: (v) => { P.T = v; rescaleTo(v * T_UNIT); } });
    const heatBy = (f) => { scaleVel(Math.sqrt(f)); P.T = clamp(Math.round(kTnow / T_UNIT), 10, 400); sT.set(P.T); };
    buttons(sg, [
      { label: '▲ Heat ×1.25', onClick: () => heatBy(1.25) },
      { label: '▼ Cool ×0.8', onClick: () => heatBy(0.8) },
    ]);

    const ss = section(panel, 'Species');
    const cTwo = checkbox(ss, { label: 'Two species (half heavy)', checked: P.two, onChange: (c) => { P.two = c; if (c && P.colorBy === 'speed') { P.colorBy = 'species'; selColor.value = 'species'; } reset(); } });
    const sRatio = slider(ss, {
      label: 'Mass ratio heavy : light', min: 1, max: 20, step: 0.5, value: P.ratio, format: (v) => `${v} : 1`,
      onInput: (v) => { P.ratio = v; for (let i = 0; i < n; i++) if (SP[i] === 1) setMass(i); },
    });

    const sf = section(panel, 'Forces & walls');
    const sG = slider(sf, { label: 'Gravity g', min: 0, max: 300, step: 5, value: P.g, format: (v) => (v ? `${v} px/s²` : 'off'), onInput: (v) => { P.g = v; densFresh = true; } });
    const cDiv = checkbox(sf, { label: 'Central divider', checked: P.divider, onChange: (c) => { P.divider = c; pistonTarget = Math.max(pistonTarget, pistonMin()); } });
    const cGap = checkbox(sf, { label: 'Gap open', checked: P.gapOpen, onChange: (c) => { P.gapOpen = c; } });
    const sGap = slider(sf, { label: 'Gap size', min: 0.04, max: 0.9, step: 0.02, value: P.gap, format: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { P.gap = v; } });
    const cDemon = checkbox(sf, { label: "Maxwell's demon at the gap", checked: P.demon, onChange: (c) => { P.demon = c; } });
    buttons(sf, [{ label: 'Reset piston', onClick: () => { pistonTarget = box.x1; if (!running) nudgePistonPaused(); } }]);
    sf.append(h('p', { class: 'hint' }, 'Drag the amber right wall to compress the gas. Pushing a moving wall into molecules speeds them up: adiabatic heating. Pull it back and the gas cools.'));

    const sv = section(panel, 'View');
    const selColor = select(sv, {
      label: 'Colour by',
      options: [['speed', 'Speed (cool → hot)'], ['species', 'Species'], ['tag', 'Starting side']],
      value: P.colorBy,
      onChange: (v) => { P.colorBy = v; },
    });
    const sSpeed = slider(sv, { label: 'Sim speed', min: 0.25, max: 3, step: 0.25, value: P.speed, format: (v) => `${v}×`, onInput: (v) => { P.speed = v; } });
    const cTrail = checkbox(sv, { label: 'Brownian trail', checked: P.trail, onChange: (c) => { P.trail = c; } });
    buttons(sv, [{ label: 'Clear trail', onClick: () => { trail = []; } }]);

    const sm = section(panel, 'Measurements');
    const ro = readout(sm, ['kT = ⟨KE⟩', 'P (wall impulse)', 'P ideal NkT/A', 'Z = PA/NkT', 'Z hard-disk', 'Area frac φ', '⟨KE⟩ light', '⟨KE⟩ heavy', 'T left', 'T right', 'Mixing S/Smax', 'M–B distance', 'Substeps']);
    sm.append(h('p', { class: 'hint', style: 'margin-top:10px' }, 'In 2D each particle has two kinetic degrees of freedom, so ⟨KE⟩ = kT and the ideal gas law reads PA = NkT. Finite disk size raises P above ideal; the hard-disk estimate is Henderson’s (1 + φ²/8)/(1 − φ)².'));

    const HINTS = {
      relax: 'Every particle starts with the same speed (red dashed line). Watch collisions spread it into the Maxwell–Boltzmann curve within a second or two.',
      species: 'Heavy and light disks start at the same speed. Collisions trade energy until both have the same mean KE (equipartition), even though heavy ones end up slower.',
      diffusion: 'Two gases at the same temperature, one on each side of an open gap. Mixing is irreversible: S/Smax climbs to 1 and never comes back.',
      hotcold: 'Hot gas on the left, cold on the right. Heat leaks through the gap until the two temperatures meet.',
      demon: 'The demon opens the gap only for fast molecules going right and slow ones going left. The right side heats and the left cools with no work done, which is why the demon needs to pay for its information.',
      atmosphere: 'Gravity pulls the gas down. The density falls off exponentially with height (barometric formula) while temperature stays uniform.',
      brownian: 'One big, heavy disk gets knocked about by many small invisible-scale ones. Its trail is a random walk: Brownian motion, as Einstein explained it in 1905.',
    };

    function syncControls() {
      presetSel.value = P.preset;
      sN.set(P.N); sRad.set(P.radius); sT.set(P.T); sRatio.set(P.ratio); sG.set(P.g);
      cTwo.checked = P.two; cDiv.checked = P.divider; cGap.checked = P.gapOpen; sGap.set(P.gap);
      cDemon.checked = P.demon; selColor.value = P.colorBy; sSpeed.set(P.speed); cTrail.checked = P.trail;
      hintEl.textContent = HINTS[P.preset] || '';
    }
    function applyPreset(key) {
      Object.assign(P, BASE, PRESETS[key], { preset: key });
      syncControls();
      reset();
      if (!running) draw();
    }
    syncControls();

    function updateReadouts() {
      const A = (pistonX - box.x0) * (box.y1 - box.y0);
      const Pid = (n * kTnow) / A;
      const phi = st.phi;
      const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
      ro.set('kT = ⟨KE⟩', `${fmt(kTnow / T_UNIT, 1)}`);
      ro.set('P (wall impulse)', fmt(Pmeas));
      ro.set('P ideal NkT/A', fmt(Pid));
      ro.set('Z = PA/NkT', fmt(Pmeas / Pid, 3));
      ro.set('Z hard-disk', fmt((1 + (phi * phi) / 8) / ((1 - phi) * (1 - phi)), 3));
      ro.set('Area frac φ', `${fmt(phi * 100, 1)}%`);
      ro.set('⟨KE⟩ light', st.c0 ? fmt(st.ke0 / st.c0 / T_UNIT, 1) : '—');
      ro.set('⟨KE⟩ heavy', st.c1 ? fmt(st.ke1 / st.c1 / T_UNIT, 1) : '—');
      ro.set('T left', P.divider && st.cL ? fmt(st.keL / st.cL / T_UNIT, 1) : '—');
      ro.set('T right', P.divider && st.cR ? fmt(st.keR / st.cR / T_UNIT, 1) : '—');
      ro.set('Mixing S/Smax', P.divider || P.colorBy === 'tag' ? fmt(mixingEntropy(), 3) : '—');
      ro.set('M–B distance', `${fmt(mbDist * 100, 1)}%`);
      ro.set('Substeps', `${lastSub}`);
    }

    // ---------------------------------------------------------------- main loop
    let roClock = 0, trailClock = 0;
    const stop = loop((dt) => {
      if (!ready) return;
      if (running) {
        step(dt);
        computeStats();
        kTs += (kTnow - kTs) * 0.05;
        updateHist();
        updateDensity();
        seriesClock += dt;
        if (seriesClock >= 0.125) { seriesClock = 0; sampleSeries(); }
        if (bigIdx >= 0) {
          trailClock += dt * P.speed;
          if (trailClock >= 1 / 30) {
            trailClock = 0;
            trail.push([X[bigIdx], Y[bigIdx]]);
            if (trail.length > 3000) trail.shift();
          }
        }
      }
      roClock += dt;
      if (roClock > 0.2) { roClock = 0; updateReadouts(); }
      draw();
    });

    return () => {
      stop();
      cv.destroy();
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
    };
  },
};
