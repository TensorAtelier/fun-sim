// Galton board (bean machine): balls hop peg-to-peg, choosing right with probability p,
// and pile up into a binomial distribution B(n, p) — which the normal curve approximates.
import { createLayout, section, slider, select, checkbox, buttons, readout, createCanvas, loop, h } from '../ui.js';

const C = {
  amber: '#f5b544',
  amberDim: '#d99a36',
  cyan: '#5ec8e5',
  muted: '#8a93a6',
  line: '#242c3b',
  good: '#7bd88f',
  danger: '#ef6b6b',
  text: '#e6e9ef',
  peg: '#3a4458',
};
const MONO = '"IBM Plex Mono", ui-monospace, monospace';

// ---------- math helpers ----------
function choose(n, k) {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return Math.round(r);
}
function binomPmf(n, p) {
  const out = [];
  for (let k = 0; k <= n; k++) out.push(choose(n, k) * p ** k * (1 - p) ** (n - k));
  return out;
}
function lnGamma(z) {
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
// Regularized upper incomplete gamma Q(a, x) -> chi-square survival function.
function gammaQ(a, x) {
  if (x <= 0) return 1;
  const gln = lnGamma(a);
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let i = 0; i < 200; i++) {
      ap += 1; del *= x / ap; sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-12) break;
    }
    return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  let b = x + 1 - a, c = 1e300, d = 1 / b, hh = d;
  for (let i = 1; i < 200; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const del = d * c;
    hh *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  return Math.exp(-x + a * Math.log(x) - gln) * hh;
}
// Pearson chi-square with adjacent bins pooled until each expected count >= 5.
function chiSquare(obs, exp) {
  const groups = [];
  let o = 0, e = 0;
  for (let k = 0; k < obs.length; k++) {
    o += obs[k]; e += exp[k];
    if (e >= 5) { groups.push([o, e]); o = 0; e = 0; }
  }
  if (e > 0 || o > 0) {
    if (groups.length) { groups[groups.length - 1][0] += o; groups[groups.length - 1][1] += e; }
    else groups.push([o, e]);
  }
  if (groups.length < 2) return null;
  let chi = 0;
  for (const [go, ge] of groups) chi += (go - ge) ** 2 / ge;
  const df = groups.length - 1;
  return { chi, df, p: gammaQ(df / 2, chi / 2) };
}
const compact = (v) => {
  if (v < 1000) return String(v);
  if (v < 1e4) return (v / 1e3).toFixed(1) + 'k';
  if (v < 1e6) return Math.round(v / 1e3) + 'k';
  if (v < 1e7) return (v / 1e6).toFixed(1) + 'M';
  return Math.round(v / 1e6) + 'M';
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const rateFromSlider = (s) => (s <= 0 ? 0 : Math.round(10 ** ((s / 100) * Math.log10(500))));

export default {
  id: 'galton',
  title: 'Galton board',
  glyph: '∴',
  tag: 'statistics',
  blurb: 'Thousands of beads bounce left or right through a triangle of pegs and pile up into a bell curve — the binomial distribution, and the central limit theorem, built out of coin flips.',

  mount(root) {
    const { stage, panel } = createLayout(root, {
      title: 'Galton board',
      desc: 'Each bead meets one peg per row and goes right with probability p. The bin it lands in counts its right-turns, so the piles follow B(n, p) — which, for many rows, looks like a normal curve.',
    });

    // ---------- state ----------
    const S = {
      n: 12,
      p: 0.5,
      rate: rateFromSlider(45),
      speed: 1,
      showBinom: true,
      showNormal: true,
      showMeans: true,
      heatmap: false,
      labels: 'off',
    };
    let bins, reserved, pegHits, glow, balls, N, barMode, dispPpc, spawnAcc, pending, pmf;

    function reset() {
      bins = new Array(S.n + 1).fill(0);
      reserved = new Array(S.n + 1).fill(0);
      pegHits = Array.from({ length: S.n }, (_, r) => new Array(r + 1).fill(0));
      glow = Array.from({ length: S.n }, (_, r) => new Float32Array(r + 1));
      balls = [];
      N = 0;
      barMode = false;
      dispPpc = 0;
      spawnAcc = 0;
      pending = 0;
      pmf = binomPmf(S.n, S.p);
      updateReadout();
    }

    // ---------- layout ----------
    const L = { w: 0, h: 0, cx: 0, dx: 20, dy: 20, R: 3, pegR: 2, funnelTop: 0, funnelW: 40, pegTop: 0, binTop: 0, binBottom: 0, binH: 0, cols: 1 };
    function computeLayout(w, hgt) {
      const n = S.n;
      L.w = w; L.h = hgt;
      const padX = 28, top = 18, bottomPad = 34;
      const avail = Math.max(60, hgt - top - bottomPad);
      let dy = (avail * 0.62) / (n + 1.1);
      let dx = Math.min((w - 2 * padX) / (n + 1), dy * 1.3);
      dy = Math.min(dy, dx * 1.05);
      L.dx = Math.max(2, dx);
      L.dy = Math.max(2, dy);
      L.cx = w / 2;
      L.R = clamp(L.dx * 0.16, 1.6, 6);
      L.pegR = clamp(L.dx * 0.085, 1.4, 4.5);
      L.funnelTop = top;
      L.funnelW = Math.min(L.dx * 2.2, w * 0.18);
      L.pegTop = top + L.dy * 1.35;
      L.binTop = L.pegTop + (n - 1) * L.dy + L.dy * 0.75;
      L.binBottom = hgt - bottomPad;
      L.binH = Math.max(10, L.binBottom - L.binTop);
      L.cols = Math.max(1, Math.floor((L.dx * 0.92) / (2 * L.R)));
    }
    const pegX = (r, k) => L.cx + (k - r / 2) * L.dx;
    const pegY = (r) => L.pegTop + r * L.dy;
    const binX = (k) => L.cx + (k - S.n / 2) * L.dx;
    const valX = (v) => L.cx + (v - S.n / 2) * L.dx;
    const restX = (r, k, j) => pegX(r, k) + j * L.dx;
    const restY = (r) => pegY(r) - L.pegR - L.R * 0.9;

    // Pixels per count in the bins (circle stacks: one level of `cols` balls per 2R).
    function currentPpc() { return barMode ? dispPpc : (2 * L.R) / L.cols; }
    function slotPos(bin, slot) {
      if (!barMode) {
        const col = slot % L.cols, lvl = Math.floor(slot / L.cols);
        const off = L.cols > 1 && lvl % 2 ? L.R * 0.35 : 0;
        return [binX(bin) + (col - (L.cols - 1) / 2) * 2 * L.R + off, L.binBottom - L.R - lvl * 2 * L.R];
      }
      return [binX(bin), L.binBottom - Math.min(L.binH, bins[bin] * dispPpc) - L.R];
    }

    // ---------- balls ----------
    const HOP = 0.13; // seconds per row at 1x
    const jit = () => (Math.random() - 0.5) * 0.07;
    function spawn() {
      balls.push({ ph: 0, r: 0, k: 0, t: 0, dur: HOP * 1.6, fx: (Math.random() - 0.5) * 0.9, j0: 0, j1: jit(), hop: 0, k2: 0, bin: 0, slot: 0 });
    }
    function hitPeg(r, k) {
      pegHits[r][k]++;
      glow[r][k] = Math.min(1, glow[r][k] + 0.6);
    }
    // Ball has arrived on peg (b.r, b.k): choose a direction and start the next leg.
    function nextLeg(b) {
      hitPeg(b.r, b.k);
      const dir = Math.random() < S.p ? 1 : 0;
      b.j0 = b.j1;
      if (b.r === S.n - 1) {
        b.ph = 2;
        b.bin = b.k + dir;
        b.slot = bins[b.bin] + reserved[b.bin];
        reserved[b.bin]++;
        const [, ty] = slotPos(b.bin, b.slot);
        const dist = Math.max(0, ty - restY(b.r));
        b.dur = HOP * (1 + Math.sqrt(dist / L.dy) * 0.7);
      } else {
        b.ph = 1;
        b.k2 = b.k + dir;
        b.j1 = jit();
        b.hop = 0.18 + Math.random() * 0.14;
        b.dur = HOP * (0.9 + Math.random() * 0.2);
      }
    }
    function ballPos(b) {
      const t = b.t;
      if (b.ph === 0) {
        const x0 = L.cx + b.fx * L.funnelW * 0.5, y0 = L.funnelTop + L.R;
        const x1 = restX(0, 0, b.j1), y1 = restY(0);
        return [x0 + (x1 - x0) * easeInOut(t), y0 + (y1 - y0) * t * t];
      }
      if (b.ph === 1) {
        const x0 = restX(b.r, b.k, b.j0), y0 = restY(b.r);
        const x1 = restX(b.r + 1, b.k2, b.j1), y1 = restY(b.r + 1);
        return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t - L.dy * b.hop * 4 * t * (1 - t)];
      }
      const x0 = restX(b.r, b.k, b.j0), y0 = restY(b.r);
      const [x1, y1] = slotPos(b.bin, b.slot);
      return [x0 + (x1 - x0) * (1 - (1 - t) ** 3), y0 + (y1 - y0) * t * t - L.dy * 0.2 * 4 * t * (1 - t) * (1 - t)];
    }
    function stepBalls(dt) {
      for (let i = balls.length - 1; i >= 0; i--) {
        const b = balls[i];
        b.t += (dt * S.speed) / b.dur;
        while (b.t >= 1) {
          const over = (b.t - 1) * b.dur;
          if (b.ph === 0) { b.r = 0; b.k = 0; nextLeg(b); }
          else if (b.ph === 1) { b.r += 1; b.k = b.k2; nextLeg(b); }
          else {
            reserved[b.bin]--;
            bins[b.bin]++;
            N++;
            balls[i] = balls[balls.length - 1];
            balls.pop();
            break;
          }
          b.t = over / b.dur;
        }
      }
    }
    // Skip the animation: run many balls straight through the board.
    function dropInstant(count) {
      const n = S.n, p = S.p;
      for (let c = 0; c < count; c++) {
        let k = 0;
        for (let r = 0; r < n; r++) {
          pegHits[r][k]++;
          if (Math.random() < p) k++;
        }
        bins[k]++;
      }
      N += count;
      // flash the board in proportion to where the burst went
      for (let r = 0; r < n; r++) {
        let m = 1;
        for (const v of pegHits[r]) m = Math.max(m, v);
        for (let k = 0; k <= r; k++) glow[r][k] = Math.min(1, glow[r][k] + (pegHits[r][k] / m) * 0.9);
      }
    }

    // ---------- controls ----------
    const board = section(panel, 'Board');
    slider(board, {
      label: 'Rows (n)', min: 4, max: 24, step: 1, value: S.n,
      onInput: (v) => { S.n = v; computeLayout(L.w, L.h); reset(); },
    });
    slider(board, {
      label: 'Bias p (go right)', min: 0.05, max: 0.95, step: 0.01, value: S.p, format: (v) => v.toFixed(2),
      onInput: (v) => { S.p = v; reset(); },
    });

    const drop = section(panel, 'Dropping');
    slider(drop, {
      label: 'Drop rate', min: 0, max: 100, step: 1, value: 45,
      format: (s) => { const r = rateFromSlider(s); return r ? `${r} /s` : 'paused'; },
      onInput: (s) => { S.rate = rateFromSlider(s); },
    });
    slider(drop, {
      label: 'Animation speed', min: 0.25, max: 4, step: 0.25, value: 1, format: (v) => `${v.toFixed(2)}×`,
      onInput: (v) => { S.speed = v; },
    });
    buttons(drop, [
      { label: 'Drop 1', onClick: () => spawn() },
      { label: 'Drop 100', primary: true, onClick: () => { pending += 100; } },
      { label: 'Drop 1000', onClick: () => dropInstant(1000) },
      { label: 'Reset', onClick: () => reset() },
    ]);
    drop.append(h('div', { class: 'hint' }, 'Drop 1000 fast-forwards: the beads skip the animation and land instantly.'));

    const ov = section(panel, 'Overlays');
    checkbox(ov, { label: 'Exact binomial B(n, p)', checked: S.showBinom, onChange: (v) => { S.showBinom = v; } });
    checkbox(ov, { label: 'Normal approx. N(np, np(1−p))', checked: S.showNormal, onChange: (v) => { S.showNormal = v; } });
    checkbox(ov, { label: 'Mean markers', checked: S.showMeans, onChange: (v) => { S.showMeans = v; } });
    checkbox(ov, { label: 'Peg heatmap (hit frequency)', checked: S.heatmap, onChange: (v) => { S.heatmap = v; } });
    select(ov, {
      label: "Peg labels (Pascal's triangle)",
      value: S.labels,
      options: [['off', 'Off'], ['pascal', "Pascal's triangle C(r, k)"], ['hits', 'Observed hit counts'], ['prob', 'P(reach peg) — theory']],
      onChange: (v) => { S.labels = v; },
    });

    const st = section(panel, 'Statistics');
    const ro = readout(st, ['Balls dropped', 'In flight', 'Mean', 'np', 'Std dev', '√(np(1−p))', 'χ² / df', 'p-value', 'Fit']);
    let roTimer = 0;
    function updateReadout() {
      if (!ro) return;
      roTimer = 0;
      const n = S.n, p = S.p;
      ro.set('Balls dropped', N.toLocaleString());
      ro.set('In flight', (balls ? balls.length : 0) + (pending ? ` (+${pending})` : ''));
      ro.set('np', (n * p).toFixed(3));
      ro.set('√(np(1−p))', Math.sqrt(n * p * (1 - p)).toFixed(3));
      if (N > 0) {
        let s = 0, s2 = 0;
        for (let k = 0; k <= n; k++) { s += k * bins[k]; s2 += k * k * bins[k]; }
        const m = s / N;
        const sd = N > 1 ? Math.sqrt(Math.max(0, (s2 - N * m * m) / (N - 1))) : 0;
        ro.set('Mean', m.toFixed(3));
        ro.set('Std dev', N > 1 ? sd.toFixed(3) : '—');
      } else { ro.set('Mean', '—'); ro.set('Std dev', '—'); }
      const cs = N >= 20 ? chiSquare(bins, pmf.map((q) => q * N)) : null;
      if (cs) {
        ro.set('χ² / df', `${cs.chi.toFixed(2)} / ${cs.df}`);
        ro.set('p-value', cs.p < 0.001 ? '< 0.001' : cs.p.toFixed(3));
        ro.set('Fit', cs.p > 0.05 ? 'consistent ✓' : cs.p > 0.001 ? 'marginal' : 'poor ✗');
      } else {
        ro.set('χ² / df', '—'); ro.set('p-value', '—'); ro.set('Fit', N ? 'need ≥ 20 balls' : '—');
      }
    }
    panel.append(h('p', { class: 'hint' },
      'Try p = 0.5 with the Pascal labels on: hit counts on each peg converge to C(r, k)/2ʳ of the total. Skew p and the pile slides to np while staying bell-shaped.'));

    // ---------- canvas ----------
    const cv = createCanvas(stage, (w, hh) => computeLayout(w, hh));
    const ctx = cv.ctx;
    reset();

    function heatColor(v) {
      // line-gray -> cyan -> amber
      const a = [58, 68, 88], b = [94, 200, 229], c = [245, 181, 68];
      let from, to, t;
      if (v < 0.5) { from = a; to = b; t = v / 0.5; } else { from = b; to = c; t = (v - 0.5) / 0.5; }
      const m = (i) => Math.round(from[i] + (to[i] - from[i]) * t);
      return `rgb(${m(0)},${m(1)},${m(2)})`;
    }

    function draw() {
      const { w, h: H } = cv.size;
      ctx.clearRect(0, 0, w, H);
      if (w < 40 || H < 60) return;
      const n = S.n, p = S.p;
      const dx = L.dx, R = L.R;

      // scale for bins
      let maxObs = 0;
      for (let k = 0; k <= n; k++) maxObs = Math.max(maxObs, bins[k] + reserved[k]);
      let maxPmf = 0;
      for (const q of pmf) maxPmf = Math.max(maxPmf, q);
      const maxExp = maxPmf * Math.max(N, 1);
      if (!barMode) {
        const fitsCircles = Math.ceil(Math.max(maxObs, maxExp) / L.cols) * 2 * R <= L.binH * 0.94 && N <= 8000;
        if (!fitsCircles) { barMode = true; dispPpc = (2 * R) / L.cols; }
      }
      if (barMode) {
        const target = (L.binH * 0.88) / Math.max(1, maxObs, maxExp);
        // Shrink instantly so bars never clip; ease only when zooming back in.
        dispPpc = target < dispPpc ? target : dispPpc + (target - dispPpc) * 0.15;
        if (!isFinite(dispPpc) || dispPpc <= 0) dispPpc = target;
      }
      const ppc = currentPpc();

      // --- bin area
      const left = binX(0) - dx / 2, right = binX(n) + dx / 2;
      const g = ctx.createLinearGradient(0, L.binTop, 0, L.binBottom);
      g.addColorStop(0, 'rgba(94,200,229,0.0)');
      g.addColorStop(1, 'rgba(94,200,229,0.05)');
      ctx.fillStyle = g;
      ctx.fillRect(left, L.binTop, right - left, L.binH);
      ctx.strokeStyle = C.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k <= n + 1; k++) {
        const x = Math.round(left + k * dx) + 0.5;
        ctx.moveTo(x, L.binTop + 4);
        ctx.lineTo(x, L.binBottom);
      }
      ctx.stroke();
      ctx.strokeStyle = '#3a4458';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(left - 6, L.binBottom + 0.5);
      ctx.lineTo(right + 6, L.binBottom + 0.5);
      ctx.stroke();

      // --- stacks
      if (!barMode) {
        ctx.fillStyle = C.amberDim;
        ctx.beginPath();
        for (let k = 0; k <= n; k++) {
          for (let s = 0; s < bins[k]; s++) {
            const [x, y] = slotPos(k, s);
            ctx.moveTo(x + R * 0.92, y);
            ctx.arc(x, y, R * 0.92, 0, Math.PI * 2);
          }
        }
        ctx.fill();
        if (R > 2.5) {
          ctx.fillStyle = 'rgba(255,240,210,0.28)';
          ctx.beginPath();
          for (let k = 0; k <= n; k++) {
            for (let s = 0; s < bins[k]; s++) {
              const [x, y] = slotPos(k, s);
              ctx.moveTo(x - R * 0.3 + R * 0.35, y - R * 0.3);
              ctx.arc(x - R * 0.3, y - R * 0.3, R * 0.35, 0, Math.PI * 2);
            }
          }
          ctx.fill();
        }
      } else {
        const bw = dx * 0.78;
        for (let k = 0; k <= n; k++) {
          const hgt = Math.min(L.binH, bins[k] * ppc);
          if (hgt <= 0) continue;
          const x = binX(k) - bw / 2, y = L.binBottom - hgt;
          const bg = ctx.createLinearGradient(0, y, 0, L.binBottom);
          bg.addColorStop(0, 'rgba(245,181,68,0.95)');
          bg.addColorStop(1, 'rgba(245,181,68,0.35)');
          ctx.fillStyle = bg;
          ctx.fillRect(x, y, bw, hgt);
          ctx.fillStyle = '#ffd98a';
          ctx.fillRect(x, y, bw, Math.min(2, hgt));
        }
      }

      // --- overlays
      if (N > 0 && S.showBinom) {
        const bw = dx * 0.86;
        ctx.strokeStyle = C.cyan;
        ctx.fillStyle = 'rgba(94,200,229,0.07)';
        ctx.lineWidth = 1.25;
        ctx.setLineDash([3, 3]);
        for (let k = 0; k <= n; k++) {
          const hgt = Math.min(L.binH, pmf[k] * N * ppc);
          if (hgt < 0.5) continue;
          const x = binX(k) - bw / 2, y = L.binBottom - hgt;
          ctx.fillRect(x, y, bw, hgt);
          ctx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(bw), Math.round(hgt));
        }
        ctx.setLineDash([]);
        ctx.fillStyle = C.cyan;
        for (let k = 0; k <= n; k++) {
          const hgt = Math.min(L.binH, pmf[k] * N * ppc);
          ctx.beginPath();
          ctx.arc(binX(k), L.binBottom - hgt, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      if (N > 0 && S.showNormal) {
        const mu = n * p, sd = Math.sqrt(n * p * (1 - p));
        ctx.save();
        ctx.beginPath();
        ctx.rect(left - 20, L.binTop - 2, right - left + 40, L.binH + 3);
        ctx.clip();
        ctx.strokeStyle = C.good;
        ctx.lineWidth = 2;
        ctx.shadowColor = 'rgba(123,216,143,0.6)';
        ctx.shadowBlur = 8;
        ctx.beginPath();
        const steps = 240;
        for (let i = 0; i <= steps; i++) {
          const v = -0.5 + ((n + 1) * i) / steps;
          const dens = Math.exp(-0.5 * ((v - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
          const y = L.binBottom - dens * N * ppc;
          const x = valX(v);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
      }

      // --- pegs
      let rowMax = null;
      if (S.heatmap) {
        rowMax = pegHits.map((row) => Math.max(1, ...row));
      }
      for (let r = 0; r < n; r++) {
        const y = pegY(r);
        for (let k = 0; k <= r; k++) {
          const x = pegX(r, k);
          const gl = glow[r][k];
          if (gl > 0.02) {
            ctx.fillStyle = `rgba(245,181,68,${0.28 * gl})`;
            ctx.beginPath();
            ctx.arc(x, y, L.pegR + 2 + 5 * gl, 0, Math.PI * 2);
            ctx.fill();
          }
          let col = C.peg;
          let pr = L.pegR;
          if (S.heatmap && N + balls.length > 0) {
            const v = pegHits[r][k] / rowMax[r];
            col = heatColor(v);
            pr = L.pegR * (0.8 + 0.5 * v);
          }
          if (gl > 0.02 && !S.heatmap) {
            const a = Math.min(1, gl);
            col = `rgb(${Math.round(58 + (245 - 58) * a)},${Math.round(68 + (181 - 68) * a)},${Math.round(88 + (68 - 88) * a)})`;
          }
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.arc(x, y, pr, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      // peg labels
      if (S.labels !== 'off') {
        const fs = Math.min(11, dx * 0.27);
        if (fs >= 6.5) {
          ctx.font = `${fs.toFixed(1)}px ${MONO}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          for (let r = 0; r < n; r++) {
            const y = pegY(r) + L.pegR + 2;
            for (let k = 0; k <= r; k++) {
              let txt;
              if (S.labels === 'pascal') txt = compact(choose(r, k));
              else if (S.labels === 'hits') txt = compact(pegHits[r][k]);
              else {
                const q = choose(r, k) * p ** k * (1 - p) ** (r - k);
                txt = q >= 0.995 ? '1' : q < 0.01 ? '<.01' : q.toFixed(2).replace(/^0/, '');
              }
              ctx.fillStyle = S.labels === 'hits' ? 'rgba(245,181,68,0.85)' : 'rgba(230,233,239,0.72)';
              ctx.fillText(txt, pegX(r, k), y);
            }
          }
          // bin row of the triangle
          ctx.textBaseline = 'bottom';
          ctx.fillStyle = 'rgba(94,200,229,0.8)';
          for (let k = 0; k <= n; k++) {
            let txt;
            if (S.labels === 'pascal') txt = compact(choose(n, k));
            else if (S.labels === 'hits') txt = compact(bins[k]);
            else { const q = pmf[k]; txt = q >= 0.995 ? '1' : q < 0.01 ? '<.01' : q.toFixed(2).replace(/^0/, ''); }
            ctx.fillText(txt, binX(k), L.binTop - 1);
          }
        } else {
          ctx.font = `11px ${MONO}`;
          ctx.fillStyle = C.muted;
          ctx.textAlign = 'right';
          ctx.textBaseline = 'top';
          ctx.fillText('labels hidden — too many rows for this width', w - 14, 14);
        }
      }

      // --- funnel
      const fy1 = L.pegTop - L.dy * 0.7;
      const mouth = Math.max(R * 1.6, dx * 0.28);
      ctx.strokeStyle = '#4a5570';
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(L.cx - L.funnelW, L.funnelTop);
      ctx.lineTo(L.cx - mouth, fy1);
      ctx.moveTo(L.cx + L.funnelW, L.funnelTop);
      ctx.lineTo(L.cx + mouth, fy1);
      ctx.stroke();

      // --- flying balls
      if (balls.length) {
        ctx.fillStyle = C.amber;
        ctx.beginPath();
        const pos = new Float32Array(balls.length * 2);
        for (let i = 0; i < balls.length; i++) {
          const [x, y] = ballPos(balls[i]);
          pos[2 * i] = x; pos[2 * i + 1] = y;
          ctx.moveTo(x + R, y);
          ctx.arc(x, y, R, 0, Math.PI * 2);
        }
        ctx.fill();
        if (R > 2.5) {
          ctx.fillStyle = 'rgba(255,248,230,0.55)';
          ctx.beginPath();
          for (let i = 0; i < balls.length; i++) {
            const x = pos[2 * i] - R * 0.32, y = pos[2 * i + 1] - R * 0.32;
            ctx.moveTo(x + R * 0.32, y);
            ctx.arc(x, y, R * 0.32, 0, Math.PI * 2);
          }
          ctx.fill();
        }
      }

      // --- bin labels + mean markers
      ctx.font = `10px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = C.muted;
      const labelEvery = dx >= 16 ? 1 : dx >= 8 ? 2 : 4;
      for (let k = 0; k <= n; k += labelEvery) ctx.fillText(String(k), binX(k), L.binBottom + 5);
      if (S.showMeans) {
        const tri = (x, color, up) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          const y = L.binBottom + (up ? 18 : 2);
          ctx.moveTo(x, y);
          ctx.lineTo(x - 5, y + 7);
          ctx.lineTo(x + 5, y + 7);
          ctx.closePath();
          ctx.fill();
        };
        const mx = valX(n * p);
        ctx.strokeStyle = 'rgba(94,200,229,0.45)';
        ctx.setLineDash([2, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mx, L.binTop);
        ctx.lineTo(mx, L.binBottom);
        ctx.stroke();
        ctx.setLineDash([]);
        tri(mx, C.cyan, true);
        if (N > 0) {
          let s = 0;
          for (let k = 0; k <= n; k++) s += k * bins[k];
          const sx = valX(s / N);
          ctx.strokeStyle = 'rgba(245,181,68,0.7)';
          ctx.beginPath();
          ctx.moveTo(sx, L.binTop);
          ctx.lineTo(sx, L.binBottom);
          ctx.stroke();
          tri(sx, C.amber, true);
        }
      }

      // --- legend / HUD
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.font = `12px ${MONO}`;
      let ly = 14;
      const lx = 14;
      ctx.fillStyle = C.text;
      ctx.fillText(`n = ${n}   p = ${p.toFixed(2)}`, lx, ly); ly += 18;
      ctx.font = `11px ${MONO}`;
      ctx.fillStyle = C.muted;
      ctx.fillText(`N = ${N.toLocaleString()}${barMode ? '   (bars scaled)' : ''}`, lx, ly); ly += 20;
      const item = (color, label, kind) => {
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = kind === 'curve' ? 2 : 1.25;
        if (kind === 'dash') { ctx.setLineDash([3, 3]); ctx.strokeRect(lx + 0.5, ly + 1.5, 14, 9); ctx.setLineDash([]); }
        else if (kind === 'curve') { ctx.beginPath(); ctx.moveTo(lx, ly + 10); ctx.quadraticCurveTo(lx + 7, ly - 4, lx + 14, ly + 10); ctx.stroke(); }
        else { ctx.beginPath(); ctx.moveTo(lx + 7, ly + 1); ctx.lineTo(lx + 2, ly + 10); ctx.lineTo(lx + 12, ly + 10); ctx.closePath(); ctx.fill(); }
        ctx.fillStyle = C.muted;
        ctx.fillText(label, lx + 22, ly);
        ly += 17;
      };
      if (S.showBinom) item(C.cyan, `B(${n}, ${p.toFixed(2)}) expected`, 'dash');
      if (S.showNormal) item(C.good, `N(${(n * p).toFixed(1)}, ${(n * p * (1 - p)).toFixed(2)})`, 'curve');
      if (S.showMeans) { item(C.amber, 'sample mean', 'tri'); item(C.cyan, 'np', 'tri'); }
    }

    const stop = loop((dt) => {
      // spawn continuous + queued bursts
      spawnAcc += S.rate * dt;
      if (pending > 0) {
        const burst = Math.min(pending, Math.max(1, Math.round(160 * dt * Math.max(1, S.speed))));
        pending -= burst;
        spawnAcc += burst;
      }
      let toSpawn = Math.floor(spawnAcc);
      spawnAcc -= toSpawn;
      const cap = 5000;
      if (balls.length + toSpawn > cap) {
        const overflow = balls.length + toSpawn - cap;
        dropInstant(Math.min(overflow, toSpawn));
        toSpawn -= Math.min(overflow, toSpawn);
      }
      for (let i = 0; i < toSpawn; i++) spawn();
      stepBalls(dt);
      const decay = Math.exp(-dt * 5);
      for (const row of glow) for (let k = 0; k < row.length; k++) row[k] *= decay;
      draw();
      roTimer += dt;
      if (roTimer > 0.12) updateReadout();
    });

    return () => {
      stop();
      cv.destroy();
    };
  },
};
