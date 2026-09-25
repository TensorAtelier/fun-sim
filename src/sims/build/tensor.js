// Minimal tensors for the Workshop's forward pass. Pure JS, no DOM.
// A tensor is { shape, data } with row-major Float32Array data and no batch dimension:
// [C, H, W] for images, [F] for vectors.
import { outSize } from './conv.js';

export const size = (shape) => shape.reduce((a, b) => a * b, 1);
export const tensor = (shape, data = new Float32Array(size(shape))) => ({ shape, data });

// ---------- seeded randomness ----------
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) {
  const u = Math.max(r(), 1e-12), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
export const mixSeed = (a, b) => (Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be5ab, 0xc2b2ae35)) >>> 0;

// ---------- weights ----------
// Classic image kernels, generated for any odd-or-even size k, so a single-channel conv on a
// digit shows recognisable feature maps before any training.
const CLASSIC = [
  (a, b, k) => Math.sign((k - 1) / 2 - b) / k,        // vertical edge
  (a, b, k) => Math.sign((k - 1) / 2 - a) / k,        // horizontal edge
  (a, b, k) => 1 / (k * k),                           // blur
  (a, b, k) => (a === b ? 1 : a === b + 1 || b === a + 1 ? -0.5 : 0) / k, // diagonal
  (a, b, k) => (a + b === k - 1 ? 1 : Math.abs(a + b - (k - 1)) === 1 ? -0.5 : 0) / k, // anti-diagonal
  (a, b, k) => (a === (k - 1) >> 1 && b === (k - 1) >> 1 ? 1 : -1 / (k * k - 1)), // spot / sharpen
];

// Conv weights laid out [cout][cin][k][k]; He-normal init, bias 0.
export function convWeights(cin, cout, k, seed) {
  const r = rng(seed), w = new Float32Array(cout * cin * k * k), b = new Float32Array(cout);
  const std = Math.sqrt(2 / (cin * k * k));
  for (let i = 0; i < w.length; i++) w[i] = gauss(r) * std;
  if (cin === 1 && k > 1) {
    for (let o = 0; o < Math.min(cout, CLASSIC.length); o++) {
      for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) w[(o * k + a) * k + c] = CLASSIC[o](a, c, k);
    }
  }
  return { w, b };
}

// Linear weights laid out [fout][fin]; He-normal init, bias 0.
export function linearWeights(fin, fout, seed) {
  const r = rng(seed), w = new Float32Array(fout * fin), b = new Float32Array(fout);
  const std = Math.sqrt(2 / fin);
  for (let i = 0; i < w.length; i++) w[i] = gauss(r) * std;
  return { w, b };
}

// Recurrent cell weights: `rows` outputs over `fin` inputs, std √(1/fin) (tanh-friendly), bias 0.
// For an LSTM, rows = 4H laid out as gates f, i, g, o; the forget gate's bias starts at `forgetBias`.
export function cellWeights(fin, rows, seed, forgetBias = 0) {
  const r = rng(seed), w = new Float32Array(rows * fin), b = new Float32Array(rows);
  const std = Math.sqrt(1 / fin);
  for (let i = 0; i < w.length; i++) w[i] = gauss(r) * std;
  if (forgetBias) b.fill(forgetBias, 0, rows / 4);
  return { w, b };
}

// ---------- ops ----------
export function conv2d(x, { w, b }, cout, k, s = 1, p = 0) {
  const [cin, H, W] = x.shape;
  const oh = outSize(H, k, s, p), ow = outSize(W, k, s, p);
  const y = tensor([cout, oh, ow]), xd = x.data, yd = y.data;
  for (let o = 0; o < cout; o++) {
    for (let i = 0; i < oh; i++) {
      for (let j = 0; j < ow; j++) {
        let sum = b[o];
        for (let c = 0; c < cin; c++) {
          const wBase = (o * cin + c) * k * k, xBase = c * H * W;
          for (let a = 0; a < k; a++) {
            const yy = i * s - p + a;
            if (yy < 0 || yy >= H) continue;
            for (let e = 0; e < k; e++) {
              const xx = j * s - p + e;
              if (xx < 0 || xx >= W) continue;
              sum += w[wBase + a * k + e] * xd[xBase + yy * W + xx];
            }
          }
        }
        yd[(o * oh + i) * ow + j] = sum;
      }
    }
  }
  return y;
}

export function relu(x) {
  const y = tensor(x.shape);
  for (let i = 0; i < x.data.length; i++) y.data[i] = x.data[i] > 0 ? x.data[i] : 0;
  return y;
}

export function maxpool(x, k, s = k) {
  const [C, H, W] = x.shape;
  const oh = outSize(H, k, s), ow = outSize(W, k, s);
  const y = tensor([C, oh, ow]);
  for (let c = 0; c < C; c++) {
    for (let i = 0; i < oh; i++) {
      for (let j = 0; j < ow; j++) {
        let m = -Infinity;
        for (let a = 0; a < k; a++) for (let e = 0; e < k; e++) {
          const v = x.data[(c * H + i * s + a) * W + j * s + e];
          if (v > m) m = v;
        }
        y.data[(c * oh + i) * ow + j] = m;
      }
    }
  }
  return y;
}

// Mean of each channel: [C, H, W] → [C].
export function gap(x) {
  const [C, H, W] = x.shape, n = H * W, y = tensor([C]);
  for (let c = 0; c < C; c++) {
    let sum = 0;
    for (let i = c * n; i < (c + 1) * n; i++) sum += x.data[i];
    y.data[c] = sum / n;
  }
  return y;
}

const map = (x, f) => { const y = tensor(x.shape); for (let i = 0; i < x.data.length; i++) y.data[i] = f(x.data[i]); return y; };
export const sigmoid = (x) => map(x, (v) => 1 / (1 + Math.exp(-v)));
export const tanh = (x) => map(x, Math.tanh);
export function mul(a, b) {
  const y = tensor(a.shape);
  for (let i = 0; i < a.data.length; i++) y.data[i] = a.data[i] * b.data[i];
  return y;
}
// Vectors end to end; matrices side by side (along features).
export function concat(a, b) {
  if (a.shape.length === 2) {
    const [T, fa] = a.shape, fb = b.shape[1], y = tensor([T, fa + fb]);
    for (let i = 0; i < T; i++) {
      y.data.set(a.data.subarray(i * fa, (i + 1) * fa), i * (fa + fb));
      y.data.set(b.data.subarray(i * fb, (i + 1) * fb), i * (fa + fb) + fa);
    }
    return y;
  }
  const y = tensor([a.data.length + b.data.length]);
  y.data.set(a.data, 0);
  y.data.set(b.data, a.data.length);
  return y;
}

// h' = tanh(W·[x, h] + b)
export function rnnCell(x, h, W) {
  return tanh(linear(concat(x, h), W, h.data.length));
}

// s = [h, c]; gates z = W·[x, h] + b split into f, i, g, o; c' = σ(f)⊙c + σ(i)⊙tanh(g); h' = σ(o)⊙tanh(c').
export function lstmCell(x, s, W) {
  const H = s.data.length / 2, h = tensor([H], s.data.slice(0, H)), c = s.data.subarray(H);
  const z = linear(concat(x, h), W, 4 * H).data, out = tensor([2 * H]);
  const sg = (v) => 1 / (1 + Math.exp(-v));
  for (let k = 0; k < H; k++) {
    const cNew = sg(z[k]) * c[k] + sg(z[H + k]) * Math.tanh(z[2 * H + k]);
    out.data[H + k] = cNew;
    out.data[k] = sg(z[3 * H + k]) * Math.tanh(cNew);
  }
  return out;
}

export function add(a, b) {
  const y = tensor(a.shape);
  for (let i = 0; i < a.data.length; i++) y.data[i] = a.data[i] + b.data[i];
  return y;
}

// Normalise to mean 0, std 1: an image or vector over all its values; a sentence matrix [T, F] row
// by row (each word on its own, as Transformers do). The learnable scale and shift start at 1 and 0
// (untrained), so they don't change the output here.
export function layernorm(x, eps = 1e-5) {
  if (x.shape.length === 2) {
    const [T, F] = x.shape, y = tensor(x.shape);
    for (let i = 0; i < T; i++) y.data.set(layernorm(tensor([F], x.data.slice(i * F, (i + 1) * F)), eps).data, i * F);
    return y;
  }
  const n = x.data.length, y = tensor(x.shape);
  let mean = 0, v = 0;
  for (const d of x.data) mean += d;
  mean /= n;
  for (const d of x.data) v += (d - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + eps);
  for (let i = 0; i < n; i++) y.data[i] = (x.data[i] - mean) * inv;
  return y;
}

// Root-mean-square of all values: the "signal strength" of an activation.
export function rms(t) {
  let s = 0;
  for (const v of t.data) s += v * v;
  return Math.sqrt(s / Math.max(1, t.data.length));
}

export const flatten = (x) => tensor([x.data.length], x.data.slice());

// On a vector [F] → [fout]; on a matrix [T, F], every row with the same weights → [T, fout].
export function linear(x, { w, b }, fout) {
  const rows = x.shape.length === 2 ? x.shape[0] : 1, fin = x.shape[x.shape.length - 1];
  const y = tensor(x.shape.length === 2 ? [rows, fout] : [fout]);
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < fout; o++) {
      let sum = b[o];
      const base = o * fin, xb = r * fin;
      for (let i = 0; i < fin; i++) sum += w[base + i] * x.data[xb + i];
      y.data[r * fout + o] = sum;
    }
  }
  return y;
}

// Q [T, d] · Kᵀ → [T, S]
export function matmulT(q, k) {
  const [T, d] = q.shape, S = k.shape[0], y = tensor([T, S]);
  for (let i = 0; i < T; i++) for (let j = 0; j < S; j++) {
    let s = 0;
    for (let c = 0; c < d; c++) s += q.data[i * d + c] * k.data[j * d + c];
    y.data[i * S + j] = s;
  }
  return y;
}

// A [T, S] · V [S, d] → [T, d]
export function matmul(a, v) {
  const [T, S] = a.shape, d = v.shape[1], y = tensor([T, d]);
  for (let i = 0; i < T; i++) for (let j = 0; j < S; j++) {
    const w = a.data[i * S + j];
    for (let c = 0; c < d; c++) y.data[i * d + c] += w * v.data[j * d + c];
  }
  return y;
}

export const scaleBy = (x, s) => { const y = tensor(x.shape); for (let i = 0; i < x.data.length; i++) y.data[i] = x.data[i] * s; return y; };

// Softmax over the last axis: a vector as a whole, a matrix row by row.
export function softmaxRows(x) {
  if (x.shape.length === 1) return softmax(x);
  const [T, S] = x.shape, y = tensor(x.shape);
  for (let i = 0; i < T; i++) {
    const row = softmax(tensor([S], x.data.slice(i * S, (i + 1) * S)));
    y.data.set(row.data, i * S);
  }
  return y;
}

// Keep each row's k largest values, zero the rest, and renormalise the kept ones to sum to 1.
export function topkRows(x, k) {
  const [T, E] = x.shape, y = tensor(x.shape);
  for (let i = 0; i < T; i++) {
    const row = [...x.data.slice(i * E, (i + 1) * E)].map((v, e) => [v, e]).sort((a, b) => b[0] - a[0]).slice(0, k);
    const z = row.reduce((s, [v]) => s + v, 0) || 1;
    for (const [v, e] of row) y.data[i * E + e] = v / z;
  }
  return y;
}

// Sparse Mixture-of-Experts dispatch. gates [T, E] (zeros = not routed); experts[e] = { w1, w2 }
// (Linear D→ffn, ReLU, Linear ffn→D). Each expert keeps its highest-gate words up to
// capacity = ceil(factor · T · k / E), k = routed experts per word; the rest are dropped (they get
// nothing from that expert). Returns the output, per-expert load and dropped assignments.
export function moeDispatch(x, gates, experts, factor) {
  const [T, D] = x.shape, E = gates.shape[1], ffn = experts[0].w1.b.length;
  let routed = 0;
  for (const v of gates.data) if (v > 0) routed++;
  const cap = Math.ceil((factor * routed) / E);
  const out = tensor([T, D]), load = new Array(E).fill(0);
  let dropped = 0;
  for (let e = 0; e < E; e++) {
    const words = [];
    for (let t = 0; t < T; t++) if (gates.data[t * E + e] > 0) words.push(t);
    words.sort((a, b) => gates.data[b * E + e] - gates.data[a * E + e]);
    const kept = words.slice(0, cap);
    dropped += words.length - kept.length;
    load[e] = kept.length;
    for (const t of kept) {
      const row = tensor([D], x.data.slice(t * D, (t + 1) * D));
      const y = linear(relu(linear(row, experts[e].w1, ffn)), experts[e].w2, D);
      const gv = gates.data[t * E + e];
      for (let c = 0; c < D; c++) out.data[t * D + c] += gv * y.data[c];
    }
  }
  return { out, load, dropped, capacity: cap, routed };
}

// Sinusoidal positional encoding [T, D]: pe[t, 2i] = sin(t / 10000^(2i/D)), pe[t, 2i+1] = cos(…).
export function positionalEncoding(T, D) {
  const pe = tensor([T, D]);
  for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) {
    const angle = t / Math.pow(10000, (2 * Math.floor(i / 2)) / D);
    pe.data[t * D + i] = i % 2 ? Math.cos(angle) : Math.sin(angle);
  }
  return pe;
}

// Columns a..b-1 of a matrix.
function cols(m, a, b) {
  const [T, F] = m.shape, y = tensor([T, b - a]);
  for (let i = 0; i < T; i++) y.data.set(m.data.subarray(i * F + a, i * F + b), i * (b - a));
  return y;
}

// Post-norm Transformer block: x₁ = LN(x + MHA(x)), out = LN(x₁ + W₂·relu(W₁·x₁)).
// MHA: full-width Q, K, V split into `heads` column slices, each attended separately, concatenated,
// then mixed by the output projection O.
export function transformerBlock(x, W, heads, ffn) {
  const D = x.shape[1], dk = D / heads;
  const q = linear(x, W.q, D), k = linear(x, W.k, D), v = linear(x, W.v, D);
  let cat = null;
  for (let h = 0; h < heads; h++) {
    const [a, b] = [h * dk, (h + 1) * dk];
    const w = softmaxRows(scaleBy(matmulT(cols(q, a, b), cols(k, a, b)), 1 / Math.sqrt(dk)));
    const o = matmul(w, cols(v, a, b));
    cat = cat ? concat(cat, o) : o;
  }
  const x1 = layernorm(add(x, linear(cat, W.o, D)));
  return layernorm(add(x1, linear(relu(linear(x1, W.w1, ffn)), W.w2, D)));
}

// One head: softmax(QKᵀ/√d)·V with Q, K, V = x·W. Returns the output and the attention weights.
export function attentionHead(x, W, d) {
  const q = linear(x, W.q, d), k = linear(x, W.k, d), v = linear(x, W.v, d);
  const weights = softmaxRows(scaleBy(matmulT(q, k), 1 / Math.sqrt(d)));
  return { out: matmul(weights, v), weights };
}

export function softmax(x) {
  const y = tensor(x.shape);
  let m = -Infinity, z = 0;
  for (const v of x.data) if (v > m) m = v;
  for (let i = 0; i < x.data.length; i++) { y.data[i] = Math.exp(x.data[i] - m); z += y.data[i]; }
  for (let i = 0; i < y.data.length; i++) y.data[i] /= z;
  return y;
}

export function range(t) {
  let lo = Infinity, hi = -Infinity;
  for (const v of t.data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}
