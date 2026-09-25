// DOM-free core of the neural-net playground: a small multilayer perceptron with manual
// backprop (Float64Array math), three optimizers, the 2D datasets and the input features.
// Imported by ../neural.js and directly by node tests.

// ---------------------------------------------------------------- random numbers
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// ---------------------------------------------------------------- activations
const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
// Numerically stable log(1 + e^z).
const softplus = (z) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));

// f(x), and f'(x) written in terms of the output y = f(x) (all four allow that).
export const ACTIVATIONS = {
  tanh: { f: Math.tanh, d: (y) => 1 - y * y, range: [-1, 1] },
  relu: { f: (x) => (x > 0 ? x : 0), d: (y) => (y > 0 ? 1 : 0), range: [0, Infinity] },
  sigmoid: { f: sigmoid, d: (y) => y * (1 - y), range: [0, 1] },
  linear: { f: (x) => x, d: () => 1, range: [-Infinity, Infinity] },
};

// ---------------------------------------------------------------- the network
// sizes = [nInputs, hidden..., 1]. Hidden layers use `activation`; the single output unit is a
// sigmoid trained with binary cross-entropy (labels 0/1). Weights are row-major out x in.
export class MLP {
  constructor(sizes, activation = 'tanh', seed = 1) {
    this.sizes = sizes.slice();
    this.activation = activation;
    this.L = sizes.length - 1;
    const mk = (n) => new Float64Array(n);
    this.W = []; this.b = []; this.gW = []; this.gB = [];
    this.mW = []; this.vW = []; this.mB = []; this.vB = [];
    this.a = [mk(sizes[0])]; this.d = [mk(sizes[0])];
    for (let l = 0; l < this.L; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      for (const arr of [this.W, this.gW, this.mW, this.vW]) arr.push(mk(nIn * nOut));
      for (const arr of [this.b, this.gB, this.mB, this.vB]) arr.push(mk(nOut));
      this.a.push(mk(nOut));
      this.d.push(mk(nOut));
    }
    this.zOut = 0;
    this.init(seed);
  }

  // Xavier/Glorot for tanh/sigmoid/linear (and the sigmoid output), He for ReLU layers.
  init(seed) {
    const rng = mulberry32(seed);
    for (let l = 0; l < this.L; l++) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1];
      const hiddenRelu = this.activation === 'relu' && l < this.L - 1;
      const std = hiddenRelu ? Math.sqrt(2 / nIn) : Math.sqrt(2 / (nIn + nOut));
      const W = this.W[l];
      for (let i = 0; i < W.length; i++) W[i] = gaussian(rng) * std;
      this.b[l].fill(hiddenRelu ? 0.01 : 0);
    }
    this.resetOptimizer();
  }

  resetOptimizer() {
    this.t = 0;
    for (let l = 0; l < this.L; l++) {
      this.mW[l].fill(0); this.vW[l].fill(0); this.mB[l].fill(0); this.vB[l].fill(0);
    }
  }

  get paramCount() {
    let n = 0;
    for (let l = 0; l < this.L; l++) n += this.W[l].length + this.b[l].length;
    return n;
  }

  // Forward pass for one input vector; activations are left in this.a. Returns P(class 1).
  forward(x, off = 0) {
    const a0 = this.a[0];
    for (let i = 0; i < a0.length; i++) a0[i] = x[off + i];
    const f = ACTIVATIONS[this.activation].f;
    for (let l = 0; l < this.L; l++) {
      const W = this.W[l], b = this.b[l], ain = this.a[l], aout = this.a[l + 1];
      const nIn = ain.length, nOut = aout.length, last = l === this.L - 1;
      for (let j = 0; j < nOut; j++) {
        let s = b[j];
        const row = j * nIn;
        for (let i = 0; i < nIn; i++) s += W[row + i] * ain[i];
        if (last) { this.zOut = s; aout[j] = sigmoid(s); } else aout[j] = f(s);
      }
    }
    return this.a[this.L][0];
  }

  // Data loss of the last forward pass against label y (0/1), computed from the logit.
  lossOfLast(y) { return softplus(this.zOut) - y * this.zOut; }

  zeroGrad() {
    for (let l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gB[l].fill(0); }
  }

  // Accumulate dLoss/dparams for the last forward pass. For sigmoid + BCE, dL/dz = p - y.
  // `scale` weights this sample's gradient (policy gradient scales it by the advantage).
  backward(y, scale = 1) {
    const d = ACTIVATIONS[this.activation].d;
    this.d[this.L][0] = (this.a[this.L][0] - y) * scale;
    for (let l = this.L - 1; l >= 0; l--) {
      const W = this.W[l], gW = this.gW[l], gB = this.gB[l];
      const ain = this.a[l], dout = this.d[l + 1], din = this.d[l];
      const nIn = ain.length, nOut = dout.length;
      for (let j = 0; j < nOut; j++) {
        const dj = dout[j];
        gB[j] += dj;
        const row = j * nIn;
        for (let i = 0; i < nIn; i++) gW[row + i] += dj * ain[i];
      }
      if (l > 0) {
        for (let i = 0; i < nIn; i++) {
          let s = 0;
          for (let j = 0; j < nOut; j++) s += W[j * nIn + i] * dout[j];
          din[i] = s * d(ain[i]);
        }
      }
    }
  }

  // Mean gradient over rows idx[from..to) of X (row-major, width nIn), plus L2 on weights.
  // Returns the mean data loss of the batch.
  gradient(X, Y, idx, from, to, l2 = 0) {
    this.zeroGrad();
    const nIn = this.sizes[0];
    let loss = 0;
    for (let k = from; k < to; k++) {
      const r = idx ? idx[k] : k;
      this.forward(X, r * nIn);
      loss += this.lossOfLast(Y[r]);
      this.backward(Y[r]);
    }
    const inv = 1 / Math.max(1, to - from);
    for (let l = 0; l < this.L; l++) {
      const W = this.W[l], gW = this.gW[l], gB = this.gB[l];
      for (let i = 0; i < gW.length; i++) gW[i] = gW[i] * inv + l2 * W[i];
      for (let j = 0; j < gB.length; j++) gB[j] *= inv;
    }
    return loss * inv;
  }

  // Objective whose gradient gradient() computes: mean BCE + (l2/2)·Σw².
  objective(X, Y, idx, from, to, l2 = 0) {
    const nIn = this.sizes[0];
    let loss = 0;
    for (let k = from; k < to; k++) {
      const r = idx ? idx[k] : k;
      this.forward(X, r * nIn);
      loss += this.lossOfLast(Y[r]);
    }
    loss /= Math.max(1, to - from);
    let w2 = 0;
    for (let l = 0; l < this.L; l++) for (const w of this.W[l]) w2 += w * w;
    return loss + 0.5 * l2 * w2;
  }

  // One parameter update from the accumulated (mean) gradient.
  applyGrad(lr, optimizer = 'sgd') {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    const upd = (P, G, M, V) => {
      if (optimizer === 'sgd') {
        for (let i = 0; i < P.length; i++) P[i] -= lr * G[i];
      } else if (optimizer === 'momentum') {
        for (let i = 0; i < P.length; i++) { M[i] = 0.9 * M[i] - lr * G[i]; P[i] += M[i]; }
      } else {
        for (let i = 0; i < P.length; i++) {
          const g = G[i];
          M[i] = b1 * M[i] + (1 - b1) * g;
          V[i] = b2 * V[i] + (1 - b2) * g * g;
          P[i] -= (lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + eps);
        }
      }
    };
    for (let l = 0; l < this.L; l++) {
      upd(this.W[l], this.gW[l], this.mW[l], this.vW[l]);
      upd(this.b[l], this.gB[l], this.mB[l], this.vB[l]);
    }
  }

  // Mean BCE and accuracy over the first n rows.
  evaluate(X, Y, n) {
    if (!n) return { loss: NaN, acc: NaN };
    const nIn = this.sizes[0];
    let loss = 0, right = 0;
    for (let r = 0; r < n; r++) {
      const p = this.forward(X, r * nIn);
      loss += this.lossOfLast(Y[r]);
      if ((p >= 0.5 ? 1 : 0) === Y[r]) right++;
    }
    return { loss: loss / n, acc: right / n };
  }

  isFinite() {
    for (let l = 0; l < this.L; l++) {
      for (const w of this.W[l]) if (!Number.isFinite(w)) return false;
      for (const w of this.b[l]) if (!Number.isFinite(w)) return false;
    }
    return true;
  }
}

// Mini-batch SGD over a fixed training set, one batch at a time so a UI can spread an epoch
// across frames. step() returns true when it finished an epoch.
export class Trainer {
  constructor(net, X, Y, n, seed = 7) {
    this.net = net;
    this.rng = mulberry32(seed);
    this.setData(X, Y, n);
  }
  setData(X, Y, n) {
    this.X = X; this.Y = Y; this.n = n;
    this.perm = new Int32Array(n);
    for (let i = 0; i < n; i++) this.perm[i] = i;
    this.cursor = n; // forces a shuffle on the first step
  }
  shuffle() {
    const p = this.perm;
    for (let i = p.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      const t = p[i]; p[i] = p[j]; p[j] = t;
    }
    this.cursor = 0;
  }
  step({ lr, batch, l2 = 0, optimizer = 'sgd' }) {
    if (!this.n) return false;
    if (this.cursor >= this.n) this.shuffle();
    const from = this.cursor, to = Math.min(this.n, from + Math.max(1, batch));
    this.net.gradient(this.X, this.Y, this.perm, from, to, l2);
    this.net.applyGrad(lr, optimizer);
    this.cursor = to;
    return to >= this.n;
  }
  epoch(opts) {
    if (!this.n) return;
    if (this.cursor >= this.n) this.shuffle();
    while (!this.step(opts));
  }
}

// ---------------------------------------------------------------- features
// The input domain is [-6, 6]². Raw coordinates are rescaled so features stay O(1).
export const DOMAIN = 6;
const S = 1 / 3;
export const FEATURES = [
  { id: 'x1', label: 'x₁', f: (x, y) => x * S },
  { id: 'x2', label: 'x₂', f: (x, y) => y * S },
  { id: 'x1sq', label: 'x₁²', f: (x, y) => (x * S) ** 2 },
  { id: 'x2sq', label: 'x₂²', f: (x, y) => (y * S) ** 2 },
  { id: 'x1x2', label: 'x₁x₂', f: (x, y) => x * y * S * S },
  { id: 'sin1', label: 'sin x₁', f: (x, y) => Math.sin(x) },
  { id: 'sin2', label: 'sin x₂', f: (x, y) => Math.sin(y) },
];
export const featureList = (ids) => FEATURES.filter((F) => ids.includes(F.id));

// pts: [{x, y, label}] -> { X: Float64Array(n * nf), Y: Float64Array(n) }
export function featurize(pts, feats) {
  const nf = feats.length;
  const X = new Float64Array(pts.length * nf), Y = new Float64Array(pts.length);
  pts.forEach((p, r) => {
    for (let k = 0; k < nf; k++) X[r * nf + k] = feats[k].f(p.x, p.y);
    Y[r] = p.label;
  });
  return { X, Y };
}

// ---------------------------------------------------------------- datasets
// label 1 = class A (amber), 0 = class B (cyan). noise in [0, 0.5].
export const DATASETS = [
  ['circle', 'Circle'],
  ['xor', 'XOR'],
  ['spiral', 'Two spirals'],
  ['gauss', 'Gaussian blobs'],
  ['moons', 'Moons'],
];

export function makeDataset(kind, n, noise, seed = 1) {
  const rng = mulberry32(seed);
  const g = () => gaussian(rng);
  const U = (a, b) => a + (b - a) * rng();
  const pts = [];
  const push = (x, y, label) => pts.push({ x: Math.max(-5.9, Math.min(5.9, x)), y: Math.max(-5.9, Math.min(5.9, y)), label });
  const half = Math.floor(n / 2);
  if (kind === 'circle') {
    for (let i = 0; i < n; i++) {
      const inner = i < half;
      const r = inner ? 2.4 * Math.sqrt(rng()) : U(3.5, 5);
      const t = U(0, 2 * Math.PI), j = noise * 2;
      push(r * Math.cos(t) + g() * j, r * Math.sin(t) + g() * j, inner ? 1 : 0);
    }
  } else if (kind === 'xor') {
    for (let i = 0; i < n; i++) {
      let x = U(-5, 5), y = U(-5, 5);
      x += x > 0 ? 0.3 : -0.3; y += y > 0 ? 0.3 : -0.3;
      const label = x * y > 0 ? 1 : 0, j = noise * 2;
      push(x + g() * j, y + g() * j, label);
    }
  } else if (kind === 'spiral') {
    for (let c = 0; c < 2; c++) {
      const m = c ? n - half : half;
      for (let i = 0; i < m; i++) {
        const r = (i / m) * 5 + 0.2;
        const t = (1.75 * i / m) * 2 * Math.PI + c * Math.PI;
        const j = noise * 1.2;
        push(r * Math.sin(t) + g() * j, r * Math.cos(t) + g() * j, c ? 0 : 1);
      }
    }
  } else if (kind === 'gauss') {
    const sd = 0.55 + noise * 4;
    for (let i = 0; i < n; i++) {
      const a = i < half, c = a ? 2 : -2;
      push(c + g() * sd, c + g() * sd, a ? 1 : 0);
    }
  } else { // moons
    for (let i = 0; i < n; i++) {
      const a = i < half, t = U(0, Math.PI), j = noise * 1.6;
      const x = a ? Math.cos(t) : 1 - Math.cos(t);
      const y = a ? Math.sin(t) : 0.5 - Math.sin(t);
      push((x - 0.5) * 3.4 + g() * j, (y - 0.25) * 3.4 + g() * j, a ? 1 : 0);
    }
  }
  return pts;
}

// Shuffle and mark the last `testFrac` of the points as held-out test data.
export function splitData(pts, testFrac, seed = 1) {
  const rng = mulberry32(seed ^ 0x9e3779b9);
  const out = pts.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  const nTest = Math.round(out.length * testFrac);
  return out.map((p, i) => ({ ...p, test: i >= out.length - nTest }));
}
