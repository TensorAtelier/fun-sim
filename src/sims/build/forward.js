// Forward pass over a Workshop graph: every block's real output on one input tensor.
// Pure (no DOM). Results are cached per block and reused until something upstream changes.
import { topo, inputsOf, BLOCKS, ancestors } from './graph.js';
import { conv2d, relu, maxpool, gap, add, layernorm, flatten, linear, softmax, convWeights, linearWeights, cellWeights, mixSeed,
  sigmoid, tanh, mul, concat, rnnCell, lstmCell, tensor, matmulT, matmul, scaleBy, softmaxRows, attentionHead,
  positionalEncoding, transformerBlock, topkRows, moeDispatch } from './tensor.js';
import { wordVector, SENTENCES, VOCAB } from './words.js';

// The rows of a Sentence block: one word vector per word. `swap` = [i, j] exchanges two words
// (used to measure word-order sensitivity).
export function sentenceMatrix(p) {
  const ids = SENTENCES[p.words].map((w) => VOCAB.indexOf(w));
  if (p.swap) [ids[p.swap[0]], ids[p.swap[1]]] = [ids[p.swap[1]], ids[p.swap[0]]];
  const t = tensor([ids.length, p.dim]);
  ids.forEach((id, i) => t.data.set(wordVector(id, p.dim).data, i * p.dim));
  return t;
}

// Untrained LSTMs start with the forget gate mostly open (a common init trick), so the cell keeps
// what it has seen — what a trained forget gate learns to do.
export const LSTM_FORGET_BIAS = 2;

const strHash = (s) => { let hv = 2166136261; for (let i = 0; i < s.length; i++) hv = Math.imul(hv ^ s.charCodeAt(i), 16777619); return hv >>> 0; };

// Creates a runner that remembers weights and outputs between calls.
export function createRunner() {
  const outCache = new Map(), wCache = new Map();

  // gain scales the He-initialised weights (1 = standard init).
  // Shared-weight types (RNNCell, LSTMCell) use one weight set per type, not per block.
  function weights(n, cin, seed, gain, make) {
    const shared = BLOCKS[n.type].shared, slot = shared ? `shared:${n.type}` : n.id;
    const byType = shared || BLOCKS[n.type].idFreeSeed; // seeded by type, not by (arbitrary) block id
    const key = `${n.type}:${cin}:${JSON.stringify(n.params)}:${seed}:${gain}`;
    let e = wCache.get(slot);
    if (!e || e.key !== key) {
      const w = make(mixSeed(seed, byType ? strHash(n.type) : n.id));
      if (gain !== 1 && w.w) for (let i = 0; i < w.w.length; i++) w.w[i] *= gain;
      e = { key, w };
      wCache.set(slot, e);
    }
    return e.w;
  }

  // input: [C,H,W] tensor for Input blocks (resized/sliced by the caller); seed: board weight seed.
  // Returns Map id -> tensor for every block whose analysis status is 'ok'.
  function run(g, analysis, input, seed, inputKey, gain = 1) {
    const out = new Map(), sig = new Map();
    for (const n of topo(g)) {
      if (analysis.get(n.id)?.status !== 'ok') continue;
      const ups = inputsOf(g, n.id, BLOCKS[n.type].inputs).map((e) => e?.from);
      const xs = ups.map((id) => out.get(id));
      if (xs.some((x) => !x)) continue;
      const x = xs[0];
      // The block's own id is part of the key (its weights are seeded by it), which also makes
      // `sig` unique per upstream block — so a rewire always invalidates downstream results.
      const key = strHash(`${n.id}|${n.type}|${JSON.stringify(n.params)}|${seed}|${gain}|${n.type === 'input' ? inputKey : ups.map((u) => sig.get(u)).join(',')}`);
      sig.set(n.id, key);
      const hit = outCache.get(n.id);
      if (hit && hit.key === key) { out.set(n.id, hit.y); continue; }
      const p = n.params;
      let y;
      switch (n.type) {
        case 'input': y = input; break;
        case 'token': y = wordVector(p.w, p.dim); break;
        case 'seq': y = sentenceMatrix(p); break;
        case 'scores': y = matmulT(xs[0], xs[1]); break;
        case 'scale': y = scaleBy(x, 1 / Math.sqrt(p.d)); break;
        case 'softmax': y = softmaxRows(x); break;
        case 'wsum': y = matmul(xs[0], xs[1]); break;
        case 'router': {
          const D = x.shape[1];
          y = softmaxRows(linear(x, weights(n, D, seed, gain, (s) => linearWeights(D, p.experts, s)), p.experts));
          break;
        }
        case 'topk': y = topkRows(x, p.k); break;
        case 'experts': case 'moe': {
          const D = x.shape[1];
          const bank = weights(n, D, seed, gain, (s) => ({
            router: linearWeights(D, p.experts, s + 97),
            experts: Array.from({ length: p.experts }, (_, e) => ({ w1: linearWeights(D, p.ffn, s + 2 * e), w2: linearWeights(p.ffn, D, s + 2 * e + 1) })),
          }));
          const gates = n.type === 'experts' ? xs[1] : topkRows(softmaxRows(linear(x, bank.router, p.experts)), p.k);
          const r = moeDispatch(x, gates, bank.experts, p.capacity);
          y = r.out;
          y.routing = { gates, load: r.load, dropped: r.dropped, capacity: r.capacity, routed: r.routed };
          break;
        }
        case 'posenc': y = add(x, positionalEncoding(x.shape[0], x.shape[1])); break;
        case 'tblock': {
          const D = x.shape[1];
          const W = weights(n, D, seed, gain, (s) => ({
            q: linearWeights(D, D, s), k: linearWeights(D, D, s + 1), v: linearWeights(D, D, s + 2), o: linearWeights(D, D, s + 3),
            w1: linearWeights(D, p.ffn, s + 4), w2: linearWeights(p.ffn, D, s + 5),
          }));
          y = transformerBlock(x, W, p.heads, p.ffn);
          break;
        }
        case 'head': {
          const D = x.shape[1];
          const W = weights(n, D, seed, gain, (s) => ({ q: linearWeights(D, p.d, s), k: linearWeights(D, p.d, s + 1), v: linearWeights(D, p.d, s + 2) }));
          const r = attentionHead(x, W, p.d);
          y = r.out;
          y.attention = r.weights; // for the inspector's attention map
          break;
        }
        case 'state0': y = tensor([p.size]); break;
        case 'sigmoid': y = sigmoid(x); break;
        case 'tanh': y = tanh(x); break;
        case 'mul': y = mul(xs[0], xs[1]); break;
        case 'concat': y = concat(xs[0], xs[1]); break;
        // a = previous state, b = word
        case 'rnn': y = rnnCell(xs[1], xs[0], weights(n, xs[1].shape[0], seed, gain, (s) => cellWeights(xs[1].shape[0] + p.hidden, p.hidden, s))); break;
        case 'lstm': y = lstmCell(xs[1], xs[0], weights(n, xs[1].shape[0], seed, gain, (s) => cellWeights(xs[1].shape[0] + p.hidden, 4 * p.hidden, s, LSTM_FORGET_BIAS))); break;
        case 'conv': y = conv2d(x, weights(n, x.shape[0], seed, gain, (s) => convWeights(x.shape[0], p.out, p.k, s)), p.out, p.k, p.s, p.p); break;
        case 'relu': y = relu(x); break;
        case 'pool': y = maxpool(x, p.k, p.s); break;
        case 'gap': y = gap(x); break;
        case 'flatten': y = flatten(x); break;
        case 'linear': {
          const fin = x.shape[x.shape.length - 1]; // features per row (a matrix's rows are words)
          y = linear(x, weights(n, fin, seed, gain, (s) => linearWeights(fin, p.out, s)), p.out);
          break;
        }
        case 'add': y = add(xs[0], xs[1]); break;
        case 'norm': y = layernorm(x); break;
        // A classifier Output turns scores into probabilities; an Output with a target shape just shows it.
        case 'output': y = p.expect ? x : softmax(x); break;
        default: continue;
      }
      outCache.set(n.id, { key, y });
      out.set(n.id, y);
    }
    // Forget deleted blocks so their (possibly large) weights can be freed.
    const alive = new Set(g.nodes.map((n) => n.id));
    for (const cache of [outCache, wCache]) for (const id of cache.keys()) if (typeof id === 'number' && !alive.has(id)) cache.delete(id);
    return out;
  }
  return { run };
}

// Rough cost of one forward pass: multiply-adds and learnable weights, from inferred shapes.
export function estimateCost(g, analysis) {
  let macs = 0, weights = 0;
  for (const n of g.nodes) {
    const r = analysis.get(n.id);
    if (r?.status !== 'ok') continue;
    if (n.type === 'conv') macs += r.shape[0] * r.shape[1] * r.shape[2] * r.inShape[0] * n.params.k * n.params.k;
    if (n.type === 'linear') macs += r.inShape.reduce((a, b) => a * b, 1) * n.params.out;
    if (n.type === 'scores' || n.type === 'wsum') macs += r.shape[0] * r.shape[1] * (n.type === 'scores' ? r.inShape[1] : r.inShape[1]);
    if (n.type === 'tblock') { const [T, D] = r.inShape; macs += 4 * T * D * D + 2 * T * T * D + 2 * T * D * n.params.ffn; }
    if (n.type === 'head') macs += 3 * r.inShape[0] * r.inShape[1] * n.params.d + 2 * r.inShape[0] * r.inShape[0] * n.params.d;
    weights += r.params;
  }
  return { macs, weights };
}

// How much of the first word survives in what reaches Output: swap word 1 for a few other words and
// measure how much that final vector changes, relative to its size (0 = forgotten entirely).
// Averaged over fixed weight seeds so a level's verdict doesn't depend on a lucky reroll.
export function firstWordMemory(g, analysis, seeds = [1, 2, 3], gain = 1, alts = [11, 12, 13]) {
  const out = g.nodes.find((n) => n.type === 'output');
  const feed = out && g.edges.find((e) => e.to === out.id)?.from;
  const first = g.nodes.filter((n) => n.type === 'token').sort((a, b) => (a.params.t ?? 0) - (b.params.t ?? 0))[0];
  if (feed == null || !first || analysis.get(feed)?.status !== 'ok') return null;
  const runner = createRunner();
  let total = 0;
  for (const seed of seeds) {
    const final = (w) => {
      const g2 = { ...g, nodes: g.nodes.map((n) => (n.id === first.id ? { ...n, params: { ...n.params, w } } : n)) };
      return runner.run(g2, analysis, null, seed, '', gain).get(feed);
    };
    const base = final(first.params.w);
    if (!base) return null;
    const size = Math.hypot(...base.data) || 1e-9;
    for (const w of alts) {
      const alt = final(w);
      let d = 0;
      for (let i = 0; i < alt.data.length; i++) d += (alt.data[i] - base.data[i]) ** 2;
      total += Math.sqrt(d) / size;
    }
  }
  return total / (alts.length * seeds.length);
}

// How peaked the attention is: the mean, over rows, of each row's largest weight — 1/T is perfectly
// even, 1.0 means every word looks at exactly one other. Taken from the last Softmax (or head)
// before Output, averaged over fixed seeds so Reroll can't change a verdict.
export function attentionPeak(g, analysis, seeds = [1, 2, 3]) {
  const out = g.nodes.find((n) => n.type === 'output');
  if (!out || analysis.get(out.id)?.status !== 'ok') return null;
  const up = ancestors(g, out.id);
  const att = topo(g).filter((n) => up.has(n.id) && (n.type === 'softmax' || n.type === 'head')).at(-1);
  if (!att) return null;
  const runner = createRunner();
  let total = 0;
  for (const seed of seeds) {
    const y = runner.run(g, analysis, null, seed, '').get(att.id);
    const w = att.type === 'head' ? y.attention : y;
    if (!w || w.shape.length !== 2) return null;
    const [T, S] = w.shape;
    let m = 0;
    for (let i = 0; i < T; i++) m += Math.max(...w.data.slice(i * S, (i + 1) * S));
    total += m / T;
  }
  return total / seeds.length;
}

// Does the model care about word order? Swap words 2 and 3 of the sentence and measure how much the
// first word's output row changes, relative to its size. Self-attention without positions is
// permutation-equivariant, so this is exactly 0 until positional encoding is added.
export function orderSensitivity(g, analysis, seeds = [1, 2, 3]) {
  const out = g.nodes.find((n) => n.type === 'output'), seq = g.nodes.find((n) => n.type === 'seq');
  const feed = out && g.edges.find((e) => e.to === out.id)?.from;
  if (feed == null || !seq || analysis.get(feed)?.status !== 'ok' || analysis.get(feed).shape.length !== 2) return null;
  const runner = createRunner();
  const row0 = (swap, seed) => {
    const g2 = { ...g, nodes: g.nodes.map((n) => (n.id === seq.id ? { ...n, params: { ...n.params, swap } } : n)) };
    const y = runner.run(g2, analysis, null, seed, '');
    return y.get(feed).data.slice(0, y.get(feed).shape[1]);
  };
  let total = 0;
  for (const seed of seeds) {
    const a = row0(undefined, seed), b = row0([1, 2], seed);
    total += Math.hypot(...a.map((v, i) => v - b[i])) / (Math.hypot(...a) || 1e-9);
  }
  return total / seeds.length;
}

// Mixture-of-Experts bookkeeping for a level: dropped word→expert assignments (worst over fixed
// seeds) across every Experts / MoE block feeding Output.
export function moeStats(g, analysis, seeds = [1, 2, 3]) {
  const out = g.nodes.find((n) => n.type === 'output');
  if (!out || analysis.get(out.id)?.status !== 'ok') return null;
  const up = ancestors(g, out.id);
  const moes = g.nodes.filter((n) => up.has(n.id) && (n.type === 'experts' || n.type === 'moe'));
  if (!moes.length) return { dropped: 0 };
  const runner = createRunner();
  let dropped = 0;
  for (const seed of seeds) {
    const y = runner.run(g, analysis, null, seed, '');
    dropped = Math.max(dropped, moes.reduce((s, n) => s + (y.get(n.id)?.routing?.dropped ?? 0), 0));
  }
  return { dropped };
}
