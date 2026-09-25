// Block graph for the Workshop: a DAG of layers joined by wires, with shape inference.
// Pure data + functions (no DOM) so it can be tested in node.
//
// Shapes omit the batch dimension: [C, H, W] for images, [F] for vectors.
import { outSize } from './conv.js';
import { SENTENCES } from './words.js';

const is3d = (s) => s.length === 3;
const fmtShape = (s) => `[${s.join(', ')}]`;

// Each type: inputs/outputs = port counts, defaults = editable params,
// infer(inShape, params) -> { shape } | { error }, params(inShape, params) -> learnable count.
export const BLOCKS = {
  input: {
    title: 'Input', inputs: 0, outputs: 1,
    defaults: { c: 1, h: 28, w: 28 },
    summary: (p) => `${p.c}×${p.h}×${p.w}`,
    infer: (_, p) => ({ shape: [p.c, p.h, p.w] }),
  },
  // A whole sentence at once: one row per word, [T, D].
  seq: {
    title: 'Sentence', inputs: 0, outputs: 1, source: true,
    defaults: { words: 'long', dim: 8 },
    summary: (p) => `"${SENTENCES[p.words].slice(0, 3).join(' ')} …" · ${SENTENCES[p.words].length} words`,
    infer: (_, p) => ({ shape: [SENTENCES[p.words].length, p.dim] }),
  },
  // Sequence sources: one Word block per time step, and a zero starting state.
  token: {
    title: 'Word', inputs: 0, outputs: 1, source: true,
    defaults: { word: 'the', w: 0, dim: 8 },
    summary: (p) => `"${p.word}" → [${p.dim}]`,
    infer: (_, p) => ({ shape: [p.dim] }),
  },
  state0: {
    title: 'Zero state', inputs: 0, outputs: 1, source: true,
    defaults: { size: 16 },
    summary: (p) => `${p.label ? `${p.label} = ` : ''}zeros [${p.size}]`,
    infer: (_, p) => ({ shape: [p.size] }),
  },
  // One step of a recurrent network: h' = tanh(W·[x, h] + b). All RNNCells share one weight set.
  rnn: {
    title: 'RNNCell', inputs: 2, outputs: 1, shared: true,
    defaults: { hidden: 16 },
    help: 'One time step: h′ = tanh(W·[x, h] + b). Input a is the previous state h, input b the word x. Every RNNCell on the board uses the same weights — that is what makes it recurrent — so its parameters count once.',
    summary: (p) => `h′ = tanh(W[x,h]) · ${p.hidden}`,
    infer: (x, p, [s, w]) => {
      if (s.length !== 1 || w.length !== 1) return { error: `RNNCell needs vectors: a state [${p.hidden}] on a and a word [D] on b` };
      if (s[0] !== p.hidden) return { error: `RNNCell's state input (a) must be [${p.hidden}], got [${s[0]}]` };
      return { shape: [p.hidden] };
    },
    params: (x, p, [, w]) => p.hidden * (w[0] + p.hidden) + p.hidden,
  },
  // LSTM step on a packed state s = [h, c] (2H values): four gates, one shared weight set.
  lstm: {
    title: 'LSTMCell', inputs: 2, outputs: 1, shared: true,
    defaults: { hidden: 16 },
    help: 'One LSTM step. Input a is the packed state [h, c] (2 × hidden values); input b is the word x. Forget, input and output gates decide what the cell state c keeps, adds and reveals. Weights are shared by every LSTMCell.',
    summary: (p) => `gates f,i,g,o · state [${2 * p.hidden}]`,
    infer: (x, p, [s, w]) => {
      if (s.length !== 1 || w.length !== 1) return { error: `LSTMCell needs vectors: a state [${2 * p.hidden}] on a and a word [D] on b` };
      if (s[0] !== 2 * p.hidden) return { error: `LSTMCell's state input (a) is [h, c] = [${2 * p.hidden}], got [${s[0]}]` };
      return { shape: [2 * p.hidden] };
    },
    params: (x, p, [, w]) => 4 * (p.hidden * (w[0] + p.hidden) + p.hidden),
  },
  // Parts for building gates by hand.
  concat: {
    title: 'Concat', inputs: 2, outputs: 1, defaults: {},
    help: 'Joins two vectors end to end ([a] and [b] become [a + b]), or two matrices side by side ([T, a] and [T, b] become [T, a + b]).',
    summary: () => '[a] ‖ [b]',
    infer: (x, p, [a, b]) => {
      if (a.length === 1 && b.length === 1) return { shape: [a[0] + b[0]] };
      if (a.length === 2 && b.length === 2 && a[0] === b[0]) return { shape: [a[0], a[1] + b[1]] };
      return { error: `Concat joins two vectors, or two matrices with the same number of rows — got ${fmtShape(a)} and ${fmtShape(b)}` };
    },
  },
  sigmoid: {
    title: 'Sigmoid', inputs: 1, outputs: 1, defaults: {},
    help: 'Squashes each value into 0…1 — a gate: 0 blocks, 1 lets through.',
    summary: () => '1 / (1 + e^−x)',
    infer: (x) => ({ shape: x }),
  },
  tanh: {
    title: 'Tanh', inputs: 1, outputs: 1, defaults: {},
    help: 'Squashes each value into −1…1.',
    summary: () => 'tanh(x)',
    infer: (x) => ({ shape: x }),
  },
  mul: {
    title: 'Multiply', inputs: 2, outputs: 1, defaults: {},
    help: 'Multiplies its two inputs value by value — how a gate scales what passes through it.',
    summary: () => 'a ⊙ b (same shape)',
    infer: (x, p, [a, b]) => (fmtShape(a) === fmtShape(b) ? { shape: a } : { error: `Multiply needs matching shapes, got ${fmtShape(a)} and ${fmtShape(b)}` }),
  },
  conv: {
    title: 'Conv2d', inputs: 1, outputs: 1,
    defaults: { out: 8, k: 3, s: 1, p: 1 },
    summary: (p) => `${p.out} filters · k${p.k} s${p.s} p${p.p}`,
    infer: (x, p) => {
      if (!is3d(x)) return { error: `Conv2d needs an image [C, H, W], got ${fmtShape(x)}` };
      const h = outSize(x[1], p.k, p.s, p.p), w = outSize(x[2], p.k, p.s, p.p);
      if (h < 1 || w < 1) return { error: `Kernel ${p.k} doesn't fit a ${x[1]}×${x[2]} input (with padding ${p.p})` };
      return { shape: [p.out, h, w] };
    },
    params: (x, p) => p.k * p.k * x[0] * p.out + p.out,
  },
  relu: {
    title: 'ReLU', inputs: 1, outputs: 1, defaults: {},
    summary: () => 'max(0, x)',
    infer: (x) => ({ shape: x }),
  },
  pool: {
    title: 'MaxPool2d', inputs: 1, outputs: 1,
    defaults: { k: 2, s: 2 },
    summary: (p) => `${p.k}×${p.k}, stride ${p.s}`,
    infer: (x, p) => {
      if (!is3d(x)) return { error: `MaxPool2d needs an image [C, H, W], got ${fmtShape(x)}` };
      const h = outSize(x[1], p.k, p.s), w = outSize(x[2], p.k, p.s);
      if (h < 1 || w < 1) return { error: `Pool ${p.k}×${p.k} doesn't fit a ${x[1]}×${x[2]} input` };
      return { shape: [x[0], h, w] };
    },
  },
  gap: {
    title: 'GlobalAvgPool', inputs: 1, outputs: 1, defaults: {},
    summary: () => '[C,H,W] → [C] (mean)',
    infer: (x) => (is3d(x) ? { shape: [x[0]] } : { error: `GlobalAvgPool needs an image [C, H, W], got ${fmtShape(x)}` }),
  },
  // Two inputs, summed element-wise: the join at the end of a skip connection.
  add: {
    title: 'Add', inputs: 2, outputs: 1, defaults: {},
    help: 'Adds its two inputs value by value. Wire the main path into a and the skip into b; both must have the same shape.',
    summary: () => 'a + b (same shape)',
    infer: (x, p, all) => {
      const [a, b] = all;
      if (fmtShape(a) === fmtShape(b)) return { shape: a };
      const hint = a.length === 3 && b.length === 3
        ? ` — put a 1×1 Conv2d on the skip path to match (${b[0] !== a[0] ? 'channels' : ''}${b[0] !== a[0] && b[1] !== a[1] ? ' and ' : ''}${b[1] !== a[1] ? 'stride for size' : ''})`
        : '';
      return { error: `Add needs matching shapes, got ${fmtShape(a)} and ${fmtShape(b)}${hint}` };
    },
  },
  // Normalise each sample over all its values, then scale/shift per channel (GroupNorm(1)-style).
  norm: {
    title: 'LayerNorm', inputs: 1, outputs: 1, defaults: {},
    help: 'Rescales values to mean 0 and std 1, then applies a learnable scale and shift (starting at 1 and 0). On an image: over the whole sample, 2 parameters per channel (like GroupNorm with one group). On a sentence: each word\'s row on its own — as in Transformers — 2 parameters per feature.',
    summary: () => 'mean 0, std 1 per sample',
    infer: (x) => ({ shape: x }),
    // Scale and shift per channel (images, vectors) or per feature (a sentence matrix, row by row).
    params: (x) => 2 * (x.length === 2 ? x[1] : x[0]),
  },
  flatten: {
    title: 'Flatten', inputs: 1, outputs: 1, defaults: {},
    summary: () => '[C,H,W] → [C·H·W]',
    infer: (x) => ({ shape: [x.reduce((a, b) => a * b, 1)] }),
  },
  // On a vector [F] → [out]; on a matrix [T, F] it maps every row with the same weights → [T, out].
  linear: {
    title: 'Linear', inputs: 1, outputs: 1,
    defaults: { out: 10 },
    summary: (p) => `→ ${p.out} features`,
    infer: (x, p) => {
      if (x.length === 1) return { shape: [p.out] };
      if (x.length === 2) return { shape: [x[0], p.out] };
      return { error: `Linear needs a vector [F], got ${fmtShape(x)} — add a Flatten first` };
    },
    params: (x, p) => x[x.length - 1] * p.out + p.out,
  },
  // Attention parts.
  scores: {
    title: 'Scores QKᵀ', inputs: 2, outputs: 1, defaults: {},
    help: 'Dot product of every query row (a) with every key row (b): entry [i, j] says how well word i\'s query matches word j\'s key. [T, d] and [S, d] give [T, S].',
    summary: () => 'Q · Kᵀ',
    infer: (x, p, [q, k]) => {
      if (q.length !== 2 || k.length !== 2) return { error: `Scores needs matrices Q [T, d] and K [S, d], got ${fmtShape(q)} and ${fmtShape(k)}` };
      if (q[1] !== k[1]) return { error: `Q and K need the same width d to be compared, got ${q[1]} and ${k[1]}` };
      return { shape: [q[0], k[0]] };
    },
  },
  scale: {
    title: 'Scale', inputs: 1, outputs: 1, defaults: { d: 8 },
    help: 'Divides by √d. Dot products of d-wide vectors grow like √d; without this, softmax saturates and each word attends to just one other.',
    summary: (p) => `÷ √${p.d}`,
    infer: (x) => ({ shape: x }),
  },
  softmax: {
    title: 'Softmax', inputs: 1, outputs: 1, defaults: {},
    help: 'Turns each row into positive weights that sum to 1 — for scores, how much each word attends to every other word.',
    summary: () => 'each row sums to 1',
    infer: (x) => ({ shape: x }),
  },
  wsum: {
    title: 'Weighted sum A·V', inputs: 2, outputs: 1, defaults: {},
    help: 'Each output row is a mix of the value rows (b), weighted by that row of attention weights (a). [T, S] · [S, d] gives [T, d].',
    summary: () => 'A · V',
    infer: (x, p, [a, v]) => {
      if (a.length !== 2 || v.length !== 2) return { error: `A·V needs matrices A [T, S] and V [S, d], got ${fmtShape(a)} and ${fmtShape(v)}` };
      if (a[1] !== v[0]) return { error: `A has ${a[1]} columns but V has ${v[0]} rows — they must match` };
      return { shape: [a[0], v[1]] };
    },
  },
  // Adds a fixed sine/cosine pattern per position, so identical words at different places differ.
  posenc: {
    title: 'Positional encoding', inputs: 1, outputs: 1, defaults: {},
    help: 'Adds a fixed pattern of sines and cosines to each word\'s row — a different one per position — so the model can tell "dog bites man" from "man bites dog". No parameters.',
    summary: () => '+ sin/cos by position',
    infer: (x) => (x.length === 2 ? { shape: x } : { error: `Positional encoding works on a sentence matrix [T, D], got ${fmtShape(x)}` }),
  },
  // Multi-head attention → Add → LayerNorm → MLP → Add → LayerNorm, packaged (post-norm).
  tblock: {
    title: 'Transformer block', inputs: 1, outputs: 1, defaults: { heads: 2, ffn: 32 },
    help: 'x₁ = LayerNorm(x + MultiHeadAttention(x)); out = LayerNorm(x₁ + MLP(x₁)), MLP = Linear(ffn) → ReLU → Linear(D). The heads split the width D evenly. Its own weights.',
    summary: (p) => `${p.heads} heads · MLP ${p.ffn}`,
    infer: (x, p) => {
      if (x.length !== 2) return { error: `A Transformer block needs a sentence matrix [T, D], got ${fmtShape(x)}` };
      if (x[1] % p.heads) return { error: `${p.heads} heads can't split a width of ${x[1]} evenly` };
      return { shape: x };
    },
    params: (x, p) => { const D = x[1]; return 4 * (D * D + D) + (D * p.ffn + p.ffn) + (p.ffn * D + D) + 2 * 2 * D; },
  },
  // Mixture of Experts. Router, experts and the packaged layer are seeded by type, not block id
  // (`idFreeSeed`), so which word goes where can't change with arbitrary ids.
  router: {
    title: 'Router', inputs: 1, outputs: 1, idFreeSeed: true, defaults: { experts: 4 },
    help: 'Scores every expert for every word: Softmax(Linear(x)) per row, so each word gets a probability for each expert. D·E + E parameters.',
    summary: (p) => `→ ${p.experts} experts`,
    infer: (x, p) => (x.length === 2 ? { shape: [x[0], p.experts] } : { error: `A Router reads a sentence matrix [T, D], got ${fmtShape(x)}` }),
    params: (x, p) => x[1] * p.experts + p.experts,
  },
  topk: {
    title: 'Top-k', inputs: 1, outputs: 1, defaults: { k: 2 },
    help: 'Keeps each word\'s k highest gates, zeroes the rest and renormalises them to sum to 1 — so each word is sent to only k experts.',
    summary: (p) => `keep top ${p.k}`,
    infer: (x, p) => {
      if (x.length !== 2) return { error: `Top-k works on gates [T, E], got ${fmtShape(x)}` };
      if (p.k > x[1]) return { error: `Top-k can't keep ${p.k} of ${x[1]} experts` };
      return { shape: x };
    },
  },
  experts: {
    title: 'Experts', inputs: 2, outputs: 1, idFreeSeed: true, defaults: { experts: 4, ffn: 16, capacity: 2 },
    help: 'E small MLPs (Linear → ReLU → Linear). Input a is the sentence, b the gates [T, E]: each word\'s output is the gate-weighted sum of the experts it was sent to. Each expert takes at most capacity × T·k/E words; the rest are dropped.',
    summary: (p) => `${p.experts} × MLP ${p.ffn} · cap ${p.capacity}×`,
    infer: (x, p, [a, gates]) => {
      if (a.length !== 2 || gates.length !== 2) return { error: `Experts need the sentence [T, D] on a and gates [T, E] on b` };
      if (gates[0] !== a[0]) return { error: `Gates have ${gates[0]} rows but the sentence has ${a[0]} words` };
      if (gates[1] !== p.experts) return { error: `Gates score ${gates[1]} experts but this block has ${p.experts} — make them match` };
      return { shape: a };
    },
    params: (x, p) => p.experts * (x[1] * p.ffn + p.ffn + p.ffn * x[1] + x[1]),
  },
  moe: {
    title: 'MoE layer', inputs: 1, outputs: 1, idFreeSeed: true, defaults: { experts: 8, k: 1, ffn: 16, capacity: 2 },
    help: 'Router → Top-k → Experts, packaged. Total parameters grow with the number of experts; each word only uses the router and k of them.',
    summary: (p) => `${p.experts} experts · top ${p.k}`,
    infer: (x, p) => {
      if (x.length !== 2) return { error: `An MoE layer reads a sentence matrix [T, D], got ${fmtShape(x)}` };
      if (p.k > p.experts) return { error: `Top ${p.k} of ${p.experts} experts isn't possible` };
      return { shape: x };
    },
    params: (x, p) => x[1] * p.experts + p.experts + p.experts * (2 * x[1] * p.ffn + p.ffn + x[1]),
  },
  // Q, K, V projections + scaled dot-product attention, packaged.
  head: {
    title: 'Attention head', inputs: 1, outputs: 1, defaults: { d: 8 },
    help: 'One self-attention head: Q, K, V = three Linear maps of the sentence, then softmax(QKᵀ/√d)·V. Its own weights; 3 × (D·d + d) parameters.',
    summary: (p) => `softmax(QKᵀ/√${p.d})·V`,
    infer: (x, p) => (x.length === 2 ? { shape: [x[0], p.d] } : { error: `An attention head needs a sentence matrix [T, D], got ${fmtShape(x)}` }),
    params: (x, p) => 3 * (x[1] * p.d + p.d),
  },
  // Output demands a target shape: `expect` (set by a level) or [classes].
  output: {
    title: 'Output', inputs: 1, outputs: 0,
    defaults: { classes: 10 },
    summary: (p) => (p.expect ? `target ${fmtShape(p.expect)}` : `${p.classes} class scores`),
    infer: (x, p) => {
      const want = p.expect || [p.classes];
      if (fmtShape(x) !== fmtShape(want)) return { error: `Output expects ${fmtShape(want)}, got ${fmtShape(x)}` };
      return { shape: x };
    },
  },
};

export function createGraph() {
  return { nodes: [], edges: [], nextId: 1 };
}

export function addNode(g, type, x = 0, y = 0, params = {}, extra = {}) {
  const node = { id: g.nextId++, type, x, y, params: { ...BLOCKS[type].defaults, ...params }, ...extra };
  g.nodes.push(node);
  return node;
}

export const getNode = (g, id) => g.nodes.find((n) => n.id === id);

export function removeNode(g, id) {
  g.nodes = g.nodes.filter((n) => n.id !== id);
  g.edges = g.edges.filter((e) => e.from !== id && e.to !== id);
}

// Would a wire from -> to close a loop? True if `from` is reachable from `to`.
function reaches(g, start, target) {
  const stack = [start], seen = new Set();
  while (stack.length) {
    const id = stack.pop();
    if (id === target) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const e of g.edges) if (e.from === id) stack.push(e.to);
  }
  return false;
}

// Wire from's output into input `port` of to. Each port takes one wire: a new one replaces the old.
export function connect(g, from, to, port = 0) {
  const a = getNode(g, from), b = getNode(g, to);
  if (!a || !b) return { ok: false, reason: 'No such block' };
  if (from === to) return { ok: false, reason: "A block can't feed itself" };
  if (!BLOCKS[a.type].outputs) return { ok: false, reason: `${BLOCKS[a.type].title} has no output` };
  if (!BLOCKS[b.type].inputs) return { ok: false, reason: `${BLOCKS[b.type].title} has no input` };
  if (port < 0 || port >= BLOCKS[b.type].inputs) return { ok: false, reason: `${BLOCKS[b.type].title} has no input ${port + 1}` };
  const same = g.edges.find((e) => e.from === from && e.to === to && e.port === port);
  if (same) return { ok: true, edge: same };
  if (reaches(g, to, from)) return { ok: false, reason: 'That wire would make a loop' };
  g.edges = g.edges.filter((e) => !(e.to === to && e.port === port));
  const edge = { id: g.nextId++, from, to, port };
  g.edges.push(edge);
  return { ok: true, edge };
}

export function disconnect(g, edgeId) {
  g.edges = g.edges.filter((e) => e.id !== edgeId);
}

// Shapes, errors and parameter counts for every node.
// status: 'ok' | 'pending' (no input yet, or upstream broken) | 'error' (a real mismatch)
export function analyze(g) {
  const res = new Map();
  for (const n of topo(g)) {
    const B = BLOCKS[n.type];
    const ups = inputsOf(g, n.id, B.inputs).map((e) => e && res.get(e.from));
    if (ups.some((u) => !u)) {
      const reason = B.inputs > 1 ? `Needs a wire on all ${B.inputs} inputs` : 'Needs an input wire';
      res.set(n.id, { status: 'pending', reason, shape: null, params: 0 });
      continue;
    }
    if (ups.some((u) => !u.shape)) { res.set(n.id, { status: 'pending', reason: 'Waiting on the block before it', shape: null, params: 0 }); continue; }
    const shapes = ups.map((u) => u.shape), inShape = shapes[0] ?? null;
    const r = B.infer(inShape, n.params, shapes);
    if (r.error) res.set(n.id, { status: 'error', reason: r.error, shape: null, inShape, params: 0 });
    else res.set(n.id, { status: 'ok', shape: r.shape, inShape, params: B.params ? B.params(inShape, n.params, shapes) : 0, rf: receptive(n, ups.map((u) => u.rf)) });
  }
  return res;
}

// The wire into each input port of a block (undefined where a port is empty).
export function inputsOf(g, id, count) {
  return Array.from({ length: count }, (_, port) => g.edges.find((e) => e.to === id && (e.port ?? 0) === port));
}

// Receptive field of one output value, in input pixels: size, jump (input pixels per output step)
// and the centre of the first output value. Standard recurrence; `global` once every value sees
// the whole input.
function receptive(n, ups) {
  if (n.type === 'input') return { size: 1, jump: 1, start: 0 };
  if (BLOCKS[n.type].source || ['rnn', 'lstm', 'concat', 'scores', 'wsum', 'head', 'posenc', 'tblock', 'router', 'topk', 'experts', 'moe'].includes(n.type)) return { global: true };
  if (ups.some((u) => u.global)) return { global: true };
  // Add: the larger view wins (matching shapes means matching steps).
  const up = ups.reduce((a, b) => (b.size > a.size ? b : a));
  if (n.type === 'gap' || n.type === 'linear') return { global: true };
  if (n.type === 'conv' || n.type === 'pool') {
    const { k, s } = n.params, p = n.params.p || 0;
    return { size: up.size + (k - 1) * up.jump, jump: up.jump * s, start: up.start + ((k - 1) / 2 - p) * up.jump };
  }
  return up;
}

export function topo(g) {
  const indeg = new Map(g.nodes.map((n) => [n.id, 0]));
  for (const e of g.edges) indeg.set(e.to, (indeg.get(e.to) || 0) + 1);
  const queue = g.nodes.filter((n) => indeg.get(n.id) === 0), out = [];
  while (queue.length) {
    const n = queue.shift();
    out.push(n);
    for (const e of g.edges) if (e.from === n.id) {
      indeg.set(e.to, indeg.get(e.to) - 1);
      if (indeg.get(e.to) === 0) queue.push(getNode(g, e.to));
    }
  }
  return out;
}

// Parameters of a set of blocks; shared-weight types (RNNCell, LSTMCell) count once per configuration.
export function countParams(nodes, analysis) {
  const seen = new Set();
  let total = 0;
  for (const n of nodes) {
    const r = analysis.get(n.id);
    if (!r?.params) continue;
    if (BLOCKS[n.type].shared) {
      const key = `${n.type}:${JSON.stringify(n.params)}:${fmtShape(r.inShape)}`;
      if (seen.has(key)) continue;
      seen.add(key);
    }
    total += r.params;
  }
  return total;
}

// Parameters each word actually uses: everything, minus the experts it isn't routed to. An Experts
// block uses k of its E experts per word (k from the Top-k feeding its gates; all E if dense).
export function activeParams(g, nodes, analysis) {
  let inactive = 0;
  for (const n of nodes) {
    const r = analysis.get(n.id);
    if (r?.status !== 'ok' || (n.type !== 'experts' && n.type !== 'moe')) continue;
    const D = r.inShape[1], per = 2 * D * n.params.ffn + n.params.ffn + D;
    if (n.type === 'experts') {
      const gate = g.edges.find((e) => e.to === n.id && e.port === 1)?.from;
      const k = g.nodes.find((m) => m.id === gate)?.type === 'topk' ? g.nodes.find((m) => m.id === gate).params.k : n.params.experts;
      inactive += (n.params.experts - k) * per;
    }
    if (n.type === 'moe') inactive += (n.params.experts - n.params.k) * per;
  }
  return countParams(nodes, analysis) - inactive;
}

const rmsOf = (t) => Math.sqrt(t.data.reduce((s, v) => s + v * v, 0) / t.data.length);

// Every node that feeds `id`, directly or not (including `id`).
export function ancestors(g, id) {
  const seen = new Set(), stack = [id];
  while (stack.length) {
    const cur = stack.pop();
    if (seen.has(cur)) continue;
    seen.add(cur);
    for (const e of g.edges) if (e.to === cur) stack.push(e.from);
  }
  return seen;
}

// Level goal: the Output block gets a valid shape, fed by a chain that meets `requires`.
// Stars by parameters upstream of Output: budgets = [1★ max, 2★ max, 3★ max].
// `outputs` (id -> tensor from the forward pass) is only needed by the signalMin rule.
// `metrics` carries measurements the Workshop computes separately (e.g. { memory }).
export function checkLevel(g, analysis, level, outputs = null, metrics = {}) {
  const out = g.nodes.find((n) => n.type === 'output');
  if (!out) return { done: false, msg: 'Add an Output block.' };
  const r = analysis.get(out.id);
  if (r.status !== 'ok') return { done: false, msg: r.reason };
  const up = [...ancestors(g, out.id)].map((id) => getNode(g, id));
  if (!up.some((n) => n.type === 'input' || (BLOCKS[n.type].source && n.type !== 'state0'))) return { done: false, msg: 'Output must be fed from the input.' };
  // useAll: every source on the board (each word, each starting state) must reach Output.
  if (level.useAll) {
    const unused = g.nodes.filter((n) => BLOCKS[n.type].source && !up.includes(n));
    const name = (n) => (n.type === 'token' ? `word ${(n.params.t ?? 0) + 1} ("${n.params.word}")` : BLOCKS[n.type].title);
    if (unused.length) return { done: false, msg: `Wire every input through to Output — ${unused.map(name).join(', ')} ${unused.length > 1 ? 'are' : 'is'} unused.` };
  }
  const srcOf = (id, port = 0) => g.edges.find((e) => e.to === id && (e.port ?? 0) === port)?.from;
  const typeOf = (id) => getNode(g, id)?.type;
  // inOrder: the recurrent chain (followed back along the state input) starts at the zero state and
  // reads every word exactly once, first word first — no reading word 1 last to keep it fresh.
  if (level.inOrder) {
    const words = g.nodes.filter((n) => n.type === 'token').sort((a, b) => (a.params.t ?? 0) - (b.params.t ?? 0));
    const read = [];
    let id = srcOf(out.id);
    while (typeOf(id) === 'rnn' || typeOf(id) === 'lstm') { read.unshift(srcOf(id, 1)); id = srcOf(id, 0); }
    if (typeOf(id) !== 'state0') return { done: false, msg: 'The recurrent chain must start from the zero state and end at Output.' };
    if (read.length !== words.length || new Set(read).size !== read.length) {
      return { done: false, msg: `Read each word exactly once: ${words.length} words, but the chain has ${read.length} steps.` };
    }
    if (read.some((w, i) => w !== words[i].id)) return { done: false, msg: 'Read the words in order, one step each, first word first.' };
  }
  // attn: the attention structure a level builds, judged by shape (like lstmShape):
  //   'scores'  Output ← Scores(Q, K)
  //   'weights' Output ← Softmax(Scale(Scores(Q, K))), Scale's d = the width of Q and K
  //   'head'    Output ← A·V(weights, V)
  //   'multi'   Output ← Linear(Concat(head₁, head₂)), two different heads both reading the Sentence
  // where Q, K, V are different Linear maps of the Sentence.
  if (level.attn) {
    const fromSentence = (id) => typeOf(id) === 'linear' && typeOf(srcOf(id)) === 'seq';
    const qk = (sc) => {
      if (typeOf(sc) !== 'scores') return 'Scores must compare queries with keys: QKᵀ.';
      const [q, k] = [srcOf(sc, 0), srcOf(sc, 1)];
      if (!fromSentence(q) || !fromSentence(k)) return 'Q and K must each be a Linear map of the Sentence.';
      if (q === k) return 'Q and K must be two different Linear maps — separate weights for asking and for being found.';
      return { q, k };
    };
    const weightsFrom = (sm) => {
      if (typeOf(sm) !== 'softmax') return 'The attention weights must come from a Softmax.';
      const s = srcOf(sm);
      if (typeOf(s) !== 'scale') return 'Scale the scores before the Softmax (÷√d).';
      const r = qk(srcOf(s));
      if (typeof r === 'string') return r;
      const d = analysis.get(r.q).shape[1];
      if (getNode(g, s).params.d !== d) return `Scale's d must be the width of Q and K (${d}), so the division is by √${d}.`;
      return r;
    };
    const feed = srcOf(out.id);
    let problem = null;
    if (level.attn === 'scores') { const r = qk(feed); if (typeof r === 'string') problem = r; }
    if (level.attn === 'weights') { const r = weightsFrom(feed); if (typeof r === 'string') problem = r; }
    if (level.attn === 'head') {
      if (typeOf(feed) !== 'wsum') problem = 'The output must be the weighted sum A·V.';
      else {
        const r = weightsFrom(srcOf(feed, 0)), v = srcOf(feed, 1);
        if (typeof r === 'string') problem = r;
        else if (!fromSentence(v) || v === r.q || v === r.k) problem = 'V must be a third Linear map of the Sentence, separate from Q and K.';
      }
    }
    if (level.attn === 'multi') {
      const c = srcOf(feed), [h1, h2] = [srcOf(c, 0), srcOf(c, 1)];
      if (typeOf(feed) !== 'linear' || typeOf(c) !== 'concat') problem = 'Mix the heads: Concat their outputs, then a Linear.';
      else if (typeOf(h1) !== 'head' || typeOf(h2) !== 'head' || h1 === h2) problem = 'Concat two different Attention heads.';
      else if (typeOf(srcOf(h1)) !== 'seq' || typeOf(srcOf(h2)) !== 'seq') problem = 'Both heads must read the Sentence directly — side by side, not stacked.';
    }
    // Without the right scaling, say how saturated the attention is right now — the lesson itself.
    if (problem && /Scale/.test(problem) && metrics?.peak != null) {
      problem += ` Right now each word puts ${(metrics.peak * 100).toFixed(0)}% of its attention on a single other word.`;
    }
    if (problem) return { done: false, msg: problem };
  }
  // tblockShape: out ← LN(A₂), A₂ = MLP(n₁) + n₁, MLP = Linear(ReLU(Linear(n₁))), n₁ = LN(A₁),
  // A₁ = attention(x) + x, x = the Sentence (or its positional encoding), attention = an
  // Attention head or Linear(Concat(heads)).
  if (level.tblockShape) {
    const ins = (id) => [srcOf(id, 0), srcOf(id, 1)];
    const isX = (id) => typeOf(id) === 'seq' || (typeOf(id) === 'posenc' && typeOf(srcOf(id)) === 'seq');
    // Attention part: a head reading x, optionally followed by an output-projection Linear; or a
    // Linear over Concats whose leaves are different heads all reading x (any number of heads).
    const headsUnder = (id) => (typeOf(id) === 'head' ? [id] : typeOf(id) === 'concat' ? ins(id).flatMap(headsUnder) : null);
    const isAttn = (id, x) => {
      if (typeOf(id) === 'head') return srcOf(id) === x;
      if (typeOf(id) !== 'linear') return false;
      const hs = headsUnder(srcOf(id));
      return !!hs && hs.every((h) => h && srcOf(h) === x) && new Set(hs).size === hs.length;
    };
    const isMlp = (id, n1) => typeOf(id) === 'linear' && typeOf(srcOf(id)) === 'relu' && typeOf(srcOf(srcOf(id))) === 'linear' && srcOf(srcOf(srcOf(id))) === n1;
    // one of an Add's two inputs satisfies `a`, the other `b`
    const either = (add, a, b) => { const [p, q] = ins(add); return (a(p) && b(q)) || (a(q) && b(p)); };
    const problem = (() => {
      const n2 = srcOf(out.id);
      if (typeOf(n2) !== 'norm' || typeOf(srcOf(n2)) !== 'add') return 'The block must end with LayerNorm(Add(…)) feeding Output.';
      const a2 = srcOf(n2);
      const n1 = ins(a2).find((i) => typeOf(i) === 'norm');
      if (!n1) return 'The second Add must take the first LayerNorm\'s output as its skip.';
      if (!either(a2, (i) => i === n1, (i) => isMlp(i, n1))) return 'The MLP must be Linear → ReLU → Linear, reading the first LayerNorm, into the second Add.';
      const a1 = srcOf(n1);
      if (typeOf(a1) !== 'add') return 'The first LayerNorm must normalise an Add: x + attention(x).';
      const x = ins(a1).find(isX);
      if (!x) return 'The first Add must take the sentence (or its positional encoding) as its skip.';
      if (!either(a1, (i) => i === x, (i) => isAttn(i, x))) return 'The first Add\'s other input must be attention over that same input: a head (optionally + Linear), or heads → Concat → Linear.';
      return null;
    })();
    if (problem) return { done: false, msg: `Not a Transformer block yet — ${problem}` };
  }
  // lstmShape: the parts must form one LSTM step: h′ = o ⊙ tanh(c′), c′ = f ⊙ c + i ⊙ g.
  if (level.lstmShape) {
    const ins = (id) => [srcOf(id, 0), srcOf(id, 1)];
    const isForget = (m) => { const [p, q] = ins(m); return [p, q].some((i) => typeOf(i) === 'state0' && getNode(g, i).params.label === 'c') && [p, q].some((i) => typeOf(i) === 'sigmoid'); };
    const isWrite = (m) => { const t = ins(m).map(typeOf); return t.includes('sigmoid') && t.includes('tanh'); };
    const shaped = (() => {
      const hNew = srcOf(out.id);
      if (typeOf(hNew) !== 'mul') return false;
      const [a, b] = ins(hNew);
      const tc = [a, b].find((i) => typeOf(i) === 'tanh');
      if (!tc || ![a, b].some((i) => typeOf(i) === 'sigmoid')) return false;
      const cNew = srcOf(tc);
      if (typeOf(cNew) !== 'add') return false;
      const [m1, m2] = ins(cNew);
      if (m1 === m2 || typeOf(m1) !== 'mul' || typeOf(m2) !== 'mul') return false;
      return (isForget(m1) && isWrite(m2)) || (isForget(m2) && isWrite(m1));
    })();
    if (!shaped) return { done: false, msg: "Not an LSTM's shape yet: h′ = o ⊙ tanh(c′) with c′ = f ⊙ c + i ⊙ g — f, i, o through Sigmoid, g through Tanh." };
  }
  // `requires` counts: ['conv', 'conv'] means at least two Conv2d blocks.
  const need = {};
  for (const t of level.requires || []) need[t] = (need[t] || 0) + 1;
  for (const [t, k] of Object.entries(need)) {
    if (up.filter((n) => n.type === t).length < k) return { done: false, msg: k === 1 ? `Use at least one ${BLOCKS[t].title}.` : `Use at least ${k} ${BLOCKS[t].title} blocks.` };
  }
  for (const t of level.forbids || []) {
    if (up.some((n) => n.type === t)) return { done: false, msg: `This level bans ${BLOCKS[t].title} — find another way.` };
  }
  if (level.maxK && up.some((n) => n.type === 'conv' && n.params.k > level.maxK)) {
    return { done: false, msg: `Only ${level.maxK}×${level.maxK} (or smaller) kernels allowed.` };
  }
  // headRf: the last feature map before the (global) classifier head must see this much input.
  if (level.headRf) {
    // Walk back along the main (port 0) path to the last feature map before the global head.
    const main = (id) => g.edges.find((e) => e.to === id && (e.port ?? 0) === 0).from;
    let id = main(out.id);
    while (analysis.get(id).rf.global) id = main(id);
    const { size } = analysis.get(id).rf;
    if (size < level.headRf) return { done: false, msg: `The features reaching the head each see only ${size}×${size} px of the digit — needs at least ${level.headRf}×${level.headRf}.` };
  }
  // chainConvs / chainBlocks: the path must be deep — this many Conv2d (or Transformer) blocks one
  // after another on a single chain.
  const chain = level.chainConvs ? ['conv', level.chainConvs] : level.chainBlocks ? ['tblock', level.chainBlocks] : null;
  if (chain) {
    const [ctype, need] = chain;
    const best = new Map();
    for (const n of topo(g)) {
      if (!up.some((u) => u.id === n.id)) continue;
      const ins = g.edges.filter((e) => e.to === n.id).map((e) => best.get(e.from) ?? 0);
      best.set(n.id, (ins.length ? Math.max(...ins) : 0) + (n.type === ctype ? 1 : 0));
    }
    const deepest = best.get(out.id) ?? 0;
    if (deepest < need) return { done: false, msg: `Stack at least ${need} ${BLOCKS[ctype].title} blocks in one chain — your deepest path has ${deepest}.` };
  }
  // signalMin: every block on the path must keep this fraction of the input's signal (RMS), so a
  // norm at the very end can't hide a stack that went dead in the middle.
  if (level.signalMin) {
    const inp = up.find((n) => n.type === 'input');
    const base = outputs?.get(inp.id);
    const path = up.filter((n) => n.type !== 'input' && n.type !== 'output');
    if (!base || path.some((n) => !outputs.get(n.id))) return { done: false, msg: "The signal can't be measured right now — values are off, paused, or still computing." };
    const b0 = rmsOf(base);
    const weakest = path.reduce((w, n) => { const r = rmsOf(outputs.get(n.id)) / b0; return r < w.r ? { n, r } : w; }, { r: Infinity });
    if (weakest.r < level.signalMin) {
      return { done: false, signal: weakest.r, msg: `The signal fades to ${(weakest.r * 100).toFixed(1)}% of the input's at a ${BLOCKS[weakest.n.type].title} — every block must keep at least ${level.signalMin * 100}%.` };
    }
  }
  // needsSkip: some Add on the path joins two different paths of unequal depth — x + f(x), where the
  // shorter side may be a plain wire or a projection (1×1 conv). Rules out x + x and twin branches.
  if (level.needsSkip) {
    const depth = new Map();
    for (const n of topo(g)) {
      const ins = g.edges.filter((e) => e.to === n.id).map((e) => depth.get(e.from) ?? 0);
      depth.set(n.id, ins.length ? Math.max(...ins) + 1 : 0);
    }
    const isSkip = (n) => {
      const [a, b] = [0, 1].map((p) => g.edges.find((e) => e.to === n.id && (e.port ?? 0) === p)?.from);
      return a !== b && depth.get(a) !== depth.get(b);
    };
    if (!up.some((n) => n.type === 'add' && isSkip(n))) return { done: false, msg: 'Wire a real skip: one Add input straight from an earlier block, the other through some layers.' };
    // needsProjection: that skip must start where the shape was different, so the skip path has to
    // reshape it (the projection shortcut) — not "downsample first, then add".
    if (level.needsProjection) {
      const reshaped = up.some((n) => {
        if (n.type !== 'add' || !isSkip(n)) return false;
        const [a, b] = [0, 1].map((p) => g.edges.find((e) => e.to === n.id && (e.port ?? 0) === p).from);
        const ancA = ancestors(g, a), common = [...ancestors(g, b)].filter((id) => ancA.has(id));
        const fork = common.reduce((f, id) => (depth.get(id) > depth.get(f) ? id : f));
        return fmtShape(analysis.get(fork).shape) !== fmtShape(analysis.get(n.id).shape);
      });
      if (!reshaped) return { done: false, msg: 'The skip must start before the downsampling and be reshaped on its way to the Add — the skip path needs a projection.' };
    }
  }
  // peakMax: attention must not saturate — rows' largest weights average at most this much.
  if (level.peakMax) {
    const pk = metrics?.peak;
    if (pk == null) return { done: false, msg: 'Wire a Softmax over the scores through to Output to see the attention weights.' };
    if (pk > level.peakMax) return { done: false, peak: pk, msg: `Attention is saturated: each word puts ${(pk * 100).toFixed(0)}% of its weight on one other word on average — needs ${level.peakMax * 100}% or less. Did you scale by √d?` };
  }
  // moe: the Mixture-of-Experts structure a level builds —
  //   'router' Output ← Softmax(Linear(x));  'sparse' Output ← Experts(x, Top-k(Router(x)))
  // with x the Sentence (or its positional encoding).
  if (level.moe) {
    const isX = (id) => typeOf(id) === 'seq' || (typeOf(id) === 'posenc' && typeOf(srcOf(id)) === 'seq');
    const feed = srcOf(out.id);
    let problem = null;
    if (level.moe === 'router') {
      if (typeOf(feed) !== 'softmax' || typeOf(srcOf(feed)) !== 'linear' || !isX(srcOf(srcOf(feed)))) problem = 'A router is Softmax(Linear(sentence)): one probability per expert for every word.';
    }
    if (level.moe === 'sparse') {
      const [x, gates] = [srcOf(feed, 0), srcOf(feed, 1)];
      if (typeOf(feed) !== 'experts') problem = 'The output must come from an Experts block.';
      else if (!isX(x)) problem = 'Experts input a must be the sentence.';
      else if (typeOf(gates) !== 'topk' || typeOf(srcOf(gates)) !== 'router' || srcOf(srcOf(gates)) !== x) problem = 'Experts input b must be Top-k(Router(the same sentence)).';
    }
    if (problem) return { done: false, msg: problem };
  }
  // activeRatioMax: each word may use at most this fraction of the parameters.
  if (level.activeRatioMax) {
    const total = countParams(up, analysis), act = activeParams(g, up, analysis);
    if (act > level.activeRatioMax * total) {
      return { done: false, msg: `Each word uses ${act.toLocaleString()} of ${total.toLocaleString()} parameters (${Math.round((100 * act) / total)}%) — needs at most ${level.activeRatioMax * 100}%. Route each word to fewer experts, or add more experts.` };
    }
  }
  // noDrops: no word may be dropped by a full expert.
  if (level.noDrops) {
    const d = metrics?.dropped;
    if (d == null) return { done: false, msg: 'Wire the experts through to Output to see their load.' };
    if (d > 0) return { done: false, dropped: d, msg: `Up to ${d} word→expert assignment${d > 1 ? 's are' : ' is'} dropped (worst of 3 weight draws): an expert hit its capacity. Raise the capacity factor.` };
  }
  // orderMin: swapping words 2 and 3 must change word 1's output by at least this much.
  if (level.orderMin) {
    const o = metrics?.order;
    if (o == null) return { done: false, msg: 'Word order can\'t be measured yet — wire the sentence through to Output.' };
    if (o < level.orderMin) return { done: false, order: o, msg: `Swapping "old" and "dog" changes word 1's output by only ${o < 1e-4 ? '0' : (o * 100).toFixed(1)}% — the model ignores word order. Needs ${level.orderMin * 100}%.` };
  }
  // memoryMin: the first word must still matter this much in what reaches Output.
  if (level.memoryMin) {
    const m = metrics?.memory;
    if (m == null) return { done: false, msg: "Memory can't be measured yet — wire the words through to Output." };
    if (m < level.memoryMin) return { done: false, memory: m, msg: `Word 1 barely survives: swapping it changes the final state by only ${(m * 100).toFixed(1)}% — needs ${level.memoryMin * 100}%.` };
  }
  if (level.rfMin) {
    const rf = analysis.get(g.edges.find((e) => e.to === out.id).from).rf;
    if (!rf.global && rf.size < level.rfMin) return { done: false, msg: `Each output value sees ${rf.size}×${rf.size} input pixels — needs at least ${level.rfMin}×${level.rfMin}.` };
  }
  const params = countParams(up, analysis), active = activeParams(g, up, analysis);
  // Stars are scored by parameters, or — on capacity levels — by the largest expert capacity used.
  const moes = up.filter((n) => n.type === 'experts' || n.type === 'moe');
  const score = level.starsFrom === 'capacity' ? Math.max(0, ...moes.map((n) => n.params.capacity)) : params;
  const stars = level.budgets.filter((b) => score <= b).length;
  if (!stars) return { done: false, params, msg: `Works, but ${params.toLocaleString()} parameters is over the ${level.budgets[0].toLocaleString()} budget.` };
  return { done: true, params, active, stars };
}
