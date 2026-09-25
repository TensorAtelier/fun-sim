// Challenge levels. Each gen() returns a fresh random question:
// { prompt (html), fields: [{key, label} | {key, label, slider, min, max, value}],
//   answer: {key: number} | check(vals) -> bool, explain: html | (vals) -> html,
//   figure(vals, revealed) -> element | null }
import { h } from '../../ui.js';
import { outSize, effK, convParams } from './conv.js';
import { convFigure, grid, col, op, pipeline, heat, diverge, zeros, fmt } from './figure.js';

const ri = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

// Blank input with the kernel's first window (amber) and next stride position (cyan).
const sizeFig = (n, k, s, p, d = 1) => (vals, revealed) =>
  convFigure({ input: zeros(n, n), kernel: zeros(k, k), S: s, P: p, D: d },
    { showValues: false, cur: 0, showStride: true, showOutput: revealed, size: 22 });

const sizeField = [{ key: 'o', label: 'Output height (= width)' }];

export const CHAPTERS = [
  { id: 'conv', title: '1 · Convolution', ready: true },
  { id: 'cnn', title: '2 · Pooling & CNN stacks', workshop: '2', teaser: 'Build CNNs from blocks' },
  { id: 'res', title: '2b · Residual blocks', workshop: '2b', teaser: 'Skips, projections, norms' },
  { id: 'rnn', title: '3 · RNN & LSTM', workshop: '3', teaser: 'Unroll a sentence; build the gates' },
  { id: 'attn', title: '4 · Attention', workshop: '4', teaser: 'Queries, keys, values' },
  { id: 'tfm', title: '5 · Transformer block', workshop: '5', teaser: 'Attention + MLP + residuals + norms' },
  { id: 'moe', title: '6 · Mixture of Experts', workshop: '6', teaser: 'Routers and sparse experts' },
  { id: 'rl', title: '7 · Learning from reward', tab: 'reward', teaser: 'Bandits to RLHF' },
];

export const LEVELS = [
  {
    id: 'slide', chapter: 'conv', title: 'Sliding window',
    intro: `A convolution slides a small grid of weights — the <b>kernel</b> — across the input. At each
      position it multiplies the overlapping numbers and adds them into one output value. With stride 1
      and no padding, a kernel of size <b>k</b> fits in <b>n − k + 1</b> positions along a side of length <b>n</b>.
      <br><span class="bn-key"><i class="a"></i> first position <i class="b"></i> next position</span>`,
    gen() {
      const n = ri(5, 10), k = ri(2, Math.min(5, n - 1)), o = n - k + 1;
      return {
        prompt: `Input <b>${n}×${n}</b>, kernel <b>${k}×${k}</b>, stride 1, no padding. How big is the output?`,
        fields: sizeField, answer: { o },
        explain: `${n} − ${k} + 1 = <b>${o}</b>. The kernel's left edge can sit at columns 0 … ${n - k}.`,
        figure: sizeFig(n, k, 1, 0),
      };
    },
  },
  {
    id: 'stride', chapter: 'conv', title: 'Stride',
    intro: `<b>Stride</b> is how far the kernel jumps between positions. Stride 2 skips every other position,
      roughly halving the output — a cheap way to shrink an image. If the last jump doesn't fit, it's dropped:
      <div class="bn-eq">out = ⌊(n − k) / s⌋ + 1</div>`,
    gen() {
      const n = ri(6, 14), s = ri(2, 3), k = ri(1, 5), o = outSize(n, k, s);
      const rem = (n - k) % s;
      return {
        prompt: `Input <b>${n}×${n}</b>, kernel <b>${k}×${k}</b>, stride <b>${s}</b>, no padding. Output size?`,
        fields: sizeField, answer: { o },
        explain: `⌊(${n} − ${k}) / ${s}⌋ + 1 = ⌊${n - k}/${s}⌋ + 1 = <b>${o}</b>.` +
          (rem ? ` The last ${rem} column${rem > 1 ? 's' : ''} can't start another full step, so ${rem > 1 ? 'they are' : 'it is'} never the kernel's left edge.` : ''),
        figure: sizeFig(n, k, s, 0),
      };
    },
  },
  {
    id: 'padding', chapter: 'conv', title: 'Padding',
    intro: `<b>Padding</b> adds a border of zeros around the input (p cells on every side), so the kernel can
      sit over the edges and the output doesn't shrink as much. Padded side length is n + 2p:
      <div class="bn-eq">out = ⌊(n + 2p − k) / s⌋ + 1</div>`,
    gen() {
      const p = ri(1, 2), k = pick([3, 5]), n = ri(4, 11), s = ri(1, 2), o = outSize(n, k, s, p);
      return {
        prompt: `Input <b>${n}×${n}</b>, kernel <b>${k}×${k}</b>, stride <b>${s}</b>, padding <b>${p}</b>. Output size?`,
        fields: sizeField, answer: { o },
        explain: `⌊(${n} + 2·${p} − ${k}) / ${s}⌋ + 1 = ⌊${n + 2 * p - k}/${s}⌋ + 1 = <b>${o}</b>.`,
        figure: sizeFig(n, k, s, p),
      };
    },
  },
  {
    id: 'same', chapter: 'conv', title: '"Same" padding',
    intro: `Most CNNs want a conv layer to keep the image size unchanged (stride 1). Solve
      n + 2p − k + 1 = n and you get <b>p = (k − 1) / 2</b>. That only works for odd k — one reason
      3×3, 5×5 and 7×7 kernels are everywhere.`,
    gen() {
      const k = pick([1, 3, 5, 7]), n = ri(6, 14), p = (k - 1) / 2;
      return {
        prompt: `Input <b>${n}×${n}</b>, kernel <b>${k}×${k}</b>, stride 1. What padding keeps the output <b>${n}×${n}</b>?`,
        fields: [{ key: 'p', label: 'Padding p' }], answer: { p },
        explain: `p = (${k} − 1) / 2 = <b>${p}</b>. Check: ${n} + 2·${p} − ${k} + 1 = ${n}.`,
        figure: (vals, revealed) => sizeFig(n, k, 1, revealed ? p : 0)(vals, revealed),
      };
    },
  },
  {
    id: 'target', chapter: 'conv', title: 'Hit the target',
    intro: `Now you're the architect: choose kernel, stride and padding to produce an exact output shape.
      There's usually more than one answer.`,
    gen() {
      const n = ri(8, 14);
      let k, s, p, t;
      do { k = ri(1, 5); s = ri(1, 3); p = ri(0, 2); t = outSize(n, k, s, p); }
      while (t < 2 || t === outSize(n, 3, 1, 0));
      return {
        prompt: `Input <b>${n}×${n}</b>. Set the layer so the output is exactly <b>${t}×${t}</b>.`,
        fields: [
          { key: 'K', label: 'Kernel size k', slider: true, min: 1, max: 7, value: 3 },
          { key: 'S', label: 'Stride s', slider: true, min: 1, max: 3, value: 1 },
          { key: 'P', label: 'Padding p', slider: true, min: 0, max: 3, value: 0 },
        ],
        check: (v) => outSize(n, v.K, v.S, v.P) === t,
        explain: (v) => `Yours: ⌊(${n} + ${2 * v.P} − ${v.K}) / ${v.S}⌋ + 1 = <b>${outSize(n, v.K, v.S, v.P)}</b>. ` +
          `One solution: k = ${k}, s = ${s}, p = ${p}.`,
        figure: (v) => convFigure({ input: zeros(n, n), kernel: zeros(v.K, v.K), S: v.S, P: v.P },
          { showValues: false, cur: 0, showStride: true, size: 22 }),
      };
    },
  },
  {
    id: 'dot', chapter: 'conv', title: 'One output value',
    intro: `Each output value is a <b>dot product</b>: multiply every kernel weight by the input value under
      it, then add everything up. (Frameworks call this convolution; mathematicians call it cross-correlation
      — the kernel isn't flipped.)`,
    gen() {
      const patch = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => ri(0, 3)));
      const kernel = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => ri(-1, 2)));
      let sum = 0; const terms = [];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const w = kernel[a][b], v = patch[a][b];
        sum += w * v;
        if (w && v) terms.push(`${w < 0 ? `(${fmt(w)})` : w}·${v}`);
      }
      return {
        prompt: 'The kernel sits on this 3×3 patch of the input. What value does it write to the output?',
        fields: [{ key: 'v', label: 'Output value' }], answer: { v: sum },
        explain: `${terms.join(' + ') || '0'} = <b>${fmt(sum)}</b> (zero terms skipped).`,
        figure: (vals, revealed) => h('div', { class: 'bn-row bn-fig' },
          col('Input patch', grid(3, 3, 40, (a, b) => ({ text: String(patch[a][b]), style: heat(patch[a][b] / 3) }))),
          op('⊙'),
          col('Kernel', grid(3, 3, 40, (a, b) => ({ text: fmt(kernel[a][b]), style: diverge(kernel[a][b] / 2) }))),
          revealed ? [op('='), col('Output', grid(1, 1, 40, () => ({ text: fmt(sum), style: diverge(Math.sign(sum)), cls: 'cur' })))] : null),
      };
    },
  },
  {
    id: 'params', chapter: 'conv', title: 'Count the parameters',
    intro: `Real inputs have <b>channels</b> (RGB = 3). One filter spans all input channels — it's
      k×k×C<sub>in</sub> weights, all summed into a single output channel. A layer with C<sub>out</sub> filters
      makes C<sub>out</sub> output channels, plus one bias each:
      <div class="bn-eq">params = k·k·C<sub>in</sub>·C<sub>out</sub> + C<sub>out</sub></div>
      The image's height and width don't appear: the same weights are reused at every position.`,
    gen() {
      const k = pick([1, 3, 5, 7]), cin = pick([1, 3, 16, 32, 64]), cout = pick([8, 16, 32, 64, 128]);
      const bias = Math.random() < 0.7, w = k * k * cin * cout, total = convParams(k, cin, cout, bias);
      return {
        prompt: `<code>Conv2d(in_channels=${cin}, out_channels=${cout}, kernel_size=${k}, bias=${bias ? 'True' : 'False'})</code><br>How many learnable parameters?`,
        fields: [{ key: 'n', label: 'Parameters' }], answer: { n: total },
        explain: `${cout} filters × (${k}·${k}·${cin} weights) = ${w}` +
          (bias ? `, plus ${cout} biases = <b>${total}</b>.` : ` = <b>${total}</b> (no bias).`),
        figure: () => pipeline([
          { text: `[${cin}, H, W]`, kind: 'shape' },
          { text: `${cout} filters of ${k}×${k}×${cin}`, kind: 'layer' },
          { text: `[${cout}, H′, W′]`, kind: 'shape' },
        ]),
      };
    },
  },
  {
    id: 'dilation', chapter: 'conv', title: 'Dilation',
    intro: `<b>Dilation</b> spreads the kernel's taps d cells apart. A 3×3 kernel with dilation 2 covers a
      5×5 area using only 9 weights — a cheap way to see more context:
      <div class="bn-eq">effective k = d(k − 1) + 1</div>`,
    gen() {
      const k = pick([2, 3]), d = pick([2, 3]), n = ri(7, 13), ke = effK(k, d), o = n - ke + 1;
      return {
        prompt: `Input <b>${n}×${n}</b>, kernel <b>${k}×${k}</b>, dilation <b>${d}</b>, stride 1, no padding. Output size?`,
        fields: sizeField, answer: { o },
        explain: `Effective size ${d}·(${k} − 1) + 1 = ${ke}, so ${n} − ${ke} + 1 = <b>${o}</b>.`,
        figure: sizeFig(n, k, 1, 0, d),
      };
    },
  },
  {
    id: 'stack', chapter: 'conv', title: 'Stack the layers',
    intro: `Layers chain: each one's output shape is the next one's input. Shapes are written
      [channels, height, width]. A conv sets channels to its number of filters; <b>MaxPool2d(2)</b>
      keeps channels and halves height and width (⌊n/2⌋).`,
    gen() {
      for (;;) {
        let c = 3, n = pick([28, 32, 64]);
        const start = `[${c}, ${n}, ${n}]`, steps = [];
        const count = ri(3, 4);
        for (let i = 0; i < count; i++) {
          if (i > 0 && steps.at(-1).pool === false && Math.random() < 0.4) {
            const o = Math.floor(n / 2);
            steps.push({ pool: true, text: 'MaxPool2d(2)', calc: `⌊${n}/2⌋ = ${o}`, c, n: o });
            n = o;
          } else {
            const k = pick([3, 5]), s = pick([1, 1, 2]), p = pick([0, 1, 2]), co = pick([8, 16, 32, 64]);
            const o = outSize(n, k, s, p);
            steps.push({ pool: false, text: `Conv2d(${c}→${co}, k=${k}, s=${s}, p=${p})`,
              calc: `⌊(${n} + ${2 * p} − ${k}) / ${s}⌋ + 1 = ${o}`, c: co, n: o });
            c = co; n = o;
          }
        }
        if (n < 2) continue;
        return {
          prompt: 'What shape comes out of this stack?',
          fields: [{ key: 'c', label: 'Channels' }, { key: 'hw', label: 'Height (= width)' }],
          answer: { c, hw: n },
          explain: `<ol class="bn-steps">` +
            steps.map((st) => `<li>${st.text}: ${st.calc} → [${st.c}, ${st.n}, ${st.n}]</li>`).join('') + '</ol>',
          figure: (vals, revealed) => pipeline([
            { text: start, kind: 'shape' },
            ...steps.flatMap((st, i) => [
              { text: st.text, kind: 'layer' },
              ...(revealed || i < steps.length - 1 ? (revealed ? [{ text: `[${st.c}, ${st.n}, ${st.n}]`, kind: 'shape' }] : []) : [{ text: '[?, ?, ?]', kind: 'q' }]),
            ]),
          ]),
        };
      }
    },
  },
];
