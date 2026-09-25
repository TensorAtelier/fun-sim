// Workshop levels (chapter 2: pooling & CNN stacks). Pure data + setup, testable in node.
import { addNode } from './graph.js';
import { CH3_LEVELS } from './chapter3.js';
import { CH4_LEVELS } from './chapter4.js';
import { CH5_LEVELS } from './chapter5.js';
import { CH6_LEVELS } from './chapter6.js';

// Inspector sliders per block type: [param, label, min, max].
export const KNOBS = {
  input: [['c', 'Channels', 1, 4], ['h', 'Height', 4, 64], ['w', 'Width', 4, 64]],
  conv: [['out', 'Filters (out channels)', 1, 64], ['k', 'Kernel size', 1, 7], ['s', 'Stride', 1, 7], ['p', 'Padding', 0, 3]],
  pool: [['k', 'Pool size', 2, 4], ['s', 'Stride', 1, 4]],
  linear: [['out', 'Out features', 1, 256]],
  rnn: [['hidden', 'Hidden size', 4, 64]],
  lstm: [['hidden', 'Hidden size', 4, 64]],
  state0: [['size', 'Size', 2, 128]],
  scale: [['d', 'd (divide by √d)', 1, 64]],
  head: [['d', 'Head width d', 1, 32]],
  tblock: [['heads', 'Heads', 1, 8], ['ffn', 'MLP width', 4, 64]],
  router: [['experts', 'Experts', 2, 32]],
  topk: [['k', 'Keep top k', 1, 8]],
  experts: [['experts', 'Experts', 2, 32], ['ffn', 'Expert MLP width', 4, 64], ['capacity', 'Capacity factor', 0.5, 16, 0.25]],
  moe: [['experts', 'Experts', 2, 32], ['k', 'Top k', 1, 8], ['ffn', 'Expert MLP width', 4, 64], ['capacity', 'Capacity factor', 0.5, 16, 0.25]],
  output: [['classes', 'Classes', 2, 100]],
};

// Chapter 2 levels, in unlock order, then free build. Rules are data judged by checkLevel:
// target shape, requires (counted), forbids, maxK, rfMin, and star budgets [1★, 2★, 3★] by params.
// `fixed` blocks can't be removed; `locked` ones can't be edited either.
export const WS_LEVELS = [
  {
    id: 'ws-digits', chapter: '2', title: 'First classifier', target: [10], requires: ['conv'], budgets: [500_000, 100_000, 25_000],
    brief: 'Turn a 28×28 grayscale digit into 10 class scores (one per digit). Use at least one Conv2d.',
    lesson: 'The Linear at the end pays one weight per input value per class — shrink what it sees.',
  },
  {
    id: 'ws-down', chapter: '2', title: 'Downsample', target: [8, 14, 14], requires: ['conv'], budgets: [100_000, 300, 80],
    brief: 'Halve the image: turn [1, 28, 28] into [8, 14, 14].',
    lesson: 'MaxPool2d(2) halves height and width with no parameters; a stride-2 conv does it while learning.',
  },
  {
    id: 'ws-stride', chapter: '2', title: 'Stride instead of pooling', target: [16, 7, 7], forbids: ['pool'], budgets: [50_000, 3_000, 300],
    brief: 'Reach [16, 7, 7] with no MaxPool2d — only strided convolutions.',
    lesson: 'Hint for ★★★: a single conv whose kernel equals its stride cuts the image into non-overlapping patches — the trick Vision Transformers start with.',
  },
  {
    id: 'ws-rf', chapter: '2', title: 'See 7×7', target: [8, 28, 28], maxK: 3, rfMin: 7, forbids: ['pool'], budgets: [50_000, 5_000, 1_300],
    brief: 'Keep the full 28×28 size, use only 3×3 convolutions (no pooling), and make each output value see at least a 7×7 patch of the input.',
    lesson: 'Each stacked 3×3 conv grows the receptive field by 2. Select a block to see its view outlined on the Input. Three 3×3 layers see as much as one 7×7 with fewer weights (the VGG insight).',
  },
  {
    id: 'ws-gap', chapter: '2', title: 'Global average pooling', target: [10], requires: ['gap'], headRf: 7, budgets: [100_000, 5_000, 1_000],
    brief: 'Classify digits with GlobalAvgPool instead of Flatten + Linear. The features you average must each see at least 7×7 pixels.',
    lesson: 'Make the last conv output 10 channels, then average each channel to one score: the head has zero parameters.',
  },
  {
    id: 'ws-tiny', chapter: '2', title: 'Tiny classifier', target: [10], requires: ['conv', 'conv', 'pool', 'pool'], headRf: 14, budgets: [60_000, 12_000, 3_000],
    brief: 'The classic shape: conv → pool → conv → pool → classifier head. At least two convs and two pools, and the head must see features that each cover 14×14 pixels — half a digit.',
    lesson: 'Each stage trades resolution for channels — the pattern behind LeNet and most CNNs since.',
  },
  // Chapter 2b: residual blocks.
  {
    id: 'rs-skip', chapter: '2b', title: 'Skip connection', target: [8, 28, 28], requires: ['add', 'conv', 'conv'], needsSkip: true,
    budgets: [100_000, 3_000, 700],
    brief: 'Build y = x + f(x): a first Conv2d makes x; f is Conv2d → ReLU; an Add joins f(x) with a skip wire straight from x.',
    lesson: "The skip is an identity shortcut: the layers only have to learn a correction to x. Drag from the first conv's output to the Add's second port (b).",
  },
  {
    id: 'rs-match', chapter: '2b', title: 'Matching shapes', target: [16, 14, 14], requires: ['add'], needsSkip: true, needsProjection: true,
    budgets: [100_000, 5_000, 1_700],
    brief: 'The main path downsamples to [16, 14, 14]. Make the skip match that shape before the Add.',
    lesson: 'A 1×1 Conv2d with stride 2 and 16 filters reshapes the skip cheaply — the "projection shortcut" in ResNet. A 3×3 works but costs more.',
  },
  {
    id: 'rs-deep', chapter: '2b', title: 'Deep & alive', target: [8, 28, 28], initScale: 0.5, signalMin: 0.1, needsSkip: true,
    chainConvs: 8, forbids: ['norm'], budgets: [100_000, 10_000, 5_000],
    brief: "Stack at least 8 Conv2d layers in one chain with weights at half the usual scale, and keep every block's signal above 10% of the input's. No LayerNorm.",
    lesson: 'Each layer multiplies the signal by about 0.5, so a plain stack fades to nothing — in training, gradients fade the same way going backwards. A skip around each stage carries the signal past it.',
  },
  {
    id: 'rs-norm', chapter: '2b', title: 'Normalize', target: [8, 28, 28], initScale: 0.5, signalMin: 0.1,
    chainConvs: 8, requires: ['norm'], forbids: ['add'], budgets: [100_000, 10_000, 5_000],
    brief: 'The same deep stack with no skips: keep every block above 10% using LayerNorm instead.',
    lesson: 'LayerNorm resets each sample to mean 0, std 1, so the signal can neither fade nor blow up — one every stage or two does it. Skips without norms have the opposite problem: at full scale the signal grows about 30× over 8 stages, which is why real ResNets use both. (BatchNorm does the same job with statistics from a whole batch of images; here one image flows at a time, so it is not offered.)',
  },
  ...CH3_LEVELS,
  ...CH4_LEVELS,
  ...CH5_LEVELS,
  ...CH6_LEVELS,
  {
    id: 'free', title: 'Free build', free: true,
    brief: 'No goal: wire anything. Select Input or Output to change their shapes.',
  },
];

export function setupLevel(g, L) {
  if (L.setup) return L.setup(g);
  if (L.free) {
    addNode(g, 'input', 40, 80, {}, { fixed: true });
    addNode(g, 'output', 40, 300, {}, { fixed: true });
    return;
  }
  addNode(g, 'input', 40, 80, { c: 1, h: 28, w: 28 }, { fixed: true, locked: true });
  const out = L.target.length === 1 ? { classes: L.target[0] } : { expect: L.target };
  addNode(g, 'output', 40, 300, out, { fixed: true, locked: true });
}
