// Workshop levels, chapter 4: attention. Boards hold the whole sentence as one [8, 8] matrix.
import { addNode } from './graph.js';

const STEP = 268;
function board(g, target) {
  addNode(g, 'seq', 40, 200, { words: 'long', dim: 8 }, { fixed: true, locked: true });
  addNode(g, 'output', 40 + STEP * 5, 200, { expect: target }, { fixed: true, locked: true });
}

export const CH4_LEVELS = [
  {
    id: 'at-scores', chapter: '4', title: 'Scores', target: [8, 8], parts: ['linear', 'scores'],
    attn: 'scores', budgets: [1e9, 1e9, 1e9],
    brief: 'Make queries Q and keys K: two Linear(8) maps of the sentence. Then score every word against every word with QKᵀ — an 8 × 8 grid.',
    lesson: 'Row i, column j = how well word i\'s query matches word j\'s key. A Linear on a sentence maps every word with the same weights, so Q has one row per word. Set each Linear to 8 outputs.',
    setup: (g) => board(g, [8, 8]),
  },
  {
    id: 'at-weights', chapter: '4', title: 'Attention weights', target: [8, 8], parts: ['linear', 'scores', 'scale', 'softmax'],
    attn: 'weights', peakMax: 0.65, budgets: [1e9, 1e9, 1e9],
    brief: 'Turn the scores into attention weights: Scale by √d (d = the width of Q and K), then a Softmax so each row sums to 1.',
    lesson: 'Try it without Scale first: most words put nearly all their weight on a single other word, because large scores make softmax winner-take-all. Dividing by √d (d = 8, the width of Q and K) keeps the weights spread. Select the Softmax to read the attention map.',
    setup: (g) => board(g, [8, 8]),
  },
  {
    id: 'at-head', chapter: '4', title: 'One head from parts', target: [8, 8], parts: ['linear', 'scores', 'scale', 'softmax', 'wsum'],
    attn: 'head', peakMax: 0.65, budgets: [1e9, 1e9, 1e9],
    brief: 'Finish a head: values V = a third Linear(8) of the sentence, and the output = attention weights · V.',
    lesson: 'Each word\'s output row is a weighted average of every word\'s value row, so any word can read any other in one step. In chapter 3, word 1 had to survive seven squashing steps to reach the end; here it is always one step away. (With untrained weights the head looks in arbitrary places — training decides where it looks.)',
    setup: (g) => board(g, [8, 8]),
  },
  {
    id: 'at-multi', chapter: '4', title: 'Two heads', target: [8, 8], parts: ['head', 'concat', 'linear'],
    attn: 'multi', budgets: [1e9, 1_000, 400],
    brief: 'Run two attention heads side by side on the sentence, Concat their outputs, and mix them with a Linear back to 8 features.',
    lesson: 'Each head has its own Q, K and V, so each can attend to different things — select each head to compare their maps. For ★★★, split the width: two heads of d = 4 cost what one head of 8 does. The Linear after Concat (the "output projection") mixes what the heads found.',
    setup: (g) => board(g, [8, 8]),
  },
];
