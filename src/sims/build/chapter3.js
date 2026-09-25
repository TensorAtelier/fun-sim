// Workshop levels, chapter 3: RNN & LSTM. Sequence boards: a row of Word blocks over a zero state.
import { addNode } from './graph.js';
import { SENTENCES, VOCAB } from './words.js';

const X0 = 40, STEP = 268; // board layout: words on top, the recurrent chain underneath

function words(g, list) {
  return list.map((word, t) => addNode(g, 'token', X0 + STEP * (t + 1), 40, { word, w: VOCAB.indexOf(word), t, dim: 8 }, { fixed: true, locked: true }));
}

export const CH3_LEVELS = [
  {
    id: 'sq-unroll', chapter: '3', title: 'Unroll', target: [16], parts: ['rnn'],
    requires: ['rnn', 'rnn', 'rnn', 'rnn'], useAll: true, inOrder: true, budgets: [1e9, 1e9, 1e9],
    brief: 'Read "the cat sat down" one word at a time: chain four RNNCells from the zero state, each taking the previous state on a and its word on b.',
    lesson: 'Select the zero state and click RNNCell to chain one step; drag each word onto its cell. Watch the Network panel: four steps, one set of 400 parameters — the same cell is reused at every step, so a longer sentence costs nothing extra.',
    setup: (g) => {
      words(g, SENTENCES.short);
      addNode(g, 'state0', X0, 240, { size: 16, label: 'h₀' }, { fixed: true, locked: true });
      addNode(g, 'output', X0 + STEP * 5, 240, { expect: [16] }, { fixed: true, locked: true });
    },
  },
  {
    id: 'sq-parts', chapter: '3', title: 'LSTM cell from parts', target: [16],
    parts: ['concat', 'linear', 'sigmoid', 'tanh', 'mul', 'add'], useAll: true, lstmShape: true, budgets: [1e9, 1e9, 1e9],
    requires: ['concat', 'linear', 'linear', 'linear', 'linear', 'sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh', 'mul', 'mul', 'mul', 'add'],
    brief: 'Wire one LSTM step from x, h and c to the new hidden state h′ [16]. Concat x and h, then four Linear(16) layers: forget f, input i, candidate g, output o.',
    lesson: 'f = σ(Linear) decides what c keeps; i = σ(·) and g = tanh(·) decide what to write: c′ = f⊙c + i⊙g. Then h′ = o⊙tanh(c′) with o = σ(·). Four Linears, three Sigmoids, two Tanhs, three Multiplies, one Add. Set each Linear to 16 outputs in the inspector.',
    setup: (g) => {
      words(g, ['the']);
      addNode(g, 'state0', X0, 200, { size: 16, label: 'h' }, { fixed: true, locked: true });
      addNode(g, 'state0', X0, 360, { size: 16, label: 'c' }, { fixed: true, locked: true });
      addNode(g, 'output', X0 + STEP * 6, 280, { expect: [16] }, { fixed: true, locked: true });
    },
  },
  {
    id: 'sq-bottleneck', chapter: '3', title: 'The bottleneck', target: [32], parts: ['rnn', 'lstm'],
    useAll: true, inOrder: true, memoryMin: 0.2, budgets: [1e9, 1e9, 1e9],
    brief: 'Read an 8-word sentence into a [32] state so that the first word, "the", still matters at the end: swapping it must change the final state by at least 20%.',
    lesson: 'A plain RNNCell(hidden 32) forgets: each step squashes the old state through tanh with new input, and word 1 fades to ~11%. An LSTMCell(hidden 16) carries [h, c] = 32 values; its forget gate (started open) lets c keep word 1 at ~31%. Either way the whole sentence must squeeze through one fixed vector — attention (next chapter) lets the output look back at every word directly.',
    setup: (g) => {
      words(g, SENTENCES.long);
      addNode(g, 'state0', X0, 240, { size: 32, label: 's₀' }, { fixed: true, locked: true });
      addNode(g, 'output', X0 + STEP * 9, 240, { expect: [32] }, { fixed: true, locked: true });
    },
  },
];
