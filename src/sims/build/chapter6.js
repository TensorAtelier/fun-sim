// Workshop levels, chapter 6: Mixture of Experts, on the 8-word sentence.
import { addNode, connect } from './graph.js';

const STEP = 268;
function board(g, target) {
  const seq = addNode(g, 'seq', 40, 200, { words: 'long', dim: 8 }, { fixed: true, locked: true });
  const out = addNode(g, 'output', 40 + STEP * 5, 40, { expect: target }, { fixed: true, locked: true });
  return { seq, out };
}

export const CH6_LEVELS = [
  {
    id: 'moe-route', chapter: '6', title: 'Router from parts', target: [8, 4], parts: ['linear', 'softmax'],
    moe: 'router', budgets: [1e9, 1e9, 1e9],
    brief: 'Build a router for 4 experts: a Linear(4) of the sentence, then a Softmax, so every word gets a probability for each expert.',
    lesson: 'Select the Softmax to read the routing table: one row per word, one column per expert. From now on the packaged Router block does exactly this. (Untrained, the preferences are arbitrary; training teaches the router which expert suits which word.)',
    setup: (g) => { board(g, [8, 4]); },
  },
  {
    id: 'moe-sparse', chapter: '6', title: 'Sparse experts', target: [8, 8], parts: ['router', 'topk', 'experts'],
    moe: 'sparse', activeRatioMax: 0.5, budgets: [1e9, 1e9, 1e9],
    brief: 'Send each word to only its top experts: Router → Top-k → the gates (input b) of an Experts block that also reads the sentence (input a). Each word may use at most half of all the parameters.',
    lesson: 'This is the point of MoE: total parameters grow with the number of experts, but each word only pays for the router and k of them. Try k = 1 with 4 experts — and compare the "active" and "total" counts.',
    setup: (g) => { board(g, [8, 8]); },
  },
  {
    id: 'moe-capacity', chapter: '6', title: 'Capacity', target: [8, 8], parts: [],
    moe: 'sparse', noDrops: true, starsFrom: 'capacity', budgets: [16, 3, 2],
    brief: 'This MoE layer is dropping words: an expert can only take capacity × (words × k / experts) of them. Raise the capacity factor on the Experts block until no word is dropped — as little as possible for ★★★.',
    lesson: 'The router sends more words to some experts than others, so a busy expert fills up and turns words away. Bigger capacity wastes compute on idle experts; smaller drops words. Real models train with an extra "load-balancing" loss that nudges the router to spread words evenly.',
    setup: (g) => {
      const { seq, out } = board(g, [8, 8]);
      // Routing is fixed (4 experts, top-1); only the capacity factor can change.
      const r = addNode(g, 'router', 40 + STEP, 40, { experts: 4 }, { fixed: true, locked: true });
      const t = addNode(g, 'topk', 40 + STEP * 2, 40, { k: 1 }, { fixed: true, locked: true });
      const x = addNode(g, 'experts', 40 + STEP * 3, 40, { experts: 4, ffn: 16, capacity: 1 }, { fixed: true, only: ['capacity'] });
      connect(g, seq.id, r.id); connect(g, r.id, t.id); connect(g, seq.id, x.id, 0); connect(g, t.id, x.id, 1); connect(g, x.id, out.id);
    },
  },
  {
    id: 'moe-scale', chapter: '6', title: 'Scale up, not out', target: [8, 8], parts: ['posenc', 'tblock', 'moe'],
    requires: ['moe'], activeRatioMax: 0.25, noDrops: true, budgets: [1e9, 1e9, 1e9],
    brief: 'Give the model at least 4× more parameters in total than any one word uses — an MoE layer with many experts — without dropping a single word.',
    lesson: 'This is how the largest models grow: many experts, few used per word (Mixtral-style models activate 2 of 8; some 1 of 100+). Now try putting a Transformer block in front: its untrained output rows look so alike that the router sends almost every word to the same two experts — "routing collapse", the failure load-balancing losses exist to prevent. With 16 experts it drops words even at capacity 8×.',
    setup: (g) => { board(g, [8, 8]); },
  },
];
