import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze, checkLevel, countParams } from '../src/sims/build/graph.js';

// Words w1..wT → RNNCell chain from a zero state → Output [16].
function unrolled(T, type = 'rnn') {
  const g = createGraph();
  const H = 16, state = type === 'lstm' ? 2 * H : H;
  let h = addNode(g, 'state0', 0, 0, { size: state });
  const cells = [];
  for (let t = 0; t < T; t++) {
    const w = addNode(g, 'token', 0, 0, { word: `w${t}`, w: t, dim: 8 });
    const c = addNode(g, type, 0, 0, { hidden: H });
    connect(g, h.id, c.id, 0); connect(g, w.id, c.id, 1);
    cells.push(c); h = c;
  }
  const o = addNode(g, 'output', 0, 0, { expect: [state] });
  connect(g, h.id, o.id);
  return { g, cells, o };
}

test('an unrolled RNN has the state shape at every step and shares its parameters', () => {
  const { g, cells } = unrolled(4);
  const a = analyze(g);
  for (const c of cells) assert.deepEqual(a.get(c.id).shape, [16]);
  const one = 16 * (8 + 16) + 16;
  assert.equal(countParams(cells, a), one, 'four steps, one weight set');
  assert.equal(countParams(unrolled(8).cells, analyze(unrolled(8).g)), one, 'length does not add parameters');
  assert.equal(checkLevel(g, a, { budgets: [1e9, 1e9, 1e9] }).params, one);
});

test('LSTMCell packs [h, c] and has four gates of parameters', () => {
  const { g, cells } = unrolled(3, 'lstm');
  const a = analyze(g);
  assert.deepEqual(a.get(cells[2].id).shape, [32]);
  assert.equal(countParams(cells, a), 4 * (16 * 24 + 16));
});

test('wrong state sizes and non-vectors are explained', () => {
  const g = createGraph();
  const w = addNode(g, 'token'), s = addNode(g, 'state0', 0, 0, { size: 10 }), c = addNode(g, 'rnn');
  connect(g, s.id, c.id, 0); connect(g, w.id, c.id, 1);
  assert.match(analyze(g).get(c.id).reason, /state input \(a\) must be \[16\]/);
});

test('gate parts: concat, sigmoid, tanh, multiply', () => {
  const g = createGraph();
  const x = addNode(g, 'token'), h = addNode(g, 'state0'), cat = addNode(g, 'concat'), lin = addNode(g, 'linear', 0, 0, { out: 16 });
  const sg = addNode(g, 'sigmoid'), m = addNode(g, 'mul');
  connect(g, x.id, cat.id, 0); connect(g, h.id, cat.id, 1); connect(g, cat.id, lin.id); connect(g, lin.id, sg.id);
  connect(g, sg.id, m.id, 0); connect(g, h.id, m.id, 1);
  const a = analyze(g);
  assert.deepEqual(a.get(cat.id).shape, [24]);
  assert.deepEqual(a.get(m.id).shape, [16]);
  connect(g, x.id, m.id, 1);
  assert.match(analyze(g).get(m.id).reason, /matching shapes/);
});

import { createRunner, firstWordMemory } from '../src/sims/build/forward.js';
import { wordVector, VOCAB } from '../src/sims/build/words.js';
import { rnnCell, lstmCell, cellWeights, tensor } from '../src/sims/build/tensor.js';

function sentence(type, words) {
  const g = createGraph();
  const H = 16, state = type === 'lstm' ? 2 * H : H;
  let h = addNode(g, 'state0', 0, 0, { size: state });
  words.forEach((word, t) => {
    const w = addNode(g, 'token', 0, 0, { word, w: VOCAB.indexOf(word), t, dim: 8 });
    const c = addNode(g, type, 0, 0, { hidden: H });
    connect(g, h.id, c.id, 0); connect(g, w.id, c.id, 1); h = c;
  });
  const o = addNode(g, 'output', 0, 0, { expect: [state] });
  connect(g, h.id, o.id);
  return g;
}
const LONG = ['the', 'old', 'dog', 'slept', 'by', 'the', 'warm', 'fire'];

test('word vectors are fixed and distinct', () => {
  assert.deepEqual([...wordVector(3).data], [...wordVector(3).data]);
  assert.notDeepEqual([...wordVector(3).data], [...wordVector(4).data]);
});

test('cells: tanh range, LSTM packs [h, c]; separate RNNCells really share weights', () => {
  const W = cellWeights(24, 16, 1), x = wordVector(1), h = tensor([16]);
  assert.ok([...rnnCell(x, h, W).data].every((v) => v > -1 && v < 1));
  assert.equal(lstmCell(x, tensor([32]), cellWeights(24, 64, 1, 2)).shape[0], 32);
  // two independent cells, same word, same zero state: identical outputs only if the weights are shared
  const g = createGraph();
  const z = addNode(g, 'state0'), w1 = addNode(g, 'token', 0, 0, { w: 2 }), w2 = addNode(g, 'token', 0, 0, { w: 2 });
  const c1 = addNode(g, 'rnn'), c2 = addNode(g, 'rnn');
  connect(g, z.id, c1.id, 0); connect(g, w1.id, c1.id, 1); connect(g, z.id, c2.id, 0); connect(g, w2.id, c2.id, 1);
  const out = createRunner().run(g, analyze(g), null, 1, '');
  assert.deepEqual([...out.get(c1.id).data], [...out.get(c2.id).data]);
  // ...whereas two Linear blocks get their own weights
  const l1 = addNode(g, 'linear', 0, 0, { out: 4 }), l2 = addNode(g, 'linear', 0, 0, { out: 4 });
  connect(g, w1.id, l1.id); connect(g, w2.id, l2.id);
  const out2 = createRunner().run(g, analyze(g), null, 1, '');
  assert.notDeepEqual([...out2.get(l1.id).data], [...out2.get(l2.id).data]);
});

test('the bottleneck: a plain RNN forgets word 1 over 8 words; an LSTM keeps it', () => {
  const rnnLong = firstWordMemory(sentence('rnn', LONG), analyze(sentence('rnn', LONG)));
  const rnnShort = firstWordMemory(sentence('rnn', LONG.slice(0, 3)), analyze(sentence('rnn', LONG.slice(0, 3))));
  const lstmLong = firstWordMemory(sentence('lstm', LONG), analyze(sentence('lstm', LONG)));
  console.log(`memory of word 1 — RNN 3 words ${(rnnShort * 100).toFixed(1)}%, RNN 8 words ${(rnnLong * 100).toFixed(1)}%, LSTM 8 words ${(lstmLong * 100).toFixed(1)}%`);
  assert.ok(rnnShort > rnnLong, 'RNN memory fades with length');
  assert.ok(rnnLong < 0.1, `rnn ${rnnLong}`);
  assert.ok(lstmLong > 0.15, `lstm ${lstmLong}`);
});

test('memoryMin rule', () => {
  const lvl = { memoryMin: 0.2, budgets: [1e9, 1e9, 1e9] };
  const g = sentence('lstm', LONG), a = analyze(g);
  assert.match(checkLevel(g, a, lvl).msg, /measured/);
  assert.equal(checkLevel(g, a, lvl, null, { memory: firstWordMemory(g, a) }).done, true);
  const r = sentence('rnn', LONG), ar = analyze(r);
  assert.match(checkLevel(r, ar, { ...lvl }, null, { memory: firstWordMemory(r, ar) }).msg, /barely survives/);
});
