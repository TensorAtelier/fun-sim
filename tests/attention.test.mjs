import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze } from '../src/sims/build/graph.js';

// sentence → Q, K, V (Linear d) → scores → scale → softmax → A·V
function head(d = 8, { kWidth = d } = {}) {
  const g = createGraph();
  const s = addNode(g, 'seq');
  const q = addNode(g, 'linear', 0, 0, { out: d }), k = addNode(g, 'linear', 0, 0, { out: kWidth }), v = addNode(g, 'linear', 0, 0, { out: d });
  for (const l of [q, k, v]) connect(g, s.id, l.id);
  const sc = addNode(g, 'scores'), scl = addNode(g, 'scale', 0, 0, { d }), sm = addNode(g, 'softmax'), ws = addNode(g, 'wsum');
  connect(g, q.id, sc.id, 0); connect(g, k.id, sc.id, 1); connect(g, sc.id, scl.id); connect(g, scl.id, sm.id);
  connect(g, sm.id, ws.id, 0); connect(g, v.id, ws.id, 1);
  return { g, s, q, sc, sm, ws };
}

test('a head built from parts: [8, 8] sentence → [8, 8] weights → [8, d] output', () => {
  const { g, s, q, sc, sm, ws } = head(8);
  const a = analyze(g);
  assert.deepEqual(a.get(s.id).shape, [8, 8]);
  assert.deepEqual(a.get(q.id).shape, [8, 8]);
  assert.equal(a.get(q.id).params, 8 * 8 + 8, 'Linear on rows: one weight set for every word');
  assert.deepEqual(a.get(sc.id).shape, [8, 8]);
  assert.deepEqual(a.get(sm.id).shape, [8, 8]);
  assert.deepEqual(a.get(ws.id).shape, [8, 8]);
});

test('Q and K must share a width; A·V needs matching inner sizes', () => {
  const { g, sc } = head(8, { kWidth: 4 });
  assert.match(analyze(g).get(sc.id).reason, /same width/);
});

test('packaged head and matrix concat', () => {
  const g = createGraph();
  const s = addNode(g, 'seq'), h1 = addNode(g, 'head', 0, 0, { d: 4 }), h2 = addNode(g, 'head', 0, 0, { d: 4 }), c = addNode(g, 'concat'), mix = addNode(g, 'linear', 0, 0, { out: 8 });
  connect(g, s.id, h1.id); connect(g, s.id, h2.id); connect(g, h1.id, c.id, 0); connect(g, h2.id, c.id, 1); connect(g, c.id, mix.id);
  const a = analyze(g);
  assert.equal(a.get(h1.id).params, 3 * (8 * 4 + 4));
  assert.deepEqual(a.get(c.id).shape, [8, 8]);
  assert.deepEqual(a.get(mix.id).shape, [8, 8]);
});

import { createRunner, firstWordMemory, sentenceMatrix } from '../src/sims/build/forward.js';
import { attentionHead, linear, linearWeights, matmulT, matmul, scaleBy, softmaxRows, tensor } from '../src/sims/build/tensor.js';

test('softmax rows sum to 1; the head equals softmax(QKᵀ/√d)·V computed by hand', () => {
  const x = sentenceMatrix({ words: 'long', dim: 8 });
  const W = { q: linearWeights(8, 4, 1), k: linearWeights(8, 4, 2), v: linearWeights(8, 4, 3) };
  const { out, weights } = attentionHead(x, W, 4);
  for (let i = 0; i < 8; i++) {
    const row = weights.data.slice(i * 8, i * 8 + 8).reduce((s, v) => s + v, 0);
    assert.ok(Math.abs(row - 1) < 1e-5);
  }
  const ref = matmul(softmaxRows(scaleBy(matmulT(linear(x, W.q, 4), linear(x, W.k, 4)), 0.5)), linear(x, W.v, 4));
  assert.deepEqual([...out.data], [...ref.data]);
});

test('parts on the board produce attention weights whose rows sum to 1', () => {
  const { g, sm, ws } = head(8);
  const out = createRunner().run(g, analyze(g), null, 1, '');
  const w = out.get(sm.id);
  for (let i = 0; i < 8; i++) assert.ok(Math.abs(w.data.slice(i * 8, i * 8 + 8).reduce((s, v) => s + v, 0) - 1) < 1e-5);
  assert.deepEqual(out.get(ws.id).shape, [8, 8]);
});

test('without ÷√d attention saturates (rows put almost all weight on one word)', () => {
  const peak = (withScale) => {
    const { g, sm } = head(8);
    const sc = g.nodes.find((n) => n.type === 'scale');
    if (!withScale) sc.params.d = 1; // ÷√1 = no scaling
    const w = createRunner().run(g, analyze(g), null, 1, '').get(sm.id);
    let m = 0;
    for (let i = 0; i < 8; i++) m += Math.max(...w.data.slice(i * 8, i * 8 + 8));
    return m / 8;
  };
  const scaled = peak(true), raw = peak(false);
  console.log(`mean top attention weight — scaled ${scaled.toFixed(2)}, unscaled ${raw.toFixed(2)}`);
  assert.ok(raw > scaled + 0.1, `raw ${raw} vs scaled ${scaled}`);
});

test('peakMax rule: unscaled scores saturate, scaled ones pass', async () => {
  const { attentionPeak } = await import('../src/sims/build/forward.js');
  const { checkLevel } = await import('../src/sims/build/graph.js');
  const lvl = { peakMax: 0.65, budgets: [1e9, 1e9, 1e9] };
  for (const [scaleD, want] of [[8, true], [1, false]]) {
    const { g, sm } = head(8);
    g.nodes.find((n) => n.type === 'scale').params.d = scaleD;
    const o = addNode(g, 'output', 0, 0, { expect: [8, 8] });
    connect(g, sm.id, o.id);
    const a = analyze(g);
    assert.equal(checkLevel(g, a, lvl, null, { peak: attentionPeak(g, a) }).done, want);
  }
});

test('Linear on a matrix uses the feature width, not the number of words (F ≠ T)', () => {
  const g = createGraph();
  const s = addNode(g, 'seq'), a = addNode(g, 'linear', 0, 0, { out: 4 }), b = addNode(g, 'linear', 0, 0, { out: 8 });
  const h1 = addNode(g, 'head', 0, 0, { d: 8 }), h2 = addNode(g, 'head', 0, 0, { d: 8 }), c = addNode(g, 'concat'), mix = addNode(g, 'linear', 0, 0, { out: 8 });
  connect(g, s.id, a.id); connect(g, a.id, b.id);
  connect(g, s.id, h1.id); connect(g, s.id, h2.id); connect(g, h1.id, c.id, 0); connect(g, h2.id, c.id, 1); connect(g, c.id, mix.id);
  const out = createRunner().run(g, analyze(g), null, 1, '');
  const ref = linear(out.get(a.id), linearWeights(4, 8, 0), 8); // right shape, arbitrary weights
  assert.equal(out.get(b.id).data.length, ref.data.length);
  for (const id of [b.id, mix.id]) assert.ok([...out.get(id).data].every(Number.isFinite), 'no NaN');
});
