import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze, countParams, activeParams } from '../src/sims/build/graph.js';

function sparse({ E = 4, k = 1, ffn = 16 } = {}) {
  const g = createGraph();
  const s = addNode(g, 'seq'), r = addNode(g, 'router', 0, 0, { experts: E }), t = addNode(g, 'topk', 0, 0, { k }), x = addNode(g, 'experts', 0, 0, { experts: E, ffn });
  connect(g, s.id, r.id); connect(g, r.id, t.id); connect(g, s.id, x.id, 0); connect(g, t.id, x.id, 1);
  return { g, r, t, x };
}

test('router → top-k → experts: shapes, total and active parameters', () => {
  const { g, r, t, x } = sparse({ E: 4, k: 1 });
  const a = analyze(g);
  assert.deepEqual(a.get(r.id).shape, [8, 4]);
  assert.deepEqual(a.get(t.id).shape, [8, 4]);
  assert.deepEqual(a.get(x.id).shape, [8, 8]);
  const per = 8 * 16 + 16 + 16 * 8 + 8; // one expert MLP
  const nodes = g.nodes;
  assert.equal(countParams(nodes, a), 8 * 4 + 4 + 4 * per);
  assert.equal(activeParams(g, nodes, a), 8 * 4 + 4 + per, 'top-1: each word uses the router and one expert');
});

test('packaged MoE layer counts the same way; mismatches are explained', () => {
  const g = createGraph();
  const s = addNode(g, 'seq'), m = addNode(g, 'moe', 0, 0, { experts: 8, k: 2, ffn: 16 });
  connect(g, s.id, m.id);
  const a = analyze(g), per = 8 * 16 + 16 + 16 * 8 + 8;
  assert.equal(countParams(g.nodes, a), 8 * 8 + 8 + 8 * per);
  assert.equal(activeParams(g, g.nodes, a), 8 * 8 + 8 + 2 * per);
  const { g: g2, x } = sparse({ E: 4 });
  x.params.experts = 6;
  assert.match(analyze(g2).get(x.id).reason, /make them match/);
});

import { createRunner, moeStats } from '../src/sims/build/forward.js';
import { topkRows, moeDispatch, linear, relu, linearWeights, tensor } from '../src/sims/build/tensor.js';

test('top-k keeps k gates per row, renormalised', () => {
  const g = topkRows(tensor([2, 4], Float32Array.from([0.1, 0.4, 0.3, 0.2, 0.7, 0.1, 0.1, 0.1])), 2);
  assert.deepEqual([...g.data].map((v) => +v.toFixed(4)), [0, 0.5714, 0.4286, 0, 0.875, 0.125, 0, 0]);
});

test('dense dispatch equals the gate-weighted sum of every expert', () => {
  const x = tensor([3, 4], Float32Array.from([1, 0, -1, 2, 0.5, 0.5, 0, 1, -1, 2, 1, 0]));
  const gates = tensor([3, 2], Float32Array.from([0.3, 0.7, 0.5, 0.5, 0.9, 0.1]));
  const ex = [0, 1].map((e) => ({ w1: linearWeights(4, 6, 10 + e), w2: linearWeights(6, 4, 20 + e) }));
  const { out, dropped } = moeDispatch(x, gates, ex, 1);
  assert.equal(dropped, 0);
  for (let t = 0; t < 3; t++) for (let c = 0; c < 4; c++) {
    let ref = 0;
    for (let e = 0; e < 2; e++) {
      const row = tensor([4], x.data.slice(t * 4, t * 4 + 4));
      ref += gates.data[t * 2 + e] * linear(relu(linear(row, ex[e].w1, 6)), ex[e].w2, 4).data[c];
    }
    assert.ok(Math.abs(out.data[t * 4 + c] - ref) < 1e-5);
  }
});

test('low capacity drops words; capacity E/k never does; routing does not depend on block ids', () => {
  const stats = (factor, pad = 0) => {
    const { g, x } = sparse({ E: 4, k: 1 });
    for (let i = 0; i < pad; i++) g.nextId++;
    x.params.capacity = factor;
    const o = addNode(g, 'output', 0, 0, { expect: [8, 8] }); connect(g, x.id, o.id);
    return moeStats(g, analyze(g)).dropped;
  };
  assert.ok(stats(0.5) > 0, 'half capacity drops words');
  assert.equal(stats(4), 0, 'capacity E/k = 4 keeps every word');
  assert.equal(stats(1), stats(1, 57), 'same routing whatever the ids');
});
