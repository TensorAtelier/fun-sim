import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze } from '../src/sims/build/graph.js';
import { createRunner } from '../src/sims/build/forward.js';
import { digit } from '../src/sims/build/digits.js';

function cnn() {
  const g = createGraph();
  const ids = [['input'], ['conv', { out: 8, k: 3, p: 1 }], ['relu'], ['pool', { k: 2 }], ['flatten'], ['linear', { out: 10 }], ['output']]
    .map(([t, p]) => addNode(g, t, 0, 0, p).id);
  for (let i = 1; i < ids.length; i++) connect(g, ids[i - 1], ids[i]);
  return { g, ids };
}

test('digits are 28×28 ink in [0, 1], distinct from each other', () => {
  const ds = Array.from({ length: 10 }, (_, d) => digit(d));
  for (const t of ds) {
    assert.deepEqual(t.shape, [1, 28, 28]);
    const ink = t.data.reduce((a, v) => a + v, 0);
    assert.ok(ink > 40 && ink < 400, `ink ${ink}`);
    assert.ok(Math.min(...t.data) >= 0 && Math.max(...t.data) <= 1);
  }
  for (let a = 0; a < 10; a++) for (let b = a + 1; b < 10; b++) {
    const diff = ds[a].data.reduce((s, v, i) => s + Math.abs(v - ds[b].data[i]), 0);
    assert.ok(diff > 30, `digits ${a} and ${b} too similar`);
  }
});

test('forward pass gives every block a tensor of its inferred shape; Output is a distribution', () => {
  const { g, ids } = cnn();
  const a = analyze(g);
  const out = createRunner().run(g, a, digit(3), 1, 'd3');
  for (const id of ids) assert.deepEqual(out.get(id).shape, a.get(id).shape);
  const probs = out.get(ids.at(-1)).data;
  assert.ok(Math.abs(probs.reduce((s, v) => s + v, 0) - 1) < 1e-5);
  assert.ok([...out.get(ids[2]).data].every((v) => v >= 0), 'ReLU output is non-negative');
});

test('deterministic per seed; cache reused until something upstream changes', () => {
  const { g, ids } = cnn();
  const a = analyze(g);
  const r = createRunner();
  const o1 = r.run(g, a, digit(3), 1, 'd3');
  const o2 = r.run(g, a, digit(3), 1, 'd3');
  assert.equal(o1.get(ids[5]), o2.get(ids[5]), 'same tensor object from cache');
  const fresh = createRunner().run(g, a, digit(3), 1, 'd3');
  assert.deepEqual([...fresh.get(ids[5]).data], [...o1.get(ids[5]).data]);
  const o3 = r.run(g, a, digit(3), 2, 'd3');
  assert.notDeepEqual([...o3.get(ids[5]).data], [...o1.get(ids[5]).data], 'new seed, new weights');
  const o4 = r.run(g, a, digit(7), 1, 'd7');
  assert.notEqual(o4.get(ids[1]), o1.get(ids[1]), 'new input recomputes');
  const base = r.run(g, a, digit(3), 1, 'd3'); // the cache keeps one result per block
  Object.assign(g.nodes.find((n) => n.id === ids[3]).params, { k: 4, s: 4 });
  const a2 = analyze(g);
  const o5 = r.run(g, a2, digit(3), 1, 'd3');
  assert.equal(o5.get(ids[1]), base.get(ids[1]), 'blocks above the change are reused');
  assert.notEqual(o5.get(ids[3]), base.get(ids[3]), 'the changed block recomputes');
  assert.deepEqual(o5.get(ids[3]).shape, [8, 7, 7]);
});

test('blocks that are pending or in error get no output', () => {
  const g = createGraph();
  const i = addNode(g, 'input'), l = addNode(g, 'linear'), o = addNode(g, 'output');
  connect(g, i.id, l.id); connect(g, l.id, o.id);
  const out = createRunner().run(g, analyze(g), digit(1), 1, 'd1');
  assert.ok(out.has(i.id));
  assert.ok(!out.has(l.id) && !out.has(o.id));
});

test('rewiring to an identically configured block recomputes downstream (no stale cache)', () => {
  const g = createGraph();
  const i = addNode(g, 'input'), c1 = addNode(g, 'conv'), c2 = addNode(g, 'conv'), r = addNode(g, 'relu');
  connect(g, i.id, c1.id); connect(g, i.id, c2.id); connect(g, c1.id, r.id);
  const runner = createRunner();
  runner.run(g, analyze(g), digit(3), 1, 'd3');
  connect(g, c2.id, r.id);
  const a = analyze(g);
  const cached = runner.run(g, a, digit(3), 1, 'd3').get(r.id);
  const fresh = createRunner().run(g, a, digit(3), 1, 'd3').get(r.id);
  assert.deepEqual([...cached.data], [...fresh.data]);
});

test('cost estimate grows with conv size', async () => {
  const { estimateCost } = await import('../src/sims/build/forward.js');
  const g = createGraph();
  const i = addNode(g, 'input', 0, 0, { c: 1, h: 28, w: 28 }), c = addNode(g, 'conv', 0, 0, { out: 8, k: 3, p: 1 });
  connect(g, i.id, c.id);
  assert.equal(estimateCost(g, analyze(g)).macs, 8 * 28 * 28 * 1 * 9);
  c.params.out = 64;
  assert.equal(estimateCost(g, analyze(g)).macs, 64 * 28 * 28 * 9);
});
