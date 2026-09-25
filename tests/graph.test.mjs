import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, disconnect, removeNode, analyze, checkLevel } from '../src/sims/build/graph.js';

const LEVEL = { requires: ['conv'], budgets: [500_000, 100_000, 25_000] };

function chain(...specs) {
  const g = createGraph();
  const nodes = specs.map(([type, params]) => addNode(g, type, 0, 0, params));
  for (let i = 1; i < nodes.length; i++) assert.ok(connect(g, nodes[i - 1].id, nodes[i].id).ok);
  return { g, nodes };
}

test('small CNN infers shapes and params, and wins the level with 3 stars', () => {
  const { g, nodes } = chain(['input'], ['conv', { out: 8, k: 3, s: 1, p: 1 }], ['relu'], ['pool', { k: 2 }], ['flatten'], ['linear', { out: 10 }], ['output']);
  const a = analyze(g);
  assert.deepEqual(nodes.map((n) => a.get(n.id).shape), [[1, 28, 28], [8, 28, 28], [8, 28, 28], [8, 14, 14], [1568], [10], [10]]);
  assert.equal(a.get(nodes[1].id).params, 3 * 3 * 1 * 8 + 8);
  assert.equal(a.get(nodes[5].id).params, 1568 * 10 + 10);
  assert.deepEqual(checkLevel(g, a, LEVEL), { done: true, params: 80 + 15690, active: 80 + 15690, stars: 3 });
});

test('Linear on an image is a real error that names the fix', () => {
  const { g, nodes } = chain(['input'], ['linear', { out: 10 }], ['output']);
  const a = analyze(g);
  assert.equal(a.get(nodes[1].id).status, 'error');
  assert.match(a.get(nodes[1].id).reason, /Flatten/);
  assert.equal(a.get(nodes[2].id).status, 'pending');
});

test('an unwired block is pending, not an error', () => {
  const g = createGraph();
  const r = addNode(g, 'relu');
  assert.equal(analyze(g).get(r.id).status, 'pending');
});

test('kernel larger than the input is an error', () => {
  const { g, nodes } = chain(['input', { c: 1, h: 4, w: 4 }], ['conv', { k: 7, p: 0 }]);
  assert.equal(analyze(g).get(nodes[1].id).status, 'error');
});

test('an input takes one wire; a new wire replaces it', () => {
  const g = createGraph();
  const a = addNode(g, 'input'), b = addNode(g, 'input'), r = addNode(g, 'relu');
  connect(g, a.id, r.id);
  connect(g, b.id, r.id);
  assert.deepEqual(g.edges.map((e) => [e.from, e.to]), [[b.id, r.id]]);
});

test('outputs fan out; loops and bad ports are refused', () => {
  const g = createGraph();
  const i = addNode(g, 'input'), r1 = addNode(g, 'relu'), r2 = addNode(g, 'relu'), o = addNode(g, 'output');
  assert.ok(connect(g, i.id, r1.id).ok);
  assert.ok(connect(g, i.id, r2.id).ok);
  assert.ok(connect(g, r1.id, r2.id).ok);
  assert.equal(connect(g, r2.id, r1.id).ok, false, 'loop');
  assert.equal(connect(g, r1.id, r1.id).ok, false, 'self');
  assert.equal(connect(g, o.id, r1.id).ok, false, 'output has no output port');
  assert.equal(connect(g, r1.id, i.id).ok, false, 'input has no input port');
});

test('removing a block or wire drops its edges', () => {
  const { g, nodes } = chain(['input'], ['relu'], ['flatten']);
  removeNode(g, nodes[1].id);
  assert.equal(g.edges.length, 0);
  const { g: g2 } = chain(['input'], ['relu']);
  disconnect(g2, g2.edges[0].id);
  assert.equal(g2.edges.length, 0);
});

test('level: needs a conv, and stars fall with parameter count', () => {
  const flat = chain(['input'], ['flatten'], ['linear', { out: 10 }], ['output']);
  assert.match(checkLevel(flat.g, analyze(flat.g), LEVEL).msg, /Conv2d/);
  const big = chain(['input'], ['conv', { out: 32, k: 3, p: 1 }], ['flatten'], ['linear', { out: 10 }], ['output']);
  const r = checkLevel(big.g, analyze(big.g), LEVEL);
  assert.equal(r.done, true);
  assert.equal(r.stars, 1); // 32·28·28·10 ≈ 251k weights in the Linear
});

test('stray blocks off the Output path do not count toward params', () => {
  const { g } = chain(['input'], ['conv', { out: 8 }], ['pool'], ['flatten'], ['linear', { out: 10 }], ['output']);
  const before = checkLevel(g, analyze(g), LEVEL).params;
  const stray = addNode(g, 'conv', 0, 0, { out: 64, k: 5 });
  connect(g, g.nodes[0].id, stray.id);
  assert.equal(checkLevel(g, analyze(g), LEVEL).params, before);
});

test('receptive field: stacked 3×3 convs grow by 2, pooling doubles the step', () => {
  const { g, nodes } = chain(['input'], ['conv', { k: 3, p: 1 }], ['conv', { k: 3, p: 1 }], ['conv', { k: 3, p: 1 }], ['pool', { k: 2, s: 2 }], ['conv', { k: 3, p: 1 }]);
  const a = analyze(g);
  assert.deepEqual(nodes.map((n) => a.get(n.id).rf.size), [1, 3, 5, 7, 8, 12]);
  assert.equal(a.get(nodes[4].id).rf.jump, 2);
  assert.equal(a.get(nodes[1].id).rf.start, 0, 'padding 1 keeps the first 3×3 window centred on pixel 0');
});

test('GlobalAvgPool makes [C]; Linear and GAP make the receptive field global', () => {
  const { g, nodes } = chain(['input'], ['conv', { out: 10, k: 3 }], ['gap'], ['output']);
  const a = analyze(g);
  assert.deepEqual(a.get(nodes[2].id).shape, [10]);
  assert.equal(a.get(nodes[2].id).rf.global, true);
  assert.equal(a.get(nodes[3].id).status, 'ok');
});

test('MaxPool stride knob and Output target shapes', () => {
  const { g, nodes } = chain(['input'], ['pool', { k: 3, s: 1 }], ['output', { expect: [1, 26, 26] }]);
  const a = analyze(g);
  assert.deepEqual(a.get(nodes[1].id).shape, [1, 26, 26]);
  assert.equal(a.get(nodes[2].id).status, 'ok');
  nodes[2].params.expect = [1, 13, 13];
  assert.match(analyze(g).get(nodes[2].id).reason, /expects \[1, 13, 13\]/);
});

test('level rules: forbids, max kernel, min receptive field', () => {
  const base = { budgets: [1e9, 1e9, 1e9] };
  const pooled = chain(['input'], ['conv', { out: 16, k: 3, p: 1 }], ['pool'], ['pool'], ['output', { expect: [16, 7, 7] }]);
  assert.match(checkLevel(pooled.g, analyze(pooled.g), { ...base, forbids: ['pool'] }).msg, /bans MaxPool2d/);
  const patch = chain(['input'], ['conv', { out: 16, k: 4, s: 4, p: 0 }], ['output', { expect: [16, 7, 7] }]);
  assert.equal(checkLevel(patch.g, analyze(patch.g), { ...base, forbids: ['pool'] }).done, true);
  assert.match(checkLevel(patch.g, analyze(patch.g), { ...base, maxK: 3 }).msg, /3×3/);
  const two = chain(['input'], ['conv', { k: 3, p: 1 }], ['conv', { k: 3, p: 1 }], ['output', { expect: [8, 28, 28] }]);
  assert.match(checkLevel(two.g, analyze(two.g), { ...base, rfMin: 7 }).msg, /5×5/);
  const three = chain(['input'], ['conv', { k: 3, p: 1 }], ['conv', { k: 3, p: 1 }], ['conv', { k: 3, p: 1 }], ['output', { expect: [8, 28, 28] }]);
  assert.equal(checkLevel(three.g, analyze(three.g), { ...base, rfMin: 7 }).done, true);
});

test('requires counts repeated block types', () => {
  const lvl = { requires: ['conv', 'conv', 'pool', 'pool'], budgets: [1e9, 1e9, 1e9] };
  const one = chain(['input'], ['conv', { p: 1 }], ['pool'], ['flatten'], ['linear'], ['output']);
  assert.match(checkLevel(one.g, analyze(one.g), lvl).msg, /at least 2 Conv2d/);
  const two = chain(['input'], ['conv', { p: 1 }], ['pool'], ['conv', { p: 1 }], ['pool'], ['flatten'], ['linear'], ['output']);
  assert.equal(checkLevel(two.g, analyze(two.g), lvl).done, true);
});

test('Add takes two inputs on separate ports and needs matching shapes', () => {
  const g = createGraph();
  const i = addNode(g, 'input'), c1 = addNode(g, 'conv', 0, 0, { out: 8, k: 3, p: 1 }), c2 = addNode(g, 'conv', 0, 0, { out: 8, k: 3, p: 1 });
  const add = addNode(g, 'add');
  connect(g, i.id, c1.id); connect(g, c1.id, c2.id);
  connect(g, c2.id, add.id, 0);
  assert.equal(analyze(g).get(add.id).status, 'pending');
  connect(g, c1.id, add.id, 1); // the skip wire
  const a = analyze(g);
  assert.equal(a.get(add.id).status, 'ok');
  assert.deepEqual(a.get(add.id).shape, [8, 28, 28]);
  assert.equal(a.get(add.id).rf.size, 5, 'Add sees as far as its longer path');
  assert.equal(g.edges.filter((e) => e.to === add.id).length, 2);
  connect(g, i.id, add.id, 1); // replaces only port 1
  const bad = analyze(g).get(add.id);
  assert.equal(bad.status, 'error');
  assert.match(bad.reason, /1×1 Conv2d.*channels/);
  assert.equal(connect(g, i.id, add.id, 2).ok, false, 'no third port');
});

test('LayerNorm keeps the shape and has 2 parameters per channel', () => {
  const { g, nodes } = chain(['input', { c: 3 }], ['norm'], ['flatten'], ['norm']);
  const a = analyze(g);
  assert.deepEqual(a.get(nodes[1].id).shape, [3, 28, 28]);
  assert.equal(a.get(nodes[1].id).params, 6);
  assert.equal(a.get(nodes[3].id).params, 2 * 3 * 28 * 28);
});

test('needsSkip: identity and projection skips count; x + x and twin branches do not', () => {
  const lvl = { needsSkip: true, budgets: [1e9, 1e9, 1e9] };
  const build = (wire) => {
    const g = createGraph();
    const i = addNode(g, 'input'), stem = addNode(g, 'conv', 0, 0, { out: 8, k: 3, p: 1 }), add = addNode(g, 'add'), o = addNode(g, 'output', 0, 0, { expect: [8, 28, 28] });
    connect(g, i.id, stem.id); wire(g, stem, add); connect(g, add.id, o.id);
    return checkLevel(g, analyze(g), lvl).done;
  };
  const conv = (g) => addNode(g, 'conv', 0, 0, { out: 8, k: 3, p: 1 });
  assert.equal(build((g, s, a) => { const c = conv(g); connect(g, s.id, c.id); connect(g, c.id, a.id, 0); connect(g, s.id, a.id, 1); }), true, 'identity');
  assert.equal(build((g, s, a) => { const c = conv(g), r = addNode(g, 'relu'), p = conv(g); connect(g, s.id, c.id); connect(g, c.id, r.id); connect(g, r.id, a.id, 0); connect(g, s.id, p.id); connect(g, p.id, a.id, 1); }), true, 'projection');
  assert.equal(build((g, s, a) => { connect(g, s.id, a.id, 0); connect(g, s.id, a.id, 1); }), false, 'x + x');
  assert.equal(build((g, s, a) => { const c1 = conv(g), c2 = conv(g); connect(g, s.id, c1.id); connect(g, s.id, c2.id); connect(g, c1.id, a.id, 0); connect(g, c2.id, a.id, 1); }), false, 'twin branches');
});
