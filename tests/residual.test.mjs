import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze, checkLevel } from '../src/sims/build/graph.js';
import { createRunner } from '../src/sims/build/forward.js';
import { digit } from '../src/sims/build/digits.js';
import { add, layernorm, rms, tensor } from '../src/sims/build/tensor.js';

// Input → stem conv → `depth` stages of (conv → relu), optionally with a skip around each stage,
// or a LayerNorm after it. Returns the graph and the last block.
function deep(depth, { skip = false, norm = false } = {}) {
  const g = createGraph();
  const inp = addNode(g, 'input');
  let prev = addNode(g, 'conv', 0, 0, { out: 8, k: 3, p: 1 });
  connect(g, inp.id, prev.id);
  for (let i = 0; i < depth; i++) {
    const c = addNode(g, 'conv', 0, 0, { out: 8, k: 3, p: 1 }), r = addNode(g, 'relu');
    connect(g, prev.id, c.id); connect(g, c.id, r.id);
    let last = r;
    if (skip) { const a = addNode(g, 'add'); connect(g, r.id, a.id, 0); connect(g, prev.id, a.id, 1); last = a; }
    if (norm) { const nm = addNode(g, 'norm'); connect(g, last.id, nm.id); last = nm; }
    prev = last;
  }
  return { g, inp, last: prev };
}

function signal(opts, gain) {
  const { g, inp, last } = deep(8, opts);
  const out = createRunner().run(g, analyze(g), digit(3), 1, 'd3', gain);
  return rms(out.get(last.id)) / rms(out.get(inp.id));
}

test('add and layernorm ops', () => {
  const a = tensor([3], Float32Array.from([1, 2, 3])), b = tensor([3], Float32Array.from([10, 20, 30]));
  assert.deepEqual([...add(a, b).data], [11, 22, 33]);
  const n = layernorm(tensor([4], Float32Array.from([1, 2, 3, 4])));
  assert.ok(Math.abs(n.data.reduce((s, v) => s + v, 0)) < 1e-5);
  assert.ok(Math.abs(rms(n) - 1) < 1e-3);
});

test('at init scale 0.5 a deep plain stack fades; skips or LayerNorm keep the signal', () => {
  const plain = signal({}, 0.5), residual = signal({ skip: true }, 0.5), normed = signal({ norm: true }, 0.5);
  assert.ok(plain < 0.02, `plain ${plain}`);
  assert.ok(residual > 0.3, `residual ${residual}`);
  assert.ok(normed > 0.3, `norm ${normed}`);
});

test('signalMin rule uses real outputs', () => {
  const lvl = { budgets: [1e9, 1e9, 1e9], signalMin: 0.1 };
  for (const [opts, want] of [[{}, false], [{ skip: true }, true]]) {
    const { g, last } = deep(8, opts);
    const o = addNode(g, 'output', 0, 0, { expect: [8, 28, 28] });
    connect(g, last.id, o.id);
    const a = analyze(g);
    assert.match(checkLevel(g, a, lvl).msg, /values/);
    const outs = createRunner().run(g, a, digit(3), 1, 'd3', 0.5);
    assert.equal(checkLevel(g, a, lvl, outs).done, want);
  }
});

import { WS_LEVELS, setupLevel } from '../src/sims/build/chapter2.js';

// Build a chapter-2b level: stem conv, then `stages` of conv→relu, each optionally wrapped in a skip
// (projection = a conv on the skip path) and/or followed by LayerNorm.
function level2b(id, { stages, skip, norm, main = { out: 8, k: 3, p: 1 }, proj }) {
  const L = WS_LEVELS.find((l) => l.id === id), g = createGraph();
  setupLevel(g, L);
  const [inp, out] = g.nodes;
  let prev = addNode(g, 'conv', 0, 0, { out: 8, k: 3, p: 1 });
  connect(g, inp.id, prev.id);
  for (let i = 0; i < stages; i++) {
    const c = addNode(g, 'conv', 0, 0, main), r = addNode(g, 'relu');
    connect(g, prev.id, c.id); connect(g, c.id, r.id);
    let last = r;
    if (skip) {
      const a = addNode(g, 'add');
      connect(g, r.id, a.id, 0);
      if (proj) { const pc = addNode(g, 'conv', 0, 0, proj); connect(g, prev.id, pc.id); connect(g, pc.id, a.id, 1); }
      else connect(g, prev.id, a.id, 1);
      last = a;
    }
    if (norm) { const nm = addNode(g, 'norm'); connect(g, last.id, nm.id); last = nm; }
    prev = last;
  }
  connect(g, prev.id, out.id);
  const a = analyze(g);
  const outs = createRunner().run(g, a, digit(3), 1, 'd3', L.initScale ?? 1);
  return checkLevel(g, a, L, outs);
}

test('chapter 2b levels each have a 3★ answer', () => {
  for (const [id, opts] of [
    ['rs-skip', { stages: 1, skip: true }],
    ['rs-match', { stages: 1, skip: true, main: { out: 16, k: 3, s: 2, p: 1 }, proj: { out: 16, k: 1, s: 2, p: 0 } }],
    ['rs-deep', { stages: 7, skip: true }],
    ['rs-norm', { stages: 7, norm: true }],
  ]) {
    const r = level2b(id, opts);
    assert.equal(r.done, true, `${id}: ${r.msg}`);
    assert.equal(r.stars, 3, `${id}: ${r.params} params`);
  }
});

test('chapter 2b rejects the shortcuts', () => {
  assert.equal(level2b('rs-deep', { stages: 7 }).done, false, 'plain deep stack');
  assert.match(level2b('rs-norm', { stages: 7, norm: false }).msg, /LayerNorm/, 'no norm on the Normalize level');
  assert.match(level2b('rs-skip', { stages: 1 }).msg, /Add|skip/i, 'no Add');
  const threeByThreeSkip = level2b('rs-match', { stages: 1, skip: true, main: { out: 16, k: 3, s: 2, p: 1 }, proj: { out: 16, k: 3, s: 2, p: 1 } });
  assert.equal(threeByThreeSkip.stars, 2, 'a 3×3 projection costs a star');
});

test('QA exploits on 2b are closed', () => {
  // Matching shapes: downsample first, then x + ReLU(x) — no shape is ever matched
  {
    const L = WS_LEVELS.find((l) => l.id === 'rs-match'), g = createGraph();
    setupLevel(g, L);
    const [inp, out] = g.nodes;
    const c = addNode(g, 'conv', 0, 0, { out: 16, k: 1, s: 2, p: 0 }), r = addNode(g, 'relu'), a = addNode(g, 'add');
    connect(g, inp.id, c.id); connect(g, c.id, r.id); connect(g, r.id, a.id, 0); connect(g, c.id, a.id, 1); connect(g, a.id, out.id);
    assert.equal(checkLevel(g, analyze(g), L).done, false);
  }
  // Deep & alive: 8 one-layer convs off the Input merged by a chain of Adds — no depth
  {
    const L = WS_LEVELS.find((l) => l.id === 'rs-deep'), g = createGraph();
    setupLevel(g, L);
    const [inp, out] = g.nodes;
    const convs = Array.from({ length: 8 }, () => { const c = addNode(g, 'conv', 0, 0, { out: 8, k: 1, p: 0 }); connect(g, inp.id, c.id); return c; });
    let acc = convs[0];
    for (const c of convs.slice(1)) { const a = addNode(g, 'add'); connect(g, acc.id, a.id, 0); connect(g, c.id, a.id, 1); acc = a; }
    connect(g, acc.id, out.id);
    const an = analyze(g);
    const r = checkLevel(g, an, L, createRunner().run(g, an, digit(3), 1, 'd3', 0.5));
    assert.equal(r.done, false);
    assert.match(r.msg, /in one chain/);
  }
  // a plain deep stack now hears about the fading signal first
  assert.match(level2b('rs-deep', { stages: 7 }).msg, /fades/);
});
