import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze, checkLevel } from '../src/sims/build/graph.js';
import { moeStats } from '../src/sims/build/forward.js';
import { WS_LEVELS, setupLevel, KNOBS } from '../src/sims/build/chapter2.js';

// Set up a level; `build` wires blocks and returns the block to feed Output (or null if the level
// is pre-wired and only knobs change).
function play(id, build) {
  const L = WS_LEVELS.find((l) => l.id === id), g = createGraph();
  setupLevel(g, L);
  const seq = g.nodes.find((n) => n.type === 'seq'), out = g.nodes.find((n) => n.type === 'output');
  const n = (t, p) => addNode(g, t, 0, 0, p);
  const last = build(g, seq, n);
  if (last) connect(g, last.id, out.id);
  const a = analyze(g);
  return checkLevel(g, a, L, null, L.noDrops ? { dropped: moeStats(g, a)?.dropped } : {});
}
function sparse(g, seq, n, { E = 4, k = 1 } = {}) {
  const r = n('router', { experts: E }), t = n('topk', { k }), x = n('experts', { experts: E });
  connect(g, seq.id, r.id); connect(g, r.id, t.id); connect(g, seq.id, x.id, 0); connect(g, t.id, x.id, 1);
  return x;
}
const capacity = (c) => (g) => { g.nodes.find((m) => m.type === 'experts').params.capacity = c; return null; };

test('every chapter 6 level has a 3★ answer', () => {
  assert.equal(play('moe-route', (g, seq, n) => { const l = n('linear', { out: 4 }), s = n('softmax'); connect(g, seq.id, l.id); connect(g, l.id, s.id); return s; }).done, true);
  const sp = play('moe-sparse', (g, seq, n) => sparse(g, seq, n));
  assert.equal(sp.done, true, sp.msg);
  assert.ok(sp.active < sp.params / 2);
  const cap = play('moe-capacity', capacity(2));
  assert.equal(cap.done, true, cap.msg);
  assert.equal(cap.stars, 3);
  const scale = play('moe-scale', (g, seq, n) => { const pe = n('posenc'), m = n('moe', { experts: 16, k: 1, capacity: 8 }); connect(g, seq.id, pe.id); connect(g, pe.id, m.id); return m; });
  assert.equal(scale.done, true, scale.msg);
  assert.ok(KNOBS.moe.find(([k]) => k === 'capacity')[3] >= 8, 'capacity 8× is reachable in the UI');
});

test('chapter 6 rejects the shortcuts and teaches with its messages', () => {
  assert.match(play('moe-capacity', capacity(1)).msg, /dropped/);
  assert.equal(play('moe-capacity', capacity(4)).stars, 1, 'over-provisioning costs stars');
  assert.match(play('moe-sparse', (g, seq, n) => sparse(g, seq, n, { k: 2 })).msg, /at most 50%/, 'top-2 of 4 uses too much');
  // dense experts (gates straight from the router, no top-k)
  assert.match(play('moe-sparse', (g, seq, n) => { const r = n('router', { experts: 4 }), x = n('experts', { experts: 4 }); connect(g, seq.id, r.id); connect(g, seq.id, x.id, 0); connect(g, r.id, x.id, 1); return x; }).msg, /Top-k/);
  // routing collapse after an untrained Transformer block
  const collapsed = play('moe-scale', (g, seq, n) => { const pe = n('posenc'), tb = n('tblock'), m = n('moe', { experts: 16, k: 1, capacity: 8 }); connect(g, seq.id, pe.id); connect(g, pe.id, tb.id); connect(g, tb.id, m.id); return m; });
  assert.match(collapsed.msg, /dropped/);
});

test('QA: Capacity only lets you change capacity, and overshooting still completes', () => {
  const L = WS_LEVELS.find((l) => l.id === 'moe-capacity'), g = createGraph();
  setupLevel(g, L);
  const router = g.nodes.find((n) => n.type === 'router'), topk = g.nodes.find((n) => n.type === 'topk'), ex = g.nodes.find((n) => n.type === 'experts');
  assert.equal(router.locked, true);
  assert.equal(topk.locked, true);
  assert.deepEqual(ex.only, ['capacity'], 'only the capacity knob is editable');
  const over = play('moe-capacity', capacity(6));
  assert.equal(over.done, true, over.msg);
  assert.equal(over.stars, 1);
  assert.ok(KNOBS.router.find(([k]) => k === 'experts')[3] >= KNOBS.experts.find(([k]) => k === 'experts')[3], 'router can match any experts count');
});
