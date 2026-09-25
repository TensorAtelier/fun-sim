import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze, checkLevel } from '../src/sims/build/graph.js';
import { attentionPeak } from '../src/sims/build/forward.js';
import { WS_LEVELS, setupLevel } from '../src/sims/build/chapter2.js';

// Set up a level, let `build(g, seq, n)` wire blocks and return the block feeding Output, then judge.
function play(id, build) {
  const L = WS_LEVELS.find((l) => l.id === id), g = createGraph();
  setupLevel(g, L);
  const seq = g.nodes.find((n) => n.type === 'seq'), out = g.nodes.find((n) => n.type === 'output');
  const n = (t, p) => addNode(g, t, 0, 0, p);
  connect(g, build(g, seq, n).id, out.id);
  const a = analyze(g);
  return checkLevel(g, a, L, null, L.peakMax ? { peak: attentionPeak(g, a) } : {});
}
const lin = (g, seq, n, d = 8) => { const l = n('linear', { out: d }); connect(g, seq.id, l.id); return l; };
function scores(g, seq, n) { const s = n('scores'); connect(g, lin(g, seq, n).id, s.id, 0); connect(g, lin(g, seq, n).id, s.id, 1); return s; }
function weights(g, seq, n, scaleD = 8) { const sc = n('scale', { d: scaleD }), sm = n('softmax'); connect(g, scores(g, seq, n).id, sc.id); connect(g, sc.id, sm.id); return sm; }

test('every chapter 4 level has a 3★ answer', () => {
  assert.equal(play('at-scores', scores).done, true);
  assert.equal(play('at-weights', weights).done, true);
  const head = play('at-head', (g, seq, n) => { const ws = n('wsum'); connect(g, weights(g, seq, n).id, ws.id, 0); connect(g, lin(g, seq, n).id, ws.id, 1); return ws; });
  assert.equal(head.done, true, head.msg);
  const multi = play('at-multi', (g, seq, n) => {
    const h1 = n('head', { d: 4 }), h2 = n('head', { d: 4 }), c = n('concat'), mix = n('linear', { out: 8 });
    connect(g, seq.id, h1.id); connect(g, seq.id, h2.id); connect(g, h1.id, c.id, 0); connect(g, h2.id, c.id, 1); connect(g, c.id, mix.id);
    return mix;
  });
  assert.equal(multi.done, true, multi.msg);
  assert.equal(multi.stars, 3, `${multi.params} params`);
});

test('chapter 4 rejects the shortcuts', () => {
  assert.match(play('at-weights', (g, seq, n) => weights(g, seq, n, 1)).msg, /width of Q and K.*% of its attention/, 'no real scaling: says so, with the measured saturation');
  // V taken from the Q projection: not three separate maps
  const sharedV = play('at-head', (g, seq, n) => {
    const q = lin(g, seq, n), k = lin(g, seq, n), s = n('scores'), sc = n('scale', { d: 8 }), sm = n('softmax'), ws = n('wsum');
    connect(g, q.id, s.id, 0); connect(g, k.id, s.id, 1); connect(g, s.id, sc.id); connect(g, sc.id, sm.id); connect(g, sm.id, ws.id, 0); connect(g, q.id, ws.id, 1);
    return ws;
  });
  assert.match(sharedV.msg, /V must be a third Linear/);
  // Two wide heads: correct but costs a star
  const wide = play('at-multi', (g, seq, n) => {
    const h1 = n('head', { d: 8 }), h2 = n('head', { d: 8 }), c = n('concat'), mix = n('linear', { out: 8 });
    connect(g, seq.id, h1.id); connect(g, seq.id, h2.id); connect(g, h1.id, c.id, 0); connect(g, h2.id, c.id, 1); connect(g, c.id, mix.id);
    return mix;
  });
  assert.equal(wide.stars, 2);
});

test('QA loopholes on chapter 4 are closed', () => {
  // Softmax not over the scores
  assert.equal(play('at-weights', (g, seq, n) => { const l = lin(g, seq, n), sm = n('softmax'), s = n('scores'); connect(g, l.id, sm.id); connect(g, sm.id, s.id, 0); connect(g, lin(g, seq, n).id, s.id, 1); return s; }).done, false);
  // narrow Q/K with no Scale
  assert.equal(play('at-weights', (g, seq, n) => { const s = n('scores'), sm = n('softmax'); connect(g, lin(g, seq, n, 2).id, s.id, 0); connect(g, lin(g, seq, n, 2).id, s.id, 1); connect(g, s.id, sm.id); return sm; }).done, false);
  // Scale with the wrong d
  assert.match(play('at-weights', (g, seq, n) => weights(g, seq, n, 64)).msg, /width of Q and K/);
  // Q and K from the same map
  assert.equal(play('at-scores', (g, seq, n) => { const l = lin(g, seq, n), s = n('scores'); connect(g, l.id, s.id, 0); connect(g, l.id, s.id, 1); return s; }).done, false);
  // stacked heads / one head twice
  const stacked = play('at-multi', (g, seq, n) => { const h1 = n('head', { d: 4 }), h2 = n('head', { d: 4 }), c = n('concat'), m = n('linear', { out: 8 }); connect(g, seq.id, h1.id); connect(g, h1.id, h2.id); connect(g, h1.id, c.id, 0); connect(g, h2.id, c.id, 1); connect(g, c.id, m.id); return m; });
  assert.equal(stacked.done, false);
  const twice = play('at-multi', (g, seq, n) => { const h = n('head', { d: 4 }), c = n('concat'), m = n('linear', { out: 8 }); connect(g, seq.id, h.id); connect(g, h.id, c.id, 0); connect(g, h.id, c.id, 1); connect(g, c.id, m.id); return m; });
  assert.equal(twice.done, false);
});
