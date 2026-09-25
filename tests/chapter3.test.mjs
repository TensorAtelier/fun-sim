import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze, checkLevel } from '../src/sims/build/graph.js';
import { firstWordMemory } from '../src/sims/build/forward.js';
import { WS_LEVELS, setupLevel } from '../src/sims/build/chapter2.js';

const level = (id) => WS_LEVELS.find((l) => l.id === id);
const byType = (g, t) => g.nodes.filter((n) => n.type === t);

// Chain `type` cells from the zero state over every word, then into Output.
function unroll(id, type, hidden) {
  const L = level(id), g = createGraph();
  setupLevel(g, L);
  let prev = byType(g, 'state0')[0];
  for (const w of byType(g, 'token')) {
    const c = addNode(g, type, 0, 0, { hidden });
    connect(g, prev.id, c.id, 0); connect(g, w.id, c.id, 1); prev = c;
  }
  connect(g, prev.id, byType(g, 'output')[0].id);
  const a = analyze(g);
  return checkLevel(g, a, L, null, L.memoryMin ? { memory: firstWordMemory(g, a) } : {});
}

test('Unroll: four shared RNN steps', () => {
  const r = unroll('sq-unroll', 'rnn', 16);
  assert.equal(r.done, true, r.msg);
  assert.equal(r.params, 400);
});

test('The bottleneck: a plain RNN is refused, an LSTM passes', () => {
  assert.match(unroll('sq-bottleneck', 'rnn', 32).msg, /barely survives/);
  const r = unroll('sq-bottleneck', 'lstm', 16);
  assert.equal(r.done, true, r.msg);
});

// One LSTM step wired from parts: z = [x, h]; f,i,o = σ(Linear z); g = tanh(Linear z);
// c' = f⊙c + i⊙g; h' = o⊙tanh(c').
function lstmFromParts({ skipOutputGate = false } = {}) {
  const L = level('sq-parts'), g = createGraph();
  setupLevel(g, L);
  const [x] = byType(g, 'token'), [h, c] = byType(g, 'state0'), [out] = byType(g, 'output');
  const node = (t, p) => addNode(g, t, 0, 0, p);
  const cat = node('concat'); connect(g, x.id, cat.id, 0); connect(g, h.id, cat.id, 1);
  const gate = (act) => { const l = node('linear', { out: 16 }), a = node(act); connect(g, cat.id, l.id); connect(g, l.id, a.id); return a; };
  const f = gate('sigmoid'), i = gate('sigmoid'), gg = gate('tanh');
  const fc = node('mul'); connect(g, f.id, fc.id, 0); connect(g, c.id, fc.id, 1);
  const ig = node('mul'); connect(g, i.id, ig.id, 0); connect(g, gg.id, ig.id, 1);
  const cNew = node('add'); connect(g, fc.id, cNew.id, 0); connect(g, ig.id, cNew.id, 1);
  const tc = node('tanh'); connect(g, cNew.id, tc.id);
  if (skipOutputGate) connect(g, tc.id, out.id);
  else { const o = gate('sigmoid'), hNew = node('mul'); connect(g, o.id, hNew.id, 0); connect(g, tc.id, hNew.id, 1); connect(g, hNew.id, out.id); }
  return checkLevel(g, analyze(g), L);
}

test('LSTM cell from parts: the full cell passes; leaving out the output gate does not', () => {
  const r = lstmFromParts();
  assert.equal(r.done, true, r.msg);
  assert.equal(r.params, 4 * (24 * 16 + 16));
  assert.equal(lstmFromParts({ skipOutputGate: true }).done, false);
});

test('Unroll needs every word', () => {
  const L = level('sq-unroll'), g = createGraph();
  setupLevel(g, L);
  let prev = byType(g, 'state0')[0];
  for (const w of byType(g, 'token').slice(0, 3)) { const c = addNode(g, 'rnn'); connect(g, prev.id, c.id, 0); connect(g, w.id, c.id, 1); prev = c; }
  const c4 = addNode(g, 'rnn'); connect(g, prev.id, c4.id, 0); connect(g, byType(g, 'token')[0].id, c4.id, 1);
  connect(g, c4.id, byType(g, 'output')[0].id);
  assert.match(checkLevel(g, analyze(g), L).msg, /word 4 \("down"\) is unused/);
});

// Build a chain over the words in a given order (indices into the level's words).
function chainInOrder(id, type, hidden, order) {
  const L = level(id), g = createGraph();
  setupLevel(g, L);
  const ws = byType(g, 'token');
  let prev = byType(g, 'state0')[0];
  for (const k of order) { const c = addNode(g, type, 0, 0, { hidden }); connect(g, prev.id, c.id, 0); connect(g, ws[k].id, c.id, 1); prev = c; }
  connect(g, prev.id, byType(g, 'output')[0].id);
  const a = analyze(g);
  return checkLevel(g, a, L, null, L.memoryMin ? { memory: firstWordMemory(g, a) } : {});
}

test('QA exploits on chapter 3 are closed', () => {
  // word 1 read last, or read twice, to keep it fresh
  assert.match(chainInOrder('sq-bottleneck', 'rnn', 32, [1, 2, 3, 4, 5, 6, 7, 0]).msg, /in order/);
  assert.match(chainInOrder('sq-bottleneck', 'rnn', 32, [0, 1, 2, 3, 4, 5, 6, 7, 0]).msg, /once/);
  assert.match(chainInOrder('sq-unroll', 'rnn', 16, [3, 2, 1, 0]).msg, /in order/);
  assert.equal(chainInOrder('sq-bottleneck', 'lstm', 16, [0, 1, 2, 3, 4, 5, 6, 7]).done, true);
  // a gate-free chain with the right part counts
  const L = level('sq-parts'), g = createGraph();
  setupLevel(g, L);
  const [x] = byType(g, 'token'), [h, c] = byType(g, 'state0'), [out] = byType(g, 'output');
  const n = (t, p) => addNode(g, t, 0, 0, p);
  const cat = n('concat'); connect(g, x.id, cat.id, 0); connect(g, h.id, cat.id, 1);
  let prev = cat;
  for (const t of ['linear', 'sigmoid', 'linear', 'sigmoid', 'linear', 'sigmoid', 'linear', 'tanh', 'tanh']) { const b = n(t, t === 'linear' ? { out: 16 } : {}); connect(g, prev.id, b.id); prev = b; }
  const m1 = n('mul'); connect(g, prev.id, m1.id, 0); connect(g, c.id, m1.id, 1);
  const m2 = n('mul'); connect(g, m1.id, m2.id, 0); connect(g, m1.id, m2.id, 1);
  const m3 = n('mul'); connect(g, m2.id, m3.id, 0); connect(g, m2.id, m3.id, 1);
  const ad = n('add'); connect(g, m3.id, ad.id, 0); connect(g, m3.id, ad.id, 1);
  connect(g, ad.id, out.id);
  assert.match(checkLevel(g, analyze(g), L).msg, /LSTM/);
});
