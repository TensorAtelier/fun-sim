import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze } from '../src/sims/build/graph.js';

test('positional encoding and the packaged block keep [T, D]; params add up', () => {
  const g = createGraph();
  const s = addNode(g, 'seq'), pe = addNode(g, 'posenc'), b = addNode(g, 'tblock', 0, 0, { heads: 2, ffn: 32 });
  connect(g, s.id, pe.id); connect(g, pe.id, b.id);
  const a = analyze(g);
  assert.deepEqual(a.get(pe.id).shape, [8, 8]);
  assert.equal(a.get(pe.id).params, 0);
  assert.deepEqual(a.get(b.id).shape, [8, 8]);
  // attention 4·(8·8+8) + MLP (8·32+32) + (32·8+8) + two LayerNorms 2·2·8
  assert.equal(a.get(b.id).params, 288 + 288 + 264 + 32);
  b.params.heads = 3;
  assert.match(analyze(g).get(b.id).reason, /evenly/);
});

test('LayerNorm on a sentence has 2 parameters per feature', () => {
  const g = createGraph();
  const s = addNode(g, 'seq'), l = addNode(g, 'linear', 0, 0, { out: 6 }), n = addNode(g, 'norm');
  connect(g, s.id, l.id); connect(g, l.id, n.id);
  assert.equal(analyze(g).get(n.id).params, 12);
});

import { createRunner, orderSensitivity } from '../src/sims/build/forward.js';
import { layernorm, positionalEncoding, tensor, rms } from '../src/sims/build/tensor.js';

function stack({ pos, blocks = 1 }) {
  const g = createGraph();
  let prev = addNode(g, 'seq');
  if (pos) { const p = addNode(g, 'posenc'); connect(g, prev.id, p.id); prev = p; }
  for (let i = 0; i < blocks; i++) { const b = addNode(g, 'tblock'); connect(g, prev.id, b.id); prev = b; }
  const o = addNode(g, 'output', 0, 0, { expect: [8, 8] });
  connect(g, prev.id, o.id);
  return g;
}

test('LayerNorm on a sentence normalises each word row; positional encoding is sin/cos', () => {
  const x = tensor([2, 4], Float32Array.from([1, 2, 3, 4, 10, 10, 10, 50]));
  const y = layernorm(x);
  for (const r of [0, 1]) {
    const row = y.data.slice(r * 4, r * 4 + 4);
    assert.ok(Math.abs(row.reduce((s, v) => s + v, 0)) < 1e-4);
    assert.ok(Math.abs(rms(tensor([4], row)) - 1) < 1e-3);
  }
  const pe = positionalEncoding(8, 8);
  assert.equal(pe.data[0], 0); assert.equal(pe.data[1], 1); // position 0: sin 0, cos 0
  assert.ok(Math.abs(pe.data[8] - Math.sin(1)) < 1e-6);
});

test('a block keeps every row normalised and finite', () => {
  const g = stack({ pos: true, blocks: 2 });
  const out = createRunner().run(g, analyze(g), null, 1, '');
  const last = g.nodes.find((n) => n.type === 'tblock' && g.edges.some((e) => e.from === n.id && g.nodes.find((m) => m.id === e.to).type === 'output'));
  const y = out.get(last.id);
  assert.ok([...y.data].every(Number.isFinite));
  assert.ok(Math.abs(rms(tensor([8], y.data.slice(0, 8))) - 1) < 1e-2);
});

test("without positions, swapping words leaves word 1's output unchanged; with them it changes", () => {
  const without = orderSensitivity(stack({ pos: false }), analyze(stack({ pos: false })));
  const withPos = orderSensitivity(stack({ pos: true }), analyze(stack({ pos: true })));
  console.log(`word-order sensitivity — no positions ${(without * 100).toExponential(1)}%, with positions ${(withPos * 100).toFixed(1)}%`);
  assert.ok(without < 1e-5, `without ${without}`);
  assert.ok(withPos > 0.05, `with ${withPos}`);
});
