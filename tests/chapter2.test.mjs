import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze, checkLevel } from '../src/sims/build/graph.js';
import { WS_LEVELS, setupLevel } from '../src/sims/build/chapter2.js';

// Build a level's board, then chain `blocks` from Input to Output.
function solve(id, blocks) {
  const L = WS_LEVELS.find((l) => l.id === id), g = createGraph();
  setupLevel(g, L);
  const [inp, out] = g.nodes;
  let prev = inp;
  for (const [t, p] of blocks) { const n = addNode(g, t, 0, 0, p); assert.ok(connect(g, prev.id, n.id).ok); prev = n; }
  connect(g, prev.id, out.id);
  return checkLevel(g, analyze(g), L);
}

const THREE_STAR = {
  'ws-digits': [['conv', { out: 8, k: 3, p: 1 }], ['relu'], ['pool'], ['flatten'], ['linear', { out: 10 }]],
  'ws-down': [['conv', { out: 8, k: 3, p: 1 }], ['pool']],
  'ws-stride': [['conv', { out: 16, k: 4, s: 4, p: 0 }]],
  'ws-rf': [['conv', { out: 8, k: 3, p: 1 }], ['relu'], ['conv', { out: 8, k: 3, p: 1 }], ['relu'], ['conv', { out: 8, k: 3, p: 1 }]],
  'ws-gap': [['conv', { out: 8, k: 3, p: 1 }], ['relu'], ['pool'], ['conv', { out: 10, k: 3, p: 1 }], ['gap']],
  'ws-tiny': [['conv', { out: 4, k: 3, p: 1 }], ['relu'], ['pool'], ['conv', { out: 8, k: 3, p: 1 }], ['relu'], ['pool', { k: 2, s: 2 }], ['conv', { out: 10, k: 3, p: 1 }], ['gap']],
};

for (const L of WS_LEVELS.filter((l) => l.chapter === '2')) {
  test(`${L.id}: a reference solution earns 3 stars`, () => {
    const r = solve(L.id, THREE_STAR[L.id]);
    assert.equal(r.done, true, r.msg);
    assert.equal(r.stars, 3, `${r.params} params`);
  });
}

test('levels reject the lazy answer', () => {
  assert.match(solve('ws-stride', [['conv', { out: 16, k: 3, p: 1 }], ['pool'], ['pool']]).msg, /bans MaxPool2d/);
  assert.match(solve('ws-rf', [['conv', { out: 8, k: 7, p: 3 }]]).msg, /3×3/);
  assert.match(solve('ws-rf', [['conv', { out: 8, k: 3, p: 1 }], ['conv', { out: 8, k: 3, p: 1 }]]).msg, /5×5/);
  assert.match(solve('ws-gap', [['conv', { out: 8, k: 3, p: 1 }], ['pool'], ['flatten'], ['linear']]).msg, /GlobalAvgPool/);
  assert.match(solve('ws-tiny', [['conv', { out: 10, k: 3, p: 1 }], ['pool'], ['gap']]).msg, /at least 2/);
});

test('QA exploits are closed', () => {
  // "See 7×7" without stacking 3×3 convs: big pools then a 1×1 conv
  assert.equal(solve('ws-rf', [['pool', { k: 4, s: 1 }], ['pool', { k: 4, s: 1 }], ['conv', { out: 8, k: 1, p: 3 }]]).done, false);
  // classifier heads must sit on features that see a real patch of the digit
  assert.match(solve('ws-gap', [['conv', { out: 10, k: 1, p: 0 }], ['gap']]).msg, /head/);
  assert.match(solve('ws-tiny', [['conv', { out: 4, k: 1 }], ['pool'], ['conv', { out: 10, k: 1 }], ['pool'], ['gap']]).msg, /head/);
});

test('the patchify trick is reachable with the stride knob', async () => {
  const { KNOBS } = await import('../src/sims/build/chapter2.js');
  const stride = KNOBS.conv.find(([k]) => k === 's');
  assert.ok(stride[3] >= 4, 'Conv stride slider reaches 4');
});
