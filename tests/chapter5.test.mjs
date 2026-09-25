import test from 'node:test';
import assert from 'node:assert/strict';
import { createGraph, addNode, connect, analyze, checkLevel } from '../src/sims/build/graph.js';
import { orderSensitivity } from '../src/sims/build/forward.js';
import { WS_LEVELS, setupLevel } from '../src/sims/build/chapter2.js';

function play(id, build) {
  const L = WS_LEVELS.find((l) => l.id === id), g = createGraph();
  setupLevel(g, L);
  const seq = g.nodes.find((n) => n.type === 'seq'), out = g.nodes.find((n) => n.type === 'output');
  const n = (t, p) => addNode(g, t, 0, 0, p);
  connect(g, build(g, seq, n).id, out.id);
  const a = analyze(g);
  return checkLevel(g, a, L, null, L.orderMin ? { order: orderSensitivity(g, a) } : {});
}

// A block from parts on top of x; `attn` builds the attention part from x.
function blockFromParts(g, x, n, attn) {
  const a1 = n('add'), n1 = n('norm');
  connect(g, attn(x).id, a1.id, 0); connect(g, x.id, a1.id, 1); connect(g, a1.id, n1.id);
  const l1 = n('linear', { out: 32 }), r = n('relu'), l2 = n('linear', { out: 8 }), a2 = n('add'), n2 = n('norm');
  connect(g, n1.id, l1.id); connect(g, l1.id, r.id); connect(g, r.id, l2.id);
  connect(g, l2.id, a2.id, 0); connect(g, n1.id, a2.id, 1); connect(g, a2.id, n2.id);
  return n2;
}

const blocks = (g, x, n, k, p = {}) => { let prev = x; for (let i = 0; i < k; i++) { const b = n('tblock', p); connect(g, prev.id, b.id); prev = b; } return prev; };
const withPos = (g, seq, n) => { const pe = n('posenc'); connect(g, seq.id, pe.id); return pe; };

test('every chapter 5 level has a 3★ answer', () => {
  const single = play('tf-parts', (g, seq, n) => blockFromParts(g, seq, n, (x) => { const h = n('head', { d: 8 }); connect(g, x.id, h.id); return h; }));
  assert.equal(single.done, true, single.msg);
  const multi = play('tf-parts', (g, seq, n) => blockFromParts(g, withPos(g, seq, n), n, (x) => {
    const h1 = n('head', { d: 4 }), h2 = n('head', { d: 4 }), c = n('concat'), o = n('linear', { out: 8 });
    connect(g, x.id, h1.id); connect(g, x.id, h2.id); connect(g, h1.id, c.id, 0); connect(g, h2.id, c.id, 1); connect(g, c.id, o.id); return o;
  }));
  assert.equal(multi.done, true, multi.msg);
  const order = play('tf-order', (g, seq, n) => blocks(g, withPos(g, seq, n), n, 1));
  assert.equal(order.done, true, order.msg);
  const stack = play('tf-stack', (g, seq, n) => blocks(g, withPos(g, seq, n), n, 3, { heads: 2, ffn: 24 }));
  assert.equal(stack.done, true, stack.msg);
  assert.equal(stack.stars, 3, `${stack.params} params`);
});

test('chapter 5 rejects the shortcuts', () => {
  assert.match(play('tf-order', (g, seq, n) => blocks(g, seq, n, 1)).msg, /ignores word order/, 'no positional encoding');
  assert.match(play('tf-stack', (g, seq, n) => blocks(g, withPos(g, seq, n), n, 2)).msg, /at least 3/);
  assert.equal(play('tf-stack', (g, seq, n) => blocks(g, withPos(g, seq, n), n, 3)).stars, 2, 'default MLP width costs a star');
  // MLP without its skip, or norms in the wrong place
  const noSkip = play('tf-parts', (g, seq, n) => {
    const h = n('head', { d: 8 }), a1 = n('add'), n1 = n('norm'), l1 = n('linear', { out: 32 }), r = n('relu'), l2 = n('linear', { out: 8 }), n2 = n('norm');
    connect(g, seq.id, h.id); connect(g, h.id, a1.id, 0); connect(g, seq.id, a1.id, 1); connect(g, a1.id, n1.id);
    connect(g, n1.id, l1.id); connect(g, l1.id, r.id); connect(g, r.id, l2.id); connect(g, l2.id, n2.id);
    return n2;
  });
  assert.match(noSkip.msg, /Not a Transformer block/);
  // attention from something other than x
  const wrongAttn = play('tf-parts', (g, seq, n) => blockFromParts(g, seq, n, (x) => { const l = n('linear', { out: 8 }); connect(g, x.id, l.id); return l; }));
  assert.equal(wrongAttn.done, false);
});

test('QA: one head + output projection, and three heads via nested Concat, are valid blocks; messages say what is wrong', () => {
  const projected = play('tf-parts', (g, seq, n) => blockFromParts(g, seq, n, (x) => { const h = n('head', { d: 8 }), o = n('linear', { out: 8 }); connect(g, x.id, h.id); connect(g, h.id, o.id); return o; }));
  assert.equal(projected.done, true, projected.msg);
  const three = play('tf-parts', (g, seq, n) => blockFromParts(g, seq, n, (x) => {
    const hs = [3, 3, 2].map((d) => { const h = n('head', { d }); connect(g, x.id, h.id); return h; });
    const c1 = n('concat'), c2 = n('concat'), o = n('linear', { out: 8 });
    connect(g, hs[0].id, c1.id, 0); connect(g, hs[1].id, c1.id, 1); connect(g, c1.id, c2.id, 0); connect(g, hs[2].id, c2.id, 1); connect(g, c2.id, o.id);
    return o;
  }));
  assert.equal(three.done, true, three.msg);
  const noRelu = play('tf-parts', (g, seq, n) => {
    const h = n('head', { d: 8 }), a1 = n('add'), n1 = n('norm'), l1 = n('linear', { out: 32 }), l2 = n('linear', { out: 8 }), a2 = n('add'), n2 = n('norm');
    connect(g, seq.id, h.id); connect(g, h.id, a1.id, 0); connect(g, seq.id, a1.id, 1); connect(g, a1.id, n1.id);
    connect(g, n1.id, l1.id); connect(g, l1.id, l2.id); connect(g, l2.id, a2.id, 0); connect(g, n1.id, a2.id, 1); connect(g, a2.id, n2.id);
    return n2;
  });
  assert.match(noRelu.msg, /MLP/);
});
