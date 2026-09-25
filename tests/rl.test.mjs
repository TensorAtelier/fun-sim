import test from 'node:test';
import assert from 'node:assert/strict';
import * as RL from '../src/sims/build/rl.js';
import { MLP } from '../src/sims/neural/net.js';

const run = (learner, n) => { for (let i = 0; i < n; i++) learner.step(); return learner; };
const avg = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const seeds = [1, 2, 3];

test('bandit: REINFORCE finds the best arm; a tiny learning rate is too slow', () => {
  for (const seed of seeds) assert.ok(run(RL.createBandit({ lr: 0.1, seed }), 500).bestProb.at(-1) > 0.9);
  assert.ok(run(RL.createBandit({ lr: 0.005, seed: 1 }), 500).bestProb.at(-1) < 0.6);
});

test('baseline: with all rewards ≈ +10, no baseline collapses onto wrong arms; a baseline fixes it', () => {
  const without = seeds.map((seed) => run(RL.createBandit({ lr: 0.1, offset: 10, seed }), 500).bestProb.at(-1));
  const withB = seeds.map((seed) => run(RL.createBandit({ lr: 0.1, offset: 10, baseline: true, seed }), 500).bestProb.at(-1));
  assert.ok(without.filter((p) => p < 0.5).length >= 2, `without ${without}`);
  assert.ok(withB.every((p) => p > 0.9), `with ${withB}`);
});

test('gridworld: γ 0.95 reaches the goal in short paths; γ 0.5 wanders', () => {
  for (const seed of seeds) {
    const g = run(RL.createGridworld({ lr: 0.2, gamma: 0.95, seed }), 1000);
    assert.ok(avg(g.successes.slice(-50)) >= 0.9 && avg(g.lengths.slice(-50)) <= 12);
  }
  const short = run(RL.createGridworld({ lr: 0.2, gamma: 0.5, seed: 1 }), 1000);
  assert.ok(avg(short.lengths.slice(-50)) > 15);
});

test('cart-pole: a small MLP learns to balance; a tiny learning rate does not', () => {
  for (const seed of seeds) assert.ok(avg(run(RL.createCartpole({ hidden: 16, lr: 0.01, seed }), 400).lengths.slice(-50)) >= 170);
  assert.ok(avg(run(RL.createCartpole({ hidden: 16, lr: 0.001, seed: 1 }), 400).lengths.slice(-20)) < 150);
});

test('cart-pole physics: an unpushed tilted pole falls', () => {
  let s = [0, 0, 0.05, 0], t = 0;
  while (t < 500) { const r = RL.cartpoleStep(s, t % 2); s = r.s; t++; if (r.done) break; }
  assert.ok(t < 500);
});

test('MLP backward(y, scale) scales the gradient (policy-gradient use)', () => {
  const net = new MLP([3, 4, 1], 'tanh', 5);
  net.zeroGrad(); net.forward([0.2, -0.1, 0.4]); net.backward(1);
  const g1 = [...net.gW[0]];
  net.zeroGrad(); net.forward([0.2, -0.1, 0.4]); net.backward(1, -2.5);
  net.gW[0].forEach((v, i) => assert.ok(Math.abs(v + 2.5 * g1[i]) < 1e-12));
});

test('RLHF: no KL penalty hacks the reward model; β ≈ 1–1.5 improves both reward and true quality', () => {
  const start = RL.REFERENCE.reduce((s, p, i) => s + p * RL.STYLES[i].quality, 0);
  const at = (beta) => run(RL.createRlhf({ beta, seed: 1 }), 2000).history.at(-1);
  assert.ok(at(0).quality < 0.2, 'β = 0: collapses onto rambling');
  const good = at(1.25);
  assert.ok(good.quality > start && good.reward > 1.9, `β 1.25: ${JSON.stringify(good)}`);
  assert.ok(at(5).reward < 1.8, 'β too large: barely moves');
});
