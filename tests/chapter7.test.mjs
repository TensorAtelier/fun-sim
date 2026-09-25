import test from 'node:test';
import assert from 'node:assert/strict';
import { RL_LEVELS, evaluate } from '../src/sims/build/rllevels.js';

const level = (id) => RL_LEVELS.find((l) => l.id === id);
const defaults = (L) => Object.fromEntries(L.knobs.map((k) => [k.key, k.value]));

test('each level fails at its starting settings and has a 3★ setting', () => {
  const cases = {
    'rl-bandit': { lr: 0.1 },
    'rl-baseline': { baseline: true },
    'rl-grid': { gamma: 0.9 },
    'rl-cart': { lr: 0.01 },
    'rl-rlhf': { beta: 1.25 },
  };
  for (const L of RL_LEVELS) {
    const start = evaluate(L, defaults(L));
    assert.equal(start.stars, 0, `${L.id} should earn nothing at its defaults: ${start.msg}`);
    const good = evaluate(L, { ...defaults(L), ...cases[L.id] });
    assert.equal(good.stars, 3, `${L.id}: ${good.msg}`);
  }
});

test('RLHF: no KL penalty is reward hacking and earns nothing', () => {
  assert.equal(evaluate(level('rl-rlhf'), { beta: 0 }).stars, 0);
});

test('QA: Gridworld is decided by γ alone; cart-pole stars are smooth near the good setting; RLHF gives nothing for barely moving', () => {
  const grid = level('rl-grid');
  for (const gamma of [0.3, 0.5, 0.6]) assert.equal(evaluate(grid, { gamma, lr: 0.1 }).stars, 0, `γ ${gamma}`);
  for (const gamma of [0.8, 0.85, 0.9, 0.95, 1]) assert.equal(evaluate(grid, { gamma, lr: 0.1 }).stars, 3, `γ ${gamma}`);
  const cart = level('rl-cart');
  for (const lr of [0.006, 0.008, 0.01]) assert.equal(evaluate(cart, { hidden: 16, activation: 'tanh', lr }).stars, 3, `lr ${lr}`);
  assert.equal(evaluate(level('rl-rlhf'), { beta: 5 }).stars, 0);
});
