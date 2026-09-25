// Chapter 7 levels (the Reward tab). Pure: each level makes a learner from its knob settings, runs a
// fixed number of steps, and is judged on fixed seeds 1–3 so a verdict never depends on luck.
import { createBandit, createGridworld, createCartpole, createRlhf, REFERENCE, STYLES, rewardModel } from './rl.js';

const avg = (xs) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
export const SEEDS = [1, 2, 3];
const START_QUALITY = REFERENCE.reduce((s, p, i) => s + p * STYLES[i].quality, 0);
const START_REWARD = REFERENCE.reduce((s, p, i) => s + p * rewardModel(i), 0);

// knobs: { key, label, min, max, step, value, log?, options? (select), locked? }
export const RL_LEVELS = [
  {
    id: 'rl-bandit', kind: 'bandit', title: 'Bandit', steps: 500, perFrame: 4,
    knobs: [{ key: 'lr', label: 'Learning rate', min: 0.001, max: 1, value: 0.005, log: true }],
    brief: 'Five slot machines pay different amounts on average (plus noise). Learn which one to play: after 500 pulls, the best arm should get at least 90% of the probability.',
    lesson: 'REINFORCE: after each pull, nudge up the probability of the arm you pulled in proportion to the reward it paid — lucky actions become more likely. With too small a learning rate the policy barely moves in 500 pulls; raise it and watch the best arm take over.',
    make: (s, seed) => createBandit({ lr: s.lr, seed }),
    score: (L) => L.bestProb.at(-1),
    judge: (xs) => {
      const m = Math.min(...xs);
      return { stars: m >= 0.9 ? 3 : m >= 0.75 ? 2 : m >= 0.5 ? 1 : 0, msg: `Best arm's probability after 500 pulls, worst of 3 runs: ${(m * 100).toFixed(0)}% (★ ≥ 50%, ★★ ≥ 75%, ★★★ ≥ 90%).` };
    },
  },
  {
    id: 'rl-baseline', kind: 'bandit', title: 'Baseline', steps: 500, perFrame: 4,
    knobs: [{ key: 'lr', label: 'Learning rate', min: 0.1, max: 0.1, value: 0.1, locked: true },
      { key: 'baseline', label: 'Subtract a baseline (running mean reward)', value: false, toggle: true }],
    brief: 'Same machines, but now every pull pays about +10 more — all rewards are large and positive. Reach 90% on the best arm again.',
    lesson: 'With every reward positive, REINFORCE pushes up whatever arm it just pulled, so it locks onto an early, often wrong, arm. Subtracting a baseline — the average reward so far — turns rewards into "better or worse than usual", which cuts the noise. The same trick (advantage = reward − baseline) is inside PPO and GRPO.',
    make: (s, seed) => createBandit({ lr: 0.1, offset: 10, baseline: s.baseline, seed }),
    score: (L) => L.bestProb.at(-1),
    judge: (xs) => {
      const m = Math.min(...xs), ok = xs.filter((x) => x >= 0.9).length;
      return { stars: m >= 0.9 ? 3 : ok >= 2 ? 1 : 0, msg: `${ok} of 3 runs reached 90% on the best arm (worst: ${(m * 100).toFixed(0)}%).` };
    },
  },
  {
    id: 'rl-grid', kind: 'grid', title: 'Gridworld', steps: 1000, perFrame: 6,
    // Learning rate locked at 0.1: measured over 5 seeds, the result then depends on γ alone
    // (γ ≤ 0.6 fails, 0.7–0.75 ★★, ≥ 0.8 ★★★ with no collapse holes up to γ = 1).
    knobs: [{ key: 'gamma', label: 'Discount γ', min: 0.3, max: 1, step: 0.05, value: 0.5 },
      { key: 'lr', label: 'Learning rate', min: 0.1, max: 0.1, value: 0.1, locked: true }],
    brief: 'Walk from the top-left to the goal (+1), avoiding the pit (−1); every step costs 0.01. Over the last 50 of 1,000 episodes, reach the goal at least 90% of the time in at most 12 steps.',
    lesson: 'The reward only arrives at the end, so each move is credited with the discounted rewards that follow it. With γ = 0.5 the goal\'s reward has shrunk to 0.4% eight steps earlier — the first moves barely learn and the agent wanders. γ near 1 carries the credit all the way back.',
    make: (s, seed) => createGridworld({ lr: s.lr, gamma: s.gamma, seed }),
    score: (L) => ({ success: avg(L.successes.slice(-50)), length: avg(L.lengths.slice(-50)) }),
    judge: (xs) => {
      const s = Math.min(...xs.map((x) => x.success)), l = Math.max(...xs.map((x) => x.length));
      const stars = s >= 0.9 && l <= 12 ? 3 : s >= 0.9 && l <= 16 ? 2 : s >= 0.7 && l <= 20 ? 1 : 0;
      return { stars, msg: `Worst of 3 runs: ${(s * 100).toFixed(0)}% reach the goal, paths up to ${l.toFixed(1)} steps (★★★ needs ≥ 90% and ≤ 12).` };
    },
  },
  {
    id: 'rl-cart', kind: 'cart', title: 'Cart-pole', steps: 400, perFrame: 2,
    knobs: [{ key: 'hidden', label: 'Hidden neurons', min: 2, max: 64, value: 16 },
      { key: 'activation', label: 'Activation', value: 'tanh', options: ['tanh', 'relu'] },
      { key: 'lr', label: 'Learning rate', min: 0.0002, max: 0.05, value: 0.0005, log: true }],
    brief: 'Balance a pole on a cart by pushing left or right. Your policy is a small MLP: 4 state numbers in, P(push right) out. Over the last 50 of 400 episodes, keep it up for 170+ of 200 steps on average.',
    lesson: 'This is the Neural Net page\'s network, trained by reward instead of labels: each episode, steps followed by a long survival get their action made more likely. The learning rate matters most here — too small and 400 episodes aren\'t enough.',
    make: (s, seed) => createCartpole({ hidden: s.hidden, activation: s.activation, lr: s.lr, seed }),
    // Judged on the mean over 3 runs of the last-50-episode average: measured to be smooth across
    // neighbouring learning rates (the worst run alone flipped stars from one slider tick to the next).
    score: (L) => avg(L.lengths.slice(-50)),
    judge: (xs) => {
      const m = avg(xs);
      return { stars: m >= 170 ? 3 : m >= 130 ? 2 : m >= 80 ? 1 : 0, msg: `Balanced for ${m.toFixed(0)} steps on average over 3 runs (★ ≥ 80, ★★ ≥ 130, ★★★ ≥ 170).` };
    },
  },
  {
    id: 'rl-rlhf', kind: 'rlhf', title: 'RLHF', steps: 2000, perFrame: 20,
    knobs: [{ key: 'beta', label: 'KL penalty β', min: 0, max: 5, step: 0.25, value: 0 }],
    brief: 'Tune a chatbot\'s reply style with a reward model that secretly loves long replies. Raise the reward-model score by at least 0.3 while the true quality (hidden from training) ends above where it started.',
    lesson: 'With β = 0 the policy finds the reward model\'s flaw and floods it with rambling — reward up, real quality down: reward hacking. The KL penalty charges for drifting from the starting policy, so it only moves where the reward model is trustworthy. Chat models are tuned this way (RLHF); GRPO uses the same group baseline as here: several replies per prompt, each judged against the group\'s average.',
    make: (s, seed) => createRlhf({ beta: s.beta, seed }),
    score: (L) => L.history.at(-1),
    judge: (xs) => {
      const q = Math.min(...xs.map((x) => x.quality)), r = Math.min(...xs.map((x) => x.reward)) - START_REWARD;
      const stars = r >= 0.3 && q > START_QUALITY + 0.03 ? 3 : r >= 0.3 && q > START_QUALITY ? 2 : r >= 0.2 && q >= START_QUALITY ? 1 : 0;
      return { stars, msg: `Reward-model score ${r >= 0 ? '+' : ''}${r.toFixed(2)}, true quality ${q.toFixed(3)} (started at ${START_QUALITY.toFixed(3)}).` };
    },
  },
];

// Run a level to completion on every judging seed.
export function evaluate(level, settings) {
  const scores = SEEDS.map((seed) => {
    const L = level.make(settings, seed);
    for (let i = 0; i < level.steps; i++) L.step();
    return level.score(L);
  });
  return { ...level.judge(scores), scores };
}

export { START_QUALITY, START_REWARD };
