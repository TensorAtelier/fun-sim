// Learning from reward: small environments and REINFORCE learners. Pure (no DOM), deterministic
// given a seed, and incremental — each learner exposes step() so the page can animate training.
import { MLP, mulberry32, gaussian } from '../neural/net.js';

const softmax = (z) => {
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
const sample = (p, r) => {
  let u = r(), i = 0;
  while (i < p.length - 1 && (u -= p[i]) > 0) i++;
  return i;
};
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

// ---------------------------------------------------------------- multi-armed bandit
// Each pull of arm i pays means[i] + noise. `offset` shifts every reward (the Baseline level makes
// them all large and positive). Policy: softmax over one logit per arm.
// REINFORCE: θ += lr · (r − b) · (onehot(a) − π), b = running mean reward (if baseline) else 0.
export const BANDIT_MEANS = [0.2, 0.5, 1.0, 0.35, 0.1];

export function createBandit({ lr = 0.1, baseline = false, offset = 0, noise = 0.5, seed = 1 } = {}) {
  const r = mulberry32(seed), K = BANDIT_MEANS.length;
  const theta = new Array(K).fill(0);
  let b = 0, n = 0;
  const rewards = [], bestProb = [];
  return {
    K,
    get probs() { return softmax(theta); },
    rewards, bestProb,
    step() {
      const p = softmax(theta), a = sample(p, r);
      const rew = BANDIT_MEANS[a] + offset + noise * gaussian(r);
      const adv = rew - (baseline ? b : 0);
      for (let i = 0; i < K; i++) theta[i] += lr * adv * ((i === a ? 1 : 0) - p[i]);
      n++;
      b += (rew - b) / Math.min(n, 50); // running mean (window ~50)
      rewards.push(rew - offset);
      bestProb.push(softmax(theta)[2]);
      return { arm: a, reward: rew };
    },
  };
}

// ---------------------------------------------------------------- gridworld
// 5×5 grid. Start top-left, goal bottom-right (+1), one pit (−1), walls block moves; −0.01 per
// step; at most 40 steps. Tabular softmax policy (4 logits per cell). REINFORCE with discounted
// returns-to-go G_t and a running-mean baseline.
export const GRID = {
  size: 5, start: [0, 0], goal: [4, 4],
  pits: [[2, 2]],
  walls: [[1, 3], [3, 1]],
};
export const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // up, right, down, left (as [dx, dy])

function gridStep([x, y], a) {
  const [dx, dy] = MOVES[a], nx = x + dx, ny = y + dy;
  const blocked = nx < 0 || ny < 0 || nx >= GRID.size || ny >= GRID.size || GRID.walls.some(([wx, wy]) => wx === nx && wy === ny);
  const pos = blocked ? [x, y] : [nx, ny];
  const at = (cells) => cells.some(([cx, cy]) => cx === pos[0] && cy === pos[1]);
  if (pos[0] === GRID.goal[0] && pos[1] === GRID.goal[1]) return { pos, reward: 1, done: true, outcome: 'goal' };
  if (at(GRID.pits)) return { pos, reward: -1, done: true, outcome: 'pit' };
  return { pos, reward: -0.01, done: false };
}

export function createGridworld({ lr = 0.5, gamma = 0.95, baseline = true, seed = 1, maxSteps = 40 } = {}) {
  const r = mulberry32(seed), S = GRID.size * GRID.size;
  const theta = Array.from({ length: S }, () => [0, 0, 0, 0]);
  let b = 0, n = 0;
  const successes = [], lengths = [];
  const idx = ([x, y]) => y * GRID.size + x;
  return {
    policy: (cell) => softmax(theta[idx(cell)]),
    successes, lengths,
    step() { // one episode
      let pos = GRID.start;
      const traj = [];
      let outcome = 'timeout';
      for (let t = 0; t < maxSteps; t++) {
        const p = softmax(theta[idx(pos)]), a = sample(p, r);
        const res = gridStep(pos, a);
        traj.push({ s: idx(pos), a, p, reward: res.reward });
        pos = res.pos;
        if (res.done) { outcome = res.outcome; break; }
      }
      let G = 0;
      const returns = traj.map(() => 0);
      for (let t = traj.length - 1; t >= 0; t--) returns[t] = G = traj[t].reward + gamma * G;
      for (let t = 0; t < traj.length; t++) {
        const { s, a, p } = traj[t], adv = returns[t] - (baseline ? b : 0);
        for (let i = 0; i < 4; i++) theta[s][i] += lr * adv * ((i === a ? 1 : 0) - p[i]);
      }
      n++;
      b += (returns[0] - b) / Math.min(n, 50);
      successes.push(outcome === 'goal' ? 1 : 0);
      lengths.push(traj.length);
      return { path: traj.map((st) => st.s), outcome };
    },
  };
}

// ---------------------------------------------------------------- cart-pole
// The classic cart-pole (Barto, Sutton & Anderson 1983): push left or right each 0.02 s; the
// episode ends when the pole tips past 12° or the cart leaves ±2.4 m; at most 200 steps.
const CP = { g: 9.8, mc: 1.0, mp: 0.1, len: 0.5, force: 10, tau: 0.02, thetaMax: (12 * Math.PI) / 180, xMax: 2.4, maxSteps: 200 };

export function cartpoleStep([x, v, th, w], a) {
  const f = a === 1 ? CP.force : -CP.force, total = CP.mc + CP.mp, pml = CP.mp * CP.len;
  const cos = Math.cos(th), sin = Math.sin(th);
  const temp = (f + pml * w * w * sin) / total;
  const alpha = (CP.g * sin - cos * temp) / (CP.len * (4 / 3 - (CP.mp * cos * cos) / total));
  const acc = temp - (pml * alpha * cos) / total;
  const s = [x + CP.tau * v, v + CP.tau * acc, th + CP.tau * w, w + CP.tau * alpha];
  return { s, done: Math.abs(s[0]) > CP.xMax || Math.abs(s[2]) > CP.thetaMax };
}

// Policy: an MLP (net.js) with a sigmoid output = P(push right | state). For a two-action policy,
// ∂(−log π(a|s))/∂z = p − a, which is net.js's backward(a); REINFORCE scales it by the advantage
// (discounted return-to-go, normalised within the episode) and Adam applies the step.
export function createCartpole({ hidden = 16, activation = 'tanh', lr = 0.01, gamma = 0.99, seed = 1 } = {}) {
  const r = mulberry32(seed);
  const net = new MLP([4, hidden, 1], activation, seed);
  const norm = ([x, v, th, w]) => [x / CP.xMax, v / 2, th / CP.thetaMax, w / 2];
  const lengths = [];
  let last = [];
  return {
    net, lengths,
    get lastEpisode() { return last; },
    step() { // one episode + one update
      let s = [(r() - 0.5) * 0.1, (r() - 0.5) * 0.1, (r() - 0.5) * 0.1, (r() - 0.5) * 0.1];
      const traj = [];
      for (let t = 0; t < CP.maxSteps; t++) {
        const p = net.forward(norm(s)), a = r() < p ? 1 : 0;
        traj.push({ s, a });
        const res = cartpoleStep(s, a);
        s = res.s;
        if (res.done) break;
      }
      const T = traj.length, G = new Array(T);
      let acc = 0;
      for (let t = T - 1; t >= 0; t--) G[t] = acc = 1 + gamma * acc;
      const m = mean(G), sd = Math.sqrt(mean(G.map((g) => (g - m) ** 2))) || 1;
      net.zeroGrad();
      for (let t = 0; t < T; t++) { net.forward(norm(traj[t].s)); net.backward(traj[t].a, (G[t] - m) / sd / T); }
      net.applyGrad(lr, 'adam');
      lengths.push(T);
      last = traj.map((st) => st.s);
      return { length: T };
    },
  };
}
export const CARTPOLE = CP;

// ---------------------------------------------------------------- toy RLHF
// A "policy" picks a reply style for a prompt. A reward model scores replies but over-rewards
// length; true quality is what we actually want (hidden from training, plotted for the lesson).
// Update = policy gradient on reward − β·(log π − log π_ref) (the KL penalty), with a group
// baseline: several replies per step, advantage = reward − the group's mean (as in GRPO).
export const STYLES = [
  { name: 'concise answer', quality: 0.8, length: 1 },
  { name: 'detailed answer', quality: 1.0, length: 2 },
  { name: 'rambling filler', quality: 0.1, length: 5 },
  { name: 'refusal', quality: 0.2, length: 0.5 },
];
export const REFERENCE = [0.35, 0.4, 0.05, 0.2]; // the starting (supervised) policy
const LENGTH_BIAS = 0.6;
export const rewardModel = (i) => STYLES[i].quality + LENGTH_BIAS * STYLES[i].length;

export function createRlhf({ beta = 0.2, lr = 0.05, group = 4, seed = 1 } = {}) {
  const r = mulberry32(seed);
  const theta = REFERENCE.map(Math.log);
  const history = [];
  const expected = (f) => softmax(theta).reduce((s, p, i) => s + p * f(i), 0);
  const kl = () => softmax(theta).reduce((s, p, i) => s + (p > 0 ? p * Math.log(p / REFERENCE[i]) : 0), 0);
  return {
    get probs() { return softmax(theta); },
    history,
    step() {
      const p = softmax(theta);
      const picks = Array.from({ length: group }, () => sample(p, r));
      const scores = picks.map((a) => rewardModel(a) - beta * Math.log(p[a] / REFERENCE[a]));
      const m = mean(scores);
      picks.forEach((a, j) => {
        const adv = scores[j] - m;
        for (let i = 0; i < p.length; i++) theta[i] += (lr * adv * ((i === a ? 1 : 0) - p[i])) / group;
      });
      history.push({ reward: expected(rewardModel), quality: expected((i) => STYLES[i].quality), kl: kl() });
    },
  };
}
