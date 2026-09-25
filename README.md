# Fun Sim

Interactive math & physics simulations in the browser. Vite + vanilla JS; three.js for 3D.

| Sim | File | What it shows |
|---|---|---|
| Fourier epicycles | `src/sims/fourier.js` | Upload an image or draw → edge path → DFT → rotating circles redraw it |
| Rocket launch | `src/sims/rocket.js` | Two-stage rocket, drag, gravity turn, orbital elements |
| Solar system | `src/sims/solar.js` | 3D N-body; edit masses, add stars/black holes/asteroids |
| Double pendulum | `src/sims/pendulum.js` | Chaos: an ensemble of near-identical pendulums diverging |
| Gas & collisions | `src/sims/gas.js` | Hard-disk gas relaxing to Maxwell–Boltzmann, pressure, diffusion |
| Galton board | `src/sims/galton.js` | Binomial → normal, Pascal's triangle |
| Stats Lab | `src/sims/clt.js` | CLT, Monte Carlo π, random walks, gambler's ruin |
| Neural Net | `src/sims/neural.js` | Live-training MLP: decision boundary, per-neuron maps, loss curves |
| Build a Net | `src/sims/build.js` (+ `build/`) | Game: learn NN building blocks. Ch. 1 convolution (sandbox + 9 levels); Ch. 2 CNN stacks (6 Workshop levels); Ch. 2b residual blocks (4 levels); Ch. 3 RNN & LSTM (3 levels); Ch. 4 attention (4 levels); Ch. 5 Transformer block (3 levels); Ch. 6 Mixture of Experts (4 levels); Ch. 7 learning from reward (5 levels, Reward tab) |

## Run
```
npm install
npm run dev        # http://localhost:8870  (or: dev start fun-sim)
npm run build      # static site in dist/
npm test           # node:test suite (tests/)
```

## Adding a simulation
Create `src/sims/<id>.js` exporting `{ id, title, glyph, tag, blurb, mount(root) }`, where
`mount` builds its UI with the helpers in `src/ui.js` and returns a cleanup function.
Then add it to `src/sims/index.js`.

## Build a Net (game)
Learn neural-network parts by using them, one chapter per building block.

- `src/sims/build.js` — page: Sandbox, Challenges (level runner, stars, unlocks) and Workshop tabs.
- `src/sims/build/graph.js` — pure block-graph model: block types, wiring rules (DAG; an input takes one
  wire), shape inference, parameter counts, level check. Tested in `tests/graph.test.mjs`.
- `src/sims/build/workshop.js` — the Workshop: parts bin, drag-to-wire board, inspector, receptive-field box.
- `src/sims/build/chapter2.js` — chapter 2 Workshop levels as data (target shape, requires, forbids, maxK,
  rfMin, headRf, needsSkip, needsProjection, chainConvs, signalMin, star budgets) plus inspector knob ranges;
  `tests/chapter2.test.mjs` and `tests/residual.test.mjs` prove each level has a 3★ answer and close known exploits.
  Blocks can have several input ports (Add takes two); the forward pass reports each block's signal (RMS).
- `src/sims/build/chapter3.js`, `words.js` — sequence levels on a toy vocabulary: Word/Zero-state sources,
  shared-weight RNNCell/LSTMCell, gate parts, and a first-word memory measure (`tests/chapter3.test.mjs`).
- `src/sims/build/chapter4.js` — attention on a whole-sentence matrix [T, F]: Linear per row, QKᵀ, Scale,
  Softmax, A·V, Attention head; levels judged structurally (`attn`) plus a saturation measure (`peakMax`).
- `src/sims/build/chapter5.js` — Transformer block (post-norm; per-row LayerNorm), positional encoding, and a
  measured word-order sensitivity (`orderMin`): exactly ~0 without positions, clearly non-zero with them.
- `src/sims/build/chapter6.js` — Mixture of Experts: Router, Top-k, capacity-limited Experts, packaged MoE layer;
  active vs total parameters, dropped-word counts (`noDrops`), routing table and expert load in the inspector.
- `src/sims/build/rl.js`, `rllevels.js`, `reward.js` — the Reward tab: REINFORCE on a bandit, gridworld,
  cart-pole (the Neural Net page's MLP, `backward(y, scale)`) and a toy RLHF with a KL penalty; levels are
  judged by full training runs on fixed seeds (`tests/rl.test.mjs`, `tests/chapter7.test.mjs`).
- `src/sims/build/tensor.js`, `forward.js`, `digits.js` — plain-JS forward pass (seeded untrained weights,
  per-block cache, cost budget) on ten toy 28×28 digits; every Workshop block previews its real output.
- `src/sims/build/conv.js` — pure conv math (`outSize`, `conv2d`, `convParams`); no DOM, testable in node.
- `src/sims/build/figure.js` — DOM grids for input ✱ kernel = output, shape pipelines.
- `src/sims/build/levels.js` — `CHAPTERS` and `LEVELS`. Each level's `gen()` returns a fresh random
  question: `{ prompt, fields, answer | check(vals), explain, figure(vals, revealed) }`.
  Three of five correct unlocks the next level; best stars persist in `localStorage` (`fun-sim.build.v1`).

Roadmap: see [`docs/agents/roadmap.md`](docs/agents/roadmap.md) — block editor → CNN stacks → residual
blocks → RNN/LSTM → attention → Transformer block → MoE → policy gradient.
