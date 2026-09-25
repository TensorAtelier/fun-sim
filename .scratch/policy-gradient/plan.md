# policy-gradient — plan

**Goal:** Ch 7: learning from reward — a policy that improves by trial and error with REINFORCE: a multi-armed bandit, the same bandit fixed by a baseline, a gridworld where discounting spreads credit over many steps, cart-pole balanced by a small MLP you size yourself (reusing the Neural Net page's backprop), and an RLHF bridge showing reward hacking and the KL penalty.
**Not yet:** PPO/actor-critic (value networks, clipping), Q-learning/DQN, training the Workshop graphs themselves, real language models.
**Stack / interfaces:** pure `src/sims/build/rl.js` (environments, softmax policies, REINFORCE, cart-pole physics, toy RLHF), a one-line extension to `src/sims/neural/net.js` (`backward(y, scale)`), and a new **Reward** tab on the Build a Net page (`reward.js`) with its own canvases; chapter 7 in the Challenges list.
**Proof of done:** in the Reward tab, watch a bandit's arm probabilities converge to the best arm; see a baseline turn a stalled, noisy run (all rewards positive) into a quick one; train a gridworld agent to reach the goal and see its policy arrows; balance cart-pole with an MLP whose size you chose; and see an RLHF policy chase a flawed reward model until the KL penalty keeps it honest.

## Tickets

- [x] 01 — rl.js core: bandit, gridworld, cart-pole, toy RLHF; softmax/tabular REINFORCE with optional baseline and discount; MLP policy via net.js (backward scale); deterministic seeds; tests that good settings succeed and bad ones fail
- [x] 02 — Reward tab: level list with stars and unlocks, run/pause/step/reset and speed, bandit + baseline views (arm probabilities, reward curve)
- [x] 03 — Gridworld (grid, policy arrows, success curve) and cart-pole (animation of the latest episode, episode-length curve, MLP knobs) views
- [x] 04 — RLHF bridge level (reward-model score vs true quality, KL knob), Challenges row, roadmap chapter, tests

## Notes

- D1 Chapter 7 lives in its own Reward tab: training needs animation and charts the block board doesn't have. "Your own MLP" = hidden size/activation knobs driving net.js, not wires.
- D2 Every verdict comes from a training run with a fixed seed (the seed shown, rerollable for exploration but not for scoring), so levels are deterministic.
- D3 Two-action policies (cart-pole) use net.js's sigmoid output: ∂(−log π(a|s))/∂z = p − a, which is exactly `backward(a)`; REINFORCE scales it by the advantage.
- D4 The RLHF level is a toy: the "policy" picks reply styles for prompts, the reward model over-rewards length, true quality is hidden but plotted.
- D5 Measured and tuned (seeds 1–5+): bandit lr 0.1 → P(best) 0.94–0.98 at 500 pulls; +10 rewards without baseline collapse onto wrong arms 4/5, with baseline 5/5 succeed. Gridworld (one pit, two walls) γ 0.95 → 96–100% success in ~9 steps; γ 0.5 → 78–94% in ~22 steps; lr 0.5 collapses 2/8. Cart-pole MLP(16, tanh), lr 0.01 → 185–200 steps; lr 0.001 → 73–109. RLHF (length bias 0.6): β 0 → quality 0.12 (hacked), β 1–1.5 → quality 0.76–0.77 and reward 1.95–2.09 (start 0.725 / 1.625), β 5 → barely moves.
