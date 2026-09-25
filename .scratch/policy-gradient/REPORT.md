# policy-gradient — report

Branch `feat/policy-gradient` (base `main` @ 4500a54). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | RL core: bandit, gridworld, cart-pole (net.js MLP policy), toy RLHF with KL; `backward(y, scale)` | 19e9f8c |
| 02–04 | Reward tab: 5 levels judged on fixed seeds; bandit/grid/cart-pole/RLHF views; Challenges row | 614e212, fix (below) |

Verified in the browser: Bandit ★★★ (best arm 96%); RLHF at β = 0 visibly hacks the reward model (98%
"rambling filler", quality 0.11) and ★★★ at β = 1.25 (+0.38 reward, quality 0.769); cart-pole's animation
and learning curve (≈20 → 200 steps) ★★★. `npm test`: 93/93. Parked tickets: none.

## QA verdict (goal mode)

GOAL VERDICT: **PARTIAL** — algorithms and physics verified (softmax/tabular REINFORCE signs, discounted
returns, cart-pole advantage sign through net.js + Adam, Barto/Sutton/Anderson dynamics, KL-regularised
REINFORCE with group baseline; RLHF optimum matches the closed form); net.js change is a no-op by default.
Defects:

1. **Gridworld winnable without discounting** (raise lr at γ = 0.5). → **Fixed**: lr locked at 0.1; γ alone decides.
2. **Gridworld earned ★ at its defaults** (unlocking cart-pole). → **Fixed**: ★ needs paths ≤ 20; all levels 0★ at defaults (tested).
3. **Collapse holes at γ = 0.95** for some lr. → **Fixed** by the locked lr (monotone across 5 seeds).
4. **Cart-pole ★★★ knife-edge.** → **Fixed**: mean of 3 runs over the last 50 episodes; smooth across lr 0.004–0.01.
5. Bandit lesson overclaimed large-lr failure. → **Fixed** (reworded, slider capped at 1).
6. RLHF ★ for "barely moves". → **Fixed**.
7. No reroll despite D2. → **Fixed** (New seed button; judging unchanged).
8. Stale "two pits" comment. → **Fixed**.
9. Judging froze the page briefly. → **Mitigated** ("Judging…" shown first).

Fixes verified by me (sweeps + tests); QA not re-run.

## Decisions made for you

Ranked costly-and-surprising first:

- D1 Chapter 7 is its own Reward tab; "your own MLP" = hidden size / activation / learning-rate knobs driving net.js.
- D2 Verdicts come from full training runs on fixed seeds 1–3; the animated run's seed is explorable.
- (after QA) Gridworld's learning rate is locked so γ is the lesson; cart-pole is judged on a mean, not the worst run.
- D3 Cart-pole uses net.js's sigmoid output as P(push right); REINFORCE = backward(a, advantage).
- D4 RLHF is a toy: reply styles, a length-biased reward model, hidden true quality, KL penalty, group baseline (GRPO-style).
- D5 All thresholds come from measured sweeps (recorded in the plan).
