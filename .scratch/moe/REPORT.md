# moe — report

Branch `feat/moe` (base `main` @ 4e9e023). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | Router, Top-k, Experts, MoE layer; active-parameter counting | 715c21a |
| 02 | Top-k routing, capacity-limited dispatch, dropped-word stats, MoE rules | 3d4f634 |
| 03 | Routing table + load bars, active readout, parts/picker/Challenges wiring | 2bd242d |
| 04 | Chapter 6 levels: Router from parts, Sparse experts, Capacity, Scale up not out | 2bd242d, fix (below) |

Verified in the browser (synthetic events): Capacity drops words at 1× (full experts in red), ★★★ at 2×
(27% of parameters active per word), ★ at 6× with a capacity-worded hint; only the capacity knob is
editable there; routing table words × E1–E4. `npm test`: 83/83. Parked tickets: none.

## QA verdict (goal mode)

GOAL VERDICT: **PARTIAL** — dispatch matched a naive reference on 200 random cases (0 mismatches in capacity,
load, drops; outputs within 1.3e-7); verdicts independent of block ids and Reroll; no regressions. Defects:

1. **Medium — Capacity winnable without raising capacity** (edit Top-k/E on the pre-wired blocks). → **Fixed**: routing locked, only capacity editable.
2. **Medium — overshooting capacity failed with a "parameters" message.** → **Fixed**: any drop-free capacity ≤ 16× completes.
3. Low — star hint said "parameters" on a capacity level. → **Fixed**.
4. Low — Router knob max 16 vs Experts 32. → **Fixed**.
5. Low — Output repeated the routing view. → **Fixed**.
6. Low — same-type MoE blocks share identical weights (consequence of D1). → **Accepted**; no level stacks MoE layers.
7. Low — Proof of done still required a Transformer block. → **Fixed** (text matches D6).

Fixes verified by me (tests + browser); QA not re-run.

## Decisions made for you

Ranked costly-and-surprising first:

- D6 "Scale up, not out" routes the sentence directly: behind an untrained Transformer block the router collapses onto ~2 experts and drops words even at 8×; the lesson demonstrates this as routing collapse.
- D1 MoE weights are seeded by block type, not id; verdicts use fixed seeds 1–3 (so identical MoE blocks share weights — fine for these levels).
- D3 Capacity = ceil(factor · routed / E); highest-gate words kept, the rest dropped.
- D5 Capacity level: routing fixed at 4 experts, top-1; stars by the smallest drop-free factor (★★★ ≤ 2×).
- D2 Active parameters = everything minus the experts a word isn't routed to.
- D4 Load balancing taught through capacity/drops; the auxiliary loss is explained in the lesson.
