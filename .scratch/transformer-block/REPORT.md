# transformer-block — report

Branch `feat/transformer-block` (base `main` @ db966a1). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | Positional encoding, packaged Transformer block, per-feature LayerNorm params on sentences | 74458ff |
| 02 | Sinusoidal encoding, per-row LayerNorm, block forward (multi-head + FFN, post-norm), word-order measure | d576634 |
| 03 | Workshop: parts bins, order readout, picker group, Challenges row | f5d0051 |
| 04 | Chapter 5 levels: Block from parts, Word order, Stack three | f5d0051, fix (below) |

Verified in the browser (synthetic pointer events): Word order refuses a block alone ("changes word 1's
output by only 0% — the model ignores word order"), passes with positional encoding (★★★, 6.5%);
inspector shows 872 params. `npm test`: 75/75. Parked tickets: none.

## QA verdict (goal mode)

GOAL VERDICT: **MET** — every component matched a float64 reference (per-row LN 4.9e-8, encoding 3e-8, full
block ≤ 2.5e-7 across head/width configs); exact permutation-equivariance without positions; tblockShape
accepted all reasonable wirings and rejected 14 wrong ones; no regressions in chapters 1–4. Low defects:

1. Word-order floor 0.5% could flip on rare block ids at Heads = 1 (0.31%). → **Fixed**: floor 0.1%.
2. Single head + output projection, and 3+ heads via nested Concat, were rejected; one generic message. → **Fixed**; messages name the failing part.
3. estimateCost ignored Transformer blocks. → **Fixed**.

## Decisions made for you

Ranked costly-and-surprising first:

- D6 Word-order floor 0.1%, set from measurements across block ids, heads and MLP widths (order-blind ≤ 1e-7; order-aware ≥ ~0.3%).
- D1 Post-norm layout (the original Transformer); the lesson notes modern pre-norm.
- D2 LayerNorm works per word row on sentences; images keep the whole-sample norm.
- D3 Word-order sensitivity is measured, not claimed.
- D4 Sinusoidal positional encoding, no parameters.
- D5 Packaged block: d_model = input width, heads split it evenly, MLP width knob (default 32 = 4·d_model).
