# attention — report

Branch `feat/attention` (base `main` @ 62a555c). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | Sentence source, row-wise Linear, Scores/Scale/Softmax/A·V, matrix Concat, Attention head | 4886b0a |
| 02 | Matrix ops, head, attentionPeak + peakMax rule | a1e5f6b, fix (below) |
| 03 | Matrix previews, labelled attention map, branching adds | 0acd707, fix (below) |
| 04 | Chapter 4 levels: Scores, Attention weights, One head from parts, Two heads | 0acd707, fix (below) |

Verified in the browser (synthetic pointer events): One head from parts ★★★ (216 params, average top
weight 39%); attention map 8×8 labelled "the old dog slept by the warm fire"; Two heads ★★ at d = 8
(568 params, finite values) and ★★★ at d = 4 (288). `npm test`: 67/67. Parked tickets: none.

## QA verdict (goal mode)

GOAL VERDICT: **PARTIAL** — maths matched a reference (~1e-6); attention map and branching verified in a
browser; chapters 1–3 unaffected. Defects:

1. **High — matrix Linear sized by word count, not features** (NaN on two default heads, still passing). → **Fixed** + F ≠ T test.
2. **Medium — Softmax not required to be over the scores.** → **Fixed** (`attn` structure rule).
3. **Low–Medium — Scale skippable with narrow Q/K, or any d.** → **Fixed**: Scale mandatory, d = width of Q/K; message reports measured saturation.
4. **Medium — Two heads accepted stacked heads / one head twice.** → **Fixed**.
5. **Low — Scores accepted Q = K.** → **Fixed**.
6. **Low — cost estimate undercounted matrix work.** → **Fixed**.
7. **Low — any 2-D Softmax labelled as an attention map.** → **Fixed**.
8. **Low — dead sentence branch in the memory measure; Proof of done vs D6.** → **Fixed** (removed; plan updated).

QA also noted: with D7, clicking a part after Input/Word now branches instead of splicing (a UX
trade-off, kept; Shift-less splicing still works from any non-source block).

Fixes verified by me (failing-then-passing tests + browser); QA not re-run.

## Decisions made for you

Ranked costly-and-surprising first:

- D6 The planned "word-1 reach" rule was dropped (untrained attention gave ~5–8%, below the RNN — it would have taught nothing true). The bottleneck lesson is carried by the text (one step vs seven); the measured rule is saturation: mean top weight 0.53 with ÷√d vs 0.78 without.
- D7 Adding a part after a data source, or Shift-clicking, starts a branch instead of splicing.
- (after QA) Chapter 4 levels are judged structurally (`attn`: scores / weights / head / multi), with Scale's d tied to the Q/K width.
- D1 Matrices are [T, F]; Linear maps each row with the same weights.
- D2 Heads are untrained; lessons say trained heads learn meaningful patterns.
- D4 Scale is its own block with a d knob.
