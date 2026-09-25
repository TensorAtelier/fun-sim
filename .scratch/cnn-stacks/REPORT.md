# cnn-stacks — report

Branch `feat/cnn-stacks` (base `main` @ 3da7fcd). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | Pool stride, GlobalAvgPool, Output targets, receptive fields, level rules | c90ecd9 |
| 02 | Pool stride + GAP in the forward pass; RF readout and box on the Input preview | 41fc533 |
| 03 | Chapter 2 levels (6) with unlocks, lessons, next-level button | 118785d, fix (see below) |
| 04 | Challenges tab lists chapter 2 with stars and opens the Workshop | c4c6860 |

Verified in the browser: RF box 10×10 on the Input for conv→conv→pool→conv; Downsample won with
the mouse (★★★, 80 params), level 3 unlocked, Next button switches level; Challenges row shows ★ 5/18
and opens the Workshop; opens on the first unfinished level; stride slider reaches 7.
`npm test`: 34/34 pass, including a reference ★★★ solution for every level. Parked tickets: none.

## QA verdict (goal mode)

GOAL VERDICT: **PARTIAL** — RF math matched a brute-force reference on 2,447 random chains; maxpool
(stride ≠ size) and GAP matched naive references on 500 tensors; unlocks, saving and the Challenges link
work. Defects:

1. **The patchify ★★★ hint was unreachable** — Conv stride slider capped at 3. → **Fixed**: max 7.
2. **"See 7×7" bypass** — 4×4 pools + a 1×1 conv won with 16 params. → **Fixed**: pooling banned there.
3. **3★ too easy with thin chains** (design call). → **Partly addressed**: new `headRf` rule makes the GAP
   and Tiny levels require real receptive fields before the head; the Stride lesson now reads as a hint.
   Budgets remain generous for 1-channel chains — noted, not changed.
4. **Workshop always opened on level 1.** → **Fixed**: opens on the first level without stars.

The fixes were verified by me (new failing-then-passing tests + browser); QA was not re-run.

## Decisions made for you

Ranked costly-and-surprising first:

- D1 Chapter 2 is played in the Workshop; the Challenges tab links to it and shows its stars.
- D6 The original "Digit classifier" kept its id (`ws-digits`) and became chapter 2's first level, so earlier stars count.
- (after QA) `headRf`: classifier levels judge the receptive field of the last feature map before the head; 3★ budgets otherwise left generous.
- D4 Level rules are data (target, requires-with-counts, forbids, maxK, rfMin, headRf, budgets) judged by checkLevel.
- D3 Receptive field uses the standard recurrence; GAP/Linear make it global.
- D2 Output blocks can demand any target shape.
- D5 MaxPool has its own stride knob (default = size).
