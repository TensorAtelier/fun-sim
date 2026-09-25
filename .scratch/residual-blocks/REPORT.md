# residual-blocks — report

Branch `feat/residual-blocks` (base `main` @ a740bb4). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | Input ports, Add and LayerNorm blocks, receptive field through Add | ea6fd03 |
| 02 | add/layernorm ops, init-scale gain, signal-strength rule | bf46e7e |
| 03 | Two-port Add in the Workshop, arcing skip wires, signal readouts, init-scale knob | 1f45d1e |
| 04 | Chapter 2b levels (Skip connection, Matching shapes, Deep & alive, Normalize) | 769f2b0, fix (below) |

Verified in the browser: skip wire into Add port b, arcing over the blocks; Add refusing
[8,14,14] + [8,28,28] with the "1×1 Conv2d on the skip path" hint; signal readouts; init scale locked
at 0.5 on Deep & alive and restored elsewhere; picker grouped by chapter; Challenges shows 2 and 2b stars
separately; LayerNorm help text. `npm test`: 43/43 pass, including a 3★ reference for every 2b level and
the measured signal numbers. Parked tickets: none.

## QA verdict (goal mode)

GOAL VERDICT: **PARTIAL** — all four levels reach 3★ in the real UI (664 / 1,392 / 4,168 / 4,280 params);
port-1 rewires invalidate the forward cache; D7 numbers reproduced exactly. Defects:

1. **Matching shapes beaten without matching** (downsample first, then x + ReLU(x): 3★, 32 params). → **Fixed**: `needsProjection` — the skip must start where the shape differs.
2. **Deep & alive beaten without depth** (8 one-layer convs merged by Adds: 3★, 128 params). → **Fixed**: `chainConvs: 8` on one chain.
3. LayerNorm inspector said "no parameters" next to "Parameters 16"; D3's GroupNorm(1) note missing. → **Fixed**: help text for LayerNorm and Add.
4. Deferred forward passes could let the signal rule judge stale values (low odds). → **Fixed**: no judging while stale.
5. Paused previews gave "turn on values". → **Fixed** message.
6. headRf walk picked an arbitrary input on two-input blocks (no current level affected). → **Fixed**: walks port 0.
7. Minor: a plain deep stack heard "wire a skip" before "signal fades". → **Fixed** (signal checked first). A ReLU-only detour still passes Skip connection — accepted (it is still x + f(x)).

Fixes verified by me with failing-then-passing tests and in the browser; QA not re-run.

## Decisions made for you

Ranked costly-and-surprising first:

- D4 BatchNorm is explained in a lesson, not offered as a block (one image at a time would make it fake).
- D2 "Why deep stacks struggle" is shown with the forward signal at init scale 0.5, not gradients; the lesson says gradients take the same paths backwards.
- (after QA) Skip rules: `needsSkip` = an Add whose inputs have unequal depth; `needsProjection` = the skip starts where the shape differs; `chainConvs` = convs on one chain; the signal floor applies to every block on the path (10%).
- D6 Init scale is a board knob; Deep & alive and Normalize pin it at 0.5.
- D3 LayerNorm = GroupNorm(1)-style, 2 params per channel, stated in its help text.
- D1 Ports are indices on the target block; single-input graphs are unchanged.
- D5 Receptive field through Add is the larger of its inputs.
- D7 Measured: plain 0.1%, residual 239%, LayerNorm 253% (scale 0.5); residual at scale 1 grows to 3,392% — used in the Normalize lesson.
