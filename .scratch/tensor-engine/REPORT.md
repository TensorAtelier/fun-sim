# tensor-engine — report

Branch `feat/tensor-engine` (base `main` @ 6e2dd96). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | Tensor ops (multi-channel conv, relu, maxpool, flatten, linear, softmax), seeded He init, classic kernels | ad58215 |
| 02 | Toy 28×28 digits + cached graph forward pass | 2701bc4, fix 883acdf |
| 03 | Value previews on blocks, digit picker, show-values toggle | 617b5f4, fix 883acdf |
| 04 | Inspector detail view (all channels, range, untrained note), Reroll weights | 8b2cf53 |

Verified in the browser: digit 3 → edge-map channels → ReLU → pooled maps → 1568-value strip →
Output probabilities; digit switch, per-channel scaling, reroll, inspector detail for every block type.
`npm test`: 20/20 pass. Parked tickets: none.

## QA verdict (goal mode)

GOAL VERDICT: **PARTIAL** — proof of done works in a real browser; maths matched reference
implementations (conv2d on 180 random configs, max error 4.7e-7; 400 random chains, all 1972 outputs
had the inferred shape, no NaN). Two defects blocked MET:

1. **High — stale outputs after a rewire.** Cache key lacked the block id, so rewiring to an identically
   configured block kept the old downstream values. → **Fixed** (883acdf): key includes the block id;
   regression test added.
2. **Medium — UI freeze on big networks.** Every slider tick ran the whole forward pass (0.7–2.2 s at the
   free-build maximum). → **Fixed**: cost estimate; ≤8M multiply-adds run immediately, ≤60M run 150 ms
   after the last change, bigger networks pause previews with an explanation. Measured after the fix:
   0–1 ms per tick at the maximum; a 29M-MAC network fills in its maps after the pause.
3. Low — caches never dropped deleted blocks. → **Fixed**.
4. Low — board Output bars lacked the "untrained" label (D2). → **Fixed**.
5. Cosmetic — non-square Inputs cropped the digit. → **Fixed** (fits it).

The fixes were verified by me (new tests + browser); QA was not re-run (lite pipeline: one QA pass).

## Decisions made for you

Ranked costly-and-surprising first:

- D4 (revised after QA) Forward pass budget: ≤8M multiply-adds immediate, ≤60M deferred 150 ms, above 60M or 5M weights → values paused with a message. Free build still allows those sizes for shape practice.
- D3 A Conv2d on a single channel starts its first six filters as classic kernels (edges, blur, diagonals, spot) so maps are readable untrained.
- D2 The network is untrained and says so on the Output block and in the inspector.
- D1 Weights are seeded per block (id + board seed); Reroll bumps the board seed.
- D6 Digits come from a stroke font with a soft brush — identical in node and the browser.
- D5 Values are shown by default; a toggle hides them.
