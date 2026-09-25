# tensor-engine — plan

**Goal:** plain-JS forward pass on toy data; blocks in the Workshop show their real outputs (feature maps, values).
**Not yet:** training / autograd (roadmap Later), user-drawn input digits, new block types (later chapters), GPU.
**Stack / interfaces:** pure modules `src/sims/build/tensor.js` (tensors, ops, seeded init) and `src/sims/build/digits.js` (toy 28×28 digits), both DOM-free and node-tested; Workshop renders previews inside blocks and a large view in the inspector.
**Proof of done:** in the Workshop pick digit "3": Input shows the 3, Conv2d shows edge-detected feature maps, ReLU blanks the negatives, MaxPool2d shows them at half size, Flatten a long strip, Linear/Output bars (Output as class probabilities). Changing a knob updates every preview downstream.

## Tickets

- [x] 01 — Tensor ops: multi-channel conv2d (stride, padding), relu, maxpool, flatten, linear, softmax; seeded He init; classic kernels for single-channel convs; tests against chapter 1's conv2d and hand-computed values
- [x] 02 — Toy digits + graph forward: ten 28×28 digits drawn procedurally; `runGraph` evaluates the DAG in order and caches per block; tests: end-to-end shapes, softmax sums to 1, determinism
- [x] 03 — Previews on the board: "Show values" toggle, feature-map thumbnails / vector strips / probability bars inside blocks, digit picker in the Level panel
- [x] 04 — Inspector detail view: the selected block's full output (channel grid, value range, colour scale), "untrained weights" note, Reroll weights

## Notes

- D1 Weights are seeded per block (block id + a board seed), so previews are stable across re-renders; changing a block's knobs re-initialises only that block. "Reroll weights" bumps the board seed.
- D2 The network is untrained and says so: Output probabilities are labelled as random until training exists (roadmap Later).
- D3 A Conv2d fed by a single channel starts its first filters as classic kernels (vertical/horizontal edge, blur, sharpen, diagonals), the rest He-random — so feature maps on a digit look meaningful without training.
- D4 Recompute only when the graph, a knob, the digit or the seed changes; per-block cache keyed by params + upstream version. The biggest allowed conv (64→64, 28×28, k7) is ~150M MACs — acceptable on a knob change, not per frame.
- D5 Values are shown by default; the toggle hides them for a compact board.
- D6 Digits are rendered from a tiny stroke font (line segments) with a soft brush — no canvas, so node tests run the same data.
