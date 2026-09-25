# transformer-block — plan

**Goal:** Ch 5: assemble a Transformer block — multi-head attention, residual Add, LayerNorm, an MLP, another Add and LayerNorm — see why positional encoding is needed (attention alone ignores word order), and stack blocks.
**Not yet:** causal masking and decoders, training, dropout, pre-norm vs post-norm as separate levels (mentioned in a lesson), MoE (chapter 6).
**Stack / interfaces:** `graph.js` blocks (Positional encoding, packaged Transformer block; LayerNorm per row on matrices); `tensor.js`/`forward.js` ops (sinusoidal encoding, per-row LayerNorm, multi-head block with FFN, post-norm) and an order-sensitivity measure; `chapter5.js` levels; Workshop readout.
**Proof of done:** in the Workshop, wire a block from parts (heads → Concat → Linear, Add the sentence back, LayerNorm, Linear → ReLU → Linear, Add, LayerNorm); on "Word order" see word 1's output stay exactly the same when two later words swap — until you add positional encoding; then stack three packaged blocks under a parameter budget.

## Tickets

- [x] 01 — Model: Positional encoding block, packaged Transformer block (heads, FFN width), LayerNorm per row on matrices (2·F params), sentence word order param; tests
- [x] 02 — Forward + order: sinusoidal encoding, per-row LayerNorm, the block's forward (multi-head attention with output projection, FFN, two Add+LayerNorm), `orderSensitivity` (swap words 2 and 3, measure word 1's output row); tests incl. exact permutation-equivariance without positions
- [x] 03 — Workshop: chapter 5 parts bins, order readout, picker group and Challenges row
- [x] 04 — Chapter 5 levels: Block from parts (structural), Word order (orderMin), Stack three (three blocks on one chain, budgets) — unlocking after chapter 4; solvability + exploit tests

## Notes

- D1 Post-norm (the original Transformer: x → LN(x + Attn(x)) → LN(· + MLP(·))), because it matches chapter 2b's Add-then-norm; the lesson notes modern models normalise first (pre-norm).
- D2 LayerNorm normalises each token row on matrices (as Transformers do); images keep chapter 2b's whole-sample norm.
- D3 Word-order sensitivity is measured, not claimed: without positions, swapping words 2 and 3 leaves word 1's output bit-for-bit identical (attention is permutation-equivariant); the level needs ≥ 0.1% change (see D6).
- D4 Sinusoidal positional encoding (no parameters), added to the word vectors.
- D5 The packaged block uses d_model = the input width, heads splitting it evenly, and an FFN width knob (default 4·d_model).
- D6 Measured: swapping words 2 and 3 changes word 1's output by ≤ 1e-7 of its size without positional encoding (float rounding — exact equivariance) and by ≥ ~0.3% with it across block ids, heads and MLP widths (QA: heads=1 dipped to 0.31% on 4 of 1,500 ids; heads ≥ 2 never below 0.86%). The Word-order floor is 0.1%: four orders of magnitude above order-blind, below every order-aware case measured.
