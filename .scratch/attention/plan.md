# attention — plan

**Goal:** Ch 4: build attention from parts — queries, keys and values from Linear layers, scaled dot-product scores, a row-wise softmax you can see as a word-by-word attention map, the weighted sum, and multiple heads — and see that attention removes chapter 3's bottleneck.
**Not yet:** positional encoding, masking/causal attention, the full Transformer block (chapter 5), training.
**Stack / interfaces:** matrix values [T, F] in `graph.js`/`tensor.js`/`forward.js` (Sentence source, Linear on each row, QKᵀ, Scale, Softmax, A·V, feature Concat, packaged Attention head); `chapter4.js` levels; Workshop matrix previews and a labelled attention-map view.
**Proof of done:** in the Workshop, turn the 8-word sentence into Q and K, score every word against every other (an 8×8 grid), scale and softmax it into rows that sum to 1 and read the attention map with word labels; finish one head with V; wire two heads, concat and mix them; and read (in the lesson, per D6) why attention removes chapter 3's bottleneck: every word is one step from every other.

## Tickets

- [x] 01 — Model: Sentence source [T, D]; Linear on each row of [T, F]; QKᵀ scores, Scale (÷√d), Softmax over the last axis, A·V, Concat along features for matrices, packaged Attention head; shapes, params; tests
- [x] 02 — Forward + saturation: matrix ops and the head; `attentionPeak` (mean top weight, fixed seeds) and a `peakMax` rule; tests incl. softmax rows sum to 1, head = its parts, unscaled attention saturates
- [x] 03 — Workshop: matrix previews (heatmaps), attention-map inspector view with word labels on both axes, per-level parts
- [x] 04 — Chapter 4 levels: Scores, Attention weights, One head from parts, Two heads — unlocking after chapter 3, grouped, listed in Challenges; solvability + exploit tests (the planned "No bottleneck" level became the One-head lesson — see D6)

## Notes

- D1 Matrices are [T, F] (tokens × features) with no batch; Linear applies to each row (like nn.Linear on the last axis) with the same weights for every token.
- D2 Heads are untrained, so attention maps show the mechanism with arbitrary patterns; the lessons say trained heads learn meaningful ones (e.g. a word attending to its noun).
- D3 The sentence is chapter 3's 8-word sentence with the same word vectors, so "reach" numbers compare directly with the RNN/LSTM memory.
- D4 Scale is a block with a d knob (÷√d) rather than hidden inside QKᵀ, so the level can show what happens without it.
- D5 "One head from parts" is judged structurally (softmax(scale(QKᵀ))·V with Q, K, V Linear projections of the sentence), like chapter 3's lstmShape.
- D6 (revised while building) The planned "word-1 reach" rule was dropped: with untrained weights the last word's output depends on word 1 only ~8% — random attention needn't look at word 1. Attention's real advantage (word 1 is one step from every position, not eight) is taught in the lesson, not faked as a rule. Measured instead and used as a rule: scaling — mean top attention weight 0.53 with ÷√d vs 0.78 without (d = 8, seeds 1–3); `peakMax` = 0.65.
- D7 (found while play-testing) Adding a part after a data source (Sentence, Input, Word, zero state) — or Shift-clicking a part anywhere — starts a new branch on a free row below instead of splicing into the existing wires; Q, K and V all come off one sentence.
