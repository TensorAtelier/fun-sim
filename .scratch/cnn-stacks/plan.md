# cnn-stacks — plan

**Goal:** Chapter 2 — conv/pool/flatten/linear levels in the Workshop that teach downsampling, stride vs pooling, receptive fields and parameter budgets, ending in a tiny digit classifier.
**Not yet:** training (roadmap Later), skip connections / normalisation (residual-blocks), average pooling with stride ≠ size variants beyond MaxPool stride, dropout.
**Stack / interfaces:** extends `graph.js` (rules, receptive field), `tensor.js`/`forward.js` (new ops), `workshop.js` (levels, RF overlay); Challenges tab links chapter 2 to the Workshop.
**Proof of done:** in the Workshop, pick chapter-2 levels in order, each unlocking the next: hit exact target shapes, find the patchify trick for 3★, stack 3×3 convs until a selected block's receptive-field box on the Input covers 7×7, shrink a classifier under 1k parameters with Global Average Pooling, and build a two-stage CNN classifier.

## Tickets

- [x] 01 — Model: MaxPool stride knob, GlobalAvgPool block, Output target shapes, receptive field per block (size, jump, centre), generalised level rules (target, requires, forbids, max kernel, min receptive field, budgets); tests
- [x] 02 — Forward + RF view: pool stride and global-average-pool ops; inspector shows receptive field; selected block's RF drawn as a box on the Input preview
- [x] 03 — Chapter 2 levels: First classifier, Downsample, Stride instead of pooling, See 7×7, Global average pooling, Tiny classifier — ordered, unlocking, starred, with a lesson line each
- [x] 04 — Challenges tab: chapter 2 listed as playable, opens the Workshop; lock icons in the Workshop level picker

## Notes

- D1 Chapter 2 lives in the Workshop (block editor), not the quiz-style Challenges tab; the Challenges roadmap links to it.
- D2 Output blocks can demand any target shape (`expect`), so levels can end in feature maps, not only class scores.
- D3 Receptive field uses the standard recurrence: rf += (k−1)·jump, jump ·= s, centre += ((k−1)/2 − p)·jump; GlobalAvgPool/Linear make it "global".
- D4 Level rules are data (target, requires, forbids, maxK, rfMin, budgets); checkLevel stays the single judge.
- D5 MaxPool gains a stride knob (default = size), closing block-editor D6.
- D6 Unlock rule matches chapter 1: ≥1★ on the previous level; the original "Digit classifier" becomes chapter 2's first level (same id, stars kept).
