# residual-blocks — plan

**Goal:** Ch 2b — skip wires and an Add block, shape-matching 1×1 convs on the skip path, normalisation, and a hands-on look at why deep plain stacks struggle (the signal fades) and how skips or norms fix it.
**Not yet:** training/gradients (roadmap Later — we show the forward signal, and say gradients use the same paths), BatchNorm as a working block (needs a batch), concatenation (DenseNet), attention.
**Stack / interfaces:** multi-input ports in `graph.js`; `add` and `layernorm` ops + init-scale gain + signal RMS in `tensor.js`/`forward.js`; two-port blocks and an Init-scale knob in the Workshop; four more levels in `chapter2.js`.
**Proof of done:** in the Workshop, wire a skip from a conv's output around conv→ReLU→conv into Add; see Add refuse mismatched shapes and fix it with a 1×1 stride-2 conv on the skip; watch an 8-conv plain stack's signal (RMS on each block) fade to ~0 at init scale 0.5, then rescue it with skips — and, in the last level, with LayerNorm instead.

## Tickets

- [x] 01 — Model: input ports (edges carry `port`; each port takes one wire), Add block (2 inputs, shapes must match, hint names the 1×1 conv fix), LayerNorm block, receptive field through Add; tests
- [x] 02 — Forward + signal: add and layernorm ops, init-scale gain on weights, per-block RMS, `signalMin` level rule judged on real outputs; tests show a plain deep stack fading and a residual one holding
- [x] 03 — Workshop: two-port Add block (drop on a port, or on the body → first free port), healing with ports, RMS on blocks, Init-scale slider (levels may lock it)
- [x] 04 — Chapter 2b levels: Skip connection, Matching shapes, Deep & alive, Normalize — unlocking after Tiny classifier, grouped in the picker, listed in the Challenges tab; every level proven 3★-solvable

## Notes

- D1 Ports are indices on the target block; `connect(g, from, to, port)` replaces only that port's wire. Existing single-input blocks use port 0, so earlier graphs and tests are unchanged.
- D2 "Why deep plain stacks struggle" is shown with the forward signal (RMS through depth at a reduced init scale), not gradients — honest without training, and the lesson says gradients travel the same paths backwards.
- D3 LayerNorm normalises each sample over all its values (mean 0, std 1) then scales/shifts per channel (2·C params; 2·F on vectors) — GroupNorm(1)-style, labelled as such in its help text.
- D4 BatchNorm is explained in a lesson, not offered as a block: with one example at a time it would be a fake.
- D5 Receptive field through Add is the larger of its inputs (same jump required — shapes match, so steps match).
- D6 Init scale is a board knob (default 1 = He init); "Deep & alive" and "Normalize" lock it at 0.5 so plain stacks visibly fade.
- D7 Measured (8 stages, digit 3): init scale 0.5 → plain 0.1%, residual 239%, LayerNorm 253% of the input's RMS; at scale 1 the residual stack grows to 3,392% — used in the lessons (why ResNets pair skips with normalisation).
