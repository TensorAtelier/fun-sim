# block-editor — plan

**Goal:** DAG block editor for Build a Net (roadmap walking skeleton): parts bin, port wiring, live shape inference, red wires on mismatch, param counter, free-build mode, one level ([1,28,28] → 10 classes).
**Not yet:** forward pass / real values (tensor-engine), multi-input blocks (Add — residual-blocks), saving/sharing graphs, touch wire editing, more block types.
**Stack / interfaces:** vanilla JS + DOM blocks over an SVG wire layer, as a third "Workshop" tab on the Build a Net page; pure graph model in `src/sims/build/graph.js` (no DOM).
**Proof of done:** in the Workshop tab you add Conv2d/ReLU/MaxPool2d/Flatten/Linear blocks, wire them, see the shape on every wire, see a red wire + reason when shapes don't fit (e.g. Linear fed [C,H,W]), watch the param count, and finish the "Digit classifier" level with 1–3 stars by param budget.

## Tickets

- [x] 01 — Graph model: block types, connect/disconnect (single input per port, no cycles), shape inference, param counts, level check; `npm test` runs node:test suite
- [x] 02 — Workshop tab: board with parts bin; add blocks (click; auto-wires from the selected block), drag to move, select, delete
- [x] 03 — Wiring: drag out-port → in-port, bezier wires labelled with shapes, red wire + reason on mismatch, click wire to remove
- [x] 04 — Inspector: edit the selected block's params in the side panel; network section with per-block and total params and problems list
- [x] 05 — Levels: "Free build" and "Digit classifier" ([1,28,28] → 10, needs a Conv2d; stars ≤500k/≤100k/≤25k params), win banner, stars saved, desktop-first note on phones

## Notes

- D1 Graph is a DAG with ports from day one (roadmap decision): an output fans out, an input takes one wire; a new wire into a taken input replaces the old one.
- D2 Shapes omit batch: [C,H,W] for images, [F] for vectors — matches chapter 1's notation.
- D3 Errors vs pending: a block with no input is grey "needs input", not red; red is reserved for real mismatches so a half-built graph doesn't scream.
- D4 Params/stars count only blocks upstream of Output, so stray blocks on the board aren't penalised.
- D5 Adding a part while a block is selected places it to the right and auto-wires it — the common "extend the chain" action is one click.
- D6 Pooling is MaxPool2d(k) with stride k; stride/padding knobs wait for cnn-stacks.
- D7 Tests: node's built-in runner (`node --test`), no new deps; seam is the pure graph module.
- D8 Graph state is in memory only (save/share is on the roadmap's Later list); level stars persist with chapter 1's progress.
- D9 Visual style anchored to the existing app (dark panel, amber layers, cyan shapes) — no taste escalation needed.
