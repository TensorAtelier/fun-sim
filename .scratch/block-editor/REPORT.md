# block-editor — report

Branch `feat/block-editor` (base `main` @ 0e16fc0). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | Graph model: types, connect/disconnect, shape inference, params, level check; `npm test` | 0ddd0ea |
| 02 | Workshop tab: parts bin, auto-chaining add, move, select, delete (chain heals) | 53d73c1 |
| 03 | Drag-to-wire, shape-labelled wires, red wires on mismatch, click to cut | 42e3995 |
| 04 | Inspector + network parameter summary | ae562b4, fix 8eaa6a1 |
| 05 | Levels: Digit classifier (param-budget stars), Free build; phone note | 416af54 |

Verified in the browser: built Conv2d→ReLU→Flatten→Linear, wired Linear→Output with the mouse
(★★☆, 62,810 params), spliced in MaxPool2d (★★★, 15,770 params, saved), stray Conv2d shown as
"not used" and excluded from the total. `npm test`: 9/9 pass.

Parked tickets: none.

## QA verdict (verbatim summary of the goal-mode review)

GOAL VERDICT: MET — `npm test` 9/9, `npx vite build` OK; all five tickets delivered with file:line
evidence; no out-of-scope creep; no crashes, listener leaks or cycle bugs.

Defects reported, and what happened:
1. Network total summed stray blocks while stars counted only blocks feeding Output (drift from D4) — **fixed** in 8eaa6a1.
2. Same-type blocks indistinguishable in the Network rows — **fixed** (numbered, board order).
3. `.scratch/block-editor/*` committed on the branch — **kept**: the lite pipeline commits plan.md ticks with each ticket by design.
4. Inspector showed "In ?" for Input/pending blocks — **fixed** (shows —).

Risks noted: UI tickets have no automated tests (only the pure graph model does, per D7); board width
could be outrun by long chains — **fixed** (board grows with its blocks).

## Decisions made for you

Ranked costly-and-surprising first:

- D1 Graph is a DAG with ports from day one: an output fans out, an input takes one wire; a new wire into a taken input replaces the old one.
- D4 Params/stars count only blocks upstream of Output; strays are listed dimmed as "not used".
- D5 Adding a part while a block is selected places it to the right and splices it into that block's outgoing wires; deleting a block heals the chain.
- D3 A block with no input is grey "needs input", not red; red is only for real mismatches.
- D6 MaxPool2d(k) always uses stride k; stride/padding knobs wait for cnn-stacks.
- D8 Graph state is in memory only; level stars persist with chapter 1's progress (key `ws-digits`).
- D2 Shapes omit batch: [C,H,W] and [F].
- D7 Tests use node's built-in runner; the seam is the pure graph module.
- D9 Visual style follows the existing app; no taste escalation.
