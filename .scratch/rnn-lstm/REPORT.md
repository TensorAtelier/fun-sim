# rnn-lstm — report

Branch `feat/rnn-lstm` (base `main` @ 9fa3503). Lite pipeline, auto mode.

## Tickets

| # | Ticket | Commit |
|---|---|---|
| 01 | Word / Zero-state sources, shared RNNCell & LSTMCell, Concat/Sigmoid/Tanh/Multiply | e7a5914 |
| 02 | Word vectors, cell ops (forget bias 2), shared weight slots, first-word memory rule | f015b73 |
| 03 | Sequence boards: per-level parts, shared badges, grouped params, memory readout | d1dd44d, fix (below) |
| 04 | Chapter 3 levels (Unroll, LSTM cell from parts, The bottleneck) + solvability tests | 1168177, fix (below) |

Verified in the browser (synthetic pointer events; the automation tool cannot deliver real mouse
presses to the background tab): Unroll completed ★★★ with 400 shared params; The bottleneck refuses a
plain RNN(32) at 11.3% memory and passes an LSTM(16) at 30.8%; words and cells show values; the
Network panel shows "LSTMCell ×8 (shared) 1,600". `npm test`: 56/56 pass. Parked tickets: none.

Found while play-testing and fixed on this branch: target-shape Outputs drew softmax bars (a chapter 2
bug on feature-map targets); adding parts pushed Output out of wiring reach; the forward pass never ran
on sequence boards (no image Input).

## QA verdict (goal mode)

GOAL VERDICT: **PARTIAL** — cell maths matched a reference (max error 3.2e-8); weights truly shared;
the memory metric is deterministic. Defects:

1. **High — a plain RNN passed The bottleneck by reading word 1 last or twice.** → **Fixed**: `inOrder` rule.
2. **Medium — LSTM-from-parts only counted parts.** → **Fixed**: `lstmShape` structural check.
3. Low — ledger D7 out of date. → **Fixed**.
4. Low — Unroll accepted any word order. → **Fixed** (inOrder).
5. Low — target-shape Output's inspector drew softmax bars. → **Fixed**.
6. Low — shared-cell rows grouped by type only; "not used" from the left-most copy. → **Fixed**.
7. Info — "unused" messages couldn't tell repeated words apart. → **Fixed** (word position).

Fixes verified by me (failing-then-passing tests + browser); QA not re-run.

## Decisions made for you

Ranked costly-and-surprising first:

- D5 Untrained LSTMs start with forget-gate bias 2 so they demonstrate what trained forget gates do; the lesson says so.
- D7 The bottleneck's verdict averages three fixed weight seeds (a single seed let a plain RNN pass by luck); floor 20% (RNN 11.3%, LSTM 30.8%).
- (after QA) Sequence rules: `inOrder` (one step per word, in order, from the zero state), `lstmShape` (h′ = o⊙tanh(c′), c′ = f⊙c + i⊙g), `useAll`.
- D2 All RNNCells share one weight set (likewise LSTMCell); params count once; blocks carry a "shared" badge.
- Cells take the state on input a and the word on input b, so "add after selected" chains through the state.
- D3 LSTM state travels packed as [h, c].
- D1 Recurrence is shown unrolled; the graph stays a DAG.
