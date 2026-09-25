# rnn-lstm — plan

**Goal:** Ch 3 (thin): unroll an RNN over a sentence, build an LSTM cell's gates from parts, and feel the fixed-size bottleneck — the whole sentence squeezed into one state vector — that motivates attention.
**Not yet:** training, GRU, bidirectional RNNs, sequence-to-sequence decoders, real text datasets, attention (next chapter).
**Stack / interfaces:** new vector blocks in `graph.js` (Token inputs, State0, RNNCell, LSTMCell, Concat, Sigmoid, Tanh, Multiply); `tensor.js`/`forward.js` ops with weights shared per cell type; a toy sentence in `words.js`; chapter 3 levels in `chapter3.js`; Workshop shows shared-weight badges and a memory readout.
**Proof of done:** in the Workshop, chain four RNNCell copies over "the cat sat down" and see the parameter count stay the same as you add steps; wire an LSTM cell from Linear/Sigmoid/Tanh/Multiply/Add parts; then on an 8-word sentence see a plain RNN's memory of the first word fade to almost nothing in the final state, while an LSTM keeps it — and read why attention (chapter 4) removes the bottleneck altogether.

## Tickets

- [x] 01 — Model: Token and State0 inputs, RNNCell and LSTMCell (2 inputs, weights shared per type, counted once), Concat, Sigmoid, Tanh, Multiply; shapes and params; tests
- [x] 02 — Forward + memory: toy sentence and word vectors; ops for every new block (LSTM forget-gate bias starts high); runner accepts one tensor per input block; `memoryMin` rule = how much swapping word 1 changes the final state; tests show RNN fading and LSTM holding
- [x] 03 — Workshop: sequence boards (a row of word inputs + a zero state), shared-weights badge and single-count params, memory readout in the goal
- [x] 04 — Chapter 3 levels: Unroll, LSTM cell from parts, The bottleneck — unlocking after chapter 2b, grouped in the picker, listed in the Challenges tab; 3★ solvability tests

## Notes

- D1 Recurrence is shown unrolled (one block per time step), keeping the graph a DAG as the roadmap decided.
- D2 All RNNCell blocks share one weight set (likewise LSTMCell) — the defining RNN idea; params count it once and blocks show a "shared" badge.
- D3 An LSTM's state (h, c) travels as one [2H] vector (h‖c), so LSTMCell has one output; the level's Output reads it whole. Building the cell from parts uses separate h and c inputs.
- D4 "Memory of word 1" = relative change in the final state when word 1 is swapped for another word, averaged over a few swaps. Honest without training: it measures how much of word 1 can still reach the end.
- D5 LSTM forget-gate bias starts high (the common init trick) so the untrained LSTM demonstrates what a trained forget gate does; the lesson says so.
- D6 The "LSTM from parts" level is judged structurally (enough of each part, correct output shape), not by numerically matching a reference cell.
- D7 Measured with the shipped code, averaged over weight seeds 1–3 (8-word sentence): plain RNN(32) keeps 11.3% of word 1 in its final state, LSTM(16) keeps 30.8% — the bottleneck level's floor is 20%. (A single seed varied 5–17% for the RNN, so the verdict averages fixed seeds and ignores Reroll.)
