# moe — plan

**Goal:** Ch 6: Mixture of Experts — a router that scores experts for each word, top-k gating that keeps only a few, sparse experts so total parameters grow while the parameters used per word stay small, and expert capacity/load (why balancing matters).
**Not yet:** training the router, auxiliary load-balancing loss (explained in a lesson), expert parallelism across devices, shared experts / fine-grained experts.
**Stack / interfaces:** `graph.js` blocks (Router, Top-k, Experts, packaged MoE layer), active-vs-total parameter counting; `tensor.js`/`forward.js` routing with capacity and dropped tokens; `chapter6.js` levels; Workshop routing-table view (words × experts) and load bars.
**Proof of done:** in the Workshop, build a router from Linear + Softmax and read which expert each word prefers; add Top-k and Experts and watch total parameters rise while active-per-word stays under half; shrink expert capacity until words start being dropped, then find the smallest capacity with no drops; finally build an MoE layer with ≥ 4× more total than active parameters and no dropped words — and (per D6) see routing collapse when an untrained Transformer block sits in front.

## Tickets

- [x] 01 — Model: Router [T, D] → [T, E], Top-k (per row, renormalised), Experts (x + gates → [T, D]; E experts, width, capacity factor), packaged MoE layer; active-parameter counting; tests
- [x] 02 — Forward: router softmax, top-k, capacity-limited expert dispatch with dropped-token count and per-expert load; weights seeded by block type (not id); tests incl. dense = sum over all experts, dropped tokens at low capacity, none at capacity E/k
- [x] 03 — Workshop: routing table (words × experts) and load bars in the inspector, active/total readout, parts, picker group, Challenges row
- [x] 04 — Chapter 6 levels: Router from parts, Sparse experts, Capacity, Scale up not out — with solvability + exploit tests

## Notes

- D1 Router and experts are seeded by block type + board seed, not block id, so routing (and dropped-token counts) can't change with arbitrary ids; verdicts use fixed seeds 1–3.
- D2 Active parameters per word = router + k experts (+ everything outside the MoE); total counts all E experts.
- D3 Capacity per expert = ceil(factor · T · k / E); each expert keeps its highest-gate words up to capacity and drops the rest (a dropped word gets no output from that expert) — the standard token-dropping scheme.
- D4 Load balancing is taught by capacity/drops, not by training; the lesson explains the auxiliary loss real models use.
- D5 Measured (seeds 1–3, worst case): with 4 experts and top-1, capacity 1× drops 4 word→expert assignments, 1.5× drops 2, 2× none — the Capacity level starts at 1× and scores stars by the smallest drop-free factor.
- D6 (found while building) After an untrained Transformer block the router collapses onto ~2 experts (rows look alike), so "Scale up, not out" routes the (position-encoded) sentence directly and teaches routing collapse in its lesson; measured: 16 experts drop words even at capacity 8× behind a block, none at 8× without it. Capacity slider goes to 16×.
