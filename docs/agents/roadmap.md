# Roadmap

goal: Build a Net teaches the building blocks of modern AI by having you assemble them — done when a player can wire a working Transformer block and an MoE layer from parts (shapes checked at every connection, real forward passes on toy data) and train a small policy from reward. (Ch 1, convolution, shipped before this roadmap: 4f53cd6.)
landing: merge

## Features

- [x] block-editor      — DAG block editor: parts bin, port wiring, live shape inference, red wires on mismatch, param counter, free-build mode, one level ([1,28,28] → 10 classes)   → .scratch/block-editor/REPORT.md
- [x] tensor-engine     — plain-JS forward pass on toy data; blocks show their real outputs (feature maps, values)   (needs: block-editor)   → .scratch/tensor-engine/REPORT.md
- [x] cnn-stacks        — Ch 2: conv/pool/flatten/linear levels, receptive fields, param budgets, a tiny digit classifier   (needs: tensor-engine)   → .scratch/cnn-stacks/REPORT.md
- [x] residual-blocks   — Ch 2b: skip wires + Add, shape-matching 1×1 convs, BatchNorm/LayerNorm, why deep plain stacks struggle   (needs: cnn-stacks)   → .scratch/residual-blocks/REPORT.md
- [x] rnn-lstm          — Ch 3 (thin): unrolled RNN, LSTM gates from parts, the fixed-size bottleneck that motivates attention   (needs: tensor-engine)   → .scratch/rnn-lstm/REPORT.md
- [x] attention         — Ch 4: Q/K/V from parts, scaled dot-product, softmax, live attention weights on a sentence, multi-head   (needs: rnn-lstm)   → .scratch/attention/REPORT.md
- [x] transformer-block — Ch 5: attention + MLP + residuals + LayerNorm, positional encoding, stacking blocks   (needs: attention, residual-blocks)   → .scratch/transformer-block/REPORT.md
- [x] moe               — Ch 6: router, top-k gating, sparse experts, load balancing, active vs total params   (needs: transformer-block)   → .scratch/moe/REPORT.md
- [x] policy-gradient   — Ch 7: learning from reward — bandit, baseline, gridworld, cart-pole with your own MLP (reuses src/sims/neural/net.js backprop), RLHF/GRPO bridge level   (needs: cnn-stacks, transformer-block)   → .scratch/policy-gradient/REPORT.md

## Later

- In-browser training with general autograd (conv, attention)
- PPO / actor-critic chapter (value network, clipping); Q-learning / DQN
- Run Reward-tab judging off the main thread (a worker) — today a ~0.15–0.5 s pause for cart-pole
- Export a built graph as PyTorch nn.Module code
- Save/share designs (link or JSON)
- Draw your own input digit in the Workshop (today: ten built-in toy digits)
- Faster forward pass for big free-build networks (worker or WebGPU) — today values pause above 60M multiply-adds
- Tune chapter 2 star budgets after playtesting (thin 1-channel chains still earn ★★★ on some levels)
- A working BatchNorm block once inputs come in batches (today only explained in the Normalize lesson)
- Concatenation joins (DenseNet / U-Net style) alongside Add
- Auto-scroll the board while dragging a wire (long sequence boards need manual scrolling today)
- GRU and bidirectional RNN levels
- Masked (causal) attention, and a trained toy head so attention maps show meaningful patterns
- Pre-norm vs post-norm as a hands-on level (today a lesson note), and decoder blocks with causal masking
- MoE inside the Transformer block (replacing its MLP) once a trained or balanced router avoids routing collapse
- DOM-level regression tests for the Workshop editor (only the pure graph model is tested today)
- Standalone sims beyond the game (e.g. attention visualizer, cart-pole playground)

## Out of scope

- Touch/phone wire editing — desktop-first; phones get a "best on desktop" note
- Accounts, multiplayer, leaderboards — personal learning tool
