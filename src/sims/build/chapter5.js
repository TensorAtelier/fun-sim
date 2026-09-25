// Workshop levels, chapter 5: the Transformer block. Boards hold the sentence as one [8, 8] matrix.
import { addNode } from './graph.js';

function board(g) {
  addNode(g, 'seq', 40, 200, { words: 'long', dim: 8 }, { fixed: true, locked: true });
  addNode(g, 'output', 40 + 268 * 5, 40, { expect: [8, 8] }, { fixed: true, locked: true });
}

export const CH5_LEVELS = [
  {
    id: 'tf-parts', chapter: '5', title: 'Block from parts', target: [8, 8], parts: ['head', 'concat', 'linear', 'add', 'norm', 'relu'],
    tblockShape: true, budgets: [1e9, 1e9, 1e9],
    brief: 'Wire one Transformer block: attention over the sentence, Add the sentence back and LayerNorm; then an MLP (Linear → ReLU → Linear back to 8), Add its input back and LayerNorm again.',
    lesson: 'Attention lets words exchange information; the MLP then processes each word on its own. The two Adds are chapter 2b\'s skip connections, the LayerNorms keep every word\'s row at a steady scale. This is the original "post-norm" layout — most modern models normalise before each part instead (pre-norm), which trains more stably when stacked deep.',
    setup: board,
  },
  {
    id: 'tf-order', chapter: '5', title: 'Word order', target: [8, 8], parts: ['posenc', 'tblock'],
    requires: ['tblock'], orderMin: 0.001, budgets: [1e9, 1e9, 1e9],
    brief: 'Make the model notice word order: swapping words 2 and 3 ("old" and "dog") must change what comes out for word 1 at all (at least 0.1%).',
    lesson: 'Try a Transformer block alone first: word 1\'s output doesn\'t change at all — attention treats the sentence as a bag of words (it is permutation-equivariant). Positional encoding adds a different sine/cosine pattern to each position, so "the old dog" and "the dog old" finally look different. (Add it from the Sentence, then drag its output onto the block — the new wire replaces the old one.)',
    setup: board,
  },
  {
    id: 'tf-stack', chapter: '5', title: 'Stack three', target: [8, 8], parts: ['posenc', 'tblock'],
    chainBlocks: 3, orderMin: 0.001, budgets: [1e9, 4_000, 2_600],
    brief: 'Stack three Transformer blocks on top of positional encoding.',
    lesson: 'Deep models are just this, repeated — GPT-style models stack dozens to about a hundred blocks. For ★★★ look at where the parameters are: the MLP (Linear to ffn and back) is most of each block. Try a smaller MLP width.',
    setup: board,
  },
];
