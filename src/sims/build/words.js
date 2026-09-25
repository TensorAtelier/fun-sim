// A toy vocabulary with fixed random word vectors (like an untrained embedding table), and the
// sentences the chapter 3 levels read.
import { rng, tensor } from './tensor.js';

export const VOCAB = ['the', 'cat', 'sat', 'down', 'old', 'dog', 'slept', 'by', 'warm', 'fire', 'a', 'bird', 'sang', 'on', 'red', 'roof'];
export const SENTENCES = {
  short: ['the', 'cat', 'sat', 'down'],
  long: ['the', 'old', 'dog', 'slept', 'by', 'the', 'warm', 'fire'],
};

// Word i as a [dim] vector: seeded standard-normal values, the same every time.
export function wordVector(i, dim = 8) {
  const r = rng(0x5eed + i * 7919), t = tensor([dim]);
  for (let k = 0; k < dim; k++) {
    const u = Math.max(r(), 1e-12), v = r();
    t.data[k] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return t;
}
