import test from 'node:test';
import assert from 'node:assert/strict';
import { tensor, conv2d, relu, maxpool, flatten, linear, softmax, convWeights, linearWeights, rng } from '../src/sims/build/tensor.js';
import { conv2d as conv1 } from '../src/sims/build/conv.js';

const close = (a, b, eps = 1e-5) => assert.ok(Math.abs(a - b) < eps, `${a} ≈ ${b}`);

test('single-channel conv matches chapter 1 conv2d (stride, padding)', () => {
  const img = [[1, 2, 0, 3, 1], [0, 1, 4, 2, 2], [3, 1, 0, 1, 0], [2, 2, 1, 0, 4], [1, 0, 3, 2, 1]];
  const ker = [[1, 0, -1], [2, 0, -2], [1, 0, -1]];
  for (const [s, p] of [[1, 0], [1, 1], [2, 1]]) {
    const ref = conv1(img, ker, s, p).out;
    const y = conv2d(tensor([1, 5, 5], Float32Array.from(img.flat())), { w: Float32Array.from(ker.flat()), b: new Float32Array(1) }, 1, 3, s, p);
    assert.deepEqual(y.shape, [1, ref.length, ref[0].length]);
    assert.deepEqual([...y.data], ref.flat());
  }
});

test('multi-channel conv sums over input channels and adds bias', () => {
  // two 2×2 channels, one 1×1 filter with weights [2, -1], bias 0.5
  const x = tensor([2, 2, 2], Float32Array.from([1, 2, 3, 4, 10, 20, 30, 40]));
  const y = conv2d(x, { w: Float32Array.from([2, -1]), b: Float32Array.from([0.5]) }, 1, 1);
  assert.deepEqual([...y.data], [2 - 10 + 0.5, 4 - 20 + 0.5, 6 - 30 + 0.5, 8 - 40 + 0.5]);
});

test('relu, maxpool, flatten', () => {
  const x = tensor([1, 2, 4], Float32Array.from([-1, 2, -3, 4, 5, -6, 7, -8]));
  assert.deepEqual([...relu(x).data], [0, 2, 0, 4, 5, 0, 7, 0]);
  const m = maxpool(x, 2);
  assert.deepEqual(m.shape, [1, 1, 2]);
  assert.deepEqual([...m.data], [5, 7]);
  assert.deepEqual(flatten(x).shape, [8]);
});

test('linear and softmax', () => {
  const y = linear(tensor([2], Float32Array.from([1, 2])), { w: Float32Array.from([1, 1, 2, -1]), b: Float32Array.from([0, 1]) }, 2);
  assert.deepEqual([...y.data], [3, 1]);
  const p = softmax(tensor([3], Float32Array.from([1000, 1000, 1000])));
  for (const v of p.data) close(v, 1 / 3);
});

test('seeded init is deterministic, He-scaled, and classic kernels lead a 1-channel conv', () => {
  assert.deepEqual([...convWeights(3, 4, 3, 7).w], [...convWeights(3, 4, 3, 7).w]);
  assert.notDeepEqual([...convWeights(3, 4, 3, 7).w], [...convWeights(3, 4, 3, 8).w]);
  const { w } = linearWeights(1000, 50, 1);
  const std = Math.sqrt(w.reduce((a, v) => a + v * v, 0) / w.length);
  close(std, Math.sqrt(2 / 1000), 0.003);
  const k = convWeights(1, 8, 3, 1).w;
  assert.deepEqual([...k.slice(0, 9)].map((v) => Math.sign(v)), [1, 0, -1, 1, 0, -1, 1, 0, -1]); // vertical edge
  const r = rng(5);
  assert.ok(r() >= 0 && r() < 1);
});
