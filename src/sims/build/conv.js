// Pure convolution math (no DOM) — shared by the sandbox, the levels and tests.

// Span of a k-wide kernel whose taps are d cells apart.
export const effK = (k, d = 1) => d * (k - 1) + 1;

// Output length along one axis: ⌊(n + 2p − d(k−1) − 1) / s⌋ + 1. ≤ 0 means the kernel doesn't fit.
export function outSize(n, k, s = 1, p = 0, d = 1) {
  return Math.floor((n + 2 * p - effK(k, d)) / s) + 1;
}

export function convParams(k, cin, cout, bias = true) {
  return k * k * cin * cout + (bias ? cout : 0);
}

// Single-channel 2D convolution (cross-correlation, like every DL framework).
// taps[idx] lists, for output cell idx = i*ow + j, each {y, x, v, w} it read (y/x in unpadded coords).
export function conv2d(input, kernel, s = 1, p = 0, d = 1) {
  const H = input.length, W = input[0].length, K = kernel.length;
  const oh = outSize(H, K, s, p, d), ow = outSize(W, K, s, p, d);
  const at = (y, x) => (y < 0 || x < 0 || y >= H || x >= W ? 0 : input[y][x]);
  const out = [], taps = [];
  for (let i = 0; i < oh; i++) {
    const row = [];
    for (let j = 0; j < ow; j++) {
      let sum = 0;
      const t = [];
      for (let a = 0; a < K; a++) {
        for (let b = 0; b < K; b++) {
          const y = i * s - p + a * d, x = j * s - p + b * d;
          const v = at(y, x), w = kernel[a][b];
          sum += v * w;
          t.push({ y, x, v, w });
        }
      }
      row.push(sum);
      taps.push(t);
    }
    out.push(row);
  }
  return { out, taps, oh, ow };
}
