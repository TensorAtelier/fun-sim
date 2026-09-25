// Ten toy 28×28 digits, drawn from a tiny stroke font with a soft brush (no canvas, so node
// tests see exactly what the browser shows). Digits sit in a 20×20 box, like MNIST.
import { tensor } from './tensor.js';

const ellipse = (cx, cy, rx, ry, n = 20) =>
  Array.from({ length: n + 1 }, (_, i) => [cx + rx * Math.cos((2 * Math.PI * i) / n), cy + ry * Math.sin((2 * Math.PI * i) / n)]);

// Each digit: a list of polylines in a unit box (x right, y down).
const STROKES = [
  [ellipse(0.5, 0.5, 0.32, 0.48)],
  [[[0.32, 0.2], [0.55, 0], [0.55, 1]]],
  [[[0.15, 0.25], [0.3, 0.05], [0.6, 0], [0.82, 0.15], [0.82, 0.35], [0.6, 0.55], [0.15, 1], [0.88, 1]]],
  [[[0.15, 0.08], [0.5, 0], [0.8, 0.12], [0.8, 0.35], [0.45, 0.48], [0.8, 0.6], [0.85, 0.85], [0.5, 1], [0.12, 0.92]]],
  [[[0.7, 1], [0.7, 0], [0.1, 0.7], [0.9, 0.7]]],
  [[[0.85, 0], [0.2, 0], [0.15, 0.45], [0.55, 0.4], [0.85, 0.6], [0.85, 0.85], [0.55, 1], [0.15, 0.92]]],
  [[[0.75, 0.02], [0.4, 0.15], [0.18, 0.5], [0.18, 0.8], [0.4, 1], [0.72, 0.95], [0.85, 0.72], [0.7, 0.52], [0.4, 0.5], [0.18, 0.68]]],
  [[[0.12, 0], [0.88, 0], [0.4, 1]]],
  [ellipse(0.5, 0.25, 0.27, 0.24), ellipse(0.5, 0.72, 0.33, 0.28)],
  [ellipse(0.5, 0.3, 0.3, 0.28), [[0.8, 0.3], [0.76, 0.7], [0.5, 1]]],
];

function segDist(px, py, [ax, ay], [bx, by]) {
  const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
  const t = len2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
}

// Returns a [1, 28, 28] tensor with ink in [0, 1].
export function digit(d, { size = 28, box = 20, brush = 1.3 } = {}) {
  const off = (size - box) / 2;
  const lines = STROKES[d].map((pl) => pl.map(([x, y]) => [off + x * box, off + y * box]));
  const t = tensor([1, size, size]);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let dmin = Infinity;
      for (const pl of lines) for (let i = 1; i < pl.length; i++) dmin = Math.min(dmin, segDist(x + 0.5, y + 0.5, pl[i - 1], pl[i]));
      t.data[y * size + x] = Math.max(0, Math.min(1, brush + 0.5 - dmin));
    }
  }
  return t;
}
