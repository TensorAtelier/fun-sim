// DOM figures for convolution: input ✱ kernel = output, plus small layout helpers.
import { h } from '../../ui.js';
import { conv2d, outSize } from './conv.js';

export const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
export const fmt = (n) => String(n).replace('-', '−');
export const cellSize = (n) => Math.max(14, Math.min(34, Math.floor(320 / n)));

// Input intensity: amber ramp. t in [0, 1].
export function heat(t) {
  return `background: rgba(245,181,68,${(0.06 + 0.6 * t).toFixed(3)}); color: ${t > 0.55 ? '#0b0e14' : '#e6e9ef'}`;
}
// Signed values: amber positive, cyan negative. t in [-1, 1].
export function diverge(t) {
  const a = (0.08 + 0.6 * Math.abs(t)).toFixed(3);
  const rgb = t >= 0 ? '245,181,68' : '94,200,229';
  return `background: rgba(${rgb},${a}); color: ${Math.abs(t) > 0.55 ? '#0b0e14' : '#e6e9ef'}`;
}

// cell(y, x) -> { cls, text, style, sub, onclick, onenter }
export function grid(rows, cols, size, cell) {
  const g = h('div', { class: 'bn-grid', style: `grid-template-columns: repeat(${cols}, ${size}px); --c: ${size}px` });
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const c = cell(y, x);
      const el = h('div', { class: 'bn-cell ' + (c.cls || ''), style: c.style || '' }, c.text ?? '');
      if (c.sub) el.append(h('span', { class: 'w' }, c.sub));
      if (c.onclick) { el.classList.add('click'); el.addEventListener('click', c.onclick); }
      if (c.onenter) el.addEventListener('mouseenter', c.onenter);
      g.append(el);
    }
  }
  return g;
}

export const col = (label, el, sub) =>
  h('div', { class: 'bn-col' }, h('div', { class: 'bn-label' }, label), el, sub ? h('div', { class: 'bn-sub' }, sub) : null);
export const op = (sym) => h('div', { class: 'bn-op' }, sym);

// items: [{ text, kind: 'shape' | 'layer' | 'q' }]
export function pipeline(items) {
  return h('div', { class: 'bn-pipe' }, items.flatMap((it, i) => [
    i ? h('span', { class: 'bn-arrow' }, '→') : null,
    h('span', { class: 'bn-chip ' + (it.kind || '') }, it.text),
  ]));
}

// Input (with padding) ✱ kernel = output.
// o: { showValues, cur, reveal, showOutput, showStride, onHoverOut(idx), onKernelClick(a, b), size }
export function convFigure({ input, kernel, S, P, D = 1 }, o = {}) {
  const H = input.length, W = input[0].length, K = kernel.length;
  const Hp = H + 2 * P, Wp = W + 2 * P;
  const oh = outSize(H, K, S, P, D), ow = outSize(W, K, S, P, D);
  const fits = oh > 0 && ow > 0;
  const res = fits ? conv2d(input, kernel, S, P, D) : null;
  const vals = o.showValues !== false;
  const size = o.size || cellSize(Math.max(Hp, K));
  const maxIn = Math.max(1, ...input.flat().map(Math.abs));

  const taps = new Map(), next = new Set();
  const cur = o.cur ?? -1;
  if (res && cur >= 0 && cur < oh * ow) for (const t of res.taps[cur]) taps.set(`${t.y + P},${t.x + P}`, t.w);
  if (res && o.showStride && ow > 1) for (const t of res.taps[1]) next.add(`${t.y + P},${t.x + P}`);

  const inGrid = grid(Hp, Wp, size, (py, px) => {
    const y = py - P, x = px - P, key = `${py},${px}`;
    const pad = y < 0 || x < 0 || y >= H || x >= W;
    let cls = pad ? 'pad' : '';
    if (taps.has(key)) cls += ' rf';
    else if (next.has(key)) cls += ' rf2';
    const v = pad ? 0 : input[y][x];
    return {
      cls,
      text: vals || pad ? fmt(v) : '',
      style: !pad && vals ? heat(v / maxIn) : '',
      sub: vals && taps.has(key) ? '×' + fmt(taps.get(key)) : '',
    };
  });

  const maxW = Math.max(1, ...kernel.flat().map(Math.abs));
  const kGrid = grid(K, K, size, (a, b) => ({
    cls: vals ? '' : 'k',
    text: vals ? fmt(kernel[a][b]) : '',
    style: vals ? diverge(kernel[a][b] / maxW) : '',
    onclick: o.onKernelClick ? () => o.onKernelClick(a, b) : null,
  }));

  const parts = [
    col('Input', inGrid, `${H}×${W}` + (P ? ` + padding ${P}` : '') + (S > 1 ? ` · stride ${S}` : '')),
    op('✱'),
    col('Kernel', kGrid, `${K}×${K}` + (D > 1 ? ` · dilation ${D}` : '')),
  ];
  if (!fits) {
    parts.push(op('='), h('div', { class: 'bn-warn' }, 'Kernel is bigger than the padded input — no valid position.'));
  } else if (o.showOutput !== false) {
    const reveal = o.reveal ?? Infinity;
    const maxOut = Math.max(1, ...res.out.flat().map(Math.abs));
    const outGrid = grid(oh, ow, size, (i, j) => {
      const idx = i * ow + j, v = res.out[i][j], shown = idx < reveal;
      return {
        cls: (shown ? '' : 'hidden') + (idx === cur ? ' cur' : ''),
        text: vals && shown ? fmt(v) : '',
        style: vals && shown ? diverge(v / maxOut) : '',
        onenter: o.onHoverOut ? () => o.onHoverOut(idx) : null,
      };
    });
    parts.push(op('='), col('Output', outGrid, `${oh}×${ow}`));
  }
  return h('div', { class: 'bn-row bn-fig' + (vals ? '' : ' novals') }, parts);
}
