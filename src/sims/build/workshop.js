// Workshop: a board where you assemble networks from blocks and wires.
import { h, section, slider, select as selectCtl, buttons, checkbox } from '../../ui.js';
import { createRunner, estimateCost, firstWordMemory, attentionPeak, orderSensitivity, moeStats } from './forward.js';
import { SENTENCES } from './words.js';
import { digit } from './digits.js';
import { tensor, range, rms } from './tensor.js';
import { WS_LEVELS, setupLevel, KNOBS } from './chapter2.js';
import { BLOCKS, createGraph, addNode, removeNode, connect, disconnect, analyze, getNode, checkLevel, ancestors, countParams } from './graph.js';

const BW = 176;        // block width (px)
const PORT_Y = 22;     // first input/output port centre, from block top
const PORT_GAP = 26;   // spacing between input ports on multi-input blocks
const GAP = 92;        // horizontal gap when auto-placing (room for the shape label)
const SVG = 'http://www.w3.org/2000/svg';
// Parts bins: image chapters by default; a level can list its own; free build gets everything.
const PARTS = ['conv', 'relu', 'pool', 'gap', 'flatten', 'linear', 'add', 'norm'];
const ALL_PARTS = [...PARTS, 'rnn', 'lstm', 'concat', 'sigmoid', 'tanh', 'mul', 'scores', 'scale', 'softmax', 'wsum', 'head', 'posenc', 'tblock', 'router', 'topk', 'experts', 'moe'];
// Forward-pass budget. Small passes run on every change; mid-size ones wait until the knob
// stops moving; anything bigger pauses the previews instead of freezing the page.
const SYNC_MACS = 8e6, MAX_MACS = 60e6, MAX_WEIGHTS = 5e6;

const fmtShape = (s) => (s ? `[${s.join(', ')}]` : '?');

// ---------- value previews ----------
// Colour a value: ink for the input image, amber (+) / cyan (−) for everything after it.
function paint(img, i, v, scale, ink) {
  const t = Math.max(-1, Math.min(1, v / scale));
  let r, g, b;
  if (ink) { r = 20 + 225 * t; g = 24 + 190 * t; b = 32 + 130 * t; }
  else if (t >= 0) { r = 11 + 234 * t; g = 14 + 167 * t; b = 20 + 48 * t; }
  else { r = 11 + 83 * -t; g = 14 + 186 * -t; b = 20 + 209 * -t; }
  img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
}
const maxAbs = (t) => { let m = 1e-9; for (const v of t.data) m = Math.max(m, Math.abs(v)); return m; };
// Largest |value| in one channel of a [C,H,W] tensor.
export function chMaxAbs(t, c) {
  const n = t.shape[1] * t.shape[2];
  let m = 1e-9;
  for (let i = c * n; i < (c + 1) * n; i++) m = Math.max(m, Math.abs(t.data[i]));
  return m;
}

// One channel of a [C,H,W] tensor as a pixelated canvas `px` wide.
export function channelCanvas(t, c, px, scale = maxAbs(t), ink = false) {
  const [, H, W] = t.shape;
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(W, H);
  for (let i = 0; i < H * W; i++) paint(img, i * 4, t.data[c * H * W + i], scale, ink);
  ctx.putImageData(img, 0, 0);
  cv.className = 'ws-map';
  cv.style.width = `${px}px`;
  cv.style.height = `${Math.round((px * H) / W)}px`;
  return cv;
}

// A vector as a strip of coloured cells, averaged down to at most `cols` cells.
export function vectorCanvas(t, cols, height) {
  const F = t.data.length, n = Math.min(F, cols), scale = maxAbs(t);
  const cv = document.createElement('canvas');
  cv.width = n; cv.height = 1;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(n, 1);
  for (let i = 0; i < n; i++) {
    const a = Math.floor((i * F) / n), b = Math.max(a + 1, Math.floor(((i + 1) * F) / n));
    let s = 0;
    for (let j = a; j < b; j++) s += t.data[j];
    paint(img, i * 4, s / (b - a), scale, false);
  }
  ctx.putImageData(img, 0, 0);
  cv.className = 'ws-map ws-strip';
  cv.style.height = `${height}px`;
  return cv;
}

function probBars(t) {
  const top = t.data.indexOf(Math.max(...t.data));
  return h('div', { class: 'ws-probs' },
    h('div', { class: 'ws-bars' }, [...t.data].map((p, i) =>
      h('div', { class: 'ws-pbar' + (i === top ? ' top' : ''), style: `height:${Math.max(2, p * 100)}%`, title: `${i}: ${(p * 100).toFixed(1)}%` }))),
    h('div', { class: 'ws-probs-cap' }, `top: ${top} · ${(t.data[top] * 100).toFixed(0)}% · untrained`));
}

// A matrix [T, F] as a heatmap: one row per word, one column per feature.
function matrixCanvas(t, px) {
  return channelCanvas({ shape: [1, ...t.shape], data: t.data }, 0, px);
}

// A labelled weight grid: attention (row = the word looking, column = the word looked at), or
// routing (row = word, column = expert).
function attentionMap(w, words, colLabels = words) {
  const [T, S] = w.shape;
  const grid = h('div', { class: 'ws-attn', style: `grid-template-columns: auto repeat(${S}, 1fr)` });
  grid.append(h('span'));
  for (let j = 0; j < S; j++) grid.append(h('span', { class: 'ws-attn-col' }, colLabels[j] ?? String(j + 1)));
  for (let i = 0; i < T; i++) {
    grid.append(h('span', { class: 'ws-attn-row' }, words[i] ?? String(i + 1)));
    for (let j = 0; j < S; j++) {
      const v = w.data[i * S + j];
      grid.append(h('span', { class: 'ws-attn-cell', style: `background: rgba(245,181,68,${Math.min(1, v * 1.4).toFixed(3)})`,
        title: `"${words[i]}" → ${colLabels[j] ?? j + 1}: ${(v * 100).toFixed(0)}%` }));
    }
  }
  return grid;
}

function preview(n, t) {
  if (n.type === 'output' && !n.params.expect) return probBars(t);
  if (t.shape.length === 2) {
    return h('div', { class: 'ws-prev' }, matrixCanvas(t, Math.min(BW - 22, Math.max(40, t.shape[1] * 8))),
      h('div', { class: 'ws-prev-cap' }, `${t.shape[0]} × ${t.shape[1]}${n.type === 'softmax' ? ' · rows sum to 1' : ''}`));
  }
  if (t.shape.length === 1) {
    return h('div', { class: 'ws-prev' }, vectorCanvas(t, BW - 22, 12),
      h('div', { class: 'ws-prev-cap' }, t.data.length > BW - 22 ? `${t.data.length} values, averaged` : `${t.data.length} values`));
  }
  // Each thumbnail uses its own channel's scale so weak channels stay readable; the input keeps 0–1 ink.
  const C = t.shape[0], shown = Math.min(C, 4);
  const px = Math.floor((BW - 22 - 4 * (shown - 1)) / shown);
  return h('div', { class: 'ws-prev' },
    h('div', { class: 'ws-maps' }, Array.from({ length: shown }, (_, c) => channelCanvas(t, c, Math.min(px, 64), n.type === 'input' ? 1 : chMaxAbs(t, c), n.type === 'input'))),
    C > shown ? h('div', { class: 'ws-prev-cap' }, `+${C - shown} more channels`) : null);
}

// Large view of one block's output for the inspector.
function detail(n, t, words = [], isAttn = false) {
  const [lo, hi] = range(t);
  const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
  const box = h('div', { class: 'ws-detail' });
  if (n.type === 'output' && !n.params.expect) {
    const bars = [...t.data].map((p, i) => h('div', { class: 'ws-dbar' },
      h('div', { class: 'ws-pbar' + (p === hi ? ' top' : ''), style: `height:${Math.max(2, p * 100)}%` }), h('span', {}, String(i))));
    box.append(h('div', { class: 'ws-dbars' }, bars),
      h('p', { class: 'hint' }, 'Class probabilities (softmax). The weights are untrained, so these guesses are random — training comes in a later chapter.'));
    return box;
  }
  // Word labels only on real attention maps: a head, or a Softmax over (scaled) scores.
  // Routing: gates per word and expert, expert load, dropped words.
  const route = (n.type !== 'output' && t.routing) || (n.type === 'router' || n.type === 'topk' || (isAttn === 'route' && n.type === 'softmax') ? { gates: t } : null);
  if (route) {
    const E = route.gates.shape[1], experts = Array.from({ length: E }, (_, e) => `E${e + 1}`);
    box.append(h('div', { class: 'bn-label' }, 'Routing table'), attentionMap(route.gates, words, experts));
    if (route.load) {
      const peak = Math.max(1, ...route.load);
      box.append(h('div', { class: 'bn-label ws-load-label' }, `Words per expert (capacity ${route.capacity})`),
        h('div', { class: 'ws-load' }, route.load.map((l, e) => h('div', { class: 'ws-load-col' },
          h('div', { class: 'ws-load-bar' + (l >= route.capacity ? ' full' : ''), style: `height:${(l / peak) * 100}%` }), h('span', {}, `E${e + 1}`)))),
        h('p', { class: 'hint' }, route.dropped ? `${route.dropped} word→expert assignment${route.dropped > 1 ? 's' : ''} dropped (a full expert turned them away).` : 'No word dropped.'));
    }
    if (!t.routing) return box;
  }
  const att = isAttn === true && n.type === 'softmax' && t.shape.length === 2 ? t : t.attention;
  if (att) {
    box.append(h('div', { class: 'bn-label' }, 'Attention map'), attentionMap(att, words),
      h('p', { class: 'hint' }, 'Row = the word doing the looking; column = the word it looks at. Untrained heads look in arbitrary places; trained ones learn useful patterns — a verb finding its subject, a pronoun its noun.'));
    if (n.type !== 'head') return box;
  }
  if (t.shape.length === 2) {
    box.append(matrixCanvas(t, 240), h('div', { class: 'ws-prev-cap' }, `${t.shape[0]} words × ${t.shape[1]} features`));
  } else if (t.shape.length === 3) {
    const C = t.shape[0], shown = Math.min(C, 16);
    box.append(h('div', { class: 'ws-dgrid' }, Array.from({ length: shown }, (_, c) =>
      h('div', { class: 'ws-dcell' }, channelCanvas(t, c, 52, n.type === 'input' ? 1 : chMaxAbs(t, c), n.type === 'input'), h('span', {}, `ch ${c}`)))));
    if (C > shown) box.append(h('div', { class: 'ws-prev-cap' }, `showing ${shown} of ${C} channels`));
  } else {
    box.append(vectorCanvas(t, 240, 22), h('div', { class: 'ws-prev-cap' }, `${t.data.length} values`));
  }
  box.append(h('div', { class: 'ws-prev-cap' }, `range ${fmt(lo)} … ${fmt(hi)} · amber +, cyan −`));
  if (n.type === 'conv' || n.type === 'linear') {
    box.append(h('p', { class: 'hint' }, n.type === 'conv'
      ? 'Untrained: weights are random (the first filters on a 1-channel input start as classic edge/blur kernels).'
      : 'Untrained: weights are random, so these values mix the input arbitrarily.'));
  }
  return box;
}



export function mountWorkshop({ stage, side, progress, saveProgress }) {
  const root = h('div', { class: 'ws' });
  const bar = h('div', { class: 'ws-bar' });
  const bin = h('div', { class: 'ws-bin' }, h('div', { class: 'bn-label' }, 'Parts'));
  const scroller = h('div', { class: 'ws-scroll' });
  const board = h('div', { class: 'ws-board' });
  const wires = document.createElementNS(SVG, 'svg');
  wires.classList.add('ws-wires');
  const layer = h('div', { class: 'ws-layer' });
  board.append(wires, layer);
  scroller.append(board);
  const status = h('span', { class: 'ws-status' });
  const msg = h('span', { class: 'ws-msg' });
  bar.append(status, msg, h('span', { class: 'ws-mobile' }, 'Best on a desktop: wiring needs a mouse.'));
  root.append(bar, h('div', { class: 'ws-main' }, bin, scroller));
  stage.append(root);

  const levelSec = section(side, 'Level');
  // Open on the first level without stars (or the last one if all are done).
  let level = WS_LEVELS.find((L) => !L.free && !(progress[L.id] >= 1)) || WS_LEVELS[WS_LEVELS.length - 2];
  const view = { show: true, digit: 3, seed: 1, gain: 1 };
  const runner = createRunner();
  let outputs = new Map();
  const stars = (n) => '★'.repeat(n) + '☆'.repeat(3 - n);
  const playable = WS_LEVELS.filter((L) => !L.free);
  const unlocked = (L) => L.free || playable.indexOf(L) === 0 || (progress[playable[playable.indexOf(L) - 1].id] || 0) >= 1;
  const levelLabel = (L) => L.free ? L.title
    : `${playable.indexOf(L) + 1}. ${L.title}  ${unlocked(L) ? stars(progress[L.id] || 0) : '🔒'}`;
  const levelSel = selectCtl(levelSec, {
    options: WS_LEVELS.map((L) => [L.id, levelLabel(L)]), value: level.id,
    onChange: (id) => { level = WS_LEVELS.find((L) => L.id === id); reset(); },
  });
  function refreshLevelOptions() {
    for (const L of WS_LEVELS) {
      const opt = levelSel.querySelector(`option[value="${L.id}"]`);
      opt.textContent = levelLabel(L);
      opt.disabled = !unlocked(L);
    }
  }
  // Group the picker by chapter.
  for (const [ch, label] of [['2', 'Chapter 2 · CNN stacks'], ['2b', 'Chapter 2b · Residual blocks'], ['3', 'Chapter 3 · RNN & LSTM'], ['4', 'Chapter 4 · Attention'], ['5', 'Chapter 5 · Transformer'], ['6', 'Chapter 6 · Mixture of Experts'], [undefined, 'Sandbox']]) {
    const grp = h('optgroup', { label });
    for (const L of WS_LEVELS.filter((l) => l.chapter === ch)) grp.append(levelSel.querySelector(`option[value="${L.id}"]`));
    levelSel.append(grp);
  }
  levelSel.value = level.id;
  refreshLevelOptions();
  const brief = h('p', { class: 'hint ws-brief' });
  const lesson = h('p', { class: 'hint ws-lesson' });
  const goal = h('div', { class: 'ws-goal' });
  levelSec.append(brief, goal, lesson);
  buttons(levelSec, [{ label: 'Reset board', onClick: () => reset() }]);

  const dataSec = section(side, 'Values');
  selectCtl(dataSec, {
    label: 'Input sample', value: '3',
    options: Array.from({ length: 10 }, (_, d) => [String(d), `Digit ${d}`]),
    onChange: (v) => { view.digit = Number(v); refresh(); },
  });
  checkbox(dataSec, { label: 'Show values on blocks', checked: view.show, onChange: (on) => { view.show = on; refresh(); } });
  buttons(dataSec, [{ label: 'Reroll weights', onClick: () => { view.seed++; refresh(); } }]);
  const gainCtl = slider(dataSec, {
    label: 'Init scale (× He)', min: 0.25, max: 1.5, step: 0.05, value: view.gain, format: (v) => `${v.toFixed(2)}×`,
    onInput: (v) => { view.gain = v; refresh(); },
  });
  const budgetNote = h('div', { class: 'hint ws-budget' });
  dataSec.append(budgetNote);
  let fwdTimer = null, outputsStale = false;
  const inspector = section(side, 'Selected block');
  const netSec = section(side, 'Network');
  let info = null, infoDetail = null;

  let g = createGraph();
  let selected = null;
  let analysis = new Map();

  function renderBin() {
    for (const el of [...bin.children].slice(1)) el.remove();
    for (const type of level.parts || (level.free ? ALL_PARTS : PARTS)) {
      const B = BLOCKS[type];
      bin.append(h('button', { class: 'ws-part', onclick: (ev) => addPart(type, ev) },
        h('b', {}, B.title), h('span', {}, B.summary(B.defaults))));
    }
    bin.append(h('div', { class: 'hint ws-help' },
      'Click a part to add it after the selected block (Shift-click, or from an input, starts a new branch). Drag from a right port to a left port to wire. Click a wire to cut it. Del removes the selected block.'));
  }

  let msgTimer = null;
  function flash(text) {
    msg.textContent = text;
    msg.classList.add('on');
    clearTimeout(msgTimer);
    msgTimer = setTimeout(() => msg.classList.remove('on'), 2600);
  }

  function reset() {
    g = createGraph();
    setupLevel(g, level);
    renderBin();
    // Some levels pin the weight scale so the effect they teach is visible.
    view.gain = level.initScale ?? 1;
    gainCtl.set(view.gain);
    gainCtl.input.disabled = level.initScale != null;
    brief.textContent = level.brief;
    lesson.textContent = level.lesson ? `💡 ${level.lesson}` : '';
    levelSel.value = level.id;
    selected = g.nodes[0].id;
    refresh();
    renderInspector();
  }

  // Add a part after the selected block: place it to the right, move later blocks along, and
  // splice it into the selected block's outgoing wires.
  // Add a part after the selected block. Normally it is spliced in (it takes over the selected
  // block's outgoing wires). From a data source, or with Shift held, it starts a new branch on a
  // free row below instead — how Q, K and V all come off one sentence.
  function addPart(type, ev) {
    const sel = selected != null ? getNode(g, selected) : null;
    const canChain = sel && BLOCKS[sel.type].outputs;
    const outs = canChain ? g.edges.filter((e) => e.from === sel.id) : [];
    const branch = canChain && outs.length > 0 && (ev?.shiftKey || BLOCKS[sel.type].source || sel.type === 'input');
    const x = canChain ? sel.x + BW + GAP : Math.max(40, ...g.nodes.map((n) => n.x + BW + GAP));
    let y = canChain ? sel.y : 80;
    if (branch) {
      const col = g.nodes.filter((n) => Math.abs(n.x - x) < BW);
      y = Math.max(sel.y, ...col.map((n) => n.y)) + 170;
    } else {
      // Make room only if something in this row would overlap the new block; then shift the rest of
      // the row along. Otherwise leave the board alone (keeps a waiting Output within reach).
      const row = g.nodes.filter((n) => Math.abs(n.y - y) < 120 && n.x >= x - 10);
      if (row.some((n) => n.x < x + BW + GAP / 2)) for (const n of row) n.x += BW + GAP;
    }
    const node = addNode(g, type, x, y);
    if (canChain) {
      connect(g, sel.id, node.id);
      if (!branch) for (const e of outs) connect(g, node.id, e.to, e.port);
    }
    selected = node.id;
    refresh();
    renderInspector();
  }

  function remove(id) {
    const n = getNode(g, id);
    if (!n || n.fixed) return;
    // Heal the chain: whatever fed this block now feeds what it fed.
    const from = g.edges.find((e) => e.to === id && (e.port ?? 0) === 0)?.from; // the main (port 0) path
    const tos = g.edges.filter((e) => e.from === id);
    removeNode(g, id);
    if (from != null) for (const e of tos) connect(g, from, e.to, e.port);
    selected = null;
    refresh();
    renderInspector();
  }

  function select(id) {
    if (selected === id) return;
    selected = id;
    for (const el of layer.querySelectorAll('.ws-block')) el.classList.toggle('sel', Number(el.dataset.id) === id);
    renderInspector();
    updateRfBox();
  }

  // Outline, on the Input preview, the pixels that one output value of the selected block sees
  // (the value at the middle of its feature map).
  function updateRfBox() {
    layer.querySelector('.ws-rfbox')?.remove();
    const n = selected != null && getNode(g, selected), r = n && analysis.get(n.id);
    const inp = g.nodes.find((m) => m.type === 'input');
    const cv = inp && layer.querySelector(`.ws-block[data-id="${inp.id}"] .ws-map`);
    if (!r || r.status !== 'ok' || !r.rf || r.rf.global || n.type === 'input' || r.shape.length !== 3 || !cv) return;
    const [H, W] = [inp.params.h, inp.params.w], scale = parseFloat(cv.style.width) / W;
    const span = (len, outLen) => {
      const c = r.rf.start + Math.floor(outLen / 2) * r.rf.jump;
      const a = Math.max(0, c - (r.rf.size - 1) / 2), b = Math.min(len - 1, c + (r.rf.size - 1) / 2);
      return [a * scale, (b - a + 1) * scale];
    };
    const [y0, hh] = span(H, r.shape[1]), [x0, ww] = span(W, r.shape[2]);
    cv.parentElement.append(h('div', { class: 'ws-rfbox', title: `One ${BLOCKS[n.type].title} value sees ${r.rf.size}×${r.rf.size} input pixels`,
      style: `left:${cv.offsetLeft + x0}px; top:${cv.offsetTop + y0}px; width:${ww}px; height:${hh}px` }));
  }

  // Selected block's knobs. Rebuilt only when the selection changes, so a slider isn't
  // replaced under the pointer while it's being dragged; refresh() updates `info` instead.
  function renderInspector() {
    for (const el of [...inspector.children].slice(1)) el.remove();
    info = null;
    const n = selected != null && getNode(g, selected);
    if (!n) { inspector.append(h('div', { class: 'hint' }, 'Click a block to edit it.')); return; }
    inspector.append(h('div', { class: 'ws-insp-title' }, BLOCKS[n.type].title));
    if (BLOCKS[n.type].help) inspector.append(h('p', { class: 'hint' }, BLOCKS[n.type].help));
    if (n.locked) inspector.append(h('div', { class: 'hint' }, 'Set by this level.'));
    else {
      // `only` restricts which knobs a level lets you touch on a pre-wired block.
      for (const [key, label, min, max, step = 1] of (KNOBS[n.type] || []).filter(([k]) => !n.only || n.only.includes(k))) {
        slider(inspector, { label, min, max, step, value: n.params[key], onInput: (v) => { n.params[key] = v; refresh(); } });
      }
      if (!KNOBS[n.type]) inspector.append(h('div', { class: 'hint' }, 'No settings to change.'));
    }
    info = h('dl', { class: 'readout ws-info' });
    infoDetail = h('div');
    inspector.append(info, infoDetail);
    updateInfo();
  }

  function updateInfo() {
    const n = selected != null && getNode(g, selected);
    if (!info || !n) return;
    const r = analysis.get(n.id);
    const rows = [['In', BLOCKS[n.type].inputs && r.inShape ? fmtShape(r.inShape) : '—'], ['Out', r.status === 'ok' ? fmtShape(r.shape) : '—'], ['Parameters', r.params.toLocaleString()]];
    if (r.rf) rows.push(['Receptive field', r.rf.global ? 'whole input' : `${r.rf.size}×${r.rf.size} px`]);
    info.replaceChildren(...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]));
    if (r.status !== 'ok') info.append(h('div', { class: `ws-reason ${r.status}` }, r.reason));
    infoDetail.replaceChildren();
    const seq = g.nodes.find((m) => m.type === 'seq');
    let up = g.edges.find((e) => e.to === n.id)?.from;
    if (getNode(g, up)?.type === 'scale') up = g.edges.find((e) => e.to === up)?.from;
    const upType = getNode(g, up)?.type;
    // A Softmax over scores is an attention map; over a Linear on a routing level, a routing table.
    const isAttn = upType === 'scores' ? true : level.chapter === '6' && upType === 'linear' ? 'route' : false;
    if (outputs.has(n.id)) infoDetail.append(detail(n, outputs.get(n.id), seq ? SENTENCES[seq.params.words] : [], isAttn));
  }

  // Same rule as the level check: once Output is wired, only blocks feeding it count;
  // strays are listed dimmed. Same-type blocks are numbered in board order.
  function renderNetwork() {
    netSec.replaceChildren(netSec.firstChild);
    const out = g.nodes.find((n) => n.type === 'output');
    const live = out && analysis.get(out.id).status === 'ok' ? ancestors(g, out.id) : null;
    const counts = {}, names = new Map();
    for (const n of [...g.nodes].sort((a, b) => a.x - b.x || a.y - b.y)) {
      counts[n.type] = (counts[n.type] || 0) + 1;
      names.set(n.id, `${BLOCKS[n.type].title} ${counts[n.type]}`);
    }
    // Shared-weight cells (RNNCell, LSTMCell) get one row for all their copies.
    const all = g.nodes.filter((n) => analysis.get(n.id).params > 0).sort((a, b) => a.x - b.x);
    // Shared copies are grouped exactly as countParams groups them (type + settings + input shape);
    // a group is "used" if any copy feeds Output.
    const skey = (n) => `${n.type}:${JSON.stringify(n.params)}:${JSON.stringify(analysis.get(n.id).inShape)}`;
    const groups = new Map();
    for (const n of all) if (BLOCKS[n.type].shared) groups.set(skey(n), [...(groups.get(skey(n)) || []), n]);
    const rows = all.filter((n) => !BLOCKS[n.type].shared || groups.get(skey(n))[0] === n);
    for (const n of rows) if (BLOCKS[n.type].shared) names.set(n.id, `${BLOCKS[n.type].title} ×${groups.get(skey(n)).length} (shared)`);
    const usedGroup = (n) => BLOCKS[n.type].shared && live && groups.get(skey(n)).some((m) => live.has(m.id));
    const total = countParams(all.filter((n) => !live || live.has(n.id)), analysis);
    const dl = h('dl', { class: 'readout' });
    for (const n of rows) {
      const stray = live && !live.has(n.id) && !usedGroup(n) ? ' ws-stray' : '';
      dl.append(h('dt', { class: stray }, names.get(n.id) + (stray ? ' (not used)' : '')),
        h('dd', { class: stray }, analysis.get(n.id).params.toLocaleString()));
    }
    dl.append(h('dt', { class: 'ws-total' }, 'Total parameters'), h('dd', { class: 'ws-total' }, total.toLocaleString()));
    netSec.append(dl);
    const problems = g.nodes.filter((n) => analysis.get(n.id).status === 'error');
    for (const n of problems) netSec.append(h('div', { class: 'ws-reason error' }, `${BLOCKS[n.type].title}: ${analysis.get(n.id).reason}`));
  }

  // ---------- rendering ----------
  // The chosen digit fitted (not cropped) into the Input block's shape, repeated across its channels.
  function inputTensor(p) {
    const side = Math.min(p.h, p.w);
    const base = digit(view.digit, { size: side, box: Math.max(3, Math.round((side * 20) / 28)), brush: Math.max(0.6, side / 22) });
    const t = tensor([p.c, p.h, p.w]);
    const oy = (p.h - side) >> 1, ox = (p.w - side) >> 1;
    for (let c = 0; c < p.c; c++) for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
      t.data[(c * p.h + y + oy) * p.w + x + ox] = base.data[y * side + x];
    }
    return t;
  }

  // Image boards feed the chosen digit into their Input block; sequence boards need no input
  // tensor (Word and Zero-state blocks make their own values).
  function runForward() {
    const inp = g.nodes.find((n) => n.type === 'input');
    const hasSource = g.nodes.some((n) => BLOCKS[n.type].source && analysis.get(n.id)?.status === 'ok');
    if (inp && analysis.get(inp.id).status === 'ok') {
      outputs = runner.run(g, analysis, inputTensor(inp.params), view.seed, `${view.digit}:${JSON.stringify(inp.params)}`, view.gain);
    } else if (hasSource) {
      outputs = runner.run(g, analysis, null, view.seed, 'seq', view.gain);
    } else outputs = new Map();
  }

  // Shapes update at once; values follow the budget rules above.
  function scheduleForward() {
    clearTimeout(fwdTimer);
    const { macs, weights } = estimateCost(g, analysis);
    if (macs > MAX_MACS || weights > MAX_WEIGHTS) {
      outputs = new Map();
      budgetNote.textContent = `Values paused: this network needs ≈${(macs / 1e6).toFixed(0)}M multiply-adds and ` +
        `${(weights / 1e6).toFixed(1)}M weights per pass (limits ${MAX_MACS / 1e6}M / ${MAX_WEIGHTS / 1e6}M). Shrink it to see values again.`;
      return;
    }
    budgetNote.textContent = '';
    if (macs <= SYNC_MACS) { runForward(); outputsStale = false; return; }
    // Previews keep showing the last values meanwhile, but levels don't judge them (outputsStale).
    outputsStale = true;
    fwdTimer = setTimeout(() => { runForward(); outputsStale = false; renderBlocks(); renderWires(); updateInfo(); updateRfBox(); renderGoal(); }, 150);
  }

  function refresh() {
    analysis = analyze(g);
    scheduleForward();
    queueMicrotask(updateRfBox);
    // Grow the board so long chains stay reachable.
    board.style.width = `${Math.max(2600, ...g.nodes.map((n) => n.x + BW + 600))}px`;
    board.style.height = `${Math.max(1400, ...g.nodes.map((n) => n.y + 600))}px`;
    renderBlocks();
    renderWires();
    updateInfo();
    renderNetwork();
    renderGoal();
  }

  // Level status in the top bar and panel; a new best star count is saved.
  function renderGoal() {
    if (level.free) {
      status.textContent = 'Free build';
      goal.replaceChildren();
      return;
    }
    const metrics = {};
    if (level.memoryMin) metrics.memory = firstWordMemory(g, analysis);
    if (level.peakMax) metrics.peak = attentionPeak(g, analysis);
    if (level.orderMin) metrics.order = orderSensitivity(g, analysis);
    if (level.noDrops) metrics.dropped = moeStats(g, analysis)?.dropped;
    const r = checkLevel(g, analysis, level, view.show && !outputsStale ? outputs : null, metrics);
    goal.className = 'ws-goal' + (r.done ? ' done' : '');
    if (r.done) {
      const best = progress[level.id] || 0;
      goal.innerHTML = `<b>✓ Complete ${stars(r.stars)}</b> ${r.params.toLocaleString()} parameters` +
        (metrics.memory != null ? `<br>Memory of word 1: <b>${(metrics.memory * 100).toFixed(1)}%</b>` : '') +
        (metrics.peak != null ? `<br>Average top attention weight: <b>${(metrics.peak * 100).toFixed(0)}%</b>` : '') +
        (metrics.order != null ? `<br>Word-order sensitivity: <b>${(metrics.order * 100).toFixed(1)}%</b>` : '') +
        (r.active != null && r.active < r.params ? `<br>Active per word: <b>${r.active.toLocaleString()}</b> of ${r.params.toLocaleString()} (${Math.round((100 * r.active) / r.params)}%)` : '') +
        (r.stars < 3 ? `<br><span class="hint">${level.starsFrom === 'capacity' ? `A capacity of ${level.budgets[r.stars]}× or less` : `Under ${level.budgets[r.stars].toLocaleString()} parameters`} earns ${stars(r.stars + 1)}.</span>` : '');
      status.innerHTML = `<b class="ws-win">✓ ${level.title} complete ${stars(r.stars)}</b> · ${r.params.toLocaleString()} params`;
      if (r.stars > best) {
        progress[level.id] = r.stars;
        saveProgress(progress);
        refreshLevelOptions();
      }
      const next = playable[playable.indexOf(level) + 1];
      if (next) goal.append(h('div', {}, h('button', { class: 'btn primary ws-next', onclick: () => { level = next; reset(); } }, `Next: ${next.title} →`)));
    } else {
      goal.textContent = r.msg;
      status.innerHTML = `<b>${level.title}</b> · <span class="bn-dim">${r.msg}</span>`;
    }
  }

  function renderBlocks() {
    layer.replaceChildren();
    for (const n of g.nodes) {
      const B = BLOCKS[n.type], r = analysis.get(n.id);
      const el = h('div', {
        class: `ws-block ws-${n.type} ${r.status}` + (n.id === selected ? ' sel' : ''),
        'data-id': n.id, style: `left:${n.x}px; top:${n.y}px; width:${BW}px`,
      },
        h('div', { class: 'ws-head' }, h('b', {}, B.title, B.shared ? h('span', { class: 'ws-shared', title: 'Every block of this type uses the same weights' }, 'shared') : null),
          n.fixed ? null : h('button', { class: 'ws-x', title: 'Remove', onclick: (e) => { e.stopPropagation(); remove(n.id); } }, '×')),
        h('div', { class: 'ws-sum' }, B.summary(n.params)),
        h('div', { class: 'ws-shape' }, r.status === 'ok' ? `→ ${fmtShape(r.shape)}` : r.reason));
      if (view.show && r.status === 'ok' && outputs.has(n.id)) {
        el.append(preview(n, outputs.get(n.id)));
        if (n.type !== 'output') el.append(h('div', { class: 'ws-rms', title: 'Signal strength: root-mean-square of this block\'s values' }, `signal ${rms(outputs.get(n.id)).toFixed(3)}`));
      }
      for (let p = 0; p < B.inputs; p++) {
        el.append(h('div', { class: 'ws-port in', 'data-id': n.id, 'data-port': p, style: `top:${PORT_Y - 7 + p * PORT_GAP}px`,
          title: B.inputs > 1 ? `Input ${p === 0 ? 'a' : 'b'}` : 'Input' }));
      }
      if (B.inputs > 1) el.style.minHeight = `${PORT_Y + (B.inputs - 1) * PORT_GAP + 24}px`;
      if (B.outputs) {
        const port = h('div', { class: 'ws-port out', 'data-id': n.id, title: 'Drag to another block to wire' });
        port.addEventListener('pointerdown', (e) => startWire(e, n, port));
        el.append(port);
      }
      el.addEventListener('pointerdown', (e) => startMove(e, n, el));
      layer.append(el);
    }
  }

  const outPt = (n) => [n.x + BW, n.y + PORT_Y];
  const inPt = (n, port = 0) => [n.x, n.y + PORT_Y + port * PORT_GAP];
  // Bezier control points for a wire. Forward wires ease out by half the gap; wires that jump over
  // blocks (skip connections) arc above them; backward ones get a short hook so they stay on the board.
  function controls([x1, y1], [x2, y2]) {
    if (x2 - x1 > BW + GAP + 40) {
      const top = Math.max(8, Math.min(y1, y2) - 110);
      return [[x1 + 60, top], [x2 - 60, top]];
    }
    const dx = x2 >= x1 ? Math.max(40, (x2 - x1) / 2) : 40;
    return [[x1 + dx, y1], [x2 - dx, y2]];
  }
  function curve(a, b) {
    const [c1, c2] = controls(a, b);
    return `M${a[0]},${a[1]} C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${b[0]},${b[1]}`;
  }
  // Point halfway along the wire (t = 0.5), where its shape label goes.
  function midpoint(a, b) {
    const [c1, c2] = controls(a, b);
    return [0, 1].map((i) => 0.125 * a[i] + 0.375 * c1[i] + 0.375 * c2[i] + 0.125 * b[i]);
  }

  const svgEl = (tag, attrs) => {
    const el = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  };

  // Each wire carries the shape leaving its source; it turns red when the block it feeds rejects it.
  function renderWires() {
    wires.replaceChildren();
    for (const lbl of layer.querySelectorAll('.ws-label')) lbl.remove();
    for (const e of g.edges) {
      const a = getNode(g, e.from), b = getNode(g, e.to);
      const d = curve(outPt(a), inPt(b, e.port ?? 0));
      const bad = analysis.get(b.id)?.status === 'error';
      const shape = analysis.get(a.id)?.shape;
      const grp = svgEl('g', { class: 'ws-wire-g' + (bad ? ' bad' : '') });
      grp.append(svgEl('path', { d, class: 'ws-wire' }), svgEl('path', { d, class: 'ws-hit' }));
      const t = document.createElementNS(SVG, 'title');
      t.textContent = bad ? analysis.get(b.id).reason : 'Click to cut this wire';
      grp.append(t);
      grp.addEventListener('click', () => { disconnect(g, e.id); refresh(); });
      wires.append(grp);
      if (shape) {
        const [mx, my] = midpoint(outPt(a), inPt(b, e.port ?? 0));
        layer.prepend(h('div', { class: 'ws-label' + (bad ? ' bad' : ''), style: `left:${mx}px; top:${my}px` }, fmtShape(shape)));
      }
    }
  }

  // ---------- interaction ----------
  function startMove(e, n, el) {
    if (e.button !== 0 || e.target.closest('.ws-port, .ws-x')) return;
    select(n.id);
    const sx = e.clientX, sy = e.clientY, ox = n.x, oy = n.y;
    el.setPointerCapture(e.pointerId);
    el.classList.add('drag');
    const move = (ev) => {
      n.x = Math.max(0, ox + ev.clientX - sx);
      n.y = Math.max(0, oy + ev.clientY - sy);
      el.style.left = `${n.x}px`;
      el.style.top = `${n.y}px`;
      renderWires();
    };
    const up = () => {
      el.classList.remove('drag');
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  }

  // Drag from an output port; drop on another block (or its input port) to wire it.
  // Move/up listen on window: captured events still bubble there, and it keeps working if capture
  // isn't available (some pen/touch stacks, synthetic events).
  function startWire(e, n, port) {
    if (e.button !== 0) return;
    e.stopPropagation();
    try { port.setPointerCapture(e.pointerId); } catch { /* fine without capture */ }
    const temp = svgEl('path', { class: 'ws-wire temp' });
    wires.append(temp);
    let hot = null;
    // Target = the input port under the pointer, or — dropping on a block's body — its first empty
    // input port (port 0 if all are taken, which replaces that wire).
    const targetAt = (ev) => {
      const hit = document.elementFromPoint(ev.clientX, ev.clientY);
      const el = hit?.closest('.ws-block');
      const id = el && Number(el.dataset.id);
      const B = id && BLOCKS[getNode(g, id).type];
      if (!B || id === n.id || !B.inputs) return null;
      let port = hit.classList.contains('in') ? Number(hit.dataset.port) : -1;
      if (port < 0) {
        const taken = new Set(g.edges.filter((e) => e.to === id).map((e) => e.port ?? 0));
        port = Math.max(0, [...Array(B.inputs).keys()].find((p) => !taken.has(p)) ?? 0);
      }
      return { el, id, port, portEl: el.querySelector(`.ws-port.in[data-port="${port}"]`) };
    };
    const move = (ev) => {
      const r = board.getBoundingClientRect();
      temp.setAttribute('d', curve(outPt(n), [ev.clientX - r.left, ev.clientY - r.top]));
      const t = targetAt(ev);
      if (t?.portEl !== hot) { hot?.classList.remove('hot'); hot = t?.portEl || null; hot?.classList.add('hot'); }
    };
    const up = (ev) => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      endWire = null;
      temp.remove();
      const t = ev.type === 'pointerup' && targetAt(ev);
      hot?.classList.remove('hot');
      if (t) {
        const res = connect(g, n.id, t.id, t.port);
        if (!res.ok) flash(res.reason);
        refresh();
      }
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
    endWire = () => up({ type: 'pointercancel' });
  }
  let endWire = null; // cancels an in-progress wire drag (used on unmount)

  const onKey = (e) => {
    if (e.target.closest?.('input, select, textarea')) return;
    if ((e.key === 'Delete' || e.key === 'Backspace') && selected != null) { e.preventDefault(); remove(selected); }
    if (e.key === 'Escape') select(null);
  };
  window.addEventListener('keydown', onKey);
  board.addEventListener('pointerdown', (e) => { if (e.target === board || e.target === layer) select(null); });

  reset();
  return () => {
    endWire?.();
    clearTimeout(msgTimer);
    clearTimeout(fwdTimer);
    window.removeEventListener('keydown', onKey);
    root.remove();
  };
}
