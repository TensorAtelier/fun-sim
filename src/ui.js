// Shared layout + control helpers. Every simulation uses these so the panels look alike.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
}

// Builds the standard stage + side-panel layout inside `root`.
export function createLayout(root, { title, desc }) {
  const stage = h('div', { class: 'stage' });
  const panel = h('aside', { class: 'panel' }, h('h2', {}, title), h('p', { class: 'desc' }, desc));
  root.append(h('div', { class: 'sim' }, stage, panel));
  return { stage, panel };
}

export function section(parent, title) {
  const s = h('div', { class: 'section' }, h('div', { class: 'section-title' }, title));
  parent.append(s);
  return s;
}

export function slider(parent, { label, min, max, step = 1, value, format = (v) => v, onInput }) {
  const val = h('span', { class: 'ctl-val' }, format(value));
  const input = h('input', { type: 'range', min, max, step, value });
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    val.textContent = format(v);
    onInput?.(v);
  });
  parent.append(h('div', { class: 'ctl' }, h('div', { class: 'ctl-row' }, h('span', {}, label), val), input));
  return {
    input,
    get value() { return parseFloat(input.value); },
    set(v) { input.value = v; val.textContent = format(parseFloat(input.value)); },
  };
}

export function select(parent, { label, options, value, onChange }) {
  const sel = h('select', {}, options.map((o) => {
    const [v, text] = Array.isArray(o) ? o : [o, o];
    const opt = h('option', { value: v }, text);
    if (v === value) opt.selected = true;
    return opt;
  }));
  sel.addEventListener('change', () => onChange?.(sel.value));
  parent.append(h('div', { class: 'ctl' }, label ? h('div', { class: 'ctl-row' }, h('span', {}, label)) : null, sel));
  return sel;
}

export function checkbox(parent, { label, checked = false, onChange }) {
  const input = h('input', { type: 'checkbox' });
  input.checked = checked;
  input.addEventListener('change', () => onChange?.(input.checked));
  parent.append(h('label', { class: 'check' }, input, label));
  return input;
}

// buttons(parent, [{label, onClick, primary}]) -> array of <button>
export function buttons(parent, defs) {
  const row = h('div', { class: 'btns' });
  const els = defs.map((d) => {
    const b = h('button', { class: 'btn' + (d.primary ? ' primary' : '') }, d.label);
    b.addEventListener('click', d.onClick);
    row.append(b);
    return b;
  });
  parent.append(row);
  return els;
}

// readout(parent, ['Altitude', 'Speed']) -> { set(key, text) }
export function readout(parent, keys) {
  const dl = h('dl', { class: 'readout' });
  const dds = {};
  for (const k of keys) {
    dds[k] = h('dd', {}, '—');
    dl.append(h('dt', {}, k), dds[k]);
  }
  parent.append(dl);
  return { set: (k, text) => { dds[k].textContent = text; } };
}

// A canvas that fills its container at devicePixelRatio. `onResize(w, h)` gets CSS pixels;
// ctx is pre-scaled so you draw in CSS pixels.
export function createCanvas(container, onResize) {
  const canvas = h('canvas');
  container.append(canvas);
  const ctx = canvas.getContext('2d');
  const size = { w: 0, h: 0, dpr: 1 };
  const resize = () => {
    const r = container.getBoundingClientRect();
    if (r.width === size.w && r.height === size.h && canvas.width > 1) return;
    size.dpr = window.devicePixelRatio || 1;
    size.w = r.width; size.h = r.height;
    canvas.width = Math.max(1, Math.round(r.width * size.dpr));
    canvas.height = Math.max(1, Math.round(r.height * size.dpr));
    ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
    onResize?.(size.w, size.h);
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  // Size once right after mount finishes, without waiting for the observer
  // (which never fires in background tabs).
  queueMicrotask(resize);
  return { canvas, ctx, size, destroy: () => ro.disconnect() };
}

// requestAnimationFrame loop with dt in seconds (clamped). Returns stop().
export function loop(fn) {
  let id, last = performance.now(), stopped = false;
  const tick = (now) => {
    if (stopped) return;
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    fn(dt, now / 1000);
    id = requestAnimationFrame(tick);
  };
  id = requestAnimationFrame(tick);
  return () => { stopped = true; cancelAnimationFrame(id); };
}
