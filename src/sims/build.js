import { createLayout, section, slider, select, buttons, h } from '../ui.js';
import { outSize, effK, convParams, conv2d } from './build/conv.js';
import { convFigure, pipeline, fmt } from './build/figure.js';
import { CHAPTERS, LEVELS } from './build/levels.js';
import { mountWorkshop } from './build/workshop.js';
import { WS_LEVELS } from './build/chapter2.js';
import { mountReward } from './build/reward.js';
import { RL_LEVELS } from './build/rllevels.js';

const STORE = 'fun-sim.build.v1';
function loadProgress() { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; } }
function saveProgress(p) { try { localStorage.setItem(STORE, JSON.stringify(p)); } catch { /* private mode */ } }

const stars = (n) => '★'.repeat(n) + '☆'.repeat(3 - n);
const html = (s, cls) => { const d = h('div', cls ? { class: cls } : {}); d.innerHTML = s; return d; };

function makeInput(n, pattern) {
  return Array.from({ length: n }, (_, y) => Array.from({ length: n }, (_, x) => {
    if (pattern === 'random') return Math.floor(Math.random() * 4);
    if (pattern === 'diag') return Math.abs(x - y) <= 0 ? 1 : 0;
    const lo = Math.round(n / 4), hi = n - 1 - lo;
    return y >= lo && y <= hi && x >= lo && x <= hi ? 1 : 0;
  }));
}

function makeKernel(k, preset) {
  const c = Math.floor(k / 2);
  return Array.from({ length: k }, (_, a) => Array.from({ length: k }, (_, b) => {
    switch (preset) {
      case 'identity': return a === c && b === c ? 1 : 0;
      case 'box': return 1;
      case 'vedge': return k === 1 ? 1 : b === 0 ? 1 : b === k - 1 ? -1 : 0;
      case 'hedge': return k === 1 ? 1 : a === 0 ? 1 : a === k - 1 ? -1 : 0;
      case 'sharpen': return a === c && b === c ? k * k : -1;
      default: return Math.floor(Math.random() * 3) - 1;
    }
  }));
}

export default {
  id: 'build',
  title: 'Build a Net',
  glyph: '⧉',
  tag: 'game',
  blurb: 'Learn neural-network building blocks by playing with them — starting with convolution: kernels, stride, padding and output shapes.',

  mount(root) {
    const { stage, panel } = createLayout(root, {
      title: 'Build a Net',
      desc: 'Neural networks are assembled from a few parts. Master each one, then wire them together. Chapter 1: convolution.',
    });
    const view = h('div', { class: 'bn-scroll' });
    stage.append(view);
    const tabs = h('div', { class: 'bn-tabs' });
    const side = h('div');
    panel.append(tabs, side);

    let timer = null;
    const stopTimer = () => { clearInterval(timer); timer = null; };
    const progress = loadProgress();

    const tabEls = {};
    for (const [m, label] of [['sandbox', 'Sandbox'], ['levels', 'Challenges'], ['workshop', 'Workshop'], ['reward', 'Reward']]) {
      tabEls[m] = h('button', { class: 'bn-tab', onclick: () => setMode(m) }, label);
      tabs.append(tabEls[m]);
    }
    let modeCleanup = null;
    function setMode(m) {
      stopTimer();
      modeCleanup?.();
      modeCleanup = null;
      for (const [k, el] of Object.entries(tabEls)) el.classList.toggle('active', k === m);
      side.replaceChildren();
      view.replaceChildren();
      view.scrollTop = 0;
      view.hidden = m === 'workshop' || m === 'reward';
      if (m === 'sandbox') mountSandbox();
      else if (m === 'levels') mountLevels();
      else if (m === 'workshop') modeCleanup = mountWorkshop({ stage, side, progress, saveProgress });
      else modeCleanup = mountReward({ stage, side, progress, saveProgress });
    }

    // ---------------- Sandbox ----------------
    function mountSandbox() {
      const s = { n: 7, pattern: 'square', K: 3, kpreset: 'vedge', S: 1, P: 0, D: 1, cin: 3, cout: 16, cur: 0, reveal: 1, speed: 4 };
      let input = makeInput(s.n, s.pattern), kernel = makeKernel(s.K, s.kpreset);
      const restart = () => { s.cur = 0; s.reveal = 1; };
      const total = () => Math.max(0, outSize(s.n, s.K, s.S, s.P, s.D)) ** 2;

      let playBtn;
      const setPlaying = (on) => {
        stopTimer();
        if (on) {
          if (s.reveal >= total()) restart();
          timer = setInterval(step, 1000 / s.speed);
        }
        playBtn.textContent = on ? 'Pause' : 'Play';
      };
      function step() {
        if (s.reveal < total()) { s.cur = s.reveal; s.reveal++; }
        else setPlaying(false);
        draw();
      }
      const changed = () => { restart(); draw(); };

      const sIn = section(side, 'Input');
      slider(sIn, { label: 'Size (H = W)', min: 3, max: 12, value: s.n, onInput: (v) => { s.n = v; input = makeInput(v, s.pattern); changed(); } });
      select(sIn, {
        label: 'Pattern', value: s.pattern,
        options: [['square', 'Bright square'], ['diag', 'Diagonal line'], ['random', 'Random 0–3']],
        onChange: (v) => { s.pattern = v; input = makeInput(s.n, v); changed(); },
      });
      slider(sIn, { label: 'Input channels', min: 1, max: 64, value: s.cin, onInput: (v) => { s.cin = v; draw(); } });

      const sK = section(side, 'Kernel');
      const kSel = select(sK, {
        label: 'Weights', value: s.kpreset,
        options: [['vedge', 'Vertical edge'], ['hedge', 'Horizontal edge'], ['box', 'Box blur'], ['identity', 'Identity'], ['sharpen', 'Sharpen'], ['random', 'Random'], ['custom', 'Custom (click cells)']],
        onChange: (v) => { s.kpreset = v; if (v !== 'custom') kernel = makeKernel(s.K, v); changed(); },
      });
      slider(sK, { label: 'Kernel size k', min: 1, max: 5, value: s.K, onInput: (v) => { s.K = v; kernel = makeKernel(v, s.kpreset === 'custom' ? 'identity' : s.kpreset); changed(); } });
      sK.append(h('div', { class: 'hint' }, 'Click a kernel cell to cycle its weight (−1, 0, 1, 2).'));

      const sL = section(side, 'Layer');
      slider(sL, { label: 'Stride s', min: 1, max: 3, value: s.S, onInput: (v) => { s.S = v; changed(); } });
      slider(sL, { label: 'Padding p', min: 0, max: 3, value: s.P, onInput: (v) => { s.P = v; changed(); } });
      slider(sL, { label: 'Dilation d', min: 1, max: 3, value: s.D, onInput: (v) => { s.D = v; changed(); } });
      slider(sL, { label: 'Filters (output channels)', min: 1, max: 64, value: s.cout, onInput: (v) => { s.cout = v; draw(); } });

      const sA = section(side, 'Animate');
      [playBtn] = buttons(sA, [
        { label: 'Play', primary: true, onClick: () => setPlaying(!timer) },
        { label: 'Step', onClick: () => { setPlaying(false); step(); } },
        { label: 'Show all', onClick: () => { setPlaying(false); s.reveal = total(); s.cur = total() - 1; draw(); } },
      ]);
      slider(sA, { label: 'Speed', min: 1, max: 12, value: s.speed, format: (v) => `${v} /s`, onInput: (v) => { s.speed = v; if (timer) setPlaying(true); } });
      sA.append(h('div', { class: 'hint' }, 'Hover any output cell to see where it came from.'));

      function draw() {
        const o = outSize(s.n, s.K, s.S, s.P, s.D), fits = o > 0;
        const params = convParams(s.K, s.cin, s.cout);
        const layer = `Conv2d(${s.cin}→${s.cout}, k=${s.K}, s=${s.S}, p=${s.P}${s.D > 1 ? `, d=${s.D}` : ''})`;
        const pipe = pipeline([
          { text: `[${s.cin}, ${s.n}, ${s.n}]`, kind: 'shape' },
          { text: layer, kind: 'layer' },
          fits ? { text: `[${s.cout}, ${o}, ${o}]`, kind: 'shape' } : { text: 'invalid', kind: 'q' },
        ]);
        const ke = effK(s.K, s.D);
        const formula = html(
          `<div>out = ⌊(n + 2p − ${s.D > 1 ? 'd(k − 1) − 1' : 'k'}) / s⌋ + 1</div>` +
          `<div>&nbsp;&nbsp;&nbsp;&nbsp;= ⌊(${s.n} + ${2 * s.P} − ${ke}) / ${s.S}⌋ + 1 = <b>${fits ? o : '—'}</b></div>` +
          `<div class="bn-dim">params = k·k·C<sub>in</sub>·C<sub>out</sub> + C<sub>out</sub> = ${s.K}·${s.K}·${s.cin}·${s.cout} + ${s.cout} = <b>${params.toLocaleString()}</b></div>`,
          'bn-formula');

        const fig = convFigure({ input, kernel, S: s.S, P: s.P, D: s.D }, {
          cur: s.cur, reveal: s.reveal,
          onHoverOut: (idx) => { setPlaying(false); s.cur = idx; s.reveal = Math.max(s.reveal, idx + 1); draw(); },
          onKernelClick: (a, b) => {
            const cycle = [-1, 0, 1, 2];
            kernel[a][b] = cycle[(cycle.indexOf(kernel[a][b]) + 1) % cycle.length] ?? 0;
            s.kpreset = 'custom'; kSel.value = 'custom';
            draw();
          },
        });

        let calc = null;
        if (fits) {
          const r = conv2d(input, kernel, s.S, s.P, s.D);
          const i = Math.floor(s.cur / r.ow), j = s.cur % r.ow;
          const terms = r.taps[s.cur].map((t) => `${t.w < 0 ? `(${fmt(t.w)})` : t.w}·${t.v}`);
          calc = html(`<span class="bn-dim">out[${i}][${j}] =</span> ${terms.join(' + ')} = <b>${fmt(r.out[i][j])}</b>`, 'bn-calc');
        }
        const note = h('div', { class: 'hint bn-note' },
          `Shown: one input channel and one filter. The real layer does this for all ${s.cin} input channel${s.cin > 1 ? 's' : ''} ` +
          `(summing them into one value) and repeats it for each of its ${s.cout} filter${s.cout > 1 ? 's' : ''}.`);
        view.replaceChildren(pipe, formula, fig, calc || '', note);
      }
      draw();
    }

    // ---------------- Challenges ----------------
    function mountLevels() {
      const head = h('div', { class: 'bn-total' });
      const list = h('div');
      side.append(head, list);
      let active = null;
      const unlocked = (i) => i === 0 || (progress[LEVELS[i - 1].id] || 0) >= 1;

      function renderSide() {
        const got = LEVELS.reduce((a, L) => a + (progress[L.id] || 0), 0);
        head.innerHTML = `<span>Stars</span><b>★ ${got} / ${LEVELS.length * 3}</b>`;
        list.replaceChildren();
        for (const ch of CHAPTERS) {
          list.append(h('div', { class: 'section-title bn-ch' }, ch.title));
          if (ch.tab === 'reward') {
            const got = RL_LEVELS.reduce((a, L) => a + (progress[L.id] || 0), 0);
            list.append(h('div', { class: 'bn-level bn-to-ws', onclick: () => setMode('reward') },
              h('span', {}, `${ch.teaser} — open Reward →`), h('span', { class: 'bn-stars' }, `★ ${got}/${RL_LEVELS.length * 3}`)));
            continue;
          }
          if (ch.workshop) {
            // Played in the Workshop tab; show its stars here so the roadmap reads as one game.
            const ws = WS_LEVELS.filter((L) => L.chapter === ch.workshop);
            const got = ws.reduce((a, L) => a + (progress[L.id] || 0), 0);
            list.append(h('div', { class: 'bn-level bn-to-ws', onclick: () => setMode('workshop') },
              h('span', {}, `${ch.teaser} — open Workshop →`), h('span', { class: 'bn-stars' }, `★ ${got}/${ws.length * 3}`)));
            continue;
          }
          if (!ch.ready) {
            list.append(h('div', { class: 'bn-level locked' }, h('span', {}, ch.teaser), h('span', { class: 'bn-soon' }, 'soon')));
            continue;
          }
          LEVELS.forEach((L, i) => {
            if (L.chapter !== ch.id) return;
            const open = unlocked(i);
            const row = h('div', { class: 'bn-level' + (open ? '' : ' locked') + (active === L ? ' active' : '') },
              h('span', {}, `${i + 1}. ${L.title}`),
              h('span', { class: 'bn-stars' }, open ? stars(progress[L.id] || 0) : '🔒'));
            if (open) row.addEventListener('click', () => start(L));
            list.append(row);
          });
        }
      }

      function start(L) {
        active = L;
        renderSide();
        const N = 5, results = [];
        let q, vals, answered;
        const levelNo = LEVELS.indexOf(L) + 1;
        next();

        function next() {
          q = L.gen();
          vals = {};
          for (const f of q.fields) vals[f.key] = f.slider ? f.value : '';
          answered = false;
          render();
        }

        function render() {
          const card = h('div', { class: 'bn-card' });
          card.append(h('div', { class: 'bn-kicker' }, `Level ${levelNo} · ${L.title}`));
          const intro = h('details', { class: 'bn-intro' }, h('summary', {}, 'The idea'), html(L.intro));
          intro.open = results.length === 0 && !answered;
          const dots = h('div', { class: 'bn-dots' });
          for (let i = 0; i < N; i++) {
            const cls = i < results.length ? (results[i] ? 'ok' : 'bad') : i === results.length ? 'cur' : '';
            dots.append(h('span', { class: 'bn-dot ' + cls }));
          }
          card.append(intro, dots, html(q.prompt, 'bn-prompt'));

          const figBox = h('div', { class: 'bn-figbox' });
          const drawFig = () => { figBox.replaceChildren(); const f = q.figure?.(vals, answered); if (f) figBox.append(f); };
          drawFig();
          card.append(figBox);

          const form = h('div', { class: 'bn-answer' });
          let firstInput = null;
          for (const f of q.fields) {
            if (f.slider) {
              const sl = slider(form, { label: f.label, min: f.min, max: f.max, value: vals[f.key], onInput: (v) => { vals[f.key] = v; drawFig(); } });
              sl.input.disabled = answered;
            } else {
              const inp = h('input', { type: 'number', inputmode: 'numeric' });
              inp.value = vals[f.key];
              inp.disabled = answered;
              inp.addEventListener('input', () => { vals[f.key] = inp.value; });
              inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit(); });
              form.append(h('label', { class: 'bn-field' }, h('span', {}, f.label), inp));
              firstInput ||= inp;
            }
          }
          card.append(form);

          const fb = h('div', { class: 'bn-feedback' });
          const row = h('div', { class: 'btns' });
          let focusEl = firstInput;
          if (!answered) {
            [focusEl] = buttons(row, [{ label: 'Check', primary: true, onClick: submit }]);
            if (firstInput) focusEl = firstInput;
          } else {
            const ok = results.at(-1);
            fb.classList.add(ok ? 'ok' : 'bad');
            const correct = q.answer && !ok
              ? 'Answer: ' + q.fields.map((f) => `<b>${fmt(q.answer[f.key])}</b>`).join(', ') + '. '
              : '';
            fb.innerHTML = `<div class="bn-verdict">${ok ? '✓ Correct' : '✗ Not quite'}</div>${correct}` +
              (typeof q.explain === 'function' ? q.explain(vals) : q.explain);
            const last = results.length >= N;
            [focusEl] = buttons(row, [{ label: last ? 'See results' : 'Next →', primary: true, onClick: last ? finish : next }]);
          }
          card.append(fb, row);
          view.replaceChildren(card);
          focusEl?.focus({ preventScroll: true });
        }

        function submit() {
          if (answered) return;
          if (q.fields.some((f) => !f.slider && String(vals[f.key]).trim() === '')) return;
          const ok = q.check ? q.check(vals) : q.fields.every((f) => Number(vals[f.key]) === q.answer[f.key]);
          results.push(ok);
          answered = true;
          render();
        }

        function finish() {
          const score = results.filter(Boolean).length;
          const st = score === 5 ? 3 : score === 4 ? 2 : score === 3 ? 1 : 0;
          const best = Math.max(st, progress[L.id] || 0);
          if (best !== (progress[L.id] || 0)) { progress[L.id] = best; saveProgress(progress); }
          renderSide();
          const i = LEVELS.indexOf(L), nextL = LEVELS[i + 1];
          const card = h('div', { class: 'bn-card bn-result' },
            h('div', { class: 'bn-kicker' }, `Level ${levelNo} · ${L.title}`),
            h('div', { class: 'bn-bigstars' }, stars(st)),
            h('div', { class: 'bn-score' }, `${score} / ${N} correct`),
            h('p', { class: 'hint' }, st === 3 ? 'Perfect.' : st ? 'Level passed — try again for 3 stars.' : 'Get 3 of 5 right to unlock the next level.'));
          if (!nextL && st) {
            card.append(html('<p>Chapter 1 complete! Next chapters will stack these into CNNs, then build up to LSTMs, attention, Transformer blocks and Mixture of Experts.</p>', 'bn-prompt'));
          }
          const defs = [{ label: 'Retry', onClick: () => start(L) }];
          if (nextL && unlocked(i + 1)) defs.unshift({ label: `Next: ${nextL.title} →`, primary: true, onClick: () => start(nextL) });
          const [first] = buttons(card, defs);
          view.replaceChildren(card);
          first.focus({ preventScroll: true });
        }
      }

      const firstOpen = LEVELS.find((L, i) => unlocked(i) && !(progress[L.id] >= 1)) || LEVELS[0];
      start(firstOpen);
    }

    setMode('sandbox');
    return () => { stopTimer(); modeCleanup?.(); };
  },
};
