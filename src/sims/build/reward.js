// The Reward tab (chapter 7): watch a policy learn from reward, then get judged on fixed seeds.
import { h, section, slider, select as selectCtl, checkbox, buttons, createCanvas, loop } from '../../ui.js';
import { RL_LEVELS, evaluate, START_QUALITY, START_REWARD } from './rllevels.js';
import { BANDIT_MEANS, GRID, MOVES, CARTPOLE, STYLES, rewardModel } from './rl.js';

const C = { bg: '#0b0e14', panel: '#151b27', line: '#242c3b', text: '#e6e9ef', muted: '#8a93a6', amber: '#f5b544', cyan: '#5ec8e5', good: '#7bd88f', danger: '#ef6b6b' };
const MONO = '11px "IBM Plex Mono", monospace';
const stars = (n) => '★'.repeat(n) + '☆'.repeat(3 - n);

// A line chart of one or more series in the box (x, y, w, h). series: [{ values, color, label }]
function chart(ctx, x, y, w, hgt, series, { min, max, title } = {}) {
  ctx.fillStyle = C.panel; ctx.fillRect(x, y, w, hgt);
  ctx.strokeStyle = C.line; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, hgt - 1);
  const all = series.flatMap((s) => s.values);
  const lo = min ?? Math.min(0, ...all), hi = max ?? Math.max(1e-9, ...all);
  const n = Math.max(2, ...series.map((s) => s.values.length));
  ctx.save();
  ctx.beginPath(); ctx.rect(x + 1, y + 1, w - 2, hgt - 2); ctx.clip(); // keep noisy lines inside the box
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
    s.values.forEach((v, i) => {
      const px = x + 8 + ((w - 16) * i) / (n - 1), py = y + hgt - 8 - ((hgt - 26) * (v - lo)) / (hi - lo || 1);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
  }
  ctx.restore();
  ctx.font = MONO; ctx.textBaseline = 'top';
  let lx = x + 8;
  if (title) { ctx.fillStyle = C.muted; ctx.fillText(title, lx, y + 5); lx += ctx.measureText(title).width + 14; }
  for (const s of series) { ctx.fillStyle = s.color; ctx.fillText(s.label, lx, y + 5); lx += ctx.measureText(s.label).width + 12; }
}

const rolling = (xs, k) => xs.map((_, i) => { const a = xs.slice(Math.max(0, i - k + 1), i + 1); return a.reduce((s, v) => s + v, 0) / a.length; });

// ---------- per-kind drawing ----------
function drawBandit(ctx, w, hgt, L) {
  const p = L.probs, K = p.length, bw = Math.min(90, (w * 0.55) / K - 14), top = 50, bh = hgt * 0.5;
  ctx.font = MONO; ctx.textBaseline = 'top';
  ctx.fillStyle = C.muted; ctx.fillText('Probability of pulling each arm (policy)', 24, 20);
  for (let i = 0; i < K; i++) {
    const x = 24 + i * (bw + 14);
    ctx.fillStyle = C.panel; ctx.fillRect(x, top, bw, bh);
    ctx.fillStyle = i === 2 ? C.amber : C.cyan; ctx.fillRect(x, top + bh * (1 - p[i]), bw, bh * p[i]);
    ctx.fillStyle = C.text; ctx.fillText(`${(p[i] * 100).toFixed(0)}%`, x, top + bh + 6);
    ctx.fillStyle = C.muted; ctx.fillText(`pays ~${BANDIT_MEANS[i]}`, x, top + bh + 20);
  }
  const cx = 24 + K * (bw + 14) + 10;
  chart(ctx, cx, top, Math.max(160, w - cx - 20), bh + 34, [
    { values: L.bestProb, color: C.amber, label: 'P(best arm)' },
    { values: rolling(L.rewards, 40), color: C.cyan, label: 'avg reward' },
  ], { min: 0, max: 1.2, title: `${L.bestProb.length} pulls` });
}

function drawGrid(ctx, w, hgt, L, last) {
  const n = GRID.size, cell = Math.min((hgt - 90) / n, (w * 0.45) / n), ox = 24, oy = 40;
  const is = (cells, x, y) => cells.some(([a, b]) => a === x && b === y);
  ctx.font = MONO; ctx.textBaseline = 'top'; ctx.fillStyle = C.muted; ctx.fillText('Policy: arrow = most likely move, brightness = how sure', ox, 16);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const px = ox + x * cell, py = oy + y * cell;
    ctx.fillStyle = is(GRID.walls, x, y) ? '#2a3140' : is(GRID.pits, x, y) ? 'rgba(239,107,107,.35)' : x === GRID.goal[0] && y === GRID.goal[1] ? 'rgba(123,216,143,.35)' : C.panel;
    ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
    if (is(GRID.walls, x, y) || is(GRID.pits, x, y) || (x === GRID.goal[0] && y === GRID.goal[1])) continue;
    const p = L.policy([x, y]), a = p.indexOf(Math.max(...p)), [dx, dy] = MOVES[a];
    ctx.strokeStyle = `rgba(245,181,68,${0.25 + 0.75 * p[a]})`; ctx.lineWidth = 2.5;
    const cx = px + cell / 2, cy = py + cell / 2, len = cell * 0.3;
    ctx.beginPath(); ctx.moveTo(cx - dx * len, cy - dy * len); ctx.lineTo(cx + dx * len, cy + dy * len);
    ctx.lineTo(cx + dx * len - dy * len * 0.4 - dx * len * 0.4, cy + dy * len + dx * len * 0.4 - dy * len * 0.4);
    ctx.stroke();
  }
  if (last) {
    ctx.strokeStyle = C.cyan; ctx.lineWidth = 2; ctx.beginPath();
    last.path.forEach((s, i) => { const x = ox + ((s % n) + 0.5) * cell, y = oy + (Math.floor(s / n) + 0.5) * cell; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  ctx.fillStyle = C.muted; ctx.fillText('start', ox + 4, oy + 4); ctx.fillText('goal', ox + 4 + (n - 1) * cell, oy + 4 + (n - 1) * cell);
  const cx = ox + n * cell + 24;
  chart(ctx, cx, oy, Math.max(160, w - cx - 20), n * cell, [
    { values: rolling(L.successes, 50), color: C.good, label: 'reached goal (last 50)' },
    { values: rolling(L.lengths, 50).map((v) => v / 40), color: C.cyan, label: 'path length ÷ 40' },
  ], { min: 0, max: 1, title: `${L.successes.length} episodes` });
}

function drawCart(ctx, w, hgt, L, frame) {
  const ep = L.lastEpisode, s = ep.length ? ep[Math.min(frame, ep.length - 1)] : [0, 0, 0, 0];
  const scale = (w * 0.45) / (2 * CARTPOLE.xMax), ground = hgt * 0.55, cx = 24 + w * 0.225 + s[0] * scale;
  ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(24, ground); ctx.lineTo(24 + w * 0.45, ground); ctx.stroke();
  ctx.fillStyle = C.cyan; ctx.fillRect(cx - 28, ground - 18, 56, 18);
  const len = CARTPOLE.len * 2 * scale * 0.9;
  ctx.strokeStyle = C.amber; ctx.lineWidth = 6; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(cx, ground - 18); ctx.lineTo(cx + Math.sin(s[2]) * len, ground - 18 - Math.cos(s[2]) * len); ctx.stroke();
  ctx.lineCap = 'butt';
  ctx.font = MONO; ctx.textBaseline = 'top'; ctx.fillStyle = C.muted;
  ctx.fillText(`Latest episode, step ${Math.min(frame, ep.length)} of ${ep.length}`, 24, 16);
  const cx2 = 24 + w * 0.45 + 24;
  chart(ctx, cx2, 40, Math.max(160, w - cx2 - 20), hgt * 0.55, [
    { values: L.lengths, color: 'rgba(94,200,229,.5)', label: 'steps balanced' },
    { values: rolling(L.lengths, 20), color: C.amber, label: 'avg of 20' },
  ], { min: 0, max: 200, title: `${L.lengths.length} episodes` });
}

function drawRlhf(ctx, w, hgt, L) {
  const p = L.probs, top = 50, bh = hgt * 0.45, bw = Math.min(110, (w * 0.5) / STYLES.length - 14);
  ctx.font = MONO; ctx.textBaseline = 'top'; ctx.fillStyle = C.muted; ctx.fillText('How often the policy picks each reply style', 24, 20);
  STYLES.forEach((st, i) => {
    const x = 24 + i * (bw + 14);
    ctx.fillStyle = C.panel; ctx.fillRect(x, top, bw, bh);
    ctx.fillStyle = st.quality >= 0.8 ? C.good : st.quality < 0.3 ? C.danger : C.cyan;
    ctx.fillRect(x, top + bh * (1 - p[i]), bw, bh * p[i]);
    ctx.fillStyle = C.text; ctx.fillText(`${(p[i] * 100).toFixed(0)}%`, x, top + bh + 6);
    ctx.fillStyle = C.muted;
    ctx.fillText(st.name, x, top + bh + 20);
    ctx.fillText(`RM ${rewardModel(i).toFixed(1)} · true ${st.quality}`, x, top + bh + 34);
  });
  const cx = 24 + STYLES.length * (bw + 14) + 10;
  chart(ctx, cx, top, Math.max(160, w - cx - 20), bh + 48, [
    { values: L.history.map((x) => x.reward - START_REWARD), color: C.amber, label: 'reward-model gain' },
    { values: L.history.map((x) => x.quality - START_QUALITY), color: C.good, label: 'true-quality change' },
    { values: L.history.map((x) => x.kl), color: C.cyan, label: 'KL from start' },
  ], { title: `${L.history.length} steps` });
}

export function mountReward({ stage, side, progress, saveProgress }) {
  const root = h('div', { class: 'rw' });
  stage.append(root);
  const { ctx, size, destroy } = createCanvas(root);
  const listSec = section(side, 'Chapter 7 · Learning from reward');
  const levelSec = section(side, 'Level');
  const knobSec = section(side, 'Settings');
  const runSec = section(side, 'Train');

  const unlocked = (i) => i === 0 || (progress[RL_LEVELS[i - 1].id] || 0) >= 1;
  let level = RL_LEVELS.find((L, i) => unlocked(i) && !(progress[L.id] >= 1)) || RL_LEVELS[0];
  let settings, learner, done, playing, frame, lastEp, verdict;

  function renderList() {
    listSec.replaceChildren(listSec.firstChild);
    RL_LEVELS.forEach((L, i) => {
      const open = unlocked(i);
      const row = h('div', { class: 'bn-level' + (open ? '' : ' locked') + (L === level ? ' active' : '') },
        h('span', {}, `${i + 1}. ${L.title}`), h('span', { class: 'bn-stars' }, open ? stars(progress[L.id] || 0) : '🔒'));
      if (open) row.addEventListener('click', () => { level = L; start(); });
      listSec.append(row);
    });
  }

  const result = h('div', { class: 'ws-goal' });
  let playBtn;
  function start() {
    settings = Object.fromEntries(level.knobs.map((k) => [k.key, k.value]));
    renderList();
    levelSec.replaceChildren(levelSec.firstChild, h('p', { class: 'hint ws-brief' }, level.brief), h('p', { class: 'hint ws-lesson' }, `💡 ${level.lesson}`));
    knobSec.replaceChildren(knobSec.firstChild);
    for (const k of level.knobs) {
      if (k.toggle) checkbox(knobSec, { label: k.label, checked: k.value, onChange: (v) => { settings[k.key] = v; reset(); } });
      else if (k.options) selectCtl(knobSec, { label: k.label, options: k.options, value: k.value, onChange: (v) => { settings[k.key] = v; reset(); } });
      else if (k.log) {
        const toV = (u) => +(k.min * Math.pow(k.max / k.min, u)).toPrecision(2);
        const sl = slider(knobSec, { label: k.label, min: 0, max: 1, step: 0.01, value: Math.log(k.value / k.min) / Math.log(k.max / k.min), format: (u) => String(toV(u)), onInput: (u) => { settings[k.key] = toV(u); reset(); } });
        sl.input.disabled = !!k.locked;
      } else {
        const sl = slider(knobSec, { label: k.label, min: k.min, max: k.max, step: k.step ?? 1, value: k.value, onInput: (v) => { settings[k.key] = v; reset(); } });
        sl.input.disabled = !!k.locked;
      }
    }
    reset();
  }

  // The animated run uses `runSeed` (New seed explores other luck); judging always uses seeds 1–3.
  let runSeed = 1;
  function reset() {
    learner = level.make(settings, runSeed);
    done = 0; frame = 0; lastEp = null; verdict = null; playing = false;
    result.textContent = `Press Train to watch it learn (seed ${runSeed}). When it finishes, it's judged on 3 fixed seeds.`;
    if (playBtn) playBtn.textContent = 'Train';
  }

  // Judging runs full training three times; show a note first so the brief pause isn't a mystery.
  function judge() {
    result.className = 'ws-goal';
    result.textContent = 'Judging on 3 fixed seeds…';
    setTimeout(judgeNow, 30);
  }
  function judgeNow() {
    verdict = evaluate(level, settings);
    const best = progress[level.id] || 0;
    if (verdict.stars > best) { progress[level.id] = verdict.stars; saveProgress(progress); }
    result.className = 'ws-goal' + (verdict.stars ? ' done' : '');
    result.innerHTML = verdict.stars ? `<b>✓ ${stars(verdict.stars)}</b> ${verdict.msg}` : `Not yet. ${verdict.msg}`;
    const i = RL_LEVELS.indexOf(level), next = RL_LEVELS[i + 1];
    if (verdict.stars && next) result.append(h('div', {}, h('button', { class: 'btn primary ws-next', onclick: () => { level = next; start(); } }, `Next: ${next.title} →`)));
    renderList();
  }

  [playBtn] = buttons(runSec, [
    { label: 'Train', primary: true, onClick: () => { if (done >= level.steps) reset(); playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Train'; } },
    { label: 'Reset', onClick: () => reset() },
    { label: 'New seed', onClick: () => { runSeed = 1 + Math.floor(Math.random() * 9999); reset(); } },
    { label: 'Judge now', onClick: () => judge() },
  ]);
  let speed = 1;
  slider(runSec, { label: 'Speed', min: 1, max: 20, value: 1, format: (v) => `${v}×`, onInput: (v) => { speed = v; } });
  runSec.append(result);

  const stop = loop(() => {
    if (playing && done < level.steps) {
      for (let i = 0; i < level.perFrame * speed && done < level.steps; i++) { lastEp = learner.step(); done++; }
      if (done >= level.steps) { playing = false; playBtn.textContent = 'Train again'; judge(); }
    }
    frame += 1;
    const { w, h: hh } = size;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, hh);
    if (!learner) return;
    if (level.kind === 'bandit') drawBandit(ctx, w, hh, learner);
    if (level.kind === 'grid') drawGrid(ctx, w, hh, learner, lastEp);
    if (level.kind === 'cart') drawCart(ctx, w, hh, learner, frame % (learner.lastEpisode.length + 30));
    if (level.kind === 'rlhf') drawRlhf(ctx, w, hh, learner);
  });

  start();
  return () => { stop(); destroy(); root.remove(); };
}
