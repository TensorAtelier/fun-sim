import { h } from './ui.js';
import sims from './sims/index.js';

const app = document.getElementById('app');
const nav = document.getElementById('nav');
let cleanup = null;

for (const s of sims) nav.append(h('a', { href: `#/${s.id}`, 'data-id': s.id }, s.title));

function renderHome() {
  const home = h('div', { class: 'home' },
    h('h1', {}, 'Math you can ', h('em', {}, 'poke at'), '.'),
    h('p', { class: 'lede' }, 'Pick a simulation. Change the numbers, drag things around, and watch what happens.'),
    h('div', { class: 'cards' }, sims.map((s) =>
      h('a', { class: 'card', href: `#/${s.id}` },
        h('div', { class: 'glyph' }, s.glyph),
        h('h3', {}, s.title),
        h('p', {}, s.blurb),
        h('span', { class: 'tag' }, s.tag)))));
  app.append(home);
}

function route() {
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } cleanup = null; }
  app.replaceChildren();
  const id = location.hash.replace(/^#\/?/, '');
  const sim = sims.find((s) => s.id === id);
  for (const a of nav.children) a.classList.toggle('active', a.dataset.id === id);
  if (!sim) { document.title = 'Fun Sim'; return renderHome(); }
  document.title = `${sim.title} · Fun Sim`;
  cleanup = sim.mount(app) || null;
}

window.addEventListener('hashchange', route);
route();
