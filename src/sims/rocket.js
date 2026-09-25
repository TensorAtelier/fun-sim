// Rocket launch to orbit — 2D two-stage ascent from a round, rotating Earth.
// Physics: inverse-square gravity, exponential atmosphere with transonic drag rise,
// pressure-dependent Isp, RK4 at a fixed 0.05 s step. Guidance: vertical rise, pitch kick,
// gravity turn, auto-cutoff once periapsis reaches the target orbit.

import { createLayout, section, slider, select, checkbox, buttons, readout, createCanvas, loop, h } from '../ui.js';

// ---------------------------------------------------------------- constants
const R_E = 6371e3;
const MU = 3.986e14;
const G0 = 9.80665;
const OMEGA = 7.2921159e-5; // sidereal rotation, rad/s
const RHO0 = 1.225;
const H_SCALE = 8500;
const DT = 0.05;
const ATMO_TOP = 140e3;

// ---------------------------------------------------------------- presets
const PRESETS = {
  falcon: {
    name: 'Falcon 9-ish',
    stages: [
      { dry: 25600, prop: 395700, thrust: 8227e3, ispSL: 282, ispVac: 311 },
      { dry: 3900, prop: 92670, thrust: 981e3, ispSL: 200, ispVac: 348 },
    ],
    payload: 15000, payloadMax: 30000, payloadStep: 250,
    diam: 3.7, cd: 0.3, geom: { l1: 42, l2: 14, fair: 13 },
    pitchAlt: 700, kick: 6, targetPe: 200,
  },
  under: {
    name: 'Underpowered',
    stages: [
      { dry: 32000, prop: 395700, thrust: 6900e3, ispSL: 265, ispVac: 290 },
      { dry: 5200, prop: 70000, thrust: 620e3, ispSL: 190, ispVac: 320 },
    ],
    payload: 22000, payloadMax: 30000, payloadStep: 250,
    diam: 3.7, cd: 0.34, geom: { l1: 42, l2: 14, fair: 13 },
    pitchAlt: 700, kick: 3.6, targetPe: 200,
  },
  sounding: {
    name: 'Sounding rocket',
    stages: [
      { dry: 420, prop: 1350, thrust: 105e3, ispSL: 230, ispVac: 255 },
      { dry: 240, prop: 780, thrust: 48e3, ispSL: 240, ispVac: 272 },
    ],
    payload: 350, payloadMax: 1000, payloadStep: 10,
    diam: 0.44, cd: 0.25, geom: { l1: 6, l2: 4.2, fair: 1.8 },
    pitchAlt: 300, kick: 0.6, targetPe: 0,
  },
};

// ---------------------------------------------------------------- physics
function buildConfig(p, o) {
  const stages = p.stages.map((st) => {
    const thrust = st.thrust * o.thrustScale;
    const prop = st.prop * o.propScale;
    return { ...st, thrust, prop, mdot: thrust / (st.ispVac * G0) };
  });
  return {
    stages,
    payload: o.payload,
    area: Math.PI * (p.diam / 2) ** 2,
    cd: p.cd,
    pitchAlt: o.pitchAlt,
    kick: (o.kick * Math.PI) / 180,
    targetPe: p.targetPe ? o.targetPe * 1000 : 0,
    rotation: o.rotation,
    guidance: o.guidance,
  };
}

const omega = (cfg) => (cfg.rotation ? OMEGA : 0);
const density = (alt) => (alt > ATMO_TOP ? 0 : RHO0 * Math.exp(-Math.max(alt, 0) / H_SCALE));
const pressFrac = (alt) => (alt > ATMO_TOP ? 0 : Math.exp(-Math.max(alt, 0) / H_SCALE));
// Drag coefficient with a transonic hump peaking just above Mach 1.
const cdMach = (cd, mach) => cd * (1 + 0.9 * Math.exp(-(((mach - 1.1) / 0.35) ** 2)) - 0.25 * Math.min(1, Math.max(0, (mach - 2) / 4)));
const soundSpeed = (alt) => (alt < 11000 ? 340.3 - 0.0041 * alt : 295);

function newState(cfg) {
  const w = omega(cfg);
  return {
    t: 0, x: 0, y: R_E, vx: w * R_E, vy: 0,
    stage: 1, prop: cfg.stages.map((s) => s.prop),
    status: 'pad', phase: 'pad', engine: false, throttle: 1, manualThrottle: true,
    heading: 0, dirx: 0, diry: 1, liftoff: false, cutoff: false, sepTimer: 0,
    maxQ: 0, maxQt: 0, maxQlogged: false, apogee: 0, maxG: 0,
    orbitLogged: false, karman: false, escapeLogged: false,
    q: 0, accel: 0, mach: 0,
    events: [], debris: [],
  };
}

function mass(s, cfg) {
  const [a, b] = cfg.stages;
  let m = cfg.payload + b.dry + Math.max(0, s.prop[1]);
  if (s.stage === 1) m += a.dry + Math.max(0, s.prop[0]);
  return m;
}

function elements(x, y, vx, vy) {
  const r = Math.hypot(x, y);
  const v2 = vx * vx + vy * vy;
  const eps = v2 / 2 - MU / r;
  const hm = x * vy - y * vx;
  const rv = x * vx + y * vy;
  const ex = ((v2 - MU / r) * x - rv * vx) / MU;
  const ey = ((v2 - MU / r) * y - rv * vy) / MU;
  const e = Math.hypot(ex, ey);
  const p = (hm * hm) / MU;
  const rp = p / (1 + e);
  const ra = e < 1 ? p / (1 - e) : Infinity;
  return { e, p, rp, ra, eps, hm, argp: Math.atan2(ey, ex), a: -MU / (2 * eps) };
}

function event(s, text, kind = 'info') { s.events.push({ t: s.t, text, kind }); }

// Local frame: u = radial up, e = east (clockwise, +x at the launch site on top).
function basis(x, y) {
  const r = Math.hypot(x, y);
  const ux = x / r, uy = y / r;
  return { r, ux, uy, ex: uy, ey: -ux };
}

function guide(s, cfg) {
  const { r, ux, uy, ex, ey } = basis(s.x, s.y);
  const alt = r - R_E;
  const w = omega(cfg);
  const rvx = s.vx - w * s.y, rvy = s.vy + w * s.x; // air-relative velocity
  const setDir = (th) => { s.dirx = Math.cos(th) * ux + Math.sin(th) * ex; s.diry = Math.cos(th) * uy + Math.sin(th) * ey; };
  const prograde = () => {
    const blend = Math.min(1, Math.max(0, (alt - 20e3) / 40e3));
    const a = Math.hypot(rvx, rvy) || 1, b = Math.hypot(s.vx, s.vy) || 1;
    let dx = (1 - blend) * rvx / a + blend * s.vx / b;
    let dy = (1 - blend) * rvy / a + blend * s.vy / b;
    const n = Math.hypot(dx, dy) || 1;
    if (Math.hypot(rvx, rvy) < 5 && blend === 0) { dx = ux; dy = uy; }
    s.dirx = dx / n; s.diry = dy / n;
  };

  if (cfg.guidance === 'manual') {
    setDir(s.heading);
    s.engine = s.manualThrottle && s.sepTimer <= 0;
    s.label = s.sepTimer > 0 ? 'staging' : s.engine ? 'manual' : 'manual coast';
    return;
  }
  if (s.sepTimer > 0) { s.engine = false; prograde(); s.label = 'staging'; return; }
  if (s.cutoff) { s.engine = false; prograde(); s.label = s.inOrbit ? 'orbit' : 'coast'; return; }
  s.engine = true;
  if (s.phase === 'pad' || s.phase === 'vertical') {
    s.phase = alt >= cfg.pitchAlt && s.liftoff ? 'kick' : 'vertical';
    if (s.phase === 'kick') { s.kickStart = s.t; event(s, `Pitch kick ${(cfg.kick * 180 / Math.PI).toFixed(1)}°`); }
  }
  if (s.phase === 'vertical') { setDir(0); }
  else if (s.phase === 'kick') {
    setDir(cfg.kick);
    const gam = Math.atan2(rvx * ex + rvy * ey, rvx * ux + rvy * uy);
    if (gam >= cfg.kick * 0.98 || s.t - s.kickStart > 25) { s.phase = 'gravity turn'; }
  } else if (s.phase === 'closed loop') {
    // Upper-stage guidance: pitch so the radial acceleration drives vertical speed toward a
    // profile that levels off at the target altitude, while the rest of the thrust goes east.
    const vr = s.vx * ux + s.vy * uy, vh = s.vx * ex + s.vy * ey;
    const st = cfg.stages[s.stage - 1];
    const aT = (st.mdot * s.throttle * st.ispVac * G0) / mass(s, cfg);
    const vrDes = Math.max(-120, Math.min(600, (cfg.targetPe - alt) / 70));
    const need = (vrDes - vr) / 15 + MU / (r * r) - (vh * vh) / r;
    const sn = Math.max(-0.35, Math.min(0.95, need / Math.max(aT, 1e-3)));
    setDir(Math.acos(sn));
  } else {
    prograde();
    if (s.phase === 'gravity turn' && alt > 45e3) s.phase = 'ascent';
    if (cfg.targetPe > 0 && s.stage === 2 && s.phase === 'ascent') { s.phase = 'closed loop'; event(s, 'Closed-loop guidance engaged'); }
  }
  s.label = s.phase;
  if (cfg.targetPe > 0) {
    const el = elements(s.x, s.y, s.vx, s.vy);
    if (el.rp - R_E >= cfg.targetPe - 3000) {
      s.cutoff = true; s.engine = false; s.label = 'coast';
      event(s, s.stage === 2 ? 'SECO — orbit insertion' : 'MECO — orbit insertion', 'good');
    }
  }
}

// Acceleration at a given state; thrust uses the pressure-interpolated Isp at that altitude.
function accel(x, y, vx, vy, m, mdot, dx, dy, st, cfg, out) {
  const r = Math.hypot(x, y);
  const alt = r - R_E;
  const gk = -MU / (r * r * r);
  let ax = gk * x, ay = gk * y;
  let F = 0;
  if (mdot > 0) {
    const isp = st.ispVac + (st.ispSL - st.ispVac) * pressFrac(alt);
    F = mdot * isp * G0;
    ax += (F / m) * dx; ay += (F / m) * dy;
  }
  const rho = density(alt);
  let q = 0, mach = 0;
  if (rho > 0) {
    const w = omega(cfg);
    const rvx = vx - w * y, rvy = vy + w * x;
    const vr = Math.hypot(rvx, rvy);
    if (vr > 1e-6) {
      mach = vr / soundSpeed(alt);
      q = 0.5 * rho * vr * vr;
      const D = q * cdMach(cfg.cd, mach) * cfg.area;
      ax -= (D / m) * rvx / vr; ay -= (D / m) * rvy / vr;
    }
  }
  if (out) { out.q = q; out.mach = mach; out.F = F; out.ax = ax - gk * x; out.ay = ay - gk * y; }
  return [ax, ay];
}

function stepRocket(s, cfg, dt) {
  if (s.status !== 'flight') return;
  guide(s, cfg);
  const si = s.stage - 1;
  const st = cfg.stages[si];
  let mdot = 0;
  const throttle = cfg.guidance === 'manual' ? 1 : s.throttle;
  if (s.engine && s.prop[si] > 0) mdot = st.mdot * throttle;
  const m0 = mass(s, cfg);
  const { dirx: dx, diry: dy } = s;
  const f = (x, y, vx, vy, tau) => accel(x, y, vx, vy, m0 - mdot * tau, mdot, dx, dy, st, cfg);

  // RK4
  const { x, y, vx, vy } = s;
  const [a1x, a1y] = f(x, y, vx, vy, 0);
  const h2 = dt / 2;
  const [a2x, a2y] = f(x + h2 * vx, y + h2 * vy, vx + h2 * a1x, vy + h2 * a1y, h2);
  const v2x = vx + h2 * a1x, v2y = vy + h2 * a1y;
  const [a3x, a3y] = f(x + h2 * v2x, y + h2 * v2y, vx + h2 * a2x, vy + h2 * a2y, h2);
  const v3x = vx + h2 * a2x, v3y = vy + h2 * a2y;
  const [a4x, a4y] = f(x + dt * v3x, y + dt * v3y, vx + dt * a3x, vy + dt * a3y, dt);
  const v4x = vx + dt * a3x, v4y = vy + dt * a3y;
  let nx = x + (dt / 6) * (vx + 2 * v2x + 2 * v3x + v4x);
  let ny = y + (dt / 6) * (vy + 2 * v2y + 2 * v3y + v4y);
  let nvx = vx + (dt / 6) * (a1x + 2 * a2x + 2 * a3x + a4x);
  let nvy = vy + (dt / 6) * (a1y + 2 * a2y + 2 * a3y + a4y);

  // Hold-down clamps: stay on the pad until thrust beats weight.
  if (!s.liftoff) {
    const out = {};
    const [ax, ay] = accel(x, y, vx, vy, m0, mdot, dx, dy, st, cfg, out);
    const { ux, uy, r } = basis(x, y);
    const w = omega(cfg);
    const ar = ax * ux + ay * uy + w * w * r; // co-rotating frame
    if (ar > 0.05) { s.liftoff = true; event(s, 'Liftoff'); }
    else {
      const phi = Math.PI / 2 - w * (s.t + dt);
      nx = R_E * Math.cos(phi); ny = R_E * Math.sin(phi);
      nvx = w * ny; nvy = -w * nx;
    }
  }

  s.x = nx; s.y = ny; s.vx = nvx; s.vy = nvy;
  s.t += dt;
  s.prop[si] = Math.max(0, s.prop[si] - mdot * dt);
  s.throttleActual = mdot > 0 ? throttle : 0;

  // Diagnostics at the new state.
  const out = {};
  accel(nx, ny, nvx, nvy, mass(s, cfg), mdot, dx, dy, st, cfg, out);
  s.q = out.q; s.mach = out.mach; s.thrustN = out.F;
  s.accel = Math.hypot(out.ax, out.ay) / G0;
  if (s.liftoff && mdot > 0) s.maxG = Math.max(s.maxG, s.accel);
  if (s.q > s.maxQ && !s.maxQlogged) { s.maxQ = s.q; s.maxQt = s.t; }
  else if (!s.maxQlogged && s.maxQ > 2000 && s.q < s.maxQ * 0.97) {
    s.maxQlogged = true;
    s.events.push({ t: s.maxQt, text: `Max-Q ${(s.maxQ / 1000).toFixed(1)} kPa`, kind: 'warn' });
  }
  const r = Math.hypot(nx, ny);
  const alt = r - R_E;
  s.apogee = Math.max(s.apogee, alt);
  if (!s.karman && alt > 100e3) { s.karman = true; event(s, 'Crossed Kármán line (100 km)'); }

  if (s.sepTimer > 0) {
    s.sepTimer -= dt;
    if (s.sepTimer <= 0 && s.prop[1] > 0) event(s, 'Stage 2 ignition');
  }
  // Staging
  if (s.stage === 1 && s.prop[0] <= 0) {
    if (!s.liftoff) {
      s.status = 'failed';
      event(s, 'Stage 1 burned out on the pad — never lifted off', 'bad');
      return;
    }
    event(s, 'MECO — stage 1 separation');
    s.stage = 2;
    s.sepTimer = 2.5;
    const a = cfg.stages[0];
    s.debris.push({ x: nx, y: ny, vx: nvx - dx * 2, vy: nvy - dy * 2, m: a.dry, area: cfg.area * 1.2, ang: Math.atan2(dy, dx), spin: 0.25, alive: true, t: 0 });
  }
  if (s.stage === 2 && s.prop[1] <= 0 && !s.depleted && s.sepTimer <= 0) {
    s.depleted = true;
    if (!s.cutoff) event(s, 'Stage 2 propellant depleted', 'warn');
  }

  // Termination / milestone checks
  const el = elements(nx, ny, nvx, nvy);
  if (!s.orbitLogged && el.e < 1 && el.rp - R_E > 100e3 && mdot === 0 && s.sepTimer <= 0) {
    s.orbitLogged = true;
    event(s, `Orbit achieved — ${((el.rp - R_E) / 1000).toFixed(0)} × ${((el.ra - R_E) / 1000).toFixed(0)} km`, 'good');
    s.inOrbit = true;
  }
  if (el.eps >= 0 && !s.escapeLogged && alt > 100e3) { s.escapeLogged = true; event(s, 'Escape trajectory — leaving Earth', 'good'); }
  if (s.liftoff && alt < 0) {
    s.status = 'crashed';
    const sub = s.inOrbit ? 'Deorbited and impacted' : s.apogee > 100e3 ? `Suborbital — impact after ${(s.apogee / 1000).toFixed(0)} km apogee` : `Impact — apogee only ${(s.apogee / 1000).toFixed(1)} km`;
    event(s, sub, s.apogee > 100e3 && cfg.targetPe === 0 ? 'info' : 'bad');
  }

  // Debris (spent stage) — ballistic with drag.
  for (const d of s.debris) {
    if (!d.alive) continue;
    const dr = Math.hypot(d.x, d.y);
    const dalt = dr - R_E;
    const gk = -MU / (dr * dr * dr);
    let ax = gk * d.x, ay = gk * d.y;
    const rho = density(dalt);
    if (rho > 0) {
      const w = omega(cfg);
      const rvx = d.vx - w * d.y, rvy = d.vy + w * d.x;
      const vr = Math.hypot(rvx, rvy);
      const D = 0.5 * rho * vr * vr * 0.9 * d.area;
      if (vr > 0) { ax -= (D / d.m) * rvx / vr; ay -= (D / d.m) * rvy / vr; }
    }
    d.vx += ax * dt; d.vy += ay * dt;
    d.x += d.vx * dt; d.y += d.vy * dt;
    d.ang += d.spin * dt; d.t += dt;
    if (dalt < 0) d.alive = false;
  }
}

// Headless run (used for tuning; exported for tests).
export function simulateFlight(presetKey, overrides = {}, tMax = 1500) {
  const p = PRESETS[presetKey];
  const o = { thrustScale: 1, propScale: 1, payload: p.payload, pitchAlt: p.pitchAlt, kick: p.kick, targetPe: p.targetPe, rotation: true, guidance: 'auto', ...overrides };
  const cfg = buildConfig(p, o);
  const s = newState(cfg);
  s.status = 'flight';
  while (s.t < tMax && s.status === 'flight') {
    stepRocket(s, cfg, DT);
    if (s.inOrbit && s.cutoff) break;
  }
  return { s, el: elements(s.x, s.y, s.vx, s.vy) };
}


// ---------------------------------------------------------------- rendering helpers
const C = {
  bg: '#0b0e14', accent: '#f5b544', cyan: '#5ec8e5', muted: '#8a93a6', line: '#242c3b',
  good: '#7bd88f', danger: '#ef6b6b', text: '#e6e9ef',
};
const FONT = "'IBM Plex Mono', ui-monospace, monospace";
const WARPS = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000];

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
function mix(a, b, t, alpha = 1) {
  const A = hex(a), B = hex(b);
  const m = A.map((v, i) => Math.round(v + (B[i] - v) * t));
  return `rgba(${m[0]},${m[1]},${m[2]},${alpha})`;
}
const rgba = (c, a) => { const [r, g, b] = hex(c); return `rgba(${r},${g},${b},${a})`; };

function fmtClock(t) {
  const s = Math.max(0, Math.floor(t));
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}
const fmtAlt = (m) => (Math.abs(m) < 10e3 ? `${m.toFixed(0)} m` : `${(m / 1000).toFixed(1)} km`);
const fmtKm = (m) => (!isFinite(m) ? '∞' : `${(m / 1000).toFixed(0)} km`);
const fmtSpeed = (v) => (v < 1000 ? `${v.toFixed(0)} m/s` : `${(v / 1000).toFixed(2)} km/s`);
const fmtMass = (m) => (m >= 1000 ? `${(m / 1000).toFixed(1)} t` : `${m.toFixed(0)} kg`);

function roundRect(ctx, x, y, w, hh, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + hh - r); ctx.quadraticCurveTo(x + w, y + hh, x + w - r, y + hh);
  ctx.lineTo(x + r, y + hh); ctx.quadraticCurveTo(x, y + hh, x, y + hh - r);
  ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
}

// Random but deterministic starfield and cloud deck.
function rng(seed) { return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646; }
const STARS = (() => { const r = rng(7); return Array.from({ length: 260 }, () => ({ x: r(), y: r(), s: r() ** 3 * 1.4 + 0.3, a: 0.3 + r() * 0.7 })); })();
const CLOUDS = (() => { const r = rng(42); return Array.from({ length: 26 }, () => ({ off: (r() - 0.3) * 90e3, alt: 1800 + r() * 8000, w: 1500 + r() * 5000, hh: 150 + r() * 450, a: 0.08 + r() * 0.16 })); })();

// ---------------------------------------------------------------- module
export default {
  id: 'rocket',
  title: 'Rocket',
  glyph: '⇡',
  tag: 'physics',
  blurb: 'Fly a two-stage rocket from a round, rotating Earth to orbit — gravity turn, max-Q, staging and live Keplerian orbit prediction.',

  mount(root) {
    const { stage, panel } = createLayout(root, {
      title: 'Rocket to Orbit',
      desc: 'Two-stage ascent with inverse-square gravity, an exponential atmosphere and pressure-dependent Isp. Vertical rise, pitch kick, gravity turn, then closed-loop upper-stage guidance to the target orbit.',
    });

    // ------------------------------------------------ state
    let presetKey = 'falcon';
    const opts = { thrustScale: 1, propScale: 1, payload: 0, pitchAlt: 0, kick: 0, targetPe: 200, rotation: true, guidance: 'auto' };
    let cfg, sim, paused = false, warpIdx = 0, mapMain = false, dirty = false;
    let acc = 0, trail = [], trailEvery = 2, nextTrail = 0, chart = [], chartEvery = 1, nextChart = 0;
    let particles = [], toasts = [], eventsSeen = 0, camLog = 0, dispAng = 0, lastReadout = 0;
    let insetRect = { x: 0, y: 0, w: 0, h: 0 };
    let mapFit = 1.2 * R_E;
    const keys = {};

    // ------------------------------------------------ panel
    const secMission = section(panel, 'Mission');
    select(secMission, {
      label: 'Preset', value: presetKey,
      options: Object.entries(PRESETS).map(([k, p]) => [k, p.name]),
      onChange: (v) => { presetKey = v; applyPreset(); },
    });
    select(secMission, {
      label: 'Guidance', value: 'auto',
      options: [['auto', 'Gravity turn (auto)'], ['manual', 'Manual — ← → steer, space throttle']],
      onChange: (v) => { opts.guidance = v; changed(); },
    });
    const [launchBtn, pauseBtn, resetBtn] = buttons(secMission, [
      { label: 'Launch', primary: true, onClick: () => launch() },
      { label: 'Pause', onClick: () => togglePause() },
      { label: 'Reset', onClick: () => reset() },
    ]);
    slider(secMission, {
      label: 'Time warp', min: 0, max: WARPS.length - 1, step: 1, value: 0,
      format: (v) => `${WARPS[v]}×`, onInput: (v) => { warpIdx = v; },
    });
    checkbox(secMission, { label: 'Earth rotation (equatorial site)', checked: true, onChange: (v) => { opts.rotation = v; changed(); } });
    const mapChk = checkbox(secMission, { label: 'Orbit map as main view', checked: false, onChange: (v) => { mapMain = v; } });

    const secVeh = section(panel, 'Vehicle');
    const thrustSl = slider(secVeh, { label: 'Thrust', min: 50, max: 150, step: 1, value: 100, format: (v) => `${v}%`, onInput: (v) => { opts.thrustScale = v / 100; changed(); } });
    const propSl = slider(secVeh, { label: 'Propellant load', min: 40, max: 130, step: 1, value: 100, format: (v) => `${v}%`, onInput: (v) => { opts.propScale = v / 100; changed(); } });
    const payloadSl = slider(secVeh, { label: 'Payload', min: 0, max: 30000, step: 250, value: 15000, format: fmtMass, onInput: (v) => { opts.payload = v; changed(); } });

    const secGd = section(panel, 'Guidance');
    const pitchSl = slider(secGd, { label: 'Pitch-over altitude', min: 100, max: 5000, step: 50, value: 700, format: (v) => `${v} m`, onInput: (v) => { opts.pitchAlt = v; changed(); } });
    const kickSl = slider(secGd, { label: 'Pitch kick angle', min: 0, max: 15, step: 0.1, value: 6, format: (v) => `${(+v).toFixed(1)}°`, onInput: (v) => { opts.kick = v; changed(); } });
    const targetSl = slider(secGd, { label: 'Target periapsis', min: 120, max: 250, step: 5, value: 200, format: (v) => `${v} km`, onInput: (v) => { opts.targetPe = v; changed(); } });
    const note = h('p', { class: 'hint', style: 'margin:0' });
    secGd.append(note);

    const secTel = section(panel, 'Telemetry');
    const RO = ['T+', 'Phase', 'Altitude', 'Surface speed', 'Orbital speed', 'Mach', 'Acceleration', 'Mass', 'Stage', 'Propellant', 'Dyn. pressure', 'Max-Q', 'Apoapsis', 'Periapsis', 'Eccentricity'];
    const ro = readout(secTel, RO);

    const secLog = section(panel, 'Flight log');
    const logEl = h('div', { style: `font-family:${FONT};font-size:11.5px;line-height:1.65;max-height:180px;overflow-y:auto` });
    secLog.append(logEl);
    const hintEl = h('p', { class: 'hint', style: 'margin-top:12px' }, 'Keys: Space launch / throttle toggle (manual) · ← → steer (manual) · P pause · M swap views. Click the inset to swap.');
    panel.append(hintEl);
    const onPanelClick = (e) => { if (e.target.tagName === 'BUTTON') e.target.blur(); };
    panel.addEventListener('click', onPanelClick);

    // ------------------------------------------------ canvas
    const cv = createCanvas(stage);
    const { canvas, ctx, size } = cv;
    canvas.style.cursor = 'default';
    const onCanvasClick = (e) => {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      const ir = insetRect;
      if (x >= ir.x && x <= ir.x + ir.w && y >= ir.y && y <= ir.y + ir.h) setMapMain(!mapMain);
    };
    const onCanvasMove = (e) => {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top, ir = insetRect;
      canvas.style.cursor = x >= ir.x && x <= ir.x + ir.w && y >= ir.y && y <= ir.y + ir.h ? 'pointer' : 'default';
    };
    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('mousemove', onCanvasMove);

    function setMapMain(v) { mapMain = v; mapChk.checked = v; }

    // ------------------------------------------------ control logic
    function applyPreset() {
      const p = PRESETS[presetKey];
      opts.thrustScale = 1; opts.propScale = 1;
      opts.payload = p.payload; opts.pitchAlt = p.pitchAlt; opts.kick = p.kick;
      thrustSl.set(100); propSl.set(100);
      payloadSl.input.max = p.payloadMax; payloadSl.input.step = p.payloadStep; payloadSl.set(p.payload);
      pitchSl.set(p.pitchAlt); kickSl.set(p.kick);
      targetSl.input.disabled = !p.targetPe;
      if (p.targetPe) { opts.targetPe = p.targetPe; targetSl.set(p.targetPe); }
      reset();
    }

    function changed() {
      if (!sim || sim.status === 'pad') reset();
      else { dirty = true; updateButtons(); }
    }

    function reset() {
      cfg = buildConfig(PRESETS[presetKey], opts);
      sim = newState(cfg);
      paused = false; dirty = false; acc = 0;
      trail = [{ x: sim.x, y: sim.y, k: 1 }]; trailEvery = 2; nextTrail = 0;
      chart = []; chartEvery = 1; nextChart = 0;
      particles = []; toasts = []; eventsSeen = 0;
      camLog = Math.log(spanTarget()); dispAng = 0;
      logEl.textContent = '';
      addLog(0, `${PRESETS[presetKey].name} on the pad — liftoff mass ${fmtMass(mass(sim, cfg))}, T/W ${twr().toFixed(2)}`, 'info');
      updateButtons();
      updateReadout(true);
    }

    function twr() {
      const st = cfg.stages[0];
      return (st.mdot * st.ispSL * G0) / (mass(sim, cfg) * G0);
    }

    function launch() {
      if (sim.status === 'pad') { sim.status = 'flight'; paused = false; }
      else if (sim.status !== 'flight') { reset(); sim.status = 'flight'; }
      updateButtons();
    }
    function togglePause() { if (sim.status === 'flight') { paused = !paused; updateButtons(); } }

    function updateButtons() {
      launchBtn.disabled = sim.status === 'flight';
      launchBtn.textContent = sim.status === 'pad' ? 'Launch' : 'Relaunch';
      pauseBtn.disabled = sim.status !== 'flight';
      pauseBtn.textContent = paused ? 'Resume' : 'Pause';
      resetBtn.classList.toggle('primary', dirty);
      note.textContent = dirty ? 'Parameter changes apply on reset.' : PRESETS[presetKey].targetPe ? '' : 'Sounding rockets fly unguided to burnout — no orbit target.';
    }

    function addLog(t, text, kind) {
      const col = kind === 'good' ? C.good : kind === 'bad' ? C.danger : kind === 'warn' ? C.accent : C.text;
      const row = h('div', {}, h('span', { style: `color:${C.muted}` }, `T+${fmtClock(t)} `), h('span', { style: `color:${col}` }, text));
      logEl.append(row);
      logEl.scrollTop = logEl.scrollHeight;
    }

    // ------------------------------------------------ keyboard
    const isField = (e) => e.target && e.target.closest && e.target.closest('input, select, textarea');
    const onKeyDown = (e) => {
      if (isField(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
        if (opts.guidance === 'manual') { keys[e.code] = true; e.preventDefault(); }
      } else if (e.code === 'Space') {
        e.preventDefault();
        if (e.repeat) return;
        if (sim.status !== 'flight') launch();
        else if (opts.guidance === 'manual') sim.manualThrottle = !sim.manualThrottle;
      } else if (e.code === 'KeyM') setMapMain(!mapMain);
      else if (e.code === 'KeyP') togglePause();
    };
    const onKeyUp = (e) => { keys[e.code] = false; };
    const onBlur = () => { for (const k in keys) keys[k] = false; };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    // ------------------------------------------------ simulation advance
    function totalLen() { const g = PRESETS[presetKey].geom; return g.l1 + g.l2 + g.fair; }
    function altOf() { return Math.hypot(sim.x, sim.y) - R_E; }
    function spanTarget() { const L = totalLen() * 4.2; return Math.min(5e6, Math.max(L, altOf() * 3.6 + L)); }

    function advance(dtReal) {
      if (opts.guidance === 'manual' && sim.status !== 'crashed') {
        const rate = 0.9 * dtReal;
        if (keys.ArrowLeft) sim.heading = clamp(sim.heading - rate, -Math.PI, Math.PI);
        if (keys.ArrowRight) sim.heading = clamp(sim.heading + rate, -Math.PI, Math.PI);
      }
      if (sim.status !== 'flight' || paused) return;
      acc += dtReal * WARPS[warpIdx];
      let n = Math.floor(acc / DT);
      acc -= n * DT;
      n = Math.min(n, 30000);
      for (let i = 0; i < n && sim.status === 'flight'; i++) {
        stepRocket(sim, cfg, DT);
        if (sim.t >= nextTrail) {
          trail.push({ x: sim.x, y: sim.y, k: sim.throttleActual > 0 ? sim.stage : 0 });
          nextTrail = sim.t + trailEvery;
          if (trail.length > 3000) { trail = trail.filter((_, j) => j % 2 === 0); trailEvery *= 2; }
        }
        if (sim.t >= nextChart) {
          chart.push({ t: sim.t, alt: altOf(), v: Math.hypot(sim.vx, sim.vy) });
          nextChart = sim.t + chartEvery;
          if (chart.length > 1200) { chart = chart.filter((_, j) => j % 2 === 0); chartEvery *= 2; }
        }
      }
      if (sim.status !== 'flight') {
        trail.push({ x: sim.x, y: sim.y, k: 0 });
        if (sim.status === 'crashed') explode();
        updateButtons();
      }
    }

    function processEvents(now) {
      while (eventsSeen < sim.events.length) {
        const ev = sim.events[eventsSeen++];
        addLog(ev.t, ev.text, ev.kind);
        toasts.push({ text: ev.text, kind: ev.kind, born: now });
        if (toasts.length > 4) toasts.shift();
      }
    }

    // ------------------------------------------------ particles (camera frame, metres from nozzle)
    function emit(dtReal) {
      const L = totalLen();
      const alt = altOf();
      const pf = pressFrac(alt);
      if (sim.status === 'flight' && sim.throttleActual > 0 && !paused) {
        const n = Math.round((mapMain ? 4 : 9) * sim.throttleActual);
        const spread = 0.07 + 0.55 * (1 - pf);
        const sa = dispAng; // screen angle of rocket body
        const bx = -Math.sin(sa), by = Math.cos(sa); // exhaust direction in screen coords
        const d = PRESETS[presetKey].diam;
        for (let i = 0; i < n; i++) {
          const a = (Math.random() - 0.5) * 2 * spread;
          const ca = Math.cos(a), sn = Math.sin(a);
          const dx = bx * ca - by * sn, dy = bx * sn + by * ca;
          const sp = L * (1.6 + Math.random() * 1.2) * (0.7 + 0.6 * (1 - pf));
          particles.push({ x: (Math.random() - 0.5) * d * 0.6, y: 0, vx: dx * sp, vy: dy * sp, life: 0, max: 0.18 + Math.random() * 0.25, size: d * (0.6 + Math.random() * 0.8), kind: 'fire' });
        }
        if (alt < 35e3) {
          const m = Math.random() < 0.8 ? 2 : 1;
          for (let i = 0; i < m; i++) {
            const sp = L * (0.3 + Math.random() * 0.5);
            particles.push({ x: bx * L * 0.35 + (Math.random() - 0.5) * d, y: by * L * 0.35, vx: bx * sp + (Math.random() - 0.5) * L * 0.2, vy: by * sp, life: 0, max: 1.6 + Math.random() * 2.2, size: d * (1.5 + Math.random() * 2), kind: 'smoke', a: pf });
          }
          if (alt < 150) {
            for (let i = 0; i < 3; i++) {
              const side = Math.random() < 0.5 ? -1 : 1;
              particles.push({ x: 0, y: alt, vx: side * L * (0.5 + Math.random() * 0.9), vy: -L * Math.random() * 0.05, life: 0, max: 2.5 + Math.random() * 2.5, size: d * (1.4 + Math.random() * 2), kind: 'smoke', a: 0.8, ground: true });
            }
          }
        }
      }
      // Move: camera follows the rocket, so the air (and smoke in it) streams past at -v_air.
      const { ux, uy, ex, ey } = basis(sim.x, sim.y);
      const w = omega(cfg);
      const rvx = sim.vx - w * sim.y, rvy = sim.vy + w * sim.x;
      const simDt = sim.status === 'flight' && !paused ? dtReal * Math.min(WARPS[warpIdx], 4) : 0;
      const airX = -(rvx * ex + rvy * ey), airY = (rvx * ux + rvy * uy); // screen-down positive
      for (const p of particles) {
        p.life += dtReal;
        p.x += p.vx * dtReal; p.y += p.vy * dtReal;
        if (p.kind === 'smoke') { p.x += airX * simDt; p.y += airY * simDt; p.vx *= 0.97; p.vy *= 0.97; }
        else { const k = 0.3 * pf; p.x += airX * simDt * k; p.y += airY * simDt * k; }
      }
      particles = particles.filter((p) => p.life < p.max);
      if (particles.length > 900) particles.splice(0, particles.length - 900);
    }

    function explode() {
      const L = totalLen();
      for (let i = 0; i < 160; i++) {
        const a = Math.random() * Math.PI * 2, sp = L * (0.5 + Math.random() * 2.5);
        const fire = i < 90;
        particles.push({ x: 0, y: -L * 0.3, vx: Math.cos(a) * sp, vy: -Math.abs(Math.sin(a)) * sp, life: 0, max: fire ? 0.6 + Math.random() * 0.8 : 2 + Math.random() * 3, size: L * (fire ? 0.12 : 0.25) * (0.5 + Math.random()), kind: fire ? 'fire' : 'smoke', a: 1 });
      }
    }

    // ------------------------------------------------ close-up camera view
    function drawClose(X, Y, W, H, inset) {
      const alt = altOf();
      const sc = H / Math.exp(camLog);
      const cx = W / 2, cy = H * (inset ? 0.6 : 0.62);
      const { ux, uy, ex, ey } = basis(sim.x, sim.y);
      const toS = (wx, wy) => {
        const rx = wx - sim.x, ry = wy - sim.y;
        return [cx + (rx * ex + ry * ey) * sc, cy - (rx * ux + ry * uy) * sc];
      };
      ctx.save();
      ctx.beginPath(); ctx.rect(X, Y, W, H); ctx.clip();
      ctx.translate(X, Y);

      // Sky
      const f = smooth(0, 75e3, alt);
      const g = ctx.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, mix('#1c406a', '#04060a', Math.pow(f, 0.5)));
      g.addColorStop(0.65, mix('#4f82b3', '#070a11', Math.pow(f, 0.7)));
      g.addColorStop(1, mix('#9cc2e0', '#0b0e14', Math.pow(f, 0.8)));
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

      // Stars
      const starA = smooth(15e3, 60e3, alt);
      if (starA > 0.01) {
        const phr = Math.atan2(sim.y, sim.x);
        for (const st of STARS) {
          const sx = (((st.x + phr * 0.6) % 1) + 1) % 1 * W, sy = st.y * H;
          ctx.fillStyle = `rgba(230,236,255,${st.a * starA})`;
          ctx.fillRect(sx, sy, st.s, st.s);
        }
      }

      // Earth arc (polyline — robust at any zoom)
      const phr = Math.atan2(sim.y, sim.x);
      const arcPts = (rad) => {
        const d = Math.min(Math.PI, (1.6 * Math.hypot(W, H)) / sc / rad + 1e-6);
        const pts = [];
        const N = 180;
        for (let i = 0; i <= N; i++) {
          const ph = phr - d + (2 * d * i) / N;
          const [sx, sy] = toS(rad * Math.cos(ph), rad * Math.sin(ph));
          pts.push([clamp(sx, -1e6, 1e6), clamp(sy, -1e6, 1e6)]);
        }
        return { pts, full: d >= Math.PI };
      };
      // Atmosphere: a band from the surface to ~100 km, shaded outward from the ground.
      const limbA = smooth(6e3, 90e3, alt);
      if (limbA > 0.01) {
        const top = arcPts(R_E + 100e3);
        ctx.beginPath();
        top.pts.forEach(([x, y], j) => (j ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        if (!top.full) { const a = top.pts[top.pts.length - 1], b = top.pts[0]; ctx.lineTo(a[0], Math.max(a[1], H) + 4 * H); ctx.lineTo(b[0], Math.max(b[1], H) + 4 * H); }
        ctx.closePath();
        const [ecx, ecy] = toS(0, 0);
        const ag = ctx.createRadialGradient(ecx, ecy, R_E * sc, ecx, ecy, (R_E + 100e3) * sc);
        ag.addColorStop(0, rgba('#7fc4ec', 0.55 * limbA));
        ag.addColorStop(0.25, rgba(C.cyan, 0.22 * limbA));
        ag.addColorStop(1, rgba(C.cyan, 0));
        ctx.fillStyle = ag;
        ctx.fill();
      }
      {
        const { pts, full } = arcPts(R_E);
        ctx.beginPath();
        pts.forEach(([x, y], j) => (j ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        if (!full) { const a = pts[pts.length - 1], b = pts[0]; ctx.lineTo(a[0], Math.max(a[1], H) + 4 * H); ctx.lineTo(b[0], Math.max(b[1], H) + 4 * H); }
        ctx.closePath();
        const eg = smooth(2e3, 150e3, alt);
        const gg = ctx.createLinearGradient(0, cy, 0, H + 40);
        gg.addColorStop(0, mix('#26323d', '#1b4a74', eg));
        gg.addColorStop(1, mix('#141b23', '#0d2440', eg));
        ctx.fillStyle = gg;
        ctx.fill();
        ctx.beginPath();
        pts.forEach(([x, y], j) => (j ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.strokeStyle = rgba('#9cc2e0', 0.35 + 0.3 * limbA);
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Launch complex & clouds (co-rotating with the ground)
      const w = omega(cfg);
      const siteAng = Math.PI / 2 - w * sim.t;
      const geom = PRESETS[presetKey].geom, diam = PRESETS[presetKey].diam, L = totalLen();
      const pxm = Math.max(sc, (inset ? 26 : 50) / L);
      for (const cl of CLOUDS) {
        const ang = siteAng - cl.off / R_E, rr = R_E + cl.alt;
        const [sx, sy] = toS(rr * Math.cos(ang), rr * Math.sin(ang));
        const rw = cl.w * sc, rh = cl.hh * sc;
        if (rw < 2 || sx < -rw || sx > W + rw || sy < -rh * 3 || sy > H + rh * 3) continue;
        const cg = ctx.createRadialGradient(sx, sy, 0, sx, sy, rw);
        const ca = cl.a * (1 - smooth(20e3, 60e3, alt));
        cg.addColorStop(0, `rgba(235,240,248,${ca})`); cg.addColorStop(1, 'rgba(235,240,248,0)');
        ctx.save(); ctx.translate(sx, sy); ctx.rotate(phr - ang); ctx.scale(1, rh / rw); ctx.translate(-sx, -sy);
        ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(sx, sy, rw, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      }
      if (L * sc > 6) {
        const [sx, sy] = toS(R_E * Math.cos(siteAng), R_E * Math.sin(siteAng));
        if (sx > -W && sx < 2 * W && sy > -H && sy < 2 * H) {
          ctx.save(); ctx.translate(sx, sy); ctx.rotate(phr - siteAng);
          const k = sc;
          ctx.fillStyle = '#2b3542'; ctx.fillRect(-L * 0.9 * k, -1.5 * k, L * 1.8 * k, 3 * k); // pad
          ctx.fillStyle = '#1b222c'; ctx.fillRect(-L * 0.5 * k, 0, L * k, 6 * k); // flame trench
          const tw = Math.max(diam * 1.6, L * 0.06) * k, tx = -diam * 1.8 * k - tw, th = L * 1.05 * k;
          ctx.strokeStyle = '#6b7482'; ctx.lineWidth = Math.max(0.6, 0.18 * diam * k);
          ctx.strokeRect(tx, -th, tw, th);
          ctx.beginPath();
          const n = Math.max(4, Math.floor(th / Math.max(tw, 4)));
          for (let i = 0; i < n; i++) { const y0 = -th + (i * th) / n, y1 = -th + ((i + 1) * th) / n; ctx.moveTo(tx, y0); ctx.lineTo(tx + tw, y1); ctx.moveTo(tx + tw, y0); ctx.lineTo(tx, y1); }
          ctx.stroke();
          ctx.fillStyle = C.danger; ctx.beginPath(); ctx.arc(tx + tw / 2, -th - 2, Math.max(1.5, tw * 0.15), 0, Math.PI * 2); ctx.fill();
          ctx.restore();
        }
      }

      // Spent stages
      for (const d of sim.debris) {
        const [sx, sy] = toS(d.x, d.y);
        if (sx < -200 || sx > W + 200 || sy < -200 || sy > H + 200) continue;
        const sdx = Math.cos(d.ang) * ex + Math.sin(d.ang) * ey, sdy = -(Math.cos(d.ang) * ux + Math.sin(d.ang) * uy);
        ctx.save(); ctx.translate(sx, sy); ctx.rotate(Math.atan2(sdx, -sdy));
        drawStage1(ctx, geom, diam, pxm, d.t < 1.2 ? 1 - d.t / 1.2 : 0);
        ctx.restore();
        if (inset) continue;
        ctx.font = `10px ${FONT}`; ctx.fillStyle = rgba(C.muted, 0.8);
        ctx.fillText(d.alive ? 'STAGE 1' : 'STAGE 1 · IMPACT', sx + 10, sy + 4);
      }

      // Particles
      const pts = particles;
      ctx.save();
      ctx.translate(cx, cy);
      for (const p of pts) if (p.kind === 'smoke') {
        const t = p.life / p.max;
        const r = p.size * pxm * (1 + t * 2.5);
        const a = (p.a ?? 1) * 0.28 * (1 - t) * (1 - t);
        if (a < 0.004) continue;
        ctx.fillStyle = `rgba(200,205,214,${a})`;
        ctx.beginPath(); ctx.arc(p.x * pxm, p.y * pxm, r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalCompositeOperation = 'lighter';
      for (const p of pts) if (p.kind === 'fire') {
        const t = p.life / p.max;
        const r = p.size * pxm * (0.6 + t * 1.6);
        const col = t < 0.25 ? '255,240,200' : t < 0.6 ? '245,181,68' : '239,107,107';
        ctx.fillStyle = `rgba(${col},${0.55 * (1 - t)})`;
        ctx.beginPath(); ctx.arc(p.x * pxm, p.y * pxm, r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.restore();

      // Rocket
      if (sim.status !== 'crashed') {
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(dispAng);
        if (sim.throttleActual > 0 && sim.status === 'flight') drawFlame(ctx, diam, L, pxm, sim.throttleActual, pressFrac(alt));
        drawRocket(ctx, geom, diam, pxm, sim.stage);
        ctx.restore();
      }

      // Altitude scale ticks on the left edge
      if (!inset) {
        ctx.font = `10px ${FONT}`;
        const span = Math.exp(camLog);
        const step = Math.pow(10, Math.floor(Math.log10(span / 3)));
        const base = Math.floor(alt / step) * step;
        ctx.strokeStyle = rgba(C.text, 0.25); ctx.fillStyle = rgba(C.text, 0.45);
        for (let k = -6; k <= 6; k++) {
          const a = base + k * step;
          if (a < 0) continue;
          const y = cy - (a - alt) * sc;
          if (y < 70 || y > H - 20) continue;
          ctx.beginPath(); ctx.moveTo(W - 14, y); ctx.lineTo(W - 4, y); ctx.stroke();
          ctx.textAlign = 'right'; ctx.fillText(fmtAlt(a), W - 18, y + 3); ctx.textAlign = 'left';
        }
      }
      ctx.restore();
    }

    function bodyGrad(c, x0, x1, light) {
      const g = ctx.createLinearGradient(x0, 0, x1, 0);
      g.addColorStop(0, light ? '#8f97a3' : '#1c2129');
      g.addColorStop(0.35, light ? '#f2f4f7' : '#3a414c');
      g.addColorStop(0.6, light ? '#dde1e7' : '#2c323b');
      g.addColorStop(1, light ? '#7d8591' : '#161a20');
      return g;
    }

    function drawStage1(c, geom, diam, k, glow = 0) {
      const d = Math.max(diam * k, 3.5), l1 = geom.l1 * k;
      c.fillStyle = '#3b424c'; // engine skirt
      c.beginPath(); c.moveTo(-d * 0.42, 0); c.lineTo(d * 0.42, 0); c.lineTo(d * 0.5, d * 0.32); c.lineTo(-d * 0.5, d * 0.32); c.closePath(); c.fill();
      c.fillStyle = bodyGrad(c, -d / 2, d / 2, true);
      c.fillRect(-d / 2, -l1, d, l1);
      c.fillStyle = bodyGrad(c, -d / 2, d / 2, false);
      c.fillRect(-d / 2, -l1, d, Math.max(1, l1 * 0.09)); // interstage
      c.fillStyle = 'rgba(20,24,30,0.35)';
      c.fillRect(-d / 2, -l1 * 0.62, d, Math.max(0.5, l1 * 0.012));
      if (PRESETS[presetKey].geom.l1 < 10) { // fins on the sounding rocket
        c.fillStyle = '#c24f4f';
        for (const s of [-1, 1]) { c.beginPath(); c.moveTo(s * d / 2, -l1 * 0.22); c.lineTo(s * d * 1.6, d * 0.1); c.lineTo(s * d / 2, 0); c.closePath(); c.fill(); }
      }
      if (glow > 0) { c.fillStyle = `rgba(245,181,68,${0.6 * glow})`; c.beginPath(); c.arc(0, d * 0.3, d * 0.6, 0, Math.PI * 2); c.fill(); }
    }

    function drawRocket(c, geom, diam, k, stageNo) {
      const d = Math.max(diam * k, 3.5);
      let y0 = 0;
      if (stageNo === 1) { drawStage1(c, geom, diam, k); y0 = -geom.l1 * k; }
      else { // stage-2 nozzle
        c.fillStyle = '#4a4f57';
        c.beginPath(); c.moveTo(-d * 0.12, 0); c.lineTo(d * 0.12, 0); c.lineTo(d * 0.4, d * 0.9); c.lineTo(-d * 0.4, d * 0.9); c.closePath(); c.fill();
        c.fillStyle = 'rgba(245,181,68,0.18)'; c.fill();
      }
      const l2 = geom.l2 * k, fr = geom.fair * k;
      c.fillStyle = bodyGrad(c, -d / 2, d / 2, true);
      c.fillRect(-d / 2, y0 - l2, d, l2);
      c.fillStyle = 'rgba(20,24,30,0.28)';
      c.fillRect(-d / 2, y0 - l2 * 0.5, d, Math.max(0.5, l2 * 0.02));
      // Fairing: slightly wider ogive
      const fd = d * 1.12, yb = y0 - l2;
      c.fillStyle = bodyGrad(c, -fd / 2, fd / 2, true);
      c.beginPath();
      c.moveTo(-fd / 2, yb); c.lineTo(-fd / 2, yb - fr * 0.45);
      c.quadraticCurveTo(-fd / 2, yb - fr * 0.95, 0, yb - fr);
      c.quadraticCurveTo(fd / 2, yb - fr * 0.95, fd / 2, yb - fr * 0.45);
      c.lineTo(fd / 2, yb); c.closePath(); c.fill();
      c.strokeStyle = 'rgba(20,24,30,0.35)'; c.lineWidth = Math.max(0.5, d * 0.03);
      c.beginPath(); c.moveTo(0, yb); c.lineTo(0, yb - fr * 0.97); c.stroke();
      c.fillStyle = C.accent; // mission stripe
      c.fillRect(-fd / 2, yb - fr * 0.18, fd, Math.max(0.8, fr * 0.04));
    }

    function drawFlame(c, diam, L, k, thr, pf) {
      const d = Math.max(diam * k, 3.5);
      const vac = 1 - pf;
      const len = L * k * (0.35 + 0.55 * vac) * (0.7 + 0.3 * thr) * (0.93 + Math.random() * 0.14);
      const wid = d * (0.8 + 3.2 * vac);
      const y0 = d * 0.3;
      c.save();
      c.globalCompositeOperation = 'lighter';
      const glow = c.createRadialGradient(0, y0 + len * 0.15, 0, 0, y0 + len * 0.15, len * 0.9);
      glow.addColorStop(0, `rgba(245,181,68,${0.35 * thr})`); glow.addColorStop(1, 'rgba(245,181,68,0)');
      c.fillStyle = glow; c.beginPath(); c.arc(0, y0 + len * 0.15, len * 0.9, 0, Math.PI * 2); c.fill();
      const g = c.createLinearGradient(0, y0, 0, y0 + len);
      g.addColorStop(0, 'rgba(255,250,230,0.95)');
      g.addColorStop(0.25, 'rgba(255,214,120,0.8)');
      g.addColorStop(0.6, 'rgba(245,140,60,0.35)');
      g.addColorStop(1, 'rgba(239,107,107,0)');
      c.fillStyle = g;
      c.beginPath();
      c.moveTo(-d * 0.4, y0);
      c.bezierCurveTo(-wid * 0.6, y0 + len * 0.3, -wid * 0.35, y0 + len * 0.8, 0, y0 + len);
      c.bezierCurveTo(wid * 0.35, y0 + len * 0.8, wid * 0.6, y0 + len * 0.3, d * 0.4, y0);
      c.closePath(); c.fill();
      // Mach diamonds at low altitude
      if (pf > 0.3) {
        c.fillStyle = `rgba(255,255,255,${0.35 * pf})`;
        for (let i = 1; i <= 4; i++) { const yy = y0 + len * 0.1 * i, r = d * 0.18 * (1 - i * 0.15); c.beginPath(); c.moveTo(0, yy - r * 1.5); c.lineTo(r, yy); c.lineTo(0, yy + r * 1.5); c.lineTo(-r, yy); c.closePath(); c.fill(); }
      }
      c.restore();
    }

    // ------------------------------------------------ orbit map view
    function drawMap(X, Y, W, H, inset, el) {
      ctx.save();
      ctx.beginPath(); ctx.rect(X, Y, W, H); ctx.clip();
      ctx.translate(X, Y);
      ctx.fillStyle = '#070a10'; ctx.fillRect(0, 0, W, H);
      for (let i = 0; i < STARS.length; i += inset ? 3 : 1) {
        const st = STARS[i];
        ctx.fillStyle = `rgba(230,236,255,${st.a * 0.5})`;
        ctx.fillRect(st.x * W, st.y * H, st.s, st.s);
      }
      const ms = (Math.min(W, H) / 2 - (inset ? 8 : 24)) / mapFit;
      const cx = W / 2, cy = H / 2;
      const toM = (x, y) => [cx + x * ms, cy - y * ms];

      // Predicted conic
      const bound = el.e < 1;
      const orbitCol = sim.inOrbit ? C.good : el.eps >= 0 ? C.accent : C.cyan;
      if (sim.status === 'flight' || sim.status === 'pad') {
        ctx.beginPath();
        const numax = bound ? Math.PI : Math.acos(-1 / el.e) - 0.02;
        const N = 360;
        let started = false;
        for (let i = 0; i <= N; i++) {
          const nu = -numax + (2 * numax * i) / N;
          const rr = el.p / (1 + el.e * Math.cos(nu));
          if (rr > 40 * R_E) { started = false; continue; }
          const th = el.argp + nu;
          const [mx, my] = toM(rr * Math.cos(th), rr * Math.sin(th));
          if (!started) { ctx.moveTo(mx, my); started = true; } else ctx.lineTo(mx, my);
        }
        ctx.strokeStyle = rgba(orbitCol, 0.85);
        ctx.lineWidth = inset ? 1 : 1.4;
        ctx.setLineDash(sim.inOrbit ? [] : [5, 4]);
        if (sim.status === 'flight') ctx.stroke();
        ctx.setLineDash([]);
      }

      // Earth
      const er = R_E * ms;
      const atm = ctx.createRadialGradient(cx, cy, er, cx, cy, er + Math.max(6, 120e3 * ms));
      atm.addColorStop(0, rgba(C.cyan, 0.35)); atm.addColorStop(1, rgba(C.cyan, 0));
      ctx.fillStyle = atm; ctx.beginPath(); ctx.arc(cx, cy, er + Math.max(6, 120e3 * ms), 0, Math.PI * 2); ctx.fill();
      const eg = ctx.createRadialGradient(cx - er * 0.35, cy - er * 0.35, er * 0.1, cx, cy, er);
      eg.addColorStop(0, '#1f4c78'); eg.addColorStop(1, '#0c1d31');
      ctx.fillStyle = eg; ctx.beginPath(); ctx.arc(cx, cy, er, 0, Math.PI * 2); ctx.fill();
      // Kármán line
      ctx.strokeStyle = rgba(C.muted, 0.25); ctx.setLineDash([2, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, (R_E + 100e3) * ms, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
      // Rotating meridian ticks + launch site
      const rot = omega(cfg) * sim.t;
      ctx.strokeStyle = rgba('#9cc2e0', 0.18);
      for (let i = 0; i < 24; i++) {
        const a = Math.PI / 2 - rot + (i * Math.PI) / 12;
        ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * er * 0.2, cy - Math.sin(a) * er * 0.2); ctx.lineTo(cx + Math.cos(a) * er, cy - Math.sin(a) * er); ctx.stroke();
      }
      for (const f of [0.4, 0.7]) { ctx.beginPath(); ctx.arc(cx, cy, er * f, 0, Math.PI * 2); ctx.stroke(); }
      {
        const a = Math.PI / 2 - rot;
        const [sx, sy] = toM(R_E * Math.cos(a), R_E * Math.sin(a));
        ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(sx, sy, inset ? 2 : 3, 0, Math.PI * 2); ctx.fill();
      }

      // Flown trail
      const col = (k) => (k === 1 ? C.accent : k === 2 ? C.cyan : C.muted);
      ctx.lineWidth = inset ? 1.2 : 2;
      ctx.lineCap = 'round';
      const tr = trail.concat([{ x: sim.x, y: sim.y, k: sim.throttleActual > 0 ? sim.stage : 0 }]);
      for (let i = 1; i < tr.length; i++) {
        const a = toM(tr[i - 1].x, tr[i - 1].y), b = toM(tr[i].x, tr[i].y);
        ctx.strokeStyle = col(tr[i].k);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      // Debris
      for (const d of sim.debris) {
        const [sx, sy] = toM(d.x, d.y);
        ctx.fillStyle = C.muted; ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
      }

      // Ap / Pe markers
      if (sim.status === 'flight' && sim.liftoff) {
        ctx.font = `${inset ? 9 : 11}px ${FONT}`;
        const mark = (rr, th, label, c) => {
          const [mx, my] = toM(rr * Math.cos(th), rr * Math.sin(th));
          ctx.fillStyle = c; ctx.beginPath(); ctx.arc(mx, my, inset ? 2.5 : 3.5, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = rgba(c, 0.5); ctx.beginPath(); ctx.arc(mx, my, inset ? 5 : 7, 0, Math.PI * 2); ctx.stroke();
          const dx = Math.cos(th), dy = -Math.sin(th);
          ctx.textAlign = dx < 0 ? 'right' : 'left';
          ctx.fillText(label, mx + dx * 12, my + dy * 12 + 4);
          ctx.textAlign = 'left';
        };
        if (bound && el.ra < 40 * R_E) mark(el.ra, el.argp + Math.PI, `Ap ${fmtKm(el.ra - R_E)}`, C.accent);
        if (el.rp > R_E) mark(el.rp, el.argp, `Pe ${fmtKm(el.rp - R_E)}`, C.cyan);
      }

      // Rocket marker + velocity vector
      if (sim.status !== 'crashed') {
        const [sx, sy] = toM(sim.x, sim.y);
        const v = Math.hypot(sim.vx, sim.vy) || 1;
        ctx.strokeStyle = C.text; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + (sim.vx / v) * (inset ? 12 : 22), sy - (sim.vy / v) * (inset ? 12 : 22)); ctx.stroke();
        const gl = ctx.createRadialGradient(sx, sy, 0, sx, sy, inset ? 8 : 12);
        gl.addColorStop(0, rgba(C.accent, 0.8)); gl.addColorStop(1, rgba(C.accent, 0));
        ctx.fillStyle = gl; ctx.beginPath(); ctx.arc(sx, sy, inset ? 8 : 12, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(sx, sy, 2.2, 0, Math.PI * 2); ctx.fill();
      } else {
        const [sx, sy] = toM(sim.x, sim.y);
        ctx.strokeStyle = C.danger; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(sx - 5, sy - 5); ctx.lineTo(sx + 5, sy + 5); ctx.moveTo(sx + 5, sy - 5); ctx.lineTo(sx - 5, sy + 5); ctx.stroke();
      }

      // Legend
      if (!inset) {
        ctx.font = `11px ${FONT}`;
        const items = [[C.accent, 'Stage 1 burn'], [C.cyan, 'Stage 2 burn'], [C.muted, 'Coast'], [orbitCol, sim.inOrbit ? 'Orbit' : el.eps >= 0 ? 'Escape conic' : 'Predicted conic']];
        let ly = H - 16 - items.length * 16;
        for (const [c, t] of items) { ctx.fillStyle = c; ctx.fillRect(14, ly - 4, 14, 2); ctx.fillStyle = C.muted; ctx.fillText(t, 34, ly); ly += 16; }
        const barM = niceBar(W * 0.18 / ms);
        ctx.strokeStyle = C.muted; ctx.beginPath(); ctx.moveTo(W - 20 - barM * ms, H - 18); ctx.lineTo(W - 20, H - 18); ctx.stroke();
        ctx.textAlign = 'right'; ctx.fillStyle = C.muted; ctx.fillText(fmtKm(barM), W - 20, H - 24); ctx.textAlign = 'left';
      }
      ctx.restore();
    }
    const niceBar = (m) => { const p = Math.pow(10, Math.floor(Math.log10(m))); const f = m / p; return (f >= 5 ? 5 : f >= 2 ? 2 : 1) * p; };

    // ------------------------------------------------ chart & HUD
    function drawChart(x, y, w, hh) {
      ctx.save();
      roundRect(ctx, x, y, w, hh, 8);
      ctx.fillStyle = 'rgba(11,14,20,0.72)'; ctx.fill();
      ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke();
      const px = x + 10, py = y + 24, pw = w - 20, ph = hh - 34;
      ctx.font = `10px ${FONT}`;
      const last = chart[chart.length - 1];
      ctx.fillStyle = C.cyan; ctx.fillText(`ALT ${last ? fmtAlt(last.alt) : '—'}`, x + 10, y + 15);
      ctx.fillStyle = C.accent; ctx.textAlign = 'right'; ctx.fillText(`VEL ${last ? fmtSpeed(last.v) : '—'}`, x + w - 10, y + 15); ctx.textAlign = 'left';
      ctx.strokeStyle = rgba(C.line, 1);
      for (let i = 0; i <= 2; i++) { const yy = py + (ph * i) / 2; ctx.beginPath(); ctx.moveTo(px, yy); ctx.lineTo(px + pw, yy); ctx.stroke(); }
      if (chart.length > 1) {
        const tMax = Math.max(60, last.t);
        let aMax = 1, vMax = 1;
        for (const c of chart) { aMax = Math.max(aMax, c.alt); vMax = Math.max(vMax, c.v); }
        for (const ev of sim.events) {
          if (!/MECO|SECO|Max-Q/.test(ev.text)) continue;
          const xx = px + (ev.t / tMax) * pw;
          ctx.strokeStyle = rgba(C.muted, 0.35); ctx.setLineDash([2, 3]);
          ctx.beginPath(); ctx.moveTo(xx, py); ctx.lineTo(xx, py + ph); ctx.stroke(); ctx.setLineDash([]);
        }
        const line = (key, max, c) => {
          ctx.beginPath();
          chart.forEach((p, i) => { const xx = px + (p.t / tMax) * pw, yy = py + ph - (Math.max(0, p[key]) / max) * ph; i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); });
          ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.stroke();
        };
        line('v', vMax, C.accent);
        line('alt', aMax, C.cyan);
        ctx.fillStyle = C.muted;
        ctx.fillText(`T+${fmtClock(tMax)}`, px + pw - 62, py + ph + 0);
      } else {
        ctx.fillStyle = C.muted; ctx.fillText('awaiting launch', px + pw / 2 - 45, py + ph / 2 + 3);
      }
      ctx.restore();
    }

    function drawHud(W, H, now, el) {
      const alt = altOf();
      const w = omega(cfg);
      const rvx = sim.vx - w * sim.y, rvy = sim.vy + w * sim.x;
      ctx.save();
      const bw = 212, bh = 104;
      roundRect(ctx, 12, 12, bw, bh, 8);
      ctx.fillStyle = 'rgba(11,14,20,0.72)'; ctx.fill(); ctx.strokeStyle = C.line; ctx.stroke();
      ctx.font = `500 18px ${FONT}`; ctx.fillStyle = C.text;
      ctx.fillText(`T+ ${fmtClock(sim.t)}`, 24, 38);
      ctx.font = `10px ${FONT}`;
      const warp = WARPS[warpIdx];
      ctx.fillStyle = paused ? C.accent : C.muted; ctx.textAlign = 'right';
      ctx.fillText(paused ? 'PAUSED' : warp > 1 ? `${warp}× WARP` : 'REAL TIME', 12 + bw - 12, 36); ctx.textAlign = 'left';
      const statusTxt = sim.status === 'pad' ? 'ON THE PAD' : sim.status === 'crashed' ? 'IMPACT' : sim.status === 'failed' ? 'SCRUB' : (sim.label || sim.phase).toUpperCase();
      const statusCol = sim.status === 'crashed' || sim.status === 'failed' ? C.danger : sim.inOrbit ? C.good : C.cyan;
      ctx.fillStyle = statusCol; ctx.fillText(`● ${statusTxt}`, 24, 56);
      ctx.fillStyle = C.muted; ctx.fillText('ALT', 24, 78); ctx.fillText('VEL', 118, 78);
      ctx.font = `500 14px ${FONT}`; ctx.fillStyle = C.text;
      ctx.fillText(fmtAlt(alt), 24, 96); ctx.fillText(fmtSpeed(Math.hypot(rvx, rvy)), 118, 96);
      if (opts.guidance === 'manual') {
        ctx.font = `10px ${FONT}`;
        ctx.fillStyle = sim.manualThrottle ? C.accent : C.muted;
        ctx.fillText(`PITCH ${(sim.heading * 180 / Math.PI).toFixed(0)}° · THR ${sim.manualThrottle ? 'ON' : 'OFF'}`, 24, 110);
      }
      // Propellant bars
      const by = 12 + bh + 8;
      roundRect(ctx, 12, by, bw, 38, 8);
      ctx.fillStyle = 'rgba(11,14,20,0.72)'; ctx.fill(); ctx.strokeStyle = C.line; ctx.stroke();
      ctx.font = `10px ${FONT}`;
      cfg.stages.forEach((st, i) => {
        const frac = st.prop > 0 ? sim.prop[i] / st.prop : 0;
        const yy = by + 13 + i * 14;
        ctx.fillStyle = C.muted; ctx.fillText(`S${i + 1}`, 22, yy + 3);
        const x0 = 44, ww = bw - 50;
        ctx.fillStyle = C.line; ctx.fillRect(x0, yy - 2, ww, 4);
        const gone = i === 0 && sim.stage > 1;
        ctx.fillStyle = gone ? rgba(C.muted, 0.4) : i === 0 ? C.accent : C.cyan;
        ctx.fillRect(x0, yy - 2, ww * frac, 4);
      });
      // Toasts
      ctx.textAlign = 'center';
      let ty = 26;
      for (const t of toasts) {
        const age = now - t.born;
        if (age > 4.5) continue;
        const a = Math.min(1, age * 4) * Math.min(1, (4.5 - age) / 0.8);
        ctx.font = `500 12px ${FONT}`;
        const tw = ctx.measureText(t.text).width + 26;
        const tx = W / 2;
        roundRect(ctx, tx - tw / 2, ty - 15, tw, 24, 12);
        ctx.fillStyle = `rgba(11,14,20,${0.8 * a})`; ctx.fill();
        const c = t.kind === 'good' ? C.good : t.kind === 'bad' ? C.danger : t.kind === 'warn' ? C.accent : C.text;
        ctx.strokeStyle = rgba(c, 0.6 * a); ctx.stroke();
        ctx.fillStyle = rgba(c, a); ctx.fillText(t.text, tx, ty + 1);
        ty += 30;
      }
      toasts = toasts.filter((t) => now - t.born < 4.5);
      // End-state banner
      if (sim.status === 'crashed' || sim.status === 'failed' || (sim.inOrbit && sim.cutoff)) {
        const good = sim.inOrbit && sim.status === 'flight';
        const title = good ? 'ORBIT ACHIEVED' : sim.status === 'failed' ? 'LAUNCH FAILURE' : sim.apogee > 100e3 && !PRESETS[presetKey].targetPe ? 'SUBORBITAL FLIGHT COMPLETE' : 'MISSION LOST';
        const sub = good ? `${fmtKm(el.rp - R_E)} × ${fmtKm(el.ra - R_E)} · e ${el.e.toFixed(4)} · period ${fmtClock(2 * Math.PI * Math.sqrt(el.a ** 3 / MU))}` : `apogee ${fmtAlt(sim.apogee)} · max-Q ${(sim.maxQ / 1000).toFixed(1)} kPa`;
        const c = good ? C.good : sim.status === 'crashed' && sim.apogee > 100e3 && !PRESETS[presetKey].targetPe ? C.cyan : C.danger;
        const yy = H * 0.2 + 20;
        ctx.font = `500 20px ${FONT}`; ctx.fillStyle = c;
        ctx.fillText(title, W / 2, yy);
        ctx.font = `11px ${FONT}`; ctx.fillStyle = C.muted; ctx.fillText(sub, W / 2, yy + 20);
      }
      ctx.restore();
    }

    function updateReadout(force) {
      const now = performance.now();
      if (!force && now - lastReadout < 90) return;
      lastReadout = now;
      const alt = altOf();
      const w = omega(cfg);
      const rvx = sim.vx - w * sim.y, rvy = sim.vy + w * sim.x;
      const el = elements(sim.x, sim.y, sim.vx, sim.vy);
      ro.set('T+', fmtClock(sim.t));
      ro.set('Phase', sim.status === 'pad' ? 'pad' : sim.status === 'flight' ? (sim.label || sim.phase) : sim.status);
      ro.set('Altitude', fmtAlt(alt));
      ro.set('Surface speed', fmtSpeed(Math.hypot(rvx, rvy)));
      ro.set('Orbital speed', fmtSpeed(Math.hypot(sim.vx, sim.vy)));
      ro.set('Mach', alt < ATMO_TOP ? sim.mach.toFixed(2) : '—');
      ro.set('Acceleration', `${sim.accel.toFixed(2)} g  (max ${sim.maxG.toFixed(1)})`);
      ro.set('Mass', fmtMass(mass(sim, cfg)));
      ro.set('Stage', `${sim.stage} / 2${sim.throttleActual > 0 ? ' · burning' : ''}`);
      ro.set('Propellant', cfg.stages.map((st, i) => `S${i + 1} ${Math.round((100 * sim.prop[i]) / (st.prop || 1))}%`).join(' · '));
      ro.set('Dyn. pressure', `${(sim.q / 1000).toFixed(2)} kPa`);
      ro.set('Max-Q', sim.maxQ > 0 ? `${(sim.maxQ / 1000).toFixed(1)} kPa @ T+${sim.maxQt.toFixed(0)}s` : '—');
      ro.set('Apoapsis', el.e >= 1 ? '∞ (escape)' : fmtKm(el.ra - R_E));
      ro.set('Periapsis', el.rp < R_E ? `${fmtKm(el.rp - R_E)} (sub)` : fmtKm(el.rp - R_E));
      ro.set('Eccentricity', el.e.toFixed(4));
    }

    // ------------------------------------------------ frame
    function frame(dtReal, now) {
      const { w: W, h: H } = size;
      if (W < 2 || H < 2) return;
      advance(dtReal);
      processEvents(now);
      emit(dtReal);
      const el = elements(sim.x, sim.y, sim.vx, sim.vy);

      // Camera smoothing
      camLog += (Math.log(spanTarget()) - camLog) * (1 - Math.exp(-dtReal * 2.5));
      const { ux, uy, ex, ey } = basis(sim.x, sim.y);
      let dx = sim.dirx, dy = sim.diry;
      if (!sim.liftoff) { dx = ux; dy = uy; }
      const sdx = dx * ex + dy * ey, sdy = -(dx * ux + dy * uy);
      const target = Math.atan2(sdx, -sdy);
      let dA = target - dispAng;
      dA = Math.atan2(Math.sin(dA), Math.cos(dA));
      dispAng += dA * (1 - Math.exp(-dtReal * 6));

      // Map fit smoothing
      const r = Math.hypot(sim.x, sim.y);
      let want = Math.max(R_E * 1.15, r * 1.12);
      if (el.e < 1 && el.ra < 12 * R_E && sim.liftoff) want = Math.max(want, el.ra * 1.08);
      fitSmooth += (want - fitSmooth) * (1 - Math.exp(-dtReal * 2));

      ctx.clearRect(0, 0, W, H);
      const small = W < 620;
      const iw = Math.round(small ? W * 0.42 : clamp(W * 0.3, 200, 320));
      const ih = Math.round(iw * 0.78);
      insetRect = { x: W - iw - 12, y: H - ih - 12, w: iw, h: ih };

      mapFit = fitSmooth;
      if (mapMain) drawMap(0, 0, W, H, false, el); else drawClose(0, 0, W, H, false);
      mapFit = fitSmooth;
      // Inset
      ctx.save();
      roundRect(ctx, insetRect.x, insetRect.y, iw, ih, 10);
      ctx.clip();
      if (mapMain) drawClose(insetRect.x, insetRect.y, iw, ih, true); else drawMap(insetRect.x, insetRect.y, iw, ih, true, el);
      ctx.restore();
      roundRect(ctx, insetRect.x + 0.5, insetRect.y + 0.5, iw - 1, ih - 1, 10);
      ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke();
      ctx.font = `10px ${FONT}`; ctx.fillStyle = C.muted;
      ctx.fillText(mapMain ? 'CAMERA' : 'ORBIT MAP', insetRect.x + 10, insetRect.y + 16);

      const cw = small ? Math.min(W - 24, 240) : 250, ch = 112;
      if (small) drawChart(12, H - ih - 24 - ch, cw, ch);
      else drawChart(W - cw - 12, 12, cw, ch);
      drawHud(W, H, now, el);
      updateReadout(false);
    }
    let fitSmooth = 1.2 * R_E;

    applyPreset();
    const stopLoop = loop(frame);

    return () => {
      stopLoop();
      cv.destroy();
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
      canvas.removeEventListener('click', onCanvasClick);
      canvas.removeEventListener('mousemove', onCanvasMove);
      panel.removeEventListener('click', onPanelClick);
    };
  },
};
