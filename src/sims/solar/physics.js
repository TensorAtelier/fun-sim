// N-body gravity in AU / days / solar masses. Massive bodies attract each other (O(N^2));
// test particles feel the massive bodies but exert nothing (O(N*M)).
// Integrator: kick-drift-kick leapfrog with a step size bounded by the shortest pairwise
// dynamical timescale, so close encounters are resolved and quiet orbits run fast.

export const G0 = 2.959122082855911e-4; // AU^3 / (Msun day^2)
export const KM_PER_AU = 1.495978707e8;
export const KMS_PER_AUD = KM_PER_AU / 86400;
export const M_EARTH = 3.00348959632e-6;
export const M_JUP = 9.547919e-4;
export const R_SUN_KM = 695700;
export const TRAIL_LEN = 480;

const DEG = Math.PI / 180;
let nextId = 1;

export function starRadiusKm(m) { return R_SUN_KM * Math.pow(Math.max(m, 0.05), 0.8); }
export function planetRadiusKm(m) {
  const me = m / M_EARTH;
  if (me <= 2) return 6371 * Math.pow(Math.max(me, 1e-4), 0.28);
  return Math.min(7730 * Math.pow(me / 2, 0.59), 72000);
}

export function collisionRadius(b) {
  if (b.kind === 'bh') return 0.012 * Math.cbrt(Math.max(b.m, 0.01) / 10);
  if (b.kind === 'star') return (b.radiusKm / KM_PER_AU) * 2;
  return Math.max((b.radiusKm / KM_PER_AU) * 20, 2e-4); // "physical-ish": 20x real radius
}

// Classical elements (radians) -> heliocentric state.
export function elementsToState({ a, e, i, Omega, omega, M }, mu) {
  let E = e < 0.8 ? M : Math.PI;
  for (let k = 0; k < 30; k++) {
    const dE = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    E -= dE;
    if (Math.abs(dE) < 1e-12) break;
  }
  const cE = Math.cos(E), sE = Math.sin(E), q = Math.sqrt(1 - e * e);
  const xp = a * (cE - e), yp = a * q * sE;
  const n = Math.sqrt(mu / (a * a * a));
  const Edot = n / (1 - e * cE);
  const vxp = -a * sE * Edot, vyp = a * q * cE * Edot;
  const cO = Math.cos(Omega), sO = Math.sin(Omega), cw = Math.cos(omega), sw = Math.sin(omega);
  const ci = Math.cos(i), si = Math.sin(i);
  const r11 = cO * cw - sO * sw * ci, r12 = -cO * sw - sO * cw * ci;
  const r21 = sO * cw + cO * sw * ci, r22 = -sO * sw + cO * cw * ci;
  const r31 = sw * si, r32 = cw * si;
  return {
    pos: [r11 * xp + r12 * yp, r21 * xp + r22 * yp, r31 * xp + r32 * yp],
    vel: [r11 * vxp + r12 * vyp, r21 * vxp + r22 * vyp, r31 * vxp + r32 * vyp],
  };
}

// J2000 mean elements (a AU, e, i deg, mean longitude L, long. of perihelion w~, node Omega),
// masses in Msun, radii km. Approximate but real.
export const PLANETS = [
  { name: 'Mercury', a: 0.38710, e: 0.20563, i: 7.005, L: 252.25, lp: 77.46, O: 48.33, mass: 1.6601e-7, R: 2440, color: '#b3aca3', style: 'mercury', tilt: 0.03 },
  { name: 'Venus', a: 0.72333, e: 0.00677, i: 3.39, L: 181.98, lp: 131.60, O: 76.68, mass: 2.4478e-6, R: 6052, color: '#ebd3a0', style: 'venus', tilt: 3.1 },
  { name: 'Earth', a: 1.00000, e: 0.01671, i: 0.00, L: 100.46, lp: 102.94, O: 0, mass: 3.0035e-6, R: 6371, color: '#5ec8e5', style: 'earth', tilt: 0.41 },
  { name: 'Mars', a: 1.52371, e: 0.09340, i: 1.85, L: 355.45, lp: 336.04, O: 49.56, mass: 3.2272e-7, R: 3390, color: '#e07a4b', style: 'mars', tilt: 0.44 },
  { name: 'Jupiter', a: 5.20289, e: 0.04839, i: 1.30, L: 34.40, lp: 14.73, O: 100.47, mass: 9.5479e-4, R: 69911, color: '#dcb88f', style: 'jupiter', tilt: 0.05 },
  { name: 'Saturn', a: 9.53668, e: 0.05386, i: 2.49, L: 49.95, lp: 92.60, O: 113.66, mass: 2.8589e-4, R: 58232, color: '#ead39a', style: 'saturn', tilt: 0.47, ring: true },
  { name: 'Uranus', a: 19.18916, e: 0.04726, i: 0.77, L: 313.24, lp: 170.95, O: 74.02, mass: 4.3662e-5, R: 25362, color: '#a3e3e8', style: 'uranus', tilt: 1.71 },
  { name: 'Neptune', a: 30.06992, e: 0.00859, i: 1.77, L: 304.88, lp: 44.96, O: 131.78, mass: 5.1514e-5, R: 24622, color: '#6b8cff', style: 'neptune', tilt: 0.49 },
  { name: 'Pluto', a: 39.482, e: 0.2488, i: 17.14, L: 238.93, lp: 224.07, O: 110.30, mass: 6.58e-9, R: 1188, color: '#cfae8c', style: 'pluto', tilt: 2.1 },
];

export function makeBody(o) {
  const kind = o.kind || 'planet';
  const [x, y, z] = o.pos || [0, 0, 0];
  const [vx, vy, vz] = o.vel || [0, 0, 0];
  const radiusKm = o.radiusKm ?? (kind === 'star' ? starRadiusKm(o.mass) : kind === 'bh' ? 0 : planetRadiusKm(o.mass));
  return {
    id: nextId++, name: o.name, kind, baseMass: o.mass, mult: 1, isSun: !!o.isSun, m: o.mass,
    x, y, z, vx, vy, vz, ax: 0, ay: 0, az: 0, radiusKm,
    color: o.color || '#cccccc', style: o.style || (kind === 'star' ? 'star' : 'rocky'),
    ring: !!o.ring, tilt: o.tilt || 0, seed: o.seed ?? Math.floor(Math.random() * 1e9),
    primary: null, view: null,
    trail: { buf: new Float32Array(TRAIL_LEN * 3), head: 0, count: 0, lx: x, ly: y, lz: z, th2: 1e-4 },
  };
}

export class NBody {
  constructor() {
    this.bodies = [];
    this.t = 0;
    this.gMult = 1;
    this.sunMult = 1;
    this.eta = 0.02;   // fraction of the shortest dynamical time per step
    this.hMax = 1;     // days
    this.tauMin = Infinity;
    this.events = [];
    this.frameBody = null;
    const cap = 8000;
    this.tp = { n: 0, cap, x: new Float64Array(cap * 3), v: new Float64Array(cap * 3), a: new Float64Array(cap * 3), c: new Float32Array(cap * 3) };
  }
  get G() { return G0 * this.gMult; }

  effMass(b) { return b.baseMass * b.mult * (b.isSun ? this.sunMult : 1); }
  refreshMasses() { for (const b of this.bodies) b.m = this.effMass(b); this.computeAccel(); }

  clear() { this.bodies.length = 0; this.tp.n = 0; this.t = 0; this.events.length = 0; }

  add(b) {
    b.m = this.effMass(b);
    b._cr = collisionRadius(b);
    const t = b.trail; t.head = 0; t.count = 0; t.lx = Infinity;
    this.bodies.push(b);
  }
  remove(b) {
    if (this.frameBody === b) { this.frameBody = null; this.clearTrails(); }
    const i = this.bodies.indexOf(b);
    if (i >= 0) this.bodies.splice(i, 1);
  }

  addParticle(x, y, z, vx, vy, vz, r, g, bl) {
    const tp = this.tp;
    if (tp.n >= tp.cap) return false;
    const k = tp.n * 3;
    tp.x[k] = x; tp.x[k + 1] = y; tp.x[k + 2] = z;
    tp.v[k] = vx; tp.v[k + 1] = vy; tp.v[k + 2] = vz;
    tp.a[k] = tp.a[k + 1] = tp.a[k + 2] = 0;
    tp.c[k] = r; tp.c[k + 1] = g; tp.c[k + 2] = bl;
    tp.n++;
    return true;
  }
  removeParticle(i) {
    const tp = this.tp, last = (tp.n - 1) * 3, k = i * 3;
    for (let d = 0; d < 3; d++) {
      tp.x[k + d] = tp.x[last + d]; tp.v[k + d] = tp.v[last + d];
      tp.a[k + d] = tp.a[last + d]; tp.c[k + d] = tp.c[last + d];
    }
    tp.n--;
  }

  barycentre() {
    let M = 0, x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0;
    for (const b of this.bodies) {
      M += b.m; x += b.m * b.x; y += b.m * b.y; z += b.m * b.z;
      vx += b.m * b.vx; vy += b.m * b.vy; vz += b.m * b.vz;
    }
    if (M > 0) { x /= M; y /= M; z /= M; vx /= M; vy /= M; vz /= M; }
    return { M, x, y, z, vx, vy, vz };
  }

  // Shift everything so the barycentre sits at the origin, at rest.
  zeroMomentum() {
    const c = this.barycentre();
    for (const b of this.bodies) {
      b.x -= c.x; b.y -= c.y; b.z -= c.z; b.vx -= c.vx; b.vy -= c.vy; b.vz -= c.vz;
    }
    const tp = this.tp;
    for (let k = 0; k < tp.n * 3; k += 3) {
      tp.x[k] -= c.x; tp.x[k + 1] -= c.y; tp.x[k + 2] -= c.z;
      tp.v[k] -= c.vx; tp.v[k + 1] -= c.vy; tp.v[k + 2] -= c.vz;
    }
  }

  // Add a uniform velocity to every existing body/particle (momentum balancing on insert).
  boostAll(dvx, dvy, dvz) {
    for (const b of this.bodies) { b.vx += dvx; b.vy += dvy; b.vz += dvz; }
    const tp = this.tp;
    for (let k = 0; k < tp.n * 3; k += 3) { tp.v[k] += dvx; tp.v[k + 1] += dvy; tp.v[k + 2] += dvz; }
  }

  computeAccel() {
    const B = this.bodies, n = B.length, Gm = this.G;
    for (let i = 0; i < n; i++) { const b = B[i]; b.ax = 0; b.ay = 0; b.az = 0; }
    let tau2 = Infinity;
    for (let i = 0; i < n; i++) {
      const bi = B[i];
      for (let j = i + 1; j < n; j++) {
        const bj = B[j];
        const dx = bj.x - bi.x, dy = bj.y - bi.y, dz = bj.z - bi.z;
        const r2 = dx * dx + dy * dy + dz * dz + 1e-10;
        const r = Math.sqrt(r2);
        const inv3 = Gm / (r2 * r);
        const fi = bj.m * inv3, fj = bi.m * inv3;
        bi.ax += fi * dx; bi.ay += fi * dy; bi.az += fi * dz;
        bj.ax -= fj * dx; bj.ay -= fj * dy; bj.az -= fj * dz;
        const mu = Gm * (bi.m + bj.m);
        if (mu > 0) { const t = (r2 * r) / mu; if (t < tau2) tau2 = t; }
        const dvx = bj.vx - bi.vx, dvy = bj.vy - bi.vy, dvz = bj.vz - bi.vz;
        const v2 = dvx * dvx + dvy * dvy + dvz * dvz;
        if (v2 > 0) { const t = r2 / v2; if (t < tau2) tau2 = t; }
      }
    }
    this.tauMin = Math.sqrt(tau2);

    // Test particles
    const tp = this.tp, X = tp.x, A = tp.a;
    let dead = null;
    for (let p = 0; p < tp.n; p++) {
      const k = p * 3;
      const px = X[k], py = X[k + 1], pz = X[k + 2];
      let ax = 0, ay = 0, az = 0, kill = px * px + py * py + pz * pz > 1e6;
      for (let i = 0; i < n; i++) {
        const b = B[i];
        const dx = b.x - px, dy = b.y - py, dz = b.z - pz;
        const d2 = dx * dx + dy * dy + dz * dz;
        const cr = b.kind === 'planet' ? b._cr || 0 : (b._cr || 0) * 1.5;
        if (d2 < cr * cr) { kill = true; break; }
        const r2 = d2 + 1e-5;
        const f = (Gm * b.m) / (r2 * Math.sqrt(r2));
        ax += f * dx; ay += f * dy; az += f * dz;
      }
      A[k] = ax; A[k + 1] = ay; A[k + 2] = az;
      if (kill) (dead ||= []).push(p);
    }
    if (dead) for (let q = dead.length - 1; q >= 0; q--) this.removeParticle(dead[q]);
  }

  updateRadii() { for (const b of this.bodies) b._cr = collisionRadius(b); }

  kick(h) {
    for (const b of this.bodies) { b.vx += b.ax * h; b.vy += b.ay * h; b.vz += b.az * h; }
    const tp = this.tp, V = tp.v, A = tp.a, N = tp.n * 3;
    for (let k = 0; k < N; k++) V[k] += A[k] * h;
  }
  drift(h) {
    for (const b of this.bodies) { b.x += b.vx * h; b.y += b.vy * h; b.z += b.vz * h; }
    const tp = this.tp, X = tp.x, V = tp.v, N = tp.n * 3;
    for (let k = 0; k < N; k++) X[k] += V[k] * h;
  }

  step(h) {
    this.kick(h / 2);
    this.drift(h);
    this.computeAccel();
    this.kick(h / 2);
    this.t += h;
    if (this.checkCollisions()) { this.updateRadii(); this.computeAccel(); }
    this.recordTrails();
  }

  // Advance by T days; bounded by maxSteps and a wall-clock deadline (ms). Returns days done.
  advance(T, maxSteps, deadline) {
    this.updateRadii();
    let done = 0, steps = 0;
    while (done < T - 1e-12 && steps < maxSteps) {
      const h = Math.min(T - done, this.hMax, this.eta * this.tauMin);
      this.step(h);
      done += h; steps++;
      if ((steps & 15) === 0 && performance.now() > deadline) break;
    }
    return { done, steps };
  }

  checkCollisions() {
    const B = this.bodies;
    let any = false;
    outer: for (;;) {
      for (let i = 0; i < B.length; i++) {
        for (let j = i + 1; j < B.length; j++) {
          const a = B[i], b = B[j];
          const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
          const rr = a._cr + b._cr;
          if (dx * dx + dy * dy + dz * dz < rr * rr) { this.merge(a, b); any = true; continue outer; }
        }
      }
      break;
    }
    return any;
  }

  merge(a, b) {
    const rank = { planet: 0, star: 1, bh: 2 };
    const s = a.m >= b.m ? a : b, o = s === a ? b : a;
    const M = a.m + b.m;
    const x = (a.m * a.x + b.m * b.x) / M, y = (a.m * a.y + b.m * b.y) / M, z = (a.m * a.z + b.m * b.z) / M;
    const vx = (a.m * a.vx + b.m * b.vx) / M, vy = (a.m * a.vy + b.m * b.vy) / M, vz = (a.m * a.vz + b.m * b.vz) / M;
    const oldKind = s.kind;
    const kind = rank[a.kind] >= rank[b.kind] ? a.kind : b.kind;
    s.kind = kind;
    if (kind === 'planet') s.radiusKm = Math.cbrt(a.radiusKm ** 3 + b.radiusKm ** 3);
    else if (kind === 'star') s.radiusKm = Math.max(starRadiusKm(M), s.radiusKm);
    else s.radiusKm = 0;
    s.baseMass = M / (s.isSun ? this.sunMult : 1);
    s.mult = 1;
    s.m = M;
    s.x = x; s.y = y; s.z = z; s.vx = vx; s.vy = vy; s.vz = vz;
    if (kind !== oldKind && kind !== 'planet') { s.style = kind === 'bh' ? 'bh' : 'star'; s.ring = false; }
    this.remove(o);
    this.events.push({ type: 'merge', survivor: s, absorbed: o, kindChanged: kind !== oldKind, x, y, z });
  }

  // Trails are stored relative to the frame reference (frameBody, else the barycentre),
  // so a heliocentric view shows heliocentric orbits even when the Sun is being dragged around.
  clearTrails() { for (const b of this.bodies) { b.trail.count = 0; b.trail.head = 0; b.trail.lx = Infinity; } }
  recordTrails() {
    const f = this.frameBody || this.barycentre();
    const fx = f.x, fy = f.y, fz = f.z;
    for (const b of this.bodies) {
      const t = b.trail;
      const x = b.x - fx, y = b.y - fy, z = b.z - fz;
      const dx = x - t.lx, dy = y - t.ly, dz = z - t.lz;
      if (dx * dx + dy * dy + dz * dz < t.th2) continue;
      const k = t.head * 3;
      t.buf[k] = x; t.buf[k + 1] = y; t.buf[k + 2] = z;
      t.head = (t.head + 1) % TRAIL_LEN;
      if (t.count < TRAIL_LEN) t.count++;
      t.lx = x; t.ly = y; t.lz = z;
    }
  }

  // Strongest attractor for each body: used for trail sampling, period estimates.
  updatePrimaries() {
    const B = this.bodies;
    for (const b of B) {
      let best = null, bestF = 0, bestD = 1;
      for (const o of B) {
        if (o === b) continue;
        const dx = o.x - b.x, dy = o.y - b.y, dz = o.z - b.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        const f = o.m / d2;
        if (f > bestF) { bestF = f; best = o; bestD = Math.sqrt(d2); }
      }
      b.primary = best;
      const th = 0.012 * Math.max(best ? bestD : 1, 0.02);
      b.trail.th2 = th * th;
    }
  }
}

// ---------- Presets ----------

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sunBody() {
  return makeBody({ name: 'Sun', kind: 'star', mass: 1, isSun: true, color: '#ffc861', style: 'sun', radiusKm: R_SUN_KM });
}

function addPlanets(sim, names) {
  for (const p of PLANETS) {
    if (!names.includes(p.name)) continue;
    const st = elementsToState({
      a: p.a, e: p.e, i: p.i * DEG, Omega: p.O * DEG, omega: (p.lp - p.O) * DEG, M: (p.L - p.lp) * DEG,
    }, G0 * (1 + p.mass));
    sim.add(makeBody({ name: p.name, mass: p.mass, radiusKm: p.R, pos: st.pos, vel: st.vel, color: p.color, style: p.style, tilt: p.tilt, ring: p.ring }));
  }
}

function addBelt(sim, n, rand, around) {
  for (let k = 0; k < n; k++) {
    const a = 2.1 + rand() * 1.2;
    const st = elementsToState({
      a, e: rand() * 0.12, i: rand() * 9 * DEG, Omega: rand() * 2 * Math.PI, omega: rand() * 2 * Math.PI, M: rand() * 2 * Math.PI,
    }, G0 * around.m);
    const s = 0.55 + rand() * 0.35;
    sim.addParticle(st.pos[0] + around.x, st.pos[1] + around.y, st.pos[2] + around.z,
      st.vel[0] + around.vx, st.vel[1] + around.vy, st.vel[2] + around.vz, 0.85 * s, 0.76 * s, 0.62 * s);
  }
}

const ALL = PLANETS.map((p) => p.name);

export const PRESETS = [
  ['solar', 'Solar System'],
  ['inner', 'Inner planets only'],
  ['binary', 'Binary star with planets'],
  ['flyby', 'Rogue star flyby'],
  ['fig8', 'Figure-8 three-body'],
];

// Populates sim; returns view hints { speed (days/s), compress, extent (AU) }.
export function buildPreset(sim, id) {
  sim.clear();
  const rand = rng(1234);
  if (id === 'solar' || id === 'flyby') {
    const sun = sunBody();
    sim.add(sun);
    addPlanets(sim, ALL);
    sim.zeroMomentum();
    addBelt(sim, 420, rand, sun);
    if (id === 'flyby') {
      // 0.8 Msun star arriving at ~20 km/s with a 10 AU impact parameter (periapsis ~7 AU).
      sim.add(makeBody({ name: 'Rogue star', kind: 'star', mass: 0.8, color: '#ff8a5c', pos: [-52, 10, 4], vel: [0.0115, 0, -0.0008] }));
      return { speed: 150, compress: true, extent: 34 };
    }
    return { speed: 20, compress: true, extent: 34 };
  }
  if (id === 'inner') {
    sim.add(sunBody());
    addPlanets(sim, ['Mercury', 'Venus', 'Earth', 'Mars']);
    sim.zeroMomentum();
    return { speed: 8, compress: false, extent: 1.7 };
  }
  if (id === 'binary') {
    const m1 = 1.0, m2 = 0.6, M = m1 + m2, ab = 0.5;
    const vrel = Math.sqrt((G0 * M) / ab);
    sim.add(makeBody({ name: 'Star A', kind: 'star', mass: m1, isSun: true, color: '#ffc861', style: 'sun', pos: [ab * m2 / M, 0, 0], vel: [0, vrel * m2 / M, 0] }));
    sim.add(makeBody({ name: 'Star B', kind: 'star', mass: m2, color: '#ff9a5c', pos: [-ab * m1 / M, 0, 0], vel: [0, -vrel * m1 / M, 0] }));
    const circ = (name, r, mass, extra) => {
      const v = Math.sqrt((G0 * M) / r), th = rand() * Math.PI * 2;
      sim.add(makeBody({ name, mass, pos: [r * Math.cos(th), r * Math.sin(th), 0], vel: [-v * Math.sin(th), v * Math.cos(th), 0], ...extra }));
    };
    circ('Planet b', 2.2, 0.5 * M_JUP, { color: '#e0b48a', style: 'gas', seed: 7 });
    circ('Planet c', 3.6, 20 * M_EARTH, { color: '#7fd6e8', style: 'ice', seed: 11 });
    for (let k = 0; k < 450; k++) {
      const r = 0.9 + rand() * 3.8, th = rand() * Math.PI * 2, v = Math.sqrt((G0 * M) / r);
      const z = (rand() - 0.5) * 0.04 * r;
      const s = 0.5 + rand() * 0.4;
      sim.addParticle(r * Math.cos(th), r * Math.sin(th), z, -v * Math.sin(th), v * Math.cos(th), 0, 0.55 * s, 0.8 * s, 0.9 * s);
    }
    sim.zeroMomentum();
    return { speed: 12, compress: false, extent: 4.5, frame: 'bary' };
  }
  if (id === 'fig8') {
    // Chenciner-Montgomery (2000), G = m = 1 units; scale length 1 AU, mass 1 Msun.
    const T = Math.sqrt(1 / G0); // time unit in days
    const x1 = [0.97000436, -0.24308753], v3 = [-0.93240737, -0.86473146];
    const u = 1 / T;
    const mk = (name, color, p, v) => sim.add(makeBody({ name, kind: 'star', mass: 1, color, pos: [p[0], p[1], 0], vel: [v[0] * u, v[1] * u, 0] }));
    mk('Alpha', '#ffc861', x1, [-v3[0] / 2, -v3[1] / 2]);
    mk('Beta', '#8fd4ff', [-x1[0], -x1[1]], [-v3[0] / 2, -v3[1] / 2]);
    mk('Gamma', '#ffe7cf', [0, 0], v3);
    return { speed: 30, compress: false, extent: 1.3, frame: 'bary' };
  }
  return buildPreset(sim, 'solar');
}
