// Procedural canvas textures: planets (sampled from 3D noise on the sphere, so no seams),
// star granulation, glows, Saturn's rings and a black-hole accretion disk.
import * as THREE from 'three';

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeNoise(seed) {
  const r = rng(seed);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  for (let i = 255; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
  const perm = new Uint16Array(512);
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  const val = new Float32Array(256);
  for (let i = 0; i < 256; i++) val[i] = r();
  const L = (x, y, z) => val[perm[perm[perm[x & 255] + (y & 255)] + (z & 255)]];
  const fade = (t) => t * t * (3 - 2 * t);
  const lerp = (a, b, t) => a + (b - a) * t;
  function noise(x, y, z) {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
    const u = fade(x - xi), v = fade(y - yi), w = fade(z - zi);
    return lerp(
      lerp(lerp(L(xi, yi, zi), L(xi + 1, yi, zi), u), lerp(L(xi, yi + 1, zi), L(xi + 1, yi + 1, zi), u), v),
      lerp(lerp(L(xi, yi, zi + 1), L(xi + 1, yi, zi + 1), u), lerp(L(xi, yi + 1, zi + 1), L(xi + 1, yi + 1, zi + 1), u), v),
      w);
  }
  function fbm(x, y, z, oct = 5) {
    let s = 0, a = 0.5, f = 1, n = 0;
    for (let o = 0; o < oct; o++) { s += a * noise(x * f + o * 17.3, y * f, z * f); n += a; a *= 0.5; f *= 2.03; }
    return s / n;
  }
  return { noise, fbm };
}

// sRGB 0..255 triplets (THREE.Color stores linear values, so go through the hex string).
const hexArr = (hs) => { const n = parseInt(hs, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
const hex = (s) => hexArr(new THREE.Color(s).getHexString());
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const sstep = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
function mixTo(out, a, b, t) { out[0] = a[0] + (b[0] - a[0]) * t; out[1] = a[1] + (b[1] - a[1]) * t; out[2] = a[2] + (b[2] - a[2]) * t; return out; }
function palette(out, pal, t) {
  t = ((t % 1) + 1) % 1;
  const f = t * pal.length, i = Math.floor(f), k = f - i;
  const s = k * k * (3 - 2 * k);
  return mixTo(out, pal[i % pal.length], pal[(i + 1) % pal.length], s);
}

function canvasTex(c) {
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function paintSphere(w, h, fn) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(w, h);
  const d = img.data, out = [0, 0, 0];
  for (let j = 0; j < h; j++) {
    const lat = (0.5 - (j + 0.5) / h) * Math.PI;
    const cl = Math.cos(lat), sl = Math.sin(lat);
    for (let i = 0; i < w; i++) {
      const lon = ((i + 0.5) / w) * Math.PI * 2;
      fn(cl * Math.cos(lon), sl, cl * Math.sin(lon), lat, lon, out);
      const k = (j * w + i) * 4;
      d[k] = out[0]; d[k + 1] = out[1]; d[k + 2] = out[2]; d[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvasTex(c);
}

const angDiff = (a, b) => { let d = a - b; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d; };

function spot(lat, lon, lat0, lon0, rlat, rlon) {
  const a = (lat - lat0) / rlat, b = angDiff(lon, lon0) / rlon;
  return a * a + b * b;
}

function gasGiant(pal, seed, { bands = 5, turb = 1, grain = 12, spots = [] } = {}) {
  const N = makeNoise(seed);
  const tmp = [0, 0, 0];
  return paintSphere(512, 256, (x, y, z, lat, lon, out) => {
    const w = N.fbm(x * 2.2, y * 7, z * 2.2, 5) - 0.5;
    let t = y * bands + w * turb + 0.18 * Math.sin(y * bands * 7.1);
    palette(out, pal, t * 0.5);
    const g = (N.noise(x * 24, y * 70, z * 24) - 0.5) * grain;
    out[0] += g; out[1] += g; out[2] += g;
    for (const s of spots) {
      const e = spot(lat, lon, s.lat, s.lon, s.rlat, s.rlon);
      if (e < 1) {
        const k = Math.pow(1 - e, 0.7) * s.k;
        mixTo(out, out, mixTo(tmp, s.edge, s.color, sstep(0.2, 0.9, 1 - e)), k);
      }
    }
  });
}

function variants(color, dl = 0.12, ds = 0.05) {
  const c = new THREE.Color(color), hsl = {};
  c.getHSL(hsl, THREE.SRGBColorSpace);
  const mk = (dh, dS, dL) => { const o = new THREE.Color(); o.setHSL((hsl.h + dh + 1) % 1, clamp01(hsl.s + dS), clamp01(hsl.l + dL), THREE.SRGBColorSpace); return hexArr(o.getHexString()); };
  return [mk(0, 0, dl), mk(0.01, ds, -dl * 0.5), mk(-0.01, -ds, dl * 0.4), mk(0.02, ds, -dl), mk(0, 0, dl * 0.8)];
}

export function planetTexture(style, seed, color) {
  const N = makeNoise(seed);
  const tmp = [0, 0, 0];
  switch (style) {
    case 'mercury': {
      const a = hex('#4a4540'), b = hex('#b8b1a8');
      return paintSphere(384, 192, (x, y, z, la, lo, out) => {
        const n = N.fbm(x * 3, y * 3, z * 3, 6) * 0.75 + N.noise(x * 18, y * 18, z * 18) * 0.25;
        mixTo(out, a, b, sstep(0.3, 0.72, n));
      });
    }
    case 'venus': {
      const a = hex('#a7793f'), b = hex('#f4e2b6');
      return paintSphere(384, 192, (x, y, z, la, lo, out) => {
        const w = N.fbm(x * 2, y * 2, z * 2, 4);
        const t = N.fbm(x * 1.2 + w * 1.6, y * 4 + w, z * 1.2 + w * 1.6, 5);
        mixTo(out, a, b, sstep(0.25, 0.75, t));
      });
    }
    case 'earth': {
      const deep = hex('#0b2350'), shallow = hex('#1f67a8'), green = hex('#35652c'), tan = hex('#a38c5a'), ice = hex('#eef4f8'), white = hex('#ffffff');
      return paintSphere(512, 256, (x, y, z, la, lo, out) => {
        const n = N.fbm(x * 1.7, y * 1.7, z * 1.7, 6);
        if (n > 0.52) {
          const m = N.fbm(x * 5 + 3, y * 5, z * 5, 4);
          mixTo(out, green, tan, sstep(0.35, 0.7, m + (1 - Math.abs(y)) * 0.15 - 0.1));
        } else mixTo(out, deep, shallow, sstep(0.3, 0.52, n));
        const ay = Math.abs(y) + (N.noise(x * 8, y * 8, z * 8) - 0.5) * 0.08;
        if (ay > 0.93) mixTo(out, out, ice, sstep(0.93, 0.96, ay));
        const cl = N.fbm(x * 2.5 + 11, y * 5, z * 2.5, 5);
        mixTo(out, out, white, clamp01((cl - 0.52) * 3.2) * 0.85);
      });
    }
    case 'mars': {
      const a = hex('#6f2a12'), b = hex('#d98a55'), dark = hex('#3f1f14'), ice = hex('#f3ebe2');
      return paintSphere(384, 192, (x, y, z, la, lo, out) => {
        const n = N.fbm(x * 2.5, y * 2.5, z * 2.5, 6);
        mixTo(out, a, b, sstep(0.25, 0.75, n));
        const d = N.fbm(x * 1.3 + 5, y * 1.3, z * 1.3, 4);
        mixTo(out, out, dark, clamp01((d - 0.55) * 4) * 0.6);
        const ay = Math.abs(y) + (N.noise(x * 9, y * 9, z * 9) - 0.5) * 0.06;
        if (ay > 0.94) mixTo(out, out, ice, sstep(0.94, 0.965, ay));
      });
    }
    case 'jupiter':
      return gasGiant(['#efe4cf', '#c79d74', '#9f6844', '#e9dbc0', '#b8875e', '#dcc7a4', '#a87a55'].map(hex), seed, {
        bands: 7, turb: 1.4, grain: 14,
        spots: [{ lat: -0.39, lon: 1.3, rlat: 0.075, rlon: 0.2, k: 0.9, color: hex('#b8472a'), edge: hex('#d8a07a') }],
      });
    case 'saturn':
      return gasGiant(['#ecdcb2', '#d6bd87', '#c6a66e', '#efe3c3', '#dcc493'].map(hex), seed, { bands: 7, turb: 0.5, grain: 8 });
    case 'uranus':
      return gasGiant(['#a9e3e9', '#9ad7df', '#b8ecef', '#a2dde4'].map(hex), seed, { bands: 3, turb: 0.2, grain: 4 });
    case 'neptune':
      return gasGiant(['#4169d9', '#2e4db3', '#5886ec', '#3558c4', '#6a93f0'].map(hex), seed, {
        bands: 4, turb: 0.8, grain: 8,
        spots: [{ lat: -0.35, lon: 2.4, rlat: 0.07, rlon: 0.16, k: 0.8, color: hex('#1c2c78'), edge: hex('#2e46a8') }],
      });
    case 'pluto': {
      const a = hex('#6f5240'), b = hex('#d8b995'), heart = hex('#f2e6d6');
      return paintSphere(256, 128, (x, y, z, la, lo, out) => {
        mixTo(out, a, b, sstep(0.3, 0.7, N.fbm(x * 3, y * 3, z * 3, 5)));
        const e = spot(la, lo, 0.2, Math.PI, 0.45, 0.5) - (N.noise(x * 6, y * 6, z * 6) - 0.5) * 0.6;
        if (e < 1) mixTo(out, out, heart, sstep(0, 0.6, 1 - e) * 0.9);
      });
    }
    case 'gas':
      return gasGiant(variants(color, 0.14, 0.08), seed, { bands: 5 + (seed % 4), turb: 1.1, grain: 10 });
    case 'ice':
      return gasGiant(variants(color, 0.06, 0.03), seed, { bands: 3, turb: 0.4, grain: 5 });
    default: { // rocky
      const v = variants(color, 0.18, 0.1);
      const lo = v[3], hi = v[0], mid = v[1];
      return paintSphere(384, 192, (x, y, z, la, lon, out) => {
        const n = N.fbm(x * 2.8, y * 2.8, z * 2.8, 6);
        mixTo(out, lo, mid, sstep(0.25, 0.55, n));
        mixTo(out, out, hi, sstep(0.55, 0.8, n));
      });
    }
  }
}

// Grayscale granulation, tinted per star in the shader.
export function starTexture() {
  const N = makeNoise(99);
  return paintSphere(512, 256, (x, y, z, la, lo, out) => {
    const g = N.fbm(x * 9, y * 9, z * 9, 4);
    const c = N.fbm(x * 2.5, y * 2.5, z * 2.5, 3);
    const v = 170 + 85 * sstep(0.2, 0.8, g * 0.75 + c * 0.25);
    out[0] = v; out[1] = v; out[2] = v;
  });
}

function radialCanvas(size, stops) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [o, col] of stops) g.addColorStop(o, col);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c); // linear: keeps soft falloffs bright
  return t;
}

export function glowTexture() {
  return radialCanvas(256, [[0, 'rgba(255,255,255,1)'], [0.12, 'rgba(255,255,255,0.55)'], [0.3, 'rgba(255,255,255,0.16)'], [0.6, 'rgba(255,255,255,0.04)'], [1, 'rgba(255,255,255,0)']]);
}
export function dotTexture() {
  return radialCanvas(64, [[0, 'rgba(255,255,255,1)'], [0.35, 'rgba(255,255,255,0.9)'], [0.6, 'rgba(255,255,255,0.25)'], [1, 'rgba(255,255,255,0)']]);
}
export function haloTexture() { // thin bright ring, for the black hole photon ring
  return radialCanvas(256, [[0, 'rgba(255,255,255,0)'], [0.36, 'rgba(255,255,255,0)'], [0.42, 'rgba(255,255,255,0.95)'], [0.5, 'rgba(255,255,255,0.25)'], [0.75, 'rgba(255,255,255,0.05)'], [1, 'rgba(255,255,255,0)']]);
}

// Planar texture for a RingGeometry(inner, outer): uv (0..1)^2 spans [-outer, outer].
function paintDisk(size, innerFrac, fn) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(size, size);
  const d = img.data, out = [0, 0, 0, 0];
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const x = ((i + 0.5) / size) * 2 - 1, y = ((j + 0.5) / size) * 2 - 1;
      const r = Math.hypot(x, y);
      const k = (j * size + i) * 4;
      if (r < innerFrac * 0.98 || r > 1) { d[k + 3] = 0; continue; }
      fn((r - innerFrac) / (1 - innerFrac), Math.atan2(y, x), r, out);
      d[k] = out[0]; d[k + 1] = out[1]; d[k + 2] = out[2]; d[k + 3] = out[3];
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvasTex(c);
}

export const SATURN_RING = [1.24, 2.27];
export function saturnRingTexture() {
  const R = rng(5);
  const fine = new Float32Array(600);
  for (let i = 0; i < fine.length; i++) fine[i] = 0.75 + R() * 0.5;
  const c0 = hex('#b9a27a'), c1 = hex('#efe2c0'), col = [0, 0, 0];
  return paintDisk(1024, SATURN_RING[0] / SATURN_RING[1], (f, th, r, out) => {
    let a;
    if (f < 0) a = 0;
    else if (f < 0.17) a = 0.22 + f * 0.8;
    else if (f < 0.52) a = 0.88;
    else if (f < 0.575) a = 0.07;
    else if (f < 0.9) a = f > 0.855 && f < 0.868 ? 0.1 : 0.62;
    else a = Math.max(0, 0.25 - (f - 0.9) * 2.5);
    const fi = Math.min(fine.length - 1, Math.floor(Math.max(f, 0) * fine.length));
    a = clamp01(a * fine[fi]);
    mixTo(col, c0, c1, sstep(0.05, 0.5, f) * (f < 0.575 ? 1 : 0.75));
    out[0] = col[0]; out[1] = col[1]; out[2] = col[2]; out[3] = a * 255;
  });
}

export const BH_DISK = [1.5, 6];
export function diskTexture() {
  const N = makeNoise(42);
  const hot = hex('#fff6e0'), mid = hex('#ffa347'), cool = hex('#b2361c'), col = [0, 0, 0];
  return paintDisk(512, BH_DISK[0] / BH_DISK[1], (f, th, r, out) => {
    if (f < 0) { out[3] = 0; return; }
    const phi = th + 3.2 * Math.log(r + 0.05);
    const s = N.fbm(Math.cos(phi) * 2.2, Math.sin(phi) * 2.2, r * 9, 4);
    const base = Math.pow(1 - f, 1.8) * (0.55 + 0.9 * s) + Math.exp(-f * 40) * 0.8;
    if (f < 0.3) mixTo(col, hot, mid, f / 0.3); else mixTo(col, mid, cool, clamp01((f - 0.3) / 0.7));
    out[0] = col[0]; out[1] = col[1]; out[2] = col[2]; out[3] = clamp01(base) * 255;
  });
}

// Approximate blackbody colour (Tanner Helland fit), returned as CSS hex.
export function kelvinToHex(T) {
  const t = T / 100;
  let r, g, b;
  if (t <= 66) { r = 255; g = 99.47 * Math.log(t) - 161.12; b = t <= 19 ? 0 : 138.52 * Math.log(t - 10) - 305.04; }
  else { r = 329.7 * Math.pow(t - 60, -0.1332); g = 288.12 * Math.pow(t - 60, -0.0755); b = 255; }
  const c = (v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0');
  return '#' + c(r) + c(g) + c(b);
}
