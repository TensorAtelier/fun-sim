import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { createLayout, section, slider, select as selectCtl, checkbox, buttons, readout, h, loop } from '../ui.js';
import {
  NBody, makeBody, buildPreset, PRESETS, TRAIL_LEN, KMS_PER_AUD, M_EARTH, M_JUP,
} from './solar/physics.js';
import * as TX from './solar/textures.js';

const CSS = `
.solar-overlay{position:absolute;inset:0;pointer-events:none;overflow:hidden;font-family:var(--mono)}
.solar-label{position:absolute;left:0;top:0;font-size:11px;line-height:16px;color:#c9d1e0;white-space:nowrap;padding:0 7px 0 6px;border-radius:9px;background:rgba(11,14,20,.62);border:1px solid rgba(58,70,92,.55);pointer-events:auto;cursor:pointer;letter-spacing:.02em;will-change:transform;transition:background .15s,color .15s}
.solar-label::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--c);margin-right:6px;vertical-align:1px;box-shadow:0 0 6px var(--c)}
.solar-label:hover{color:#fff;border-color:#5ec8e5}
.solar-label.sel{color:#0b0e14;background:#f5b544;border-color:#f5b544}
.solar-label.sel::before{background:#0b0e14;box-shadow:none}
.solar-sel{position:absolute;left:0;top:0;border:1.5px solid #f5b544;border-radius:50%;box-shadow:0 0 14px rgba(245,181,68,.45),inset 0 0 8px rgba(245,181,68,.25);pointer-events:none;will-change:transform}
.solar-hud b{color:#e6e9ef;font-weight:500}
.solar-hud .paused{color:#f5b544}
.solar-hint{position:absolute;left:12px;bottom:10px;font-family:var(--mono);font-size:11px;color:#5d6679;pointer-events:none}
.solar-banner{position:absolute;top:12px;left:50%;transform:translateX(-50%);background:#f5b544;color:#0b0e14;font-family:var(--mono);font-size:12px;padding:6px 12px;border-radius:14px;box-shadow:0 4px 18px rgba(245,181,68,.3);pointer-events:none;display:none;white-space:nowrap}
.solar-placing canvas{cursor:crosshair}
.solar-selname{display:flex;align-items:center;gap:8px;font-family:var(--serif);font-size:19px;margin:2px 0 10px}
.solar-swatch{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.solar-kind{font-family:var(--mono);font-size:10px;text-transform:uppercase;color:#5ec8e5;letter-spacing:.08em;margin-left:auto}
.solar-sec .readout{margin-bottom:12px}
`;

const EPOCH = Date.UTC(2000, 0, 1, 12);
const MAX_LIGHTS = 4;

const fmtSpeed = (v) => (v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : Math.round(v)) + ' d/s';
const fmtMult = (x) => (x < 0.1 ? x.toFixed(3) : x < 10 ? x.toFixed(2) : x < 100 ? x.toFixed(1) : Math.round(x)) + '×';
function fmtMass(m) {
  if (m < 10 * M_EARTH) return (m / M_EARTH).toPrecision(3) + ' M⊕';
  if (m < 0.03) return (m / M_JUP).toPrecision(3) + ' MJup';
  return m.toPrecision(3) + ' M☉';
}
const fmtAU = (d) => (d < 0.1 ? d.toFixed(4) : d < 10 ? d.toFixed(3) : d.toFixed(2)) + ' AU';
function fmtDays(d) {
  if (d < 1000) return d.toFixed(d < 10 ? 2 : 1) + ' d';
  return (d / 365.25).toFixed(d < 36525 ? 2 : 0) + ' yr';
}

function mount(root) {
  const { stage, panel } = createLayout(root, {
    title: 'Solar System',
    desc: 'Real N-body gravity: every body pulls on every other. Distances and masses are real; planet sizes are exaggerated so you can see them.',
  });
  const styleEl = document.createElement('style');
  styleEl.textContent = CSS;
  document.head.append(styleEl);

  // ---------- state ----------
  const S = { playing: true, speed: 20, sizeScale: 1, compress: true, trails: true, labels: true, grid: true, follow: false, preset: 'solar', frame: 'sun' };
  const sim = new NBody();
  let selected = null;

  // ---------- three setup ----------
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, logarithmicDepthBuffer: true });
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(dpr);
  renderer.setClearColor(0x000000, 0);
  stage.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1e5);
  camera.position.set(0, 18, 28);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.02;
  controls.maxDistance = 2500;
  controls.zoomSpeed = 1.2;

  const trash = []; // global disposables
  const keep = (x) => { trash.push(x); return x; };
  scene.add(new THREE.AmbientLight(0x8a93a6, 0.14));

  const sphereGeo = keep(new THREE.SphereGeometry(1, 64, 40));
  const dotGeo = keep(new THREE.BufferGeometry());
  dotGeo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
  const glowTex = keep(TX.glowTexture());
  const dotTex = keep(TX.dotTexture());
  const haloTex = keep(TX.haloTexture());
  const starTex = keep(TX.starTexture());
  let ringTex = null, diskTex = null;
  const getRingTex = () => ringTex || (ringTex = keep(TX.saturnRingTexture()));
  const getDiskTex = () => diskTex || (diskTex = keep(TX.diskTexture()));

  // Starfield: a dim sky sphere with a Milky-Way-ish band, kept centred on the camera.
  const sky = new THREE.Group();
  scene.add(sky);
  {
    const R = TX.rng(77);
    const band = new THREE.Matrix4().makeRotationX(1.05).multiply(new THREE.Matrix4().makeRotationZ(0.4));
    const layer = (n, size, bright, bandFrac) => {
      const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), v = new THREE.Vector3();
      for (let i = 0; i < n; i++) {
        if (R() < bandFrac) {
          const th = R() * Math.PI * 2, z = (R() + R() + R() - 1.5) * 0.18;
          v.set(Math.cos(th), z, Math.sin(th)).normalize().applyMatrix4(band);
        } else {
          const u = R() * 2 - 1, th = R() * Math.PI * 2, s = Math.sqrt(1 - u * u);
          v.set(s * Math.cos(th), u, s * Math.sin(th));
        }
        v.multiplyScalar(4000);
        pos.set([v.x, v.y, v.z], i * 3);
        const t = R(), b = bright * (0.35 + 0.65 * R());
        const c = t < 0.15 ? [1, 0.82, 0.62] : t < 0.35 ? [0.7, 0.85, 1] : [1, 0.97, 0.93];
        col.set([c[0] * b, c[1] * b, c[2] * b], i * 3);
      }
      const g = keep(new THREE.BufferGeometry());
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const m = keep(new THREE.PointsMaterial({ size: size * dpr, sizeAttenuation: false, vertexColors: true, map: dotTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
      const p = new THREE.Points(g, m);
      p.frustumCulled = false;
      p.renderOrder = -10;
      sky.add(p);
    };
    layer(7000, 1.6, 0.55, 0.55);
    layer(700, 2.6, 0.9, 0.3);
    layer(60, 4, 1, 0);
  }

  // Ecliptic grid rings (rebuilt when the distance mapping changes).
  const gridGroup = new THREE.Group();
  scene.add(gridGroup);
  const gridMat = keep(new THREE.LineBasicMaterial({ color: 0x5ec8e5, transparent: true, opacity: 0.07, depthWrite: false }));
  const gridMat2 = keep(new THREE.LineBasicMaterial({ color: 0x5ec8e5, transparent: true, opacity: 0.16, depthWrite: false }));

  // Test particles
  const tpGeo = keep(new THREE.BufferGeometry());
  const tpPos = new Float32Array(sim.tp.cap * 3);
  const tpCol = new Float32Array(sim.tp.cap * 3);
  tpGeo.setAttribute('position', new THREE.BufferAttribute(tpPos, 3).setUsage(THREE.DynamicDrawUsage));
  tpGeo.setAttribute('color', new THREE.BufferAttribute(tpCol, 3).setUsage(THREE.DynamicDrawUsage));
  const tpMat = keep(new THREE.PointsMaterial({ size: 2.4 * dpr, sizeAttenuation: false, vertexColors: true, map: dotTex, transparent: true, depthWrite: false }));
  const tpPoints = new THREE.Points(tpGeo, tpMat);
  tpPoints.frustumCulled = false;
  scene.add(tpPoints);

  // Placement helpers
  const markerGeo = keep(new THREE.RingGeometry(0.75, 1, 48));
  const markerMat = keep(new THREE.MeshBasicMaterial({ color: 0xf5b544, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthTest: false }));
  const marker = new THREE.Mesh(markerGeo, markerMat);
  marker.rotation.x = -Math.PI / 2;
  marker.visible = false;
  marker.renderOrder = 10;
  scene.add(marker);
  const previewGeo = keep(new THREE.BufferGeometry());
  previewGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(129 * 3), 3));
  const previewMat = keep(new THREE.LineDashedMaterial({ color: 0xf5b544, transparent: true, opacity: 0.55, dashSize: 0.1, gapSize: 0.07, depthTest: false }));
  const preview = new THREE.Line(previewGeo, previewMat);
  preview.visible = false;
  preview.frustumCulled = false;
  scene.add(preview);
  const arrowGeo = keep(new THREE.BufferGeometry());
  arrowGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(5 * 3), 3));
  const arrowMat = keep(new THREE.LineBasicMaterial({ color: 0x5ec8e5, depthTest: false }));
  const arrow = new THREE.Line(arrowGeo, arrowMat);
  arrow.visible = false;
  arrow.frustumCulled = false;
  scene.add(arrow);

  const flashes = [];

  // ---------- overlay DOM ----------
  const overlay = h('div', { class: 'solar-overlay' });
  const selRing = h('div', { class: 'solar-sel' });
  selRing.style.display = 'none';
  overlay.append(selRing);
  const hud = h('div', { class: 'hud solar-hud' });
  const banner = h('div', { class: 'solar-banner' });
  const hint = h('div', { class: 'solar-hint' }, 'drag orbit · right-drag pan · scroll zoom · click select · dbl-click follow · space pause');
  stage.append(overlay, hud, banner, hint);

  // ---------- distance mapping ----------
  // compress: r -> 2(sqrt(1+r)-1). Linear near the Sun (no fake wobble), ~sqrt far out.
  const mapR = (r) => (S.compress ? 2 * (Math.sqrt(1 + r) - 1) : r);
  const unmapR = (s) => (S.compress ? s + (s * s) / 4 : s);
  // Display frame origin (physics coords): the Sun, or the barycentre when there is no Sun.
  const C = { x: 0, y: 0, z: 0 };
  function updateCenter() {
    const sun = S.frame === 'sun' ? sim.bodies.find((b) => b.isSun) || null : null;
    if (sim.frameBody !== sun) { sim.frameBody = sun; sim.clearTrails(); }
    const c = sun || sim.barycentre();
    C.x = c.x; C.y = c.y; C.z = c.z;
  }
  function toDisplay(x, y, z, out) {
    x -= C.x; y -= C.y; z -= C.z;
    let f = 1;
    if (S.compress) { const r = Math.sqrt(x * x + y * y + z * z); if (r > 1e-12) f = mapR(r) / r; }
    return out.set(x * f, z * f, -y * f);
  }
  function displayRadius(b) {
    const k = Math.sqrt(S.sizeScale);
    if (b.kind === 'star') return Math.min(0.13 * Math.pow(b.m, 0.35) * k, 1.0);
    if (b.kind === 'bh') return 0.02 * Math.cbrt(b.m / 10) * k;
    return 0.00055 * Math.sqrt(b.radiusKm) * S.sizeScale;
  }

  function rebuildGrid() {
    for (const c of gridGroup.children) c.geometry.dispose();
    gridGroup.clear();
    for (const r of [1, 2, 5, 10, 20, 30, 50]) {
      const pts = [];
      const s = mapR(r);
      for (let i = 0; i <= 256; i++) { const a = (i / 256) * Math.PI * 2; pts.push(s * Math.cos(a), 0, s * Math.sin(a)); }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      gridGroup.add(new THREE.Line(g, r === 1 || r === 10 ? gridMat2 : gridMat));
    }
    gridGroup.visible = S.grid;
  }

  // ---------- body visuals ----------
  function starMaterial(color) {
    return new THREE.ShaderMaterial({
      uniforms: { map: { value: starTex }, tint: { value: new THREE.Color(color) } },
      vertexShader: `
        #include <common>
        #include <logdepthbuf_pars_vertex>
        varying vec2 vUv; varying vec3 vN; varying vec3 vV;
        void main() {
          vUv = uv;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vN = normalize(normalMatrix * normal);
          vV = normalize(-mv.xyz);
          gl_Position = projectionMatrix * mv;
          #include <logdepthbuf_vertex>
        }`,
      fragmentShader: `
        #include <common>
        #include <logdepthbuf_pars_fragment>
        uniform sampler2D map; uniform vec3 tint;
        varying vec2 vUv; varying vec3 vN; varying vec3 vV;
        void main() {
          #include <logdepthbuf_fragment>
          float mu = clamp(dot(normalize(vN), normalize(vV)), 0.0, 1.0);
          float limb = 0.3 + 0.7 * pow(mu, 0.5);
          vec3 g = texture2D(map, vUv).rgb;
          vec3 c = tint * g * limb * 1.35 + vec3(1.0, 0.95, 0.85) * pow(mu, 4.0) * 0.35;
          gl_FragColor = vec4(c, 1.0);
          #include <colorspace_fragment>
        }`,
    });
  }

  let lightCount = 0;
  function createView(b) {
    const v = { group: new THREE.Group(), scaler: new THREE.Group(), tilt: new THREE.Group(), own: [], spin: 0.3 + Math.random() * 0.4 };
    const own = (x) => { v.own.push(x); return x; };
    v.group.add(v.scaler);
    v.scaler.add(v.tilt);
    const col = new THREE.Color(b.color);
    if (b.kind === 'star') {
      v.mesh = new THREE.Mesh(sphereGeo, own(starMaterial(b.color)));
      v.tilt.add(v.mesh);
      const sprite = (scale, opacity, c) => {
        const s = new THREE.Sprite(own(new THREE.SpriteMaterial({ map: glowTex, color: c, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false })));
        s.scale.setScalar(scale);
        v.scaler.add(s);
        return s;
      };
      sprite(4.5, 0.95, col);
      sprite(14, 0.35, col);
      sprite(40, 0.12, col);
      if (lightCount < MAX_LIGHTS) {
        v.light = new THREE.PointLight(col.clone().lerp(new THREE.Color(0xffffff), 0.5), 2.4, 0, 0);
        v.group.add(v.light);
        lightCount++;
      }
      v.spin = 0.08;
    } else if (b.kind === 'bh') {
      v.mesh = new THREE.Mesh(sphereGeo, own(new THREE.MeshBasicMaterial({ color: 0x000000 })));
      v.tilt.add(v.mesh);
      v.tilt.rotation.x = 0.35;
      const dg = own(new THREE.RingGeometry(TX.BH_DISK[0], TX.BH_DISK[1], 160, 1));
      v.disk = new THREE.Mesh(dg, own(new THREE.MeshBasicMaterial({ map: getDiskTex(), transparent: true, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false })));
      v.disk.rotation.x = -Math.PI / 2;
      v.tilt.add(v.disk);
      const halo = new THREE.Sprite(own(new THREE.SpriteMaterial({ map: haloTex, color: 0xffd9b0, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false })));
      halo.scale.setScalar(3.2);
      v.scaler.add(halo);
      const glow = new THREE.Sprite(own(new THREE.SpriteMaterial({ map: glowTex, color: 0xff8a4a, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false })));
      glow.scale.setScalar(16);
      v.scaler.add(glow);
      v.spin = 0;
    } else {
      const tex = own(TX.planetTexture(b.style, b.seed, b.color));
      v.mesh = new THREE.Mesh(sphereGeo, own(new THREE.MeshStandardMaterial({ map: tex, roughness: 0.92, metalness: 0 })));
      v.tilt.add(v.mesh);
      v.tilt.rotation.z = b.tilt;
      if (b.ring) {
        const rg = own(new THREE.RingGeometry(TX.SATURN_RING[0], TX.SATURN_RING[1], 160, 1));
        const ring = new THREE.Mesh(rg, own(new THREE.MeshStandardMaterial({ map: getRingTex(), transparent: true, side: THREE.DoubleSide, roughness: 1, metalness: 0, depthWrite: false })));
        ring.rotation.x = -Math.PI / 2;
        v.tilt.add(ring);
      }
      v.dot = new THREE.Points(dotGeo, own(new THREE.PointsMaterial({ color: col, size: 5 * dpr, sizeAttenuation: false, map: dotTex, transparent: true, depthWrite: false })));
      v.group.add(v.dot);
    }
    // trail
    v.trailGeo = own(new THREE.BufferGeometry());
    v.trailPos = new Float32Array((TRAIL_LEN + 1) * 3);
    v.trailCol = new Float32Array((TRAIL_LEN + 1) * 3);
    v.trailGeo.setAttribute('position', new THREE.BufferAttribute(v.trailPos, 3).setUsage(THREE.DynamicDrawUsage));
    v.trailGeo.setAttribute('color', new THREE.BufferAttribute(v.trailCol, 3).setUsage(THREE.DynamicDrawUsage));
    v.trail = new THREE.Line(v.trailGeo, own(new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })));
    v.trail.frustumCulled = false;
    const tc = col.clone().lerp(new THREE.Color(0xffffff), 0.15);
    v.trailRGB = [tc.r, tc.g, tc.b];
    scene.add(v.trail);
    // label
    v.label = h('div', { class: 'solar-label', style: `--c:${b.color}` }, b.name);
    v.label.addEventListener('click', (e) => { e.stopPropagation(); selectBody(b); });
    v.label.addEventListener('dblclick', (e) => { e.stopPropagation(); selectBody(b); setFollow(true); });
    overlay.append(v.label);
    scene.add(v.group);
    b.view = v;
  }

  function destroyView(b) {
    const v = b.view;
    if (!v) return;
    scene.remove(v.group);
    scene.remove(v.trail);
    if (v.light) { v.light.dispose(); lightCount--; }
    v.label.remove();
    for (const o of v.own) o.dispose();
    b.view = null;
  }

  function spawnFlash(b, x, y, z) {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: 0xffe2b0, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    toDisplay(x, y, z, s.position);
    scene.add(s);
    flashes.push({ s, age: 0, life: 1.3, R: Math.max(displayRadius(b), 0.05) });
  }

  // ---------- selection ----------
  function selectBody(b) {
    if (selected?.view) selected.view.label.classList.remove('sel');
    selected = b;
    if (b?.view) b.view.label.classList.add('sel');
    if (!b) setFollow(false);
    followLocked = false;
    renderSelPanel();
  }
  let followLocked = false;
  let followBox = null;
  function setFollow(on) {
    S.follow = on && !!selected;
    followLocked = false;
    if (followBox) followBox.checked = S.follow;
  }

  // ---------- presets ----------
  function loadPreset(id) {
    for (const b of sim.bodies) destroyView(b);
    selectBody(null);
    const hints = buildPreset(sim, id);
    S.preset = id;
    sim.gMult = 1; sim.sunMult = 1;
    gSlider?.set(0); sunSlider?.set(0);
    S.speed = hints.speed; speedSlider?.set(Math.log10(hints.speed));
    S.compress = hints.compress; if (compressBox) compressBox.checked = S.compress;
    S.frame = hints.frame || 'sun'; if (frameSel) frameSel.value = S.frame;
    for (const b of sim.bodies) createView(b);
    sim.refreshMasses();
    sim.updatePrimaries();
    updateCenter();
    rebuildGrid();
    // fit camera
    const d = Math.max(mapR(hints.extent) * 2.3, 2);
    camera.position.set(0, d * 0.5, d * 0.87);
    controls.target.set(0, 0, 0);
    controls.update();
    counters.planet = counters.star = counters.bh = 0;
  }

  // ---------- adding bodies ----------
  const counters = { planet: 0, star: 0, bh: 0 };
  const ADD = {
    planet: { label: 'Mass', min: -1, max: 3.5, def: 1, fmt: (v) => { const me = 10 ** v; return me < 300 ? me.toPrecision(3) + ' M⊕' : (me * M_EARTH / M_JUP).toPrecision(3) + ' MJup'; } },
    star: { label: 'Mass', min: -1, max: 1.3, def: 0, fmt: (v) => (10 ** v).toPrecision(3) + ' M☉' },
    bh: { label: 'Mass', min: 0, max: 2, def: 1, fmt: (v) => (10 ** v).toPrecision(3) + ' M☉' },
    swarm: { label: 'Particles', min: 50, max: 800, def: 250, step: 10, fmt: (v) => Math.round(v) },
  };
  const add = { type: 'planet', vel: 'circular', placing: false, vals: {} };
  for (const k in ADD) add.vals[k] = ADD[k].def;

  function newBody(type, px, py) {
    const v = add.vals[type];
    const R = TX.rng(Math.floor(Math.random() * 1e9));
    if (type === 'planet') {
      const m = 10 ** v * M_EARTH, me = 10 ** v;
      const style = me < 8 ? 'rocky' : me < 60 ? 'ice' : 'gas';
      const hue = style === 'ice' ? 0.48 + R() * 0.16 : style === 'gas' ? (R() < 0.6 ? 0.05 + R() * 0.08 : R()) : R();
      const color = '#' + new THREE.Color().setHSL(hue, style === 'rocky' ? 0.35 : 0.5, 0.6).getHexString();
      return makeBody({ name: `Planet ${++counters.planet}`, mass: m, pos: [px, py, 0], color, style, ring: style === 'gas' && R() < 0.35, tilt: R() * 0.5 });
    }
    if (type === 'star') {
      const m = 10 ** v;
      const color = TX.kelvinToHex(Math.min(30000, Math.max(2600, 5778 * Math.pow(m, 0.55))));
      return makeBody({ name: `Star ${++counters.star}`, kind: 'star', mass: m, pos: [px, py, 0], color });
    }
    return makeBody({ name: `Black hole ${++counters.bh}`, kind: 'bh', mass: 10 ** v, pos: [px, py, 0], color: '#ff9a5a', style: 'bh' });
  }

  function placeAt(px, py, pz, dragVel) {
    const bc = sim.barycentre();
    const rx = px - bc.x, ry = py - bc.y, r = Math.hypot(rx, ry) || 1e-6;
    if (add.type === 'swarm') {
      const n = Math.round(add.vals.swarm);
      const R = TX.rng(Math.floor(Math.random() * 1e9));
      const hue = R();
      const c = new THREE.Color();
      for (let i = 0; i < n; i++) {
        const rr = r * (0.9 + R() * 0.2), th = R() * Math.PI * 2, z = (R() - 0.5) * 0.03 * rr;
        const s = Math.sqrt((sim.G * bc.M) / rr) * (1 + (R() - 0.5) * 0.02);
        c.setHSL((hue + R() * 0.08) % 1, 0.55, 0.55 + R() * 0.2);
        sim.addParticle(bc.x + rr * Math.cos(th), bc.y + rr * Math.sin(th), pz + z,
          bc.vx - s * Math.sin(th), bc.vy + s * Math.cos(th), 0, c.r, c.g, c.b);
      }
      sim.computeAccel();
      return;
    }
    const b = newBody(add.type, px, py);
    b.z = pz;
    const m = b.m, M = bc.M;
    let vx = bc.vx, vy = bc.vy;
    if (dragVel) {
      vx += dragVel[0]; vy += dragVel[1];
      if (M > 0) sim.boostAll(-(m / M) * dragVel[0], -(m / M) * dragVel[1], 0);
    } else if (add.vel === 'circular' && M > 0) {
      // Two-body circular orbit between the new body and the existing system; the system
      // gets the recoil so total momentum stays put (a new star makes a real binary).
      const vrel = Math.sqrt((sim.G * (M + m)) / r);
      const tx = -ry / r, ty = rx / r;
      vx += tx * vrel * M / (M + m); vy += ty * vrel * M / (M + m);
      sim.boostAll(-tx * vrel * m / (M + m), -ty * vrel * m / (M + m), 0);
    }
    b.vx = vx; b.vy = vy;
    sim.add(b);
    createView(b);
    sim.computeAccel();
    sim.updatePrimaries();
    selectBody(b);
  }

  function setPlacing(on) {
    add.placing = on;
    placeBtn.textContent = on ? 'Cancel' : 'Place on ecliptic';
    stage.classList.toggle('solar-placing', on);
    banner.style.display = on ? 'block' : 'none';
    banner.textContent = add.type === 'swarm' ? 'Click the ecliptic to set the ring radius · Esc to cancel'
      : add.vel === 'drag' ? 'Press on the ecliptic and drag to aim the velocity · Esc to cancel'
        : 'Click the ecliptic to place · Esc to cancel';
    marker.visible = false; preview.visible = false; arrow.visible = false;
    if (!on) { drag = null; controls.enabled = true; }
  }

  // ---------- pointer handling ----------
  const raycaster = new THREE.Raycaster();
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const ndc = new THREE.Vector2();
  const hit = new THREE.Vector3();
  let down = null, drag = null;
  let W = 1, H = 1;

  function planePoint(e) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    return raycaster.ray.intersectPlane(plane, hit) ? hit : null;
  }
  const displayToPhys = (dx, dz) => {
    const s = Math.hypot(dx, dz);
    const f = s > 1e-12 ? unmapR(s) / s : 1;
    return [dx * f + C.x, -dz * f + C.y, C.z];
  };

  function updatePreview(p) {
    const camD = camera.position.distanceTo(p);
    marker.position.copy(p);
    marker.scale.setScalar(camD * 0.012);
    marker.visible = true;
    const showOrbit = add.type === 'swarm' || add.vel === 'circular';
    preview.visible = showOrbit;
    if (showOrbit) {
      const [px, py, pz] = displayToPhys(p.x, p.z);
      const bc = sim.barycentre();
      const r = Math.hypot(px - bc.x, py - bc.y);
      const arr = previewGeo.attributes.position.array, t = new THREE.Vector3();
      for (let i = 0; i <= 128; i++) {
        const a = (i / 128) * Math.PI * 2;
        toDisplay(bc.x + r * Math.cos(a), bc.y + r * Math.sin(a), pz, t);
        arr.set([t.x, t.y, t.z], i * 3);
      }
      previewGeo.attributes.position.needsUpdate = true;
      previewMat.dashSize = camD * 0.012; previewMat.gapSize = camD * 0.008;
      preview.computeLineDistances();
    }
  }

  const DRAG_SCALE = 0.008; // AU/day of velocity per display-AU dragged (~14 km/s)
  function updateArrow(a, b) {
    const arr = arrowGeo.attributes.position.array;
    const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz) || 1e-9;
    const ux = dx / len, uz = dz / len, hl = Math.min(len * 0.3, camera.position.distanceTo(a) * 0.03);
    arr.set([a.x, 0, a.z, b.x, 0, b.z, b.x - ux * hl - uz * hl * 0.5, 0, b.z - uz * hl + ux * hl * 0.5, b.x, 0, b.z, b.x - ux * hl + uz * hl * 0.5, 0, b.z - uz * hl - ux * hl * 0.5]);
    arrowGeo.attributes.position.needsUpdate = true;
    arrow.visible = true;
  }

  const onDown = (e) => {
    down = { x: e.clientX, y: e.clientY };
    if (add.placing && add.vel === 'drag' && add.type !== 'swarm' && e.button === 0) {
      const p = planePoint(e);
      if (p) { drag = { start: p.clone(), end: p.clone() }; controls.enabled = false; }
    }
  };
  const onMove = (e) => {
    if (!add.placing) return;
    const p = planePoint(e);
    if (!p) return;
    if (drag) { drag.end.copy(p); updateArrow(drag.start, drag.end); } else updatePreview(p);
  };
  const onUp = (e) => {
    const moved = down ? Math.hypot(e.clientX - down.x, e.clientY - down.y) : 99;
    down = null;
    if (drag) {
      const { start, end } = drag;
      drag = null;
      controls.enabled = true;
      arrow.visible = false;
      const [px, py, pz] = displayToPhys(start.x, start.z);
      const vel = [(end.x - start.x) * DRAG_SCALE, -(end.z - start.z) * DRAG_SCALE];
      placeAt(px, py, pz, vel);
      if (!e.shiftKey) setPlacing(false);
      return;
    }
    if (moved > 5 || e.button !== 0) return;
    if (add.placing) {
      const p = planePoint(e);
      if (p) {
        const [px, py, pz] = displayToPhys(p.x, p.z);
        placeAt(px, py, pz, null);
        if (!e.shiftKey) setPlacing(false);
      }
      return;
    }
    const r = renderer.domElement.getBoundingClientRect();
    selectBody(pick(e.clientX - r.left, e.clientY - r.top));
  };
  const onDbl = (e) => {
    if (add.placing) return;
    const r = renderer.domElement.getBoundingClientRect();
    const b = pick(e.clientX - r.left, e.clientY - r.top);
    if (b) { selectBody(b); setFollow(true); }
  };
  const onKey = (e) => {
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'Escape') { if (add.placing) setPlacing(false); else selectBody(null); }
  };
  const cvs = renderer.domElement;
  cvs.addEventListener('pointerdown', onDown);
  cvs.addEventListener('pointermove', onMove);
  cvs.addEventListener('pointerup', onUp);
  cvs.addEventListener('dblclick', onDbl);
  window.addEventListener('keydown', onKey);

  function pick(mx, my) {
    let best = null, bd = Infinity;
    for (const b of sim.bodies) {
      const v = b.view;
      if (!v?.onScreen) continue;
      const d = Math.hypot(v.sx - mx, v.sy - my);
      if (d < Math.max(14, v.rpx + 6) && d - v.rpx < bd) { bd = d - v.rpx; best = b; }
    }
    return best;
  }

  // ---------- panel ----------
  const simSec = section(panel, 'Simulation');
  selectCtl(simSec, { label: 'Preset', options: PRESETS, value: 'solar', onChange: (v) => loadPreset(v) });
  const [playBtn] = buttons(simSec, [
    { label: 'Pause', primary: true, onClick: () => togglePlay() },
    { label: 'Reset', onClick: () => loadPreset(S.preset) },
  ]);
  function togglePlay() { S.playing = !S.playing; playBtn.textContent = S.playing ? 'Pause' : 'Play'; }
  const speedSlider = slider(simSec, { label: 'Time speed', min: -1, max: 3, step: 0.01, value: Math.log10(S.speed), format: (v) => fmtSpeed(10 ** v), onInput: (v) => { S.speed = 10 ** v; } });
  const simRO = readout(simSec, ['Date', 'Elapsed', 'Bodies', 'Test particles']);

  const viewSec = section(panel, 'View');
  slider(viewSec, { label: 'Planet size', min: -0.6, max: 0.8, step: 0.01, value: 0, format: (v) => fmtMult(10 ** v), onInput: (v) => { S.sizeScale = 10 ** v; } });
  const compressBox = checkbox(viewSec, { label: 'Compress distances (√-ish)', checked: S.compress, onChange: (v) => { S.compress = v; rebuildGrid(); } });
  const frameSel = selectCtl(viewSec, { label: 'Centre on', value: 'sun', options: [['sun', 'Sun (heliocentric)'], ['bary', 'Barycentre']], onChange: (v) => { S.frame = v; } });
  checkbox(viewSec, { label: 'Orbit trails', checked: S.trails, onChange: (v) => { S.trails = v; } });
  checkbox(viewSec, { label: 'Labels', checked: S.labels, onChange: (v) => { S.labels = v; } });
  checkbox(viewSec, { label: 'Ecliptic grid', checked: S.grid, onChange: (v) => { S.grid = v; gridGroup.visible = v; } });

  const physSec = section(panel, 'Gravity');
  const sunSlider = slider(physSec, { label: 'Sun mass', min: -1, max: 1, step: 0.01, value: 0, format: (v) => fmtMult(10 ** v), onInput: (v) => { sim.sunMult = 10 ** v; sim.refreshMasses(); } });
  const gSlider = slider(physSec, { label: 'G strength', min: -1, max: 0.7, step: 0.01, value: 0, format: (v) => fmtMult(10 ** v), onInput: (v) => { sim.gMult = 10 ** v; sim.computeAccel(); } });

  const selSec = section(panel, 'Selected body');
  selSec.classList.add('solar-sec');
  const selBody = h('div');
  selSec.append(selBody);
  let selRO = null;
  function renderSelPanel() {
    selBody.replaceChildren();
    selRO = null; followBox = null;
    const b = selected;
    if (!b) {
      selBody.append(h('p', { class: 'hint' }, 'Click a body (or its label) to inspect it, change its mass or delete it. Double-click to follow.'));
      return;
    }
    const kindName = b.kind === 'bh' ? 'black hole' : b.kind;
    selBody.append(h('div', { class: 'solar-selname' },
      h('span', { class: 'solar-swatch', style: `background:${b.color};box-shadow:0 0 8px ${b.color}` }), b.name,
      h('span', { class: 'solar-kind' }, kindName)));
    selRO = readout(selBody, ['Mass', 'Distance', 'Speed', 'Period']);
    slider(selBody, {
      label: 'Mass multiplier', min: -2, max: 3, step: 0.01, value: Math.log10(b.mult), format: (v) => fmtMult(10 ** v),
      onInput: (v) => { b.mult = 10 ** v; sim.refreshMasses(); },
    });
    followBox = checkbox(selBody, { label: 'Camera follows this body', checked: S.follow, onChange: (v) => setFollow(v) });
    buttons(selBody, [
      { label: 'Delete', onClick: () => { const x = selected; selectBody(null); sim.remove(x); destroyView(x); sim.computeAccel(); } },
      { label: 'Deselect', onClick: () => selectBody(null) },
    ]);
    updateSelReadout();
  }

  const addSec = section(panel, 'Add body');
  selectCtl(addSec, {
    label: 'Type', value: 'planet',
    options: [['planet', 'Planet'], ['star', 'Star'], ['bh', 'Black hole'], ['swarm', 'Asteroid swarm (test particles)']],
    onChange: (t) => { add.type = t; buildAddMass(); if (add.placing) setPlacing(true); },
  });
  const addMassWrap = h('div');
  addSec.append(addMassWrap);
  function buildAddMass() {
    addMassWrap.replaceChildren();
    const c = ADD[add.type];
    slider(addMassWrap, { label: c.label, min: c.min, max: c.max, step: c.step || 0.01, value: add.vals[add.type], format: c.fmt, onInput: (v) => { add.vals[add.type] = v; } });
  }
  buildAddMass();
  selectCtl(addSec, {
    label: 'Initial velocity', value: 'circular',
    options: [['circular', 'Circular orbit around barycentre'], ['drag', 'Drag to aim'], ['rest', 'At rest']],
    onChange: (v) => { add.vel = v; if (add.placing) setPlacing(true); },
  });
  const [placeBtn] = buttons(addSec, [
    { label: 'Place on ecliptic', primary: true, onClick: () => setPlacing(!add.placing) },
    { label: 'Clear particles', onClick: () => { sim.tp.n = 0; } },
  ]);
  addSec.append(h('p', { class: 'hint' }, 'Shift-click keeps placing. Swarms are massless: they feel gravity but exert none. Bodies that touch merge, conserving momentum.'));

  // ---------- readouts ----------
  function updateSelReadout() {
    const b = selected;
    if (!b || !selRO) return;
    selRO.set('Mass', fmtMass(b.m));
    const sun = sim.bodies.find((o) => o.isSun && o !== b);
    const bc = sim.barycentre();
    if (sun) selRO.set('Distance', fmtAU(Math.hypot(b.x - sun.x, b.y - sun.y, b.z - sun.z)) + ' · Sun');
    else if (b.primary) selRO.set('Distance', fmtAU(Math.hypot(b.x - b.primary.x, b.y - b.primary.y, b.z - b.primary.z)) + ' · ' + b.primary.name);
    else selRO.set('Distance', fmtAU(Math.hypot(b.x - bc.x, b.y - bc.y, b.z - bc.z)) + ' · bary');
    const sp = Math.hypot(b.vx - bc.vx, b.vy - bc.vy, b.vz - bc.vz) * KMS_PER_AUD;
    selRO.set('Speed', sp.toFixed(sp < 10 ? 2 : 1) + ' km/s');
    const p = b.primary;
    if (!p) { selRO.set('Period', '—'); return; }
    const mu = sim.G * (b.m + p.m);
    const r = Math.hypot(b.x - p.x, b.y - p.y, b.z - p.z);
    const v2 = (b.vx - p.vx) ** 2 + (b.vy - p.vy) ** 2 + (b.vz - p.vz) ** 2;
    const eps = v2 / 2 - mu / r;
    if (eps >= 0) selRO.set('Period', 'unbound');
    else {
      const a = -mu / (2 * eps);
      selRO.set('Period', fmtDays(2 * Math.PI * Math.sqrt((a * a * a) / mu)) + ' · ' + p.name);
    }
  }
  function updateReadouts() {
    const date = new Date(EPOCH + sim.t * 86400000);
    const ds = isFinite(date) ? date.toISOString().slice(0, 10) : '—';
    const yrs = sim.t / 365.25;
    simRO.set('Date', ds);
    simRO.set('Elapsed', yrs < 1 ? sim.t.toFixed(0) + ' days' : yrs.toFixed(2) + ' yr');
    simRO.set('Bodies', String(sim.bodies.length));
    simRO.set('Test particles', String(sim.tp.n));
    hud.innerHTML = `<b>${ds}</b> · +${yrs.toFixed(2)} yr<br>${S.playing ? fmtSpeed(S.speed) : '<span class="paused">paused</span>'} · ${sim.bodies.length} bodies`;
    updateSelReadout();
  }

  // ---------- per-frame ----------
  const tmp = new THREE.Vector3(), prevTarget = new THREE.Vector3();
  const tanHalf = () => Math.tan((camera.fov * Math.PI) / 360);

  function syncViews(dt) {
    for (const b of sim.bodies) {
      const v = b.view;
      toDisplay(b.x, b.y, b.z, v.group.position);
      v.R = displayRadius(b);
      v.scaler.scale.setScalar(v.R);
      if (S.playing) {
        v.mesh.rotation.y += dt * v.spin;
        if (v.disk) v.disk.rotation.z += dt * 0.9;
      }
      if (v.light) v.light.intensity = 2.4 * Math.min(Math.max(Math.sqrt(b.m), 0.5), 3);
      v.trail.visible = S.trails;
      if (S.trails) {
        const t = b.trail, n = t.count, P = v.trailPos, Q = v.trailCol, [cr, cg, cb] = v.trailRGB;
        for (let i = 0; i < n; i++) {
          const idx = ((t.head - n + i + TRAIL_LEN) % TRAIL_LEN) * 3;
          toDisplay(t.buf[idx] + C.x, t.buf[idx + 1] + C.y, t.buf[idx + 2] + C.z, tmp);
          P[i * 3] = tmp.x; P[i * 3 + 1] = tmp.y; P[i * 3 + 2] = tmp.z;
          const f = Math.pow((i + 1) / (n + 1), 1.3) * 1.25;
          Q[i * 3] = cr * f; Q[i * 3 + 1] = cg * f; Q[i * 3 + 2] = cb * f;
        }
        const p = v.group.position;
        P[n * 3] = p.x; P[n * 3 + 1] = p.y; P[n * 3 + 2] = p.z;
        Q[n * 3] = cr * 1.25; Q[n * 3 + 1] = cg * 1.25; Q[n * 3 + 2] = cb * 1.25;
        v.trailGeo.setDrawRange(0, n + 1);
        v.trailGeo.attributes.position.needsUpdate = true;
        v.trailGeo.attributes.color.needsUpdate = true;
      }
    }
    // test particles
    const tp = sim.tp;
    for (let i = 0; i < tp.n; i++) {
      const k = i * 3;
      toDisplay(tp.x[k], tp.x[k + 1], tp.x[k + 2], tmp);
      tpPos[k] = tmp.x; tpPos[k + 1] = tmp.y; tpPos[k + 2] = tmp.z;
      tpCol[k] = tp.c[k]; tpCol[k + 1] = tp.c[k + 1]; tpCol[k + 2] = tp.c[k + 2];
    }
    tpGeo.setDrawRange(0, tp.n);
    tpGeo.attributes.position.needsUpdate = true;
    tpGeo.attributes.color.needsUpdate = true;
    // flashes
    for (let i = flashes.length - 1; i >= 0; i--) {
      const f = flashes[i];
      f.age += dt;
      const k = f.age / f.life;
      if (k >= 1) { scene.remove(f.s); f.s.material.dispose(); flashes.splice(i, 1); continue; }
      f.s.scale.setScalar(f.R * (4 + 30 * Math.sqrt(k)));
      f.s.material.opacity = (1 - k) * (1 - k);
    }
  }

  function updateFollow(dt) {
    if (!S.follow || !selected?.view) return;
    const p = selected.view.group.position;
    prevTarget.copy(controls.target);
    if (followLocked) controls.target.copy(p);
    else {
      controls.target.lerp(p, 1 - Math.exp(-dt * 6));
      if (controls.target.distanceTo(p) < camera.position.distanceTo(p) * 0.002) followLocked = true;
    }
    camera.position.add(tmp.subVectors(controls.target, prevTarget));
  }

  function updateOverlay() {
    const th = tanHalf();
    for (const b of sim.bodies) {
      const v = b.view;
      tmp.copy(v.group.position).project(camera);
      const vis = tmp.z < 1 && tmp.z > -1 && Math.abs(tmp.x) < 1.2 && Math.abs(tmp.y) < 1.2;
      v.onScreen = vis;
      if (vis) {
        v.sx = (tmp.x * 0.5 + 0.5) * W;
        v.sy = (-tmp.y * 0.5 + 0.5) * H;
        const dist = camera.position.distanceTo(v.group.position);
        v.rpx = (v.R / (dist * th)) * (H / 2);
      }
      if (v.dot) v.dot.visible = !vis || v.rpx < 3;
      const showLabel = S.labels && vis;
      v.label.style.display = showLabel ? '' : 'none';
      if (showLabel) v.label.style.transform = `translate(${(v.sx + Math.max(v.rpx, 3) + 6).toFixed(1)}px, ${(v.sy - 8).toFixed(1)}px)`;
    }
    const sv = selected?.view;
    if (sv?.onScreen) {
      const d = Math.max(sv.rpx * 2 + 12, 20);
      selRing.style.display = '';
      selRing.style.width = selRing.style.height = d + 'px';
      selRing.style.transform = `translate(${(sv.sx - d / 2).toFixed(1)}px, ${(sv.sy - d / 2).toFixed(1)}px)`;
    } else selRing.style.display = 'none';
  }

  function drainEvents() {
    for (const ev of sim.events) {
      if (ev.type !== 'merge') continue;
      const { survivor: s, absorbed: o } = ev;
      destroyView(o);
      if (ev.kindChanged && sim.bodies.includes(s)) { destroyView(s); createView(s); }
      spawnFlash(s, ev.x, ev.y, ev.z);
      if (selected === o || selected === s) selectBody(s);
      if (s.view && selected === s) s.view.label.classList.add('sel');
    }
    sim.events.length = 0;
  }

  // ---------- resize ----------
  const ro = new ResizeObserver(() => {
    const r = stage.getBoundingClientRect();
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    renderer.setSize(W, H, false);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
  });
  ro.observe(stage);

  // ---------- go ----------
  loadPreset('solar');
  let roTimer = 0;
  const frame = (dt) => {
    if (S.playing && sim.bodies.length + sim.tp.n > 0) {
      sim.updatePrimaries();
      sim.advance(S.speed * dt, 4000, performance.now() + 14);
      if (sim.events.length) drainEvents();
    }
    updateCenter();
    syncViews(dt);
    updateFollow(dt);
    controls.update();
    sky.position.copy(camera.position);
    renderer.render(scene, camera);
    updateOverlay();
    roTimer -= dt;
    if (roTimer <= 0) { roTimer = 0.1; updateReadouts(); }
  };
  const stopLoop = loop(frame);

  return () => {
    stopLoop();
    ro.disconnect();
    cvs.removeEventListener('pointerdown', onDown);
    cvs.removeEventListener('pointermove', onMove);
    cvs.removeEventListener('pointerup', onUp);
    cvs.removeEventListener('dblclick', onDbl);
    window.removeEventListener('keydown', onKey);
    controls.dispose();
    for (const b of sim.bodies) destroyView(b);
    for (const f of flashes) f.s.material.dispose();
    for (const c of gridGroup.children) c.geometry.dispose();
    for (const x of trash) x.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
    styleEl.remove();
  };
}

export default {
  id: 'solar',
  title: 'Solar System',
  glyph: '☉',
  tag: 'astronomy',
  blurb: 'A live N-body solar system where every body tugs on every other. Crank up the Sun, drop in stars and black holes, or fling a rogue star through the planets.',
  mount,
};
