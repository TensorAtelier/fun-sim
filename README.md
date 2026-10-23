# Fun Sim

Interactive math & physics simulations in the browser. Vite + vanilla JS; three.js for 3D.

**Play it: <https://tensoratelier.com/fun-sim/>** — part of [Tensor Atelier](https://tensoratelier.com).

| Sim | File | What it shows |
|---|---|---|
| Fourier epicycles | `src/sims/fourier.js` | Upload an image or draw → edge path → DFT → rotating circles redraw it |
| Rocket launch | `src/sims/rocket.js` | Two-stage rocket, drag, gravity turn, orbital elements |
| Solar system | `src/sims/solar.js` | 3D N-body; edit masses, add stars/black holes/asteroids |
| Double pendulum | `src/sims/pendulum.js` | Chaos: an ensemble of near-identical pendulums diverging |
| Gas & collisions | `src/sims/gas.js` | Hard-disk gas relaxing to Maxwell–Boltzmann, pressure, diffusion |
| Galton board | `src/sims/galton.js` | Binomial → normal, Pascal's triangle |
| Stats Lab | `src/sims/clt.js` | CLT, Monte Carlo π, random walks, gambler's ruin |
| Neural Net | `src/sims/neural.js` | Live-training MLP: decision boundary, per-neuron maps, loss curves |

## Run
```
npm install
npm run dev        # http://localhost:8870  (or: dev start fun-sim)
npm run build      # static site in dist/
```

Pushing to `main` deploys to GitHub Pages (`.github/workflows/pages.yml`, built with `--base=/fun-sim/`).

## Adding a simulation
Create `src/sims/<id>.js` exporting `{ id, title, glyph, tag, blurb, mount(root) }`, where
`mount` builds its UI with the helpers in `src/ui.js` and returns a cleanup function.
Then add it to `src/sims/index.js`.

Entries with an `href` instead of `mount` are links out: they show on the home page and in the nav,
and their old `#/<id>` route redirects there.

## Build a Net
The Build a Net game started here and moved to its own repo on 2026-10-23:
<https://github.com/TensorAtelier/build-a-net>, played at <https://tensoratelier.com/build-a-net/>.
Old `#/build` links redirect there.

## License
Apache-2.0
