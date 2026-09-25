// Guided presets for the neural-net playground (DOM-free so tests can import them).
// Each preset is a full configuration plus a one-line explanation shown in the panel.

export const DEFAULT_CONFIG = {
  dataset: 'circle', noise: 0.05, points: 300, testFrac: 0.5, seed: 11,
  features: ['x1', 'x2'], hidden: [4, 2], activation: 'tanh',
  optimizer: 'sgd', lr: 0.03, batch: 10, l2: 0,
};

export const PRESETS = [
  {
    id: 'custom', name: 'Free play',
    hint: 'Pick a dataset, shape the network, press play. Click the heatmap to add amber points, shift- or right-click for cyan.',
    config: {},
  },
  {
    id: 'xor-linear', name: 'A linear model can’t do XOR',
    hint: 'No hidden layers = logistic regression: one straight boundary, but XOR needs two. It stalls near 50 %. Now tick the x₁x₂ feature — one product term makes XOR separable.',
    config: { dataset: 'xor', noise: 0, points: 300, seed: 3, features: ['x1', 'x2'], hidden: [], optimizer: 'sgd', lr: 0.03, batch: 10 },
  },
  {
    id: 'features', name: 'Features vs. hidden layers',
    hint: 'Still no hidden layers, but with x₁² and x₂² the model can draw a circle: a boundary that is linear in the features can be curved in the plane.',
    config: { dataset: 'circle', noise: 0.05, points: 300, seed: 5, features: ['x1', 'x2', 'x1sq', 'x2sq'], hidden: [], optimizer: 'sgd', lr: 0.1, batch: 10 },
  },
  {
    id: 'spiral-depth', name: 'Spirals need depth',
    hint: 'Three hidden layers fold the plane again and again until the arms separate. Remove layers (–) and watch it give up; add sin x₁, sin x₂ and a shallow net suddenly copes.',
    config: { dataset: 'spiral', noise: 0, points: 400, testFrac: 0.3, seed: 2, features: ['x1', 'x2'], hidden: [8, 8, 6], activation: 'tanh', optimizer: 'adam', lr: 0.01, batch: 10 },
  },
  {
    id: 'overfit', name: 'Overfitting',
    hint: 'Sixty noisy points, 177 parameters, no regularisation: train loss keeps falling while test loss turns and climbs — the net is memorising noise. Raise L2 to 0.01–0.03 and reset.',
    config: { dataset: 'circle', noise: 0.45, points: 60, testFrac: 0.5, seed: 1, features: ['x1', 'x2'], hidden: [8, 8, 8], activation: 'tanh', optimizer: 'adam', lr: 0.01, batch: 10, l2: 0 },
  },
  {
    id: 'lr-high', name: 'Learning rate too high',
    hint: 'At rate 3 each step overshoots the valley: loss jumps around at several times its starting value and accuracy never settles. Drag the rate down to ~0.03 and it converges within seconds.',
    config: { dataset: 'xor', noise: 0, points: 300, seed: 3, features: ['x1', 'x2'], hidden: [4], activation: 'tanh', optimizer: 'momentum', lr: 3, batch: 10 },
  },
  {
    id: 'relu', name: 'ReLU draws polygons',
    hint: 'Every ReLU neuron is a crease along a line, so the boundary is piecewise linear — look at the straight facets. Switch to tanh for a smooth curve.',
    config: { dataset: 'moons', noise: 0.1, points: 300, seed: 4, features: ['x1', 'x2'], hidden: [6, 6], activation: 'relu', optimizer: 'adam', lr: 0.01, batch: 10 },
  },
];

export const presetConfig = (p) => ({ ...DEFAULT_CONFIG, ...p.config, features: [...(p.config.features || DEFAULT_CONFIG.features)], hidden: [...(p.config.hidden || DEFAULT_CONFIG.hidden)] });
