// Registry: order here is the order in the nav and on the home page.
import fourier from './fourier.js';
import rocket from './rocket.js';
import solar from './solar.js';
import pendulum from './pendulum.js';
import gas from './gas.js';
import galton from './galton.js';
import clt from './clt.js';
import neural from './neural.js';

// Build a Net moved to its own app; this entry links there (and redirects old #/build links).
const build = {
  id: 'build',
  title: 'Build a Net',
  glyph: '⧉',
  tag: 'game',
  blurb: 'Learn how modern AI works by building it: wire networks from parts, from a convolution up to a Transformer block.',
  href: 'https://tensoratelier.com/build-a-net/',
};

export default [fourier, rocket, solar, pendulum, gas, galton, clt, neural, build];
