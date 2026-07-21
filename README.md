# Liquid Glass Type

An interactive WebGL liquid-glass typography experiment with live camera refraction, hand tracking, per-letter rigid-body physics, and liquid merging.

## Features

- Live mirrored camera background with a procedural fallback
- Local MediaPipe hand tracking with pinch-to-grab interaction
- Independent gravity, rotation, collision, friction, and velocity-based bounce for every letter
- Mouse and touch dragging with release momentum
- Transparent glass refraction, narrow white highlights, RGB spectral edging, caustics, and soft shadows
- Liquid neck formation when letters touch
- Real-time controls for size, spacing, gloop, refraction, chromatic offset, frost, iridescence, and more

Camera frames and hand landmarks stay in the browser. The hand-tracking model and WebAssembly runtime are served locally from `public/mediapipe`.

## Run locally

```bash
pnpm install
pnpm dev
```

Open the local URL shown by Vite and allow camera access. Pinch your thumb and index finger over a letter, move while pinching, and release to throw it. Mouse and touch dragging use the same physics path.

## Commands

```bash
pnpm lint
pnpm build
```

Keyboard shortcuts:

- `H` — hide or show the interface
- `R` — reset letters and physics
- `S` — save the current canvas

## Stack

React, TypeScript, Vite, WebGL2, MediaPipe Tasks Vision, Tailwind CSS, and Motion.
