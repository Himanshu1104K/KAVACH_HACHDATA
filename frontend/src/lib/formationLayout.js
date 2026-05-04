/** @typedef {{ id: number; x: number; y: number; efficiency: number }} SoldierSlot */

function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function randomEfficiency() {
  return Math.floor(Math.random() * 100);
}

/**
 * Random formation: anchor (id 5) near center, others placed by polar coords with spacing.
 * Coordinates in 0–100 (same space as SVG viewBox).
 * @returns {SoldierSlot[]}
 */
export function generateRandomFormation() {
  const cx = 34 + Math.random() * 28;
  const cy = 34 + Math.random() * 24;

  const others = shuffle([1, 2, 3, 4, 6, 7, 8, 9, 10]);
  /** @type {{ x: number; y: number }[]} */
  const placed = [{ x: cx, y: cy }];
  /** @type {SoldierSlot[]} */
  const positions = [{ id: 5, x: cx, y: cy, efficiency: randomEfficiency() }];

  for (const id of others) {
    let x = cx;
    let y = cy;
    let placedOk = false;

    for (let attempt = 0; attempt < 90; attempt++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 9 + Math.random() * 28;
      x = clamp(cx + radius * Math.cos(angle), 7, 93);
      y = clamp(cy + radius * Math.sin(angle), 10, 90);
      if (placed.every((p) => dist(p, { x, y }) >= 7)) {
        placedOk = true;
        break;
      }
    }

    if (!placedOk) {
      const baseA = Math.random() * Math.PI * 2;
      for (let k = 0; k < 40; k++) {
        const t = baseA + k * 0.55;
        const radius = 11 + (k % 7) * 2.8;
        x = clamp(cx + radius * Math.cos(t), 7, 93);
        y = clamp(cy + radius * Math.sin(t), 10, 90);
        if (placed.every((p) => dist(p, { x, y }) >= 6)) break;
      }
    }

    placed.push({ x, y });
    positions.push({ id, x, y, efficiency: randomEfficiency() });
  }

  return positions.sort((a, b) => a.id - b.id);
}

/**
 * Convex hull (monotone chain) for perimeter polygon. Points: {x,y}.
 * @param {{ x: number; y: number }[]} points
 */
export function convexHull(points) {
  if (points.length < 3) return points;
  const pts = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}
