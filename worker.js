/* ============================================================
   EnsetScan Vision — Dedicated ML Worker
   Offloads image processing + heuristic inference off the main
   thread so camera acquisition and UI rendering never stutter.

   Protocol:
   { type: 'infer', imageData, width, height }
     → { type: 'result', probs, engine, elapsedMs }

   NOTE: The shipped model is a placeholder (placeholder=true),
   so the app always uses the deterministic heuristic classifier.
   When a real trained model is added, extend this worker with a
   neural path (importScripts TF.js + load model inside worker).
   ============================================================ */

'use strict';

const HEURISTIC_SIZE = 96; // heuristic working resolution

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'infer') return;

  const t0 = performance.now();

  let probs;
  try {
    probs = analyzeHeuristic(msg.imageData, msg.width, msg.height);
  } catch (err) {
    // If anything fails, return a uniform distribution so the
    // main thread can still render an "inconclusive" result.
    probs = uniformProbs(11);
  }

  self.postMessage({
    type: 'result',
    probs,
    engine: 'heuristic',
    elapsedMs: performance.now() - t0
  });
};

/** Uniform probability vector (fallback when analysis fails). */
function uniformProbs(n) {
  const p = 1 / n;
  return new Array(n).fill(p);
}

/** Center-square-crop raw pixels to `size` and return {data,width,height}. */
function resizeToSquare(data, w, h, size) {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const src = new ImageData(new Uint8ClampedArray(data), w, h);
  const temp = new OffscreenCanvas(w, h);
  temp.getContext('2d').putImageData(src, 0, 0);

  const side = Math.min(w, h);
  const sx = (w - side) / 2;
  const sy = (h - side) / 2;
  ctx.drawImage(temp, sx, sy, side, side, 0, 0, size, size);
  const out = ctx.getImageData(0, 0, size, size);
  return { data: out.data, width: size, height: size };
}

/** Simple Sobel-like edge density on luminance, normalized to [0,1]. */
function computeEdgeDensity(data, w, h) {
  let sum = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = (y * w + x) * 4;
      const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      const lr = 0.299 * data[i + 4] + 0.587 * data[i + 5] + 0.114 * data[i + 6];
      const ld = 0.299 * data[i + w * 4] + 0.587 * data[i + w * 4 + 1] + 0.114 * data[i + w * 4 + 2];
      sum += Math.abs(l - lr) + Math.abs(l - ld);
      count++;
    }
  }
  const avg = count > 0 ? sum / count : 0;
  return Math.min(1, avg / 110);
}

/** Numerical softmax with optional temperature. */
function softmax(values, temperature = 1) {
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp((v - max) / temperature));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

/**
 * Heuristic fallback classifier (11 classes).
 * Extracts color-region + texture features and scores each class
 * with deterministic evidence rules.
 */
function analyzeHeuristic(rawData, srcW, srcH) {
  const { data, width, height } = resizeToSquare(rawData, srcW, srcH, HEURISTIC_SIZE);

  let green = 0, yellow = 0, orange = 0, brown = 0, gray = 0, total = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    total++;
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    const sat = mx === 0 ? 0 : (mx - mn) / mx;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;

    if (g > 70 && g >= r && g >= b && g - b > 15 && g - r > 8) {
      green++;
    } else if (r > 140 && g > 130 && b < 115 && r - g < 60 && g - b > 25) {
      yellow++;
    } else if (r > 150 && g > 70 && g < 170 && b < 90 && r - g > 40) {
      orange++;
    } else if (r > 90 && g > 45 && g < r && b < g && r - g > 20) {
      brown++;
    } else if (sat < 0.18 && lum > 90 && lum < 210) {
      gray++;
    }
  }

  const edgeDensity = computeEdgeDensity(data, width, height);

  const fg = green / total;
  const fy = yellow / total;
  const fo = orange / total;
  const fb = brown / total;
  const fgray = gray / total;
  const fe = edgeDensity;

  const rawScores = [
    fy * 3.0 + (1 - fg) * 1.5 + fb * 0.3 - fo * 2.2 - fgray * 0.5,        // 0 Enset BW
    fy * 2.6 + fe * 1.6 + fg * 0.4 - fo * 1.6 - fb * 0.5,                  // 1 Enset Streak
    fo * 4.2 + fy * 1.8 - fg * 1.2 - fb * 0.6,                             // 2 Coffee Rust
    fb * 3.2 + fy * 1.2 + fg * 0.3 - fo * 1.0 - fgray * 0.3,               // 3 Brown Eye Spot
    fgray * 3.4 + fg * 0.4 - fo * 1.2 - fy * 0.8 - fb * 0.3,               // 4 N. Corn Blight
    fgray * 2.6 + fg * 0.9 - fo * 0.8 - fy * 0.4 - fb * 0.2,               // 5 Gray Leaf Spot
    fg * 3.6 + (1 - fy - fo - fb - fgray) * 1.5 - fe * 0.8,                // 6 Healthy
    fb * 3.0 + fo * 1.2 + fe * 0.6 - fg * 1.0 - fy * 0.4,                  // 7 Coffee Berry Dis
    fy * 2.2 + (1 - fg) * 1.4 + fb * 0.8 - fe * 1.2 - fo * 0.8,            // 8 Coffee Wilt
    fy * 2.4 + fb * 1.8 + fe * 1.4 - fg * 1.2 - fo * 0.6,                  // 9 Maize MLN
    fy * 2.2 + (1 - fg) * 1.6 - fe * 1.4 - fo * 1.0 - fb * 0.3             // 10 Mealybug
  ];

  return softmax(rawScores, 0.7);
}