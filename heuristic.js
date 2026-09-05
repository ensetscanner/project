/* ============================================================
   EnsetScan Vision — Shared Heuristic Classifier
   Loaded by BOTH the main thread (app.js via <script>) and the
   ML worker (worker.js via importScripts). Single source of truth
   for the deterministic 11-class fallback engine (bugs #4, #15).
   ============================================================ */

'use strict';

const HEURISTIC_CLASS_COUNT = 17;   // number of catalog classes (11 original + 6 new)
const HEURISTIC_TEMPERATURE = 0.7;  // softmax temperature (sharpens decisive cases)

/** Uniform probability vector (fallback when analysis fails). */
function uniformProbs(n) {
  const p = 1 / n;
  return new Array(n).fill(p);
}

/** Numerical softmax with optional temperature. */
function softmax(values, temperature = 1) {
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp((v - max) / temperature));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
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
  return Math.min(1, avg / 110); // normalize gradient energy
}

/**
 * Core heuristic classifier over raw RGBA pixel data (square crop).
 * Extracts color-region + texture features and scores each class
 * with deterministic evidence rules. Returns a probability array
 * of length HEURISTIC_CLASS_COUNT.
 */
function analyzeHeuristicFromPixels(data) {
  let green = 0, yellow = 0, orange = 0, brown = 0, gray = 0, total = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    total++;
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    const sat = mx === 0 ? 0 : (mx - mn) / mx;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;

    if (g > 70 && g >= r && g >= b && g - b > 15 && g - r > 8) {
      green++;                                  // healthy green tissue
    } else if (r > 140 && g > 130 && b < 115 && r - g < 60 && g - b > 25) {
      yellow++;                                 // chlorotic yellowing / streak
    } else if (r > 150 && g > 70 && g < 170 && b < 90 && r - g > 40) {
      orange++;                                 // rust-orange pustules
    } else if (r > 90 && g > 45 && g < r && b < g && r - g > 20) {
      brown++;                                  // necrotic brown spots
    } else if (sat < 0.18 && lum > 90 && lum < 210) {
      gray++;                                   // pale gray/tan lesions
    }
  }

  const side = Math.round(Math.sqrt(data.length / 4));
  const edgeDensity = computeEdgeDensity(data, side, side);

  const fg = green / total;
  const fy = yellow / total;
  const fo = orange / total;
  const fb = brown / total;
  const fgray = gray / total;
  const fe = edgeDensity;

  const rawScores = [
    fy * 3.0 + (1 - fg) * 1.5 + fb * 0.3 - fo * 2.2 - fgray * 0.5,  // 0 Enset BW
    fy * 2.6 + fe * 1.6 + fg * 0.4 - fo * 1.6 - fb * 0.5,           // 1 Enset Streak
    fo * 4.2 + fy * 1.8 - fg * 1.2 - fb * 0.6,                      // 2 Coffee Rust
    fb * 3.2 + fy * 1.2 + fg * 0.3 - fo * 1.0 - fgray * 0.3,        // 3 Brown Eye Spot
    fgray * 3.4 + fg * 0.4 - fo * 1.2 - fy * 0.8 - fb * 0.3,        // 4 N. Corn Blight
    fgray * 2.6 + fg * 0.9 - fo * 0.8 - fy * 0.4 - fb * 0.2,        // 5 Gray Leaf Spot
    fg * 3.6 + (1 - fy - fo - fb - fgray) * 1.5 - fe * 0.8,         // 6 Healthy
    fb * 3.0 + fo * 1.2 + fe * 0.6 - fg * 1.0 - fy * 0.4,           // 7 Coffee Berry Dis
    fy * 2.2 + (1 - fg) * 1.4 + fb * 0.8 - fe * 1.2 - fo * 0.8,     // 8 Coffee Wilt
    fy * 2.4 + fb * 1.8 + fe * 1.4 - fg * 1.2 - fo * 0.6,           // 9 Maize MLN
    fy * 2.2 + (1 - fg) * 1.6 - fe * 1.4 - fo * 1.0 - fb * 0.3,     // 10 Mealybug
    fb * 4.0 + fgray * 1.5 + fe * 0.6 - fg * 1.0 - fy * 0.4,        // 11 Black Leaf Spot
    fy * 2.4 + fb * 2.6 - fg * 0.8 - fe * 1.4 - fgray * 0.4,        // 12 Leaf Tip Dieback
    fe * 3.6 + (1 - fg) * 1.2 - fo * 1.4 - fgray * 0.6 - fb * 0.2,  // 13 Leaf Miner
    fgray * 3.8 + fe * 1.0 - fg * 1.4 - fy * 0.8 - fo * 0.4,        // 14 Sooty Mold
    fo * 4.4 + fe * 1.2 - fg * 0.8 - fgray * 1.0 - fb * 0.4,        // 15 Corn Rust
    fy * 3.2 + fe * 2.6 - fo * 1.6 - fb * 0.8 - fgray * 0.6         // 16 Maize Streak Virus
  ];

  return softmax(rawScores, HEURISTIC_TEMPERATURE);
}
