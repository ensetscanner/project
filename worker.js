/* ============================================================
   EnsetScan Vision — Dedicated ML Worker
   Offloads image processing + heuristic inference off the main
   thread so camera acquisition and UI rendering never stutter.

   Protocol:
   { type: 'infer', imageData, width, height }
     → { type: 'result', probs, engine, elapsedMs, unsupported }

   NOTE: The shipped model is a placeholder (placeholder=true),
   so the app always uses the deterministic heuristic classifier.
   When a real trained model is added, extend this worker with a
   neural path (importScripts TF.js + load model inside worker).

   Safari fallback (bug #1): OffscreenCanvas is unsupported there.
   If canvas operations fail, this worker posts back a uniform
   distribution with `unsupported: true` so the main thread can
   render an explicit "please use Chrome" inconclusive result.
   ============================================================ */

'use strict';

importScripts('heuristic.js'); // shared classifier (bugs #4, #15)

const HEURISTIC_SIZE = 96; // heuristic working resolution

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'infer') return;
  const reqId = msg.reqId; // echoed back so the main thread can discard stale results

  const t0 = performance.now();

  let probs = null;
  let unsupported = false;
  try {
    const { data } = resizeToSquare(msg.imageData, msg.width, msg.height, HEURISTIC_SIZE);
    probs = analyzeHeuristicFromPixels(data);
  } catch (err) {
    // OffscreenCanvas missing (Safari) or any analysis failure → uniform
    // distribution so the main thread can render an "inconclusive" result.
    unsupported = (typeof OffscreenCanvas === 'undefined');
    probs = uniformProbs(HEURISTIC_CLASS_COUNT);
  }

  self.postMessage({
    type: 'result',
    reqId,
    probs,
    engine: 'heuristic',
    unsupported,
    elapsedMs: performance.now() - t0
  });
};

/** Center-square-crop raw pixels to `size` and return {data,width,height}.
 *  Throws where OffscreenCanvas is unavailable (bug #1). */
function resizeToSquare(data, w, h, size) {
  if (typeof OffscreenCanvas === 'undefined') {
    throw new Error('OffscreenCanvas is not supported in this browser');
  }
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
