/* ============================================================
   EnsetScan Vision — Service Worker
   Cache-first offline strategy for the ~4.0 MB app shell.
   ============================================================ */

const CACHE_NAME = 'ensetscan-vision-v11';

// Same-origin app shell — REQUIRED for offline mode. Precached atomically.
const PRECACHE_URLS = [
  './',
  './index.html',
  './style.css',
  './logo.svg',
  './app.js',
  './heuristic.js',
  './icons.js',
  './worker.js',
  './manifest.webmanifest',
  './icon-192.png',
  './icon-512.png',
  './icon-192-maskable.png',
  './icon-512-maskable.png',
  './disease-catalog.json',
  './national-dashboard.json',
  './model.json',
  './group1-shard1of1.bin',
];

// Cross-origin CDN assets (Noto Sans Ethiopic CSS + font files + TensorFlow.js)
// — best-effort precache. A failed font fetch must NEVER abort the install and
// disable offline mode for the whole app (audit fix). These are also runtime-
// cached on first use by the CDN branch of the fetch handler.
const CDN_PRECACHE_URLS = [
  'https://fonts.googleapis.com/css2?family=Noto+Sans+Ethiopic:wght@400;600;700&display=swap',
  // Verified font file URLs served by Google Fonts (the CSS references these;
  // browsers requesting .woff2 variants get them via the runtime cache rule).
  'https://fonts.gstatic.com/s/notosansethiopic/v50/7cHPv50vjIepfJVOZZgcpQ5B9FBTH9KGNfhSTgtoow1KVnIvyBoMSzUMacb-T35OK6Dj.ttf',
  'https://fonts.gstatic.com/s/notosansethiopic/v50/7cHPv50vjIepfJVOZZgcpQ5B9FBTH9KGNfhSTgtoow1KVnIvyBoMSzUMacb-T36QLKDj.ttf',
  'https://fonts.gstatic.com/s/notosansethiopic/v50/7cHPv50vjIepfJVOZZgcpQ5B9FBTH9KGNfhSTgtoow1KVnIvyBoMSzUMacb-T36pLKDj.ttf',
  // TensorFlow.js — best-effort precache so the neural/model path (if a real
  // model is ever shipped) also works offline. jsdelivr sends CORS headers, so
  // add() can cache it.
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js',
];

// ---------- Install: pre-cache the app shell ----------
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) =>
        // App shell: atomic addAll (all same-origin, must succeed).
        cache.addAll(PRECACHE_URLS)
          // CDN assets: best-effort — tolerate individual failures (audit fix).
          .then(() => Promise.allSettled(CDN_PRECACHE_URLS.map((u) => cache.add(u))))
      )
      .then(() => self.skipWaiting())
  );
});

// ---------- Activate: clean up old caches ----------
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

// ---------- Fetch: cache-first with network fallback ----------
self.addEventListener('fetch', (event) => {
  // Only handle GET requests within the app scope
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Cache-first for same-origin app assets
  if (url.origin === self.location.origin) {
    // Network-first for data files so we always get fresh content when online
    // (Files are flat at the root after restructuring)
    if (url.pathname.endsWith('.json') && !url.pathname.endsWith('manifest.webmanifest')) {
      event.respondWith(
        fetch(event.request)
          .then((response) => {
            if (response && response.status === 200) {
              const copy = response.clone();
              caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
            }
            return response;
          })
          .catch(() =>
            caches.match(event.request).then((m) => m || new Response(
              '{"offline":true}',
              { status: 503, headers: { 'Content-Type': 'application/json' } }
            ))
          )
      );
      return;
    }

    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request)
          .then((response) => {
            // Cache successful same-origin responses for offline use
            if (response && response.status === 200 && response.type === 'basic') {
              const copy = response.clone();
              caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
            }
            return response;
          })
          .catch(() => {
            // Only fall back to the app shell for navigation requests;
            // for assets, return a 404-style empty response instead of HTML.
            if (event.request.mode === 'navigate') {
              return caches.match('./index.html');
            }
            return new Response('', { status: 404, statusText: 'Not Found' });
          });
      })
    );
    return;
  }

  // Network-first for CDN (TensorFlow.js + Google Fonts), with runtime cache fallback
  if (url.hostname.includes('jsdelivr.net') ||
      url.hostname.includes('cdn.') ||
      url.hostname.includes('fonts.googleapis.com') ||
      url.hostname.includes('fonts.gstatic.com')) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        const networkFetch = fetch(event.request)
          .then((response) => {
            if (response && response.status === 200) {
              const copy = response.clone();
              caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
            }
            return response;
          })
          .catch(() => cached || new Response('', { status: 504, statusText: 'Offline' }));
        return cached || networkFetch;
      })
    );
  }
});