/* ============================================================
   EnsetScan Vision — Service Worker
   Cache-first offline strategy for the ~4.0 MB app shell.
   ============================================================ */

const CACHE_NAME = 'ensetscan-vision-v8';

const PRECACHE_URLS = [
  './',
  './index.html',
  './style.css',
  './app.js',
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
  // Noto Sans Ethiopic CSS + TensorFlow.js — runtime-cached on first fetch
  'https://fonts.googleapis.com/css2?family=Noto+Sans+Ethiopic:wght@400;600;700&display=swap',
];

// ---------- Install: pre-cache the app shell ----------
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
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
          .catch(() => caches.match(event.request))
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
          .catch(() => cached);
        return cached || networkFetch;
      })
    );
  }
});