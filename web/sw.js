/* Service worker: model + icons cache-first (big, never change),
 * code files (index.html, app.js, manifest) network-first so updates arrive
 * on every visit; cache is the offline fallback. v4 = auto-update. */
const CACHE = "transformer-v4";
const CORE = ["./", "index.html", "app.js", "manifest.json",
  "model/model.onnx", "model/vocab.json",
  "icons/icon-192.png", "icons/icon-512.png"];
/* cross-origin files we need offline (WASM runtime + fonts) */
const CDN = /jsdelivr\.net$|unpkg\.com$|fonts\.gstatic\.com$|fonts\.googleapis\.com$/;
/* always-fresh files: fetch from network first, update cache, fall back offline */
const FRESH = ["index.html", "app.js", "manifest.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(CORE)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) =>
    Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) {
    if (CDN.test(url.hostname)) {
      e.respondWith(caches.match(e.request).then((hit) => hit ||
        fetch(e.request).then((r) => {
          const copy = r.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
          return r;
        })));
    }
    return; // everything else passes through
  }
  if (FRESH.some(f => url.pathname.endsWith(f))) {
    /* network-first: users always get the newest code, offline falls back to cache */
    e.respondWith(fetch(e.request).then((r) => {
      const copy = r.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy));
      return r;
    }).catch(() => caches.match(e.request)));
    return;
  }
  e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request)));
});
