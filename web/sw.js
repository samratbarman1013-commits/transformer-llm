/* Service worker: cache-first for the app shell + model + CDN scripts,
 * so Transformer works fully offline after the first load.
 * v3 = CDN runtime caching (onnxruntime + fonts) + queue/stop UI. */
const CACHE = "transformer-v3";
const CORE = ["./", "index.html", "app.js", "manifest.json",
  "model/model.onnx", "model/vocab.json",
  "icons/icon-192.png", "icons/icon-512.png"];
/* cross-origin files we need offline (WASM runtime + fonts) */
const CDN = /jsdelivr\.net$|unpkg\.com$|fonts\.gstatic\.com$|fonts\.googleapis\.com$/;

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
      /* cache-first for CDN assets, fill on first success */
      e.respondWith(caches.match(e.request).then((hit) => hit ||
        fetch(e.request).then((r) => {
          const copy = r.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
          return r;
        })));
    }
    return; // everything else passes through
  }
  e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request)));
});
