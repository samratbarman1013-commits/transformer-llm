/* Service worker: cache-first for the app shell + model, so Transformer
 * works offline after the first load. */
const CACHE = "transformer-v1";
const CORE = ["./", "index.html", "app.js", "manifest.json",
  "model/model.onnx", "model/vocab.json",
  "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(CORE)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) =>
    Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  if (new URL(e.request.url).origin !== location.origin) return; // let CDN fetches pass through
  e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request)));
});
