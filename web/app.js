/* Transformer web app — the model runs entirely in your browser via
 * onnxruntime-web (WebAssembly). Chat history is stored ONLY in
 * localStorage; nothing is ever uploaded. */

const CTX = 256;
const $ = (id) => document.getElementById(id);
const msgs = $("msgs"), input = $("input"), send = $("send"), statusEl = $("status");

let session = null, vocab = null, itos = [], stoi = {};
let busy = false;

/* ---------- persistence (on-device only) ---------- */
const LS_KEY = "transformer.chats.v1";
let chats = JSON.parse(localStorage.getItem(LS_KEY) || "[]");   // [{id,title,log:[...]}]
let cur = null;

function saveChats() { localStorage.setItem(LS_KEY, JSON.stringify(chats)); }
function newChat(title) {
  cur = { id: Date.now().toString(36), title: title || "New chat", log: [] };
  chats.unshift(cur); saveChats(); renderSidebar(); renderThread();
}
function renderSidebar() {
  const el = $("chats"); el.innerHTML = "";
  for (const c of chats) {
    const d = document.createElement("div");
    d.className = "chat-item" + (cur && c.id === cur.id ? " active" : "");
    d.textContent = c.title; d.title = c.title;
    d.onclick = () => { cur = c; renderSidebar(); renderThread(); if (innerWidth <= 760) $("side").classList.remove("open"); };
    el.appendChild(d);
  }
}
function renderThread() {
  msgs.innerHTML = "";
  for (const m of (cur ? cur.log : [])) addMsg(m.role, m.text, false);
  if (!cur) msgs.appendChild(buildHello());
  scrollDown();
}
function buildHello() {
  // rebuild the welcome screen from the template in index.html
  const h = document.createElement("div"); h.className = "hello";
  h.innerHTML = document.getElementById("hello").innerHTML;
  h.querySelectorAll("button[data-q]").forEach(b =>
    b.onclick = () => { input.value = b.dataset.q; submit(); });
  return h;
}
function addMsg(role, text, persist = true) {
  const d = document.createElement("div");
  d.className = "msg " + (role === "user" ? "user" : role === "sys" ? "sys" : "ai");
  d.textContent = text;
  if (role === "ai") d.appendChild(Object.assign(document.createElement("span"), { className: "cursor" }));
  msgs.appendChild(d); scrollDown();
  if (persist && cur) { cur.log.push({ role, text }); saveChats(); }
  return d;
}
function scrollDown() { $("thread").scrollTop = $("thread").scrollHeight; }

/* ---------- model loading ---------- */
async function loadModel() {
  try {
    const v = await (await fetch("model/vocab.json")).json();
    vocab = v.vocab; itos = vocab; stoi = {};
    vocab.forEach((ch, i) => stoi[ch] = i);
    session = await ort.InferenceSession.create("model/model.onnx",
      { executionProviders: ["wasm"] });
    statusEl.textContent = "1M params · on-device · ready";
    statusEl.className = "ready";
  } catch (e) {
    statusEl.textContent = "model failed to load";
    statusEl.className = "error";
    console.error(e);
  }
}

/* ---------- sampling ---------- */
function softmaxTopK(logits, temperature, k) {
  const t = Math.max(temperature, 1e-6);
  const scaled = logits.map(x => x / t);
  const idx = scaled.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]).slice(0, k);
  const max = idx[0][0];
  const exps = idx.map(([x]) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let i = 0; i < exps.length; i++) { r -= exps[i]; if (r <= 0) return idx[i][1]; }
  return idx[idx.length - 1][1];
}

async function nextToken(ids) {
  const trimmed = ids.length > CTX ? ids.slice(ids.length - CTX) : ids;
  const data = new BigInt64Array(trimmed.map(BigInt));
  const tensor = new ort.Tensor("int64", data, [1, trimmed.length]);
  const out = await session.run({ input_ids: tensor });
  const logits = Array.from(out.logits.data).slice((trimmed.length - 1) * vocab.length);
  return softmaxTopK(logits, 0.8, 40);
}

/* ---------- generation loop ---------- */
async function generate(prompt, maxNew = 220) {
  let ids = [];
  for (const ch of ("\n" + prompt)) ids.push(stoi[ch] !== undefined ? stoi[ch] : stoi[" "] ?? 0);
  let text = "";
  const stops = ["\nUser:", "\nTransformer:"];
  for (let i = 0; i < maxNew; i++) {
    const t = await nextToken(ids);
    ids.push(t);
    const ch = itos[t];
    text += ch;
    if (stops.some(s => text.includes(s))) { text = text.split(stops.find(s => text.includes(s)))[0]; break; }
    if (ch === "\n" && text.trim().split("\n").length > 4) break;
  }
  return text.trim();
}

/* ---------- chat wiring ---------- */
async function submit() {
  const text = input.value.trim();
  if (!text || busy || !session) return;
  if (!cur) newChat(text.slice(0, 40));
  if (cur.log.length === 0) { cur.title = text.slice(0, 40); saveChats(); renderSidebar(); }
  input.value = ""; autosize();
  addMsg("user", text);
  busy = true; send.disabled = true;

  const history = cur.log.map(m => (m.role === "user" ? "User: " : "Transformer:") + " " + m.text).join("\n");
  const bubble = addMsg("ai", "");
  const cursor = document.createElement("span"); cursor.className = "cursor";
  try {
    const prompt = history + "\nTransformer:";
    const result = await generate(prompt);
    for (const ch of result) {
      bubble.textContent += ch;
      bubble.appendChild(cursor);
      scrollDown();
      await new Promise(r => setTimeout(r, 8));
    }
    bubble.textContent = result;
    cur.log.push({ role: "ai", text: result }); saveChats();
  } catch (e) {
    bubble.textContent = "generation error: " + e.message;
    console.error(e);
  }
  busy = false; send.disabled = false;
  scrollDown();
}

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

send.onclick = submit;
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
});
input.addEventListener("input", autosize);
$("newchat").onclick = () => newChat();
$("menu-btn").onclick = () => $("side").classList.toggle("open");

renderSidebar();
loadModel();
