/* Transformer web app — dark UI edition.
 * The model runs entirely in your browser via onnxruntime-web (WASM).
 * Chat history + pinned chats live ONLY in localStorage. */

const CTX = 256;
const $ = id => document.getElementById(id);
const chatEl = $('chat'), input = $('msgInput'), sendBtn = $('sendBtn'),
      toast = $('toast'), statusDot = $('statusDot');

let session = null, vocab = null, itos = [], stoi = {};
let busy = false;

/* ---------- persistence (on-device only) ---------- */
const LS_CHATS = 'transformer.chats.v2';
const LS_SET = 'transformer.settings.v1';

let chats;
try { chats = JSON.parse(localStorage.getItem(LS_CHATS) || '[]'); }
catch { chats = []; }
/* migrate old v1 history (no pinned flag) */
if (!localStorage.getItem(LS_CHATS)) {
  try {
    const old = JSON.parse(localStorage.getItem('transformer.chats.v1') || '[]');
    chats = old.map(c => ({ ...c, pinned: false }));
  } catch {}
}
let cur = null;

let settings = { temp: 0.8, topk: 40, maxnew: 220 };
try { Object.assign(settings, JSON.parse(localStorage.getItem(LS_SET) || '{}')); } catch {}

function saveChats() { localStorage.setItem(LS_CHATS, JSON.stringify(chats)); }
function saveSettings() { localStorage.setItem(LS_SET, JSON.stringify(settings)); }

const chatIcon = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
const pinIcon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1"/></svg>';

function histItem(c) {
  const d = document.createElement('div');
  d.className = 'hist-item' + (cur && c.id === cur.id ? ' active' : '') + (c.pinned ? ' pinned' : '');
  d.setAttribute('role', 'button'); d.tabIndex = 0; d.dataset.id = c.id;
  d.innerHTML = chatIcon + `<span></span><button class="pin-btn" aria-label="${c.pinned ? 'Unpin chat' : 'Pin chat'}" title="${c.pinned ? 'Unpin' : 'Pin'}">${pinIcon}</button>`;
  d.querySelector('span').textContent = c.title || 'New chat';
  return d;
}

function renderDrawer() {
  const pins = chats.filter(c => c.pinned), recent = chats.filter(c => !c.pinned);
  const pinList = $('pinList'), hist = $('hist');
  pinList.innerHTML = ''; hist.innerHTML = '';
  for (const c of pins) pinList.appendChild(histItem(c));
  for (const c of recent) hist.appendChild(histItem(c));
  const hasPins = pins.length > 0;
  $('pinLabel').hidden = pinList.hidden = !hasPins;
}

function newChat() {
  cur = { id: Date.now().toString(36), title: 'New chat', pinned: false, log: [] };
  chats.unshift(cur); saveChats(); renderDrawer(); renderThread();
}

function renderThread() {
  chatEl.querySelectorAll('.msg-user,.msg-ai,.typing').forEach(n => n.remove());
  const hello = $('hello');
  const msgs = cur ? cur.log : [];
  hello.style.display = msgs.length ? 'none' : 'flex';
  $('chatTitle').textContent = cur ? (cur.title === 'New chat' ? 'Transformer' : cur.title) : 'Transformer';
  for (const m of msgs) addMsg(m.role, m.text, false);
  scrollDown();
}

function addMsg(role, text, persist = true) {
  const d = document.createElement('div');
  if (role === 'user') {
    d.className = 'msg-user';
    const b = document.createElement('div'); b.className = 'bubble'; b.textContent = text;
    const cp = document.createElement('button'); cp.className = 'copy'; cp.dataset.copy = text;
    cp.setAttribute('aria-label', 'Copy');
    cp.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg><span>Copy</span>';
    d.append(b, cp);
  } else {
    d.className = 'msg-ai';
    const t = document.createElement('div'); t.className = 'ai-text'; t.textContent = text;
    const a = document.createElement('div'); a.className = 'actions';
    a.innerHTML = `<button class="act" data-copy aria-label="Copy" title="Copy"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg></button>`;
    a.querySelector('button').dataset.copy = text;
    d.append(t, a);
  }
  chatEl.appendChild(d); scrollDown();
  if (persist && cur) { cur.log.push({ role, text }); saveChats(); }
  return d;
}

function scrollDown() { chatEl.scrollTop = chatEl.scrollHeight; }

/* ---------- model loading ---------- */
async function loadModel() {
  try {
    const v = await (await fetch('model/vocab.json')).json();
    vocab = v.vocab; itos = vocab; stoi = {};
    vocab.forEach((ch, i) => stoi[ch] = i);
    session = await ort.InferenceSession.create('model/model.onnx',
      { executionProviders: ['wasm'] });
    statusDot.className = 'statusdot ready';
    statusDot.title = 'Model ready — runs on your device';
    sendBtn.disabled = false;
  } catch (e) {
    statusDot.className = 'statusdot error';
    statusDot.title = 'Model failed to load';
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
  const tensor = new ort.Tensor('int64', data, [1, trimmed.length]);
  const out = await session.run({ input_ids: tensor });
  const logits = Array.from(out.logits.data).slice((trimmed.length - 1) * vocab.length);
  return softmaxTopK(logits, settings.temp, settings.topk);
}

async function generate(prompt, maxNew) {
  let ids = [];
  for (const ch of ('\n' + prompt)) ids.push(stoi[ch] !== undefined ? stoi[ch] : (stoi[' '] ?? 0));
  let text = '';
  const stops = ['\nUser:', '\nTransformer:'];
  for (let i = 0; i < maxNew; i++) {
    const t = await nextToken(ids);
    ids.push(t);
    text += itos[t];
    if (stops.some(s => text.includes(s))) { text = text.split(stops.find(s => text.includes(s)))[0]; break; }
    if (text.trim().split('\n').length > 5) break;
  }
  return text.trim();
}

/* ---------- chat wiring ---------- */
async function submit() {
  const text = input.value.trim();
  if (!text || busy || !session) return;
  if (!cur) newChat();
  if (cur.log.length === 0) {
    cur.title = text.slice(0, 40);
    $('chatTitle').textContent = cur.title;
    saveChats(); renderDrawer();
  }
  input.value = '';
  addMsg('user', text);
  busy = true; sendBtn.disabled = true;

  const history = cur.log.map(m => (m.role === 'user' ? 'User: ' : 'Transformer:') + ' ' + m.text).join('\n');

  /* typing indicator while the model computes */
  const ty = document.createElement('div');
  ty.className = 'typing'; ty.innerHTML = '<span></span><span></span><span></span>';
  chatEl.appendChild(ty); scrollDown();

  try {
    const result = await generate(history + '\nTransformer:', settings.maxnew);
    ty.remove();
    const d = document.createElement('div');
    d.className = 'msg-ai';
    const t = document.createElement('div'); t.className = 'ai-text';
    const caret = document.createElement('span'); caret.className = 'caret';
    d.appendChild(t); chatEl.appendChild(d);
    for (const ch of result) {
      t.textContent += ch;
      t.appendChild(caret);
      scrollDown();
      await new Promise(r => setTimeout(r, 10));
    }
    caret.remove();
    const a = document.createElement('div'); a.className = 'actions';
    a.innerHTML = `<button class="act" data-copy aria-label="Copy" title="Copy"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg></button>`;
    a.querySelector('button').dataset.copy = result;
    d.appendChild(a);
    cur.log.push({ role: 'ai', text: result }); saveChats();
    if (cur.title === 'New chat') { cur.title = result.slice(0, 40) || 'New chat'; renderDrawer(); }
  } catch (e) {
    ty.remove();
    addMsg('ai', 'generation error: ' + e.message, false);
    console.error(e);
  }
  busy = false; sendBtn.disabled = false;
  scrollDown();
}

/* ---------- UI: drawer / menus / toast / sheet ---------- */
const drawer = $('drawer'), overlay = $('overlay'), menu = $('menu'), panel = $('settingsPanel');

const openDrawer = () => { closeMenus(); drawer.classList.add('show'); overlay.classList.add('show'); };
const closeDrawer = () => { drawer.classList.remove('show'); if (!$('sheetWrap').classList.contains('show')) overlay.classList.remove('show'); };
const closeMenus = () => { menu.classList.remove('show'); panel.classList.remove('show'); };

$('menuBtn').addEventListener('click', openDrawer);
$('drawerClose').addEventListener('click', closeDrawer);
overlay.addEventListener('click', () => { closeDrawer(); closeMenus(); closeSheet(); });

$('dotsBtn').addEventListener('click', e => {
  e.stopPropagation();
  const willShow = !menu.classList.contains('show');
  closeMenus();
  if (willShow) menu.classList.add('show');
});
document.addEventListener('click', e => {
  if (!menu.contains(e.target) && !panel.contains(e.target) && e.target.closest('#dotsBtn') === null) {
    closeMenus();
  }
});

/* toast */
let toastTimer;
function showToast(msg) {
  toast.textContent = msg; toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1400);
}

/* copy buttons (delegated) */
document.addEventListener('click', e => {
  const btn = e.target.closest('[data-copy]');
  if (!btn) return;
  const txt = btn.getAttribute('data-copy');
  if (navigator.clipboard) navigator.clipboard.writeText(txt).catch(() => {});
  showToast('Copied');
});

/* history + pin clicks (delegated) */
function bindHistLists() {
  [$('hist'), $('pinList')].forEach(list => {
    list.addEventListener('click', e => {
      const pin = e.target.closest('.pin-btn');
      const item = e.target.closest('.hist-item');
      if (!item) return;
      const id = item.dataset.id;
      const c = chats.find(x => x.id === id);
      if (!c) return;
      if (pin) {
        c.pinned = !c.pinned;
        saveChats(); renderDrawer();
        showToast(c.pinned ? 'Chat pinned' : 'Chat unpinned');
        return;
      }
      cur = c; renderDrawer(); renderThread(); closeDrawer();
    });
  });
}
bindHistLists();

/* new chat */
function startNewChat() {
  newChat(); closeDrawer(); closeMenus();
  $('msgInput').focus();
}
$('newChatBtn').addEventListener('click', startNewChat);
$('drawerNewChat').addEventListener('click', startNewChat);

/* settings menu items */
$('menuSettings').addEventListener('click', e => {
  e.stopPropagation();
  menu.classList.remove('show');
  panel.classList.add('show');
});
$('menuHelp').addEventListener('click', () => { closeMenus(); openSheet('Help', `
  <p>Everything on this page runs <b>locally in your browser</b> — the language model is downloaded once and generates text on your device.</p>
  <p>Chat history is stored only in this browser's localStorage. Clearing site data erases it permanently.</p>
  <p>Tip: the model is a small prototype — it continues your text creatively rather than answering questions.</p>`); });
$('menuAbout').addEventListener('click', () => { closeMenus(); openSheet('About', `
  <p><b>Transformer</b> — a decoder-only GPT trained from scratch.</p>
  <p>Model: 1M parameters · runs via onnxruntime-web (WebAssembly) · works offline after first load.</p>
  <p>No accounts, no servers, no telemetry. Your conversations never leave this device.</p>
  <p><a href="https://github.com/samratbarman1013-commits/transformer-llm" target="_blank" rel="noopener" style="color:var(--send)">View source on GitHub</a></p>`); });

/* settings controls */
const tempR = $('tempRange'), topkR = $('topkRange'), maxnewR = $('maxnewRange');
function syncSettingsUI() {
  tempR.value = settings.temp; $('tempVal').textContent = (+settings.temp).toFixed(2);
  topkR.value = settings.topk; $('topkVal').textContent = settings.topk;
  maxnewR.value = settings.maxnew; $('maxnewVal').textContent = settings.maxnew;
}
tempR.addEventListener('input', () => { settings.temp = +tempR.value; $('tempVal').textContent = settings.temp.toFixed(2); saveSettings(); });
topkR.addEventListener('input', () => { settings.topk = +topkR.value; $('topkVal').textContent = settings.topk; saveSettings(); });
maxnewR.addEventListener('input', () => { settings.maxnew = +maxnewR.value; $('maxnewVal').textContent = settings.maxnew; saveSettings(); });
syncSettingsUI();

$('clearAllBtn').addEventListener('click', e => {
  e.stopPropagation();
  closeMenus();
  openSheet('Clear all chats?', `
    <p>This permanently deletes every chat stored on this device. There is no undo.</p>`,
    'Delete everything', () => {
      chats = []; cur = null; saveChats();
      renderDrawer(); renderThread();
      showToast('All chats cleared');
    });
});

/* info sheet */
function openSheet(title, bodyHtml, okLabel, onOk) {
  $('sheetTitle').textContent = title;
  $('sheetBody').innerHTML = bodyHtml;
  const ok = $('sheetOk');
  ok.textContent = okLabel || 'OK';
  ok.onclick = () => { closeSheet(); if (onOk) onOk(); };
  $('sheetWrap').classList.add('show');
  overlay.classList.add('show');
}
function closeSheet() {
  $('sheetWrap').classList.remove('show');
  if (!drawer.classList.contains('show')) overlay.classList.remove('show');
}
$('sheetOk').addEventListener('click', closeSheet);
$('sheetWrap').addEventListener('click', e => { if (e.target === $('sheetWrap')) closeSheet(); });

/* hello chips */
$('hello').querySelectorAll('button[data-q]').forEach(b => {
  b.addEventListener('click', () => { input.value = b.dataset.q; submit(); });
});

/* send */
sendBtn.addEventListener('click', submit);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});

/* keep the app pinned above the on-screen keyboard */
const vv = window.visualViewport;
if (vv) {
  const syncViewport = () => {
    document.documentElement.style.setProperty('--app-height', vv.height + 'px');
    if (window.scrollX !== 0 || window.scrollY !== 0) window.scrollTo(0, 0);
  };
  vv.addEventListener('resize', syncViewport);
  vv.addEventListener('scroll', syncViewport);
  window.addEventListener('orientationchange', syncViewport);
  syncViewport();
}

/* go */
renderDrawer();
loadModel();
