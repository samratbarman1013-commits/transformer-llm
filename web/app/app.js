/* Transformer — thin-client app.
 * No model on this device: every reply comes from the owner's API server
 * (URL + key configured on first run). History stays in localStorage only. */

const $ = id => document.getElementById(id);
const chatEl = $('chat'), input = $('msgInput'), sendBtn = $('sendBtn'),
      toast = $('toast'), statusDot = $('statusDot');

let busy = false, abortCtl = null, tyEl = null, lastGenError = 0, stoppedAt = -1;

/* ---------- server config (on-device only) ---------- */
const LS_CFG = 'transformer.api.v1';
let cfg = null;
try { cfg = JSON.parse(localStorage.getItem(LS_CFG) || 'null'); } catch {}

/* ---------- chat history (on-device only) ---------- */
const LS_CHATS = 'transformer.chats.v2';
const LS_SET = 'transformer.settings.v1';

let chats;
try { chats = JSON.parse(localStorage.getItem(LS_CHATS) || '[]'); }
catch { chats = []; }
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
  $('pinLabel').hidden = pinList.hidden = pins.length === 0;
}

function newChat() {
  cur = { id: Date.now().toString(36), title: 'New chat', pinned: false, log: [] };
  chats.unshift(cur); saveChats(); renderDrawer(); renderThread();
}

function renderThread() {
  chatEl.querySelectorAll('.msg-user,.msg-ai,.msg-sys,.typing').forEach(n => n.remove());
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
  } else if (role === 'ai') {
    d.className = 'msg-ai';
    const t = document.createElement('div'); t.className = 'ai-text'; t.textContent = text;
    const a = document.createElement('div'); a.className = 'actions';
    a.innerHTML = `<button class="act" data-copy aria-label="Copy" title="Copy"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg></button>`;
    a.querySelector('button').dataset.copy = text;
    d.append(t, a);
  } else {
    d.className = 'msg-sys'; d.textContent = text;
    chatEl.appendChild(d); scrollDown();
    return d;
  }
  chatEl.appendChild(d); scrollDown();
  if (persist && cur) { cur.log.push({ role, text }); saveChats(); }
  return d;
}

function scrollDown() { chatEl.scrollTop = chatEl.scrollHeight; }

function showTyping() {
  if (!tyEl) {
    tyEl = document.createElement('div');
    tyEl.className = 'typing';
    tyEl.innerHTML = '<span></span><span></span><span></span>';
    chatEl.appendChild(tyEl);
  }
  scrollDown();
}
function hideTyping() { if (tyEl) { tyEl.remove(); tyEl = null; } }

/* ---------- server connection ---------- */
function apiBase() { return cfg.url.replace(/\/+$/, ''); }

async function pingServer(silent) {
  if (!cfg) {
    statusDot.className = 'statusdot error';
    statusDot.title = 'No server configured';
    return false;
  }
  statusDot.className = 'statusdot loading';
  statusDot.title = 'Connecting…';
  try {
    const r = await fetch(apiBase() + '/api/verify', { headers: { 'X-API-Key': cfg.key } });
    if (r.status === 401) throw new Error('The API key was rejected.');
    if (!r.ok) throw new Error('Server answered with error ' + r.status + '.');
    const d = await r.json();
    if (!d.ok) throw new Error('Server rejected the request.');
    statusDot.className = 'statusdot ready';
    const m = d.model || {};
    statusDot.title = m.loaded
      ? `Connected — ${m.config} (${(m.params / 1e6).toFixed(1)}M params on server)`
      : 'Connected, but the server has no model loaded yet';
    if (!silent) showToast(m.loaded ? 'Connected to server' : 'Server has no model loaded');
    return true;
  } catch (e) {
    statusDot.className = 'statusdot error';
    statusDot.title = 'Cannot reach server';
    if (!silent) showToast('Cannot reach server — check settings');
    return false;
  }
}

/* ---------- setup screen ---------- */
const setupWrap = $('setupWrap');
function openSetup() {
  $('cfgUrl').value = cfg ? cfg.url : '';
  $('cfgKey').value = cfg ? cfg.key : '';
  $('setupErr').textContent = '';
  $('setupTitle').textContent = cfg ? 'Server settings' : 'Connect to server';
  setupWrap.classList.add('show'); overlay.classList.add('show');
  setTimeout(() => $('cfgUrl').focus(), 150);
}
function closeSetup() {
  setupWrap.classList.remove('show');
  if (!overlayNeeded()) overlay.classList.remove('show');
}
$('cfgSave').addEventListener('click', async () => {
  const url = $('cfgUrl').value.trim(), key = $('cfgKey').value.trim();
  const err = $('setupErr');
  if (!/^https?:\/\/.+/.test(url)) { err.textContent = 'Enter a full server URL starting with http:// or https://'; return; }
  const btn = $('cfgSave');
  btn.disabled = true; btn.textContent = 'Connecting…';
  const prev = cfg;
  cfg = { url, key };
  const ok = await pingServer(true);
  btn.disabled = false; btn.textContent = 'Connect';
  if (!ok) {
    cfg = prev;
    err.textContent = 'Could not reach the server. Check the URL, the API key, and that the server is running.';
    return;
  }
  localStorage.setItem(LS_CFG, JSON.stringify(cfg));
  closeSetup();
  showToast('Server connected');
});

/* ---------- chat via API ---------- */
function setBusyUI(b) {
  busy = b;
  sendBtn.innerHTML = b
    ? '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>'
    : '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>';
  sendBtn.title = b ? 'Stop' : 'Send';
  sendBtn.setAttribute('aria-label', b ? 'Stop' : 'Send');
}

function maybeAutoContinue() {
  if (busy || !cur) return;
  if (Date.now() - lastGenError < 5000) return;
  const i = cur.log.length - 1;
  if (i <= stoppedAt) return;                 // user stopped this message on purpose
  const last = cur.log[i];
  if (last && last.role === 'user') runGeneration();
}

async function submit() {
  const text = input.value.trim();
  if (!text) return;
  if (!cfg) { openSetup(); showToast('Connect to a server first'); return; }
  input.value = '';
  if (!cur) newChat();
  if (cur.log.length === 0) {
    cur.title = text.slice(0, 40);
    $('chatTitle').textContent = cur.title;
    saveChats(); renderDrawer();
  }
  addMsg('user', text);
  if (busy) { showToast('Queued — will answer after this reply'); return; }
  runGeneration();
}

async function runGeneration() {
  setBusyUI(true);
  abortCtl = new AbortController();
  showTyping();

  /* history = everything before the message being answered */
  const hist = cur.log.slice(0, -1)
    .filter(m => m.role === 'user' || m.role === 'ai')
    .map(m => ({ role: m.role, text: m.text }));

  try {
    const r = await fetch(apiBase() + '/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': cfg.key },
      signal: abortCtl.signal,
      body: JSON.stringify({
        message: cur.log[cur.log.length - 1].text,
        history: hist,
        max_new: settings.maxnew,
        temperature: settings.temp,
        top_k: settings.topk,
      }),
    });
    if (r.status === 401) throw new Error('The server rejected the API key. Check Server settings.');
    if (r.status === 503) throw new Error('The server is running but no model is loaded on it.');
    if (!r.ok) throw new Error('Server error ' + r.status + '.');
    const d = await r.json();
    hideTyping();
    addMsg('ai', d.reply || '(empty reply)');
    if (cur.title === 'New chat' && d.reply) {
      cur.title = d.reply.slice(0, 40); saveChats(); renderDrawer();
      $('chatTitle').textContent = cur.title;
    }
    stoppedAt = -1;
  } catch (e) {
    hideTyping();
    if (e.name === 'AbortError') {
      stoppedAt = cur.log.length - 1;   // leave this message unanswered
      showToast('Stopped');
    } else {
      lastGenError = Date.now();
      addMsg('sys', String(e.message || e));
    }
  }
  abortCtl = null;
  setBusyUI(false);
  scrollDown();
  maybeAutoContinue();
}

/* ---------- UI: drawer / menus / toast / sheets ---------- */
const drawer = $('drawer'), overlay = $('overlay'), menu = $('menu'), panel = $('settingsPanel');

const openDrawer = () => { closeMenus(); drawer.classList.add('show'); overlay.classList.add('show'); };
const closeDrawer = () => { drawer.classList.remove('show'); if (!overlayNeeded()) overlay.classList.remove('show'); };
const closeMenus = () => { menu.classList.remove('show'); panel.classList.remove('show'); };
const overlayNeeded = () => drawer.classList.contains('show') || setupWrap.classList.contains('show') || $('sheetWrap').classList.contains('show');

$('menuBtn').addEventListener('click', openDrawer);
$('drawerClose').addEventListener('click', closeDrawer);
overlay.addEventListener('click', () => { closeDrawer(); closeMenus(); closeSheet(); closeSetup(); });

$('dotsBtn').addEventListener('click', e => {
  e.stopPropagation();
  const willShow = !menu.classList.contains('show');
  closeMenus();
  if (willShow) menu.classList.add('show');
});
document.addEventListener('click', e => {
  if (!menu.contains(e.target) && !panel.contains(e.target) && e.target.closest('#dotsBtn') === null) closeMenus();
});

let toastTimer;
function showToast(msg) {
  toast.textContent = msg; toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
}

document.addEventListener('click', e => {
  const btn = e.target.closest('[data-copy]');
  if (!btn) return;
  if (navigator.clipboard) navigator.clipboard.writeText(btn.getAttribute('data-copy')).catch(() => {});
  showToast('Copied');
});

function bindHistLists() {
  [$('hist'), $('pinList')].forEach(list => {
    list.addEventListener('click', e => {
      const pin = e.target.closest('.pin-btn');
      const item = e.target.closest('.hist-item');
      if (!item) return;
      const c = chats.find(x => x.id === item.dataset.id);
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

function startNewChat() {
  newChat(); closeDrawer(); closeMenus();
  $('msgInput').focus();
}
$('newChatBtn').addEventListener('click', startNewChat);
$('drawerNewChat').addEventListener('click', startNewChat);

/* menu items */
$('menuServer').addEventListener('click', () => { closeMenus(); openSetup(); });
$('menuSettings').addEventListener('click', e => {
  e.stopPropagation();
  menu.classList.remove('show');
  panel.classList.add('show');
});
$('menuAbout').addEventListener('click', () => {
  closeMenus();
  openSheet('About', `
  <p><b>Transformer</b> — a decoder-only GPT trained from scratch.</p>
  <p>This app is a thin client: it holds no model and works only online. The language model runs on the owner's private server.</p>
  <p>Your chat history is stored only on this device and never uploaded. Each message is sent to the server only to generate a reply.</p>`);
});

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
  if (!overlayNeeded()) overlay.classList.remove('show');
}
$('sheetOk').addEventListener('click', closeSheet);
$('sheetWrap').addEventListener('click', e => { if (e.target === $('sheetWrap')) closeSheet(); });

/* send / stop */
sendBtn.addEventListener('click', () => {
  if (busy) { if (abortCtl) abortCtl.abort(); }
  else submit();
});
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
renderThread();
if (cfg) pingServer(false);
else openSetup();
