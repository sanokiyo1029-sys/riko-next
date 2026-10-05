import { AudioEngine } from './audio.js';
import { createAvatar } from './avatar.js';

const $ = (id) => document.getElementById(id);
const token = new URLSearchParams(location.search).get('token');
const withToken = (url) => (token ? `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}` : url);
const TAG_RE = /\[\s*(neutral|happy|excited|smug|embarrassed|sad|angry|surprised|thinking|concerned)\s*(:\s*\d*\.?\d+)?\s*\]\s*/gi;
const stripTags = (s) => s.replace(TAG_RE, '').replace(/<think>[\s\S]*?(<\/think>|$)/g, '');
const LAYER_LABEL = { semantic: 'Fact', episodic: 'Event', working: 'Current goal' };
const HINTS = {
  offline: 'Reconnecting to Riko…',
  listening: 'Just talk. Hold Space to talk over her, Esc to stop her.',
  transcribing: 'Catching that…',
  thinking: 'Thinking…',
  speaking: 'Start talking to interrupt.',
};

const state = {
  ws: null, settings: {}, activeTurn: 0, rikoLine: null, tokenText: '', sentences: new Map(),
  started: false, retry: 0, avatar: null, audio: null, speaker: 'none', userSpeaking: false,
  pttHeld: false, historyLoaded: false, memIds: new Set(),
};

// ------------------------------------------------------------------ connection
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(withToken(`${proto}://${location.host}/ws`));
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  setConn(null, 'Connecting');
  ws.onopen = () => { state.retry = 0; setConn(true, 'Connected'); };
  ws.onclose = (e) => {
    setConn(false, e.code === 1008 ? 'Access denied (check ?token=)' : 'Disconnected');
    document.body.dataset.state = 'offline';
    $('hint').textContent = HINTS.offline;
    if (e.code !== 1008) setTimeout(connect, Math.min(8000, 500 * 2 ** state.retry++));
  };
  ws.onmessage = (e) => (typeof e.data === 'string' ? onEvent(JSON.parse(e.data)) : onAudio(e.data));
}
const send = (obj) => state.ws?.readyState === 1 && state.ws.send(JSON.stringify(obj));
function setConn(ok, text) {
  $('conn').dataset.ok = ok === null ? '' : String(ok);
  $('connText').textContent = text;
}

// ------------------------------------------------------------------ server events
function onAudio(buf) {
  const dv = new DataView(buf);
  if (buf.byteLength < 16 || dv.getUint32(0, true) !== 0x31414b52) return;   // "RKA1"
  const turn = dv.getUint32(4, true), idx = dv.getUint32(8, true), sr = dv.getUint32(12, true);
  state.audio?.playChunk(turn, idx, sr, new Int16Array(buf, 16));
}

function onEvent(ev) {
  switch (ev.type) {
    case 'hello':
    case 'settings':
      applySettings(ev.settings);
      if (ev.emotion) setEmotion(ev.emotion);
      if (!state.historyLoaded) loadHistory();
      break;
    case 'state':
      document.body.dataset.state = ev.state;
      $('hint').textContent = HINTS[ev.state] || '';
      break;
    case 'user_speech_started':
      state.userSpeaking = true;
      break;
    case 'user_speech_stopped':
      state.userSpeaking = false;
      break;
    case 'transcription':
      if (ev.text) { addLine('you', ev.text); subtitle(ev.text, 'you'); }
      else if (ev.reason === 'no_speech') note("Didn't catch that. Try again a little closer to the mic.");
      break;
    case 'llm_started':
      state.activeTurn = ev.turn_id;
      state.audio?.setActiveTurn(ev.turn_id);
      state.tokenText = '';
      state.rikoLine = addLine('riko', '');
      break;
    case 'llm_token':
      if (ev.turn_id !== state.activeTurn || !state.rikoLine) break;
      state.tokenText += ev.text;
      state.rikoLine.querySelector('.text').textContent = stripTags(state.tokenText).trimStart();
      scrollLog();
      break;
    case 'llm_sentence':
      state.sentences.set(`${ev.turn_id}:${ev.index}`, ev.text);
      if (!state.settings.show_tokens && state.rikoLine) {
        const t = state.rikoLine.querySelector('.text');
        t.textContent = (t.textContent + ' ' + ev.text).trim();
      }
      if (state.settings.tts_provider === 'none' || state.settings.muted) subtitle(ev.text, 'riko');
      break;
    case 'llm_finished':
      if (state.rikoLine && ev.text) state.rikoLine.querySelector('.text').textContent = ev.text;
      break;
    case 'tts_speak':
      if (ev.turn_id === state.activeTurn) {
        state.audio?.speak(ev.text, { speed: ev.speed, onStart: () => subtitle(ev.text, 'riko') });
      }
      break;
    case 'assistant_stopped':
      if (ev.reason !== 'completed' && ev.reason !== 'error') {
        state.audio?.flush();
        if (ev.turn_id === state.activeTurn) state.rikoLine?.classList.add('interrupted');
        subtitle('', 'riko');
      }
      break;
    case 'assistant_duck':
      state.audio?.duck(ev.volume);
      break;
    case 'assistant_unduck':
      state.audio?.duck(1);
      break;
    case 'emotion_changed':
      setEmotion(ev);
      break;
    case 'memory_created':
      note(`Remembered: ${ev.content}`, 'memory');
      addMemory(ev, 'fresh', true);
      break;
    case 'memory_retrieved':
      for (const m of ev.items || []) addMemory(m, 'recalled');
      break;
    case 'metrics':
      if (ev.first_audio_ms) $('latencyInfo').textContent =
        `Last reply: first word ${ev.llm_first_token_ms ?? '–'} ms, first audio ${ev.first_audio_ms} ms after your words were transcribed.`;
      break;
    case 'error':
      note(errorText(ev), 'error');
      break;
  }
}

function errorText(ev) {
  const fixes = {
    tts: 'Riko will keep replying in text. Start the GPT-SoVITS API or switch tts.provider to "browser".',
    llm: 'Check that your model server is running, or run "riko check".',
    asr: 'Speech recognition failed. Run "riko check" for details.',
  };
  return `${ev.component ? ev.component.toUpperCase() + ': ' : ''}${ev.message}${fixes[ev.component] ? ' ' + fixes[ev.component] : ''}`;
}

// ------------------------------------------------------------------ UI helpers
function addLine(who, text) {
  $('logPanel').querySelector('.empty')?.remove();
  const el = document.createElement('p');
  el.className = `line ${who}`;
  el.innerHTML = `<span class="who"></span><span class="text"></span>`;
  el.querySelector('.who').textContent = who === 'you' ? 'You' : (state.settings.character_name || 'Riko');
  el.querySelector('.text').textContent = text;
  $('logPanel').appendChild(el);
  scrollLog();
  return el;
}
function note(text, kind = '') {
  const el = document.createElement('p');
  el.className = `note ${kind}`;
  el.textContent = text;
  $('logPanel').appendChild(el);
  scrollLog();
}
const scrollLog = () => { const l = $('logPanel'); l.scrollTop = l.scrollHeight; };

let subtitleTimer = 0;
function subtitle(text, who) {
  const s = $('subtitle');
  s.textContent = text;
  s.dataset.who = who;
  clearTimeout(subtitleTimer);
  if (text) subtitleTimer = setTimeout(() => { if (!state.audio?.playing) s.textContent = ''; }, 6000);
}

function setEmotion(e) {
  $('emotion').textContent = e.emotion === 'neutral' ? 'calm' : e.emotion;
  state.avatar?.setEmotion(e);
}

function fillSelect(sel, items, current, label = (x) => x) {
  sel.replaceChildren(...items.map((v) => { const o = new Option(label(v), v); o.selected = v === current; return o; }));
}

function applySettings(s) {
  state.settings = s;
  $('charName').textContent = s.character_name;
  document.title = `${s.character_name} · Riko Next`;
  fillSelect($('presetSelect'), s.presets, s.preset, (p) => p[0].toUpperCase() + p.slice(1));
  fillSelect($('characterSelect'), s.characters, s.character);
  fillSelect($('voiceSelect'), s.voices, s.voice);
  if (!$('modelSelect').options.length) fillSelect($('modelSelect'), [s.model], s.model);
  document.querySelectorAll('input[name=inputMode]').forEach((r) => (r.checked = r.value === s.input_mode));
  $('talkBtn').textContent = s.input_mode === 'ptt' ? 'Hold to talk' : 'Hold to talk over';
  $('muteBtn').setAttribute('aria-pressed', String(!s.muted));
  $('muteBtn').textContent = s.muted ? 'Voice off' : 'Voice on';
  $('engineInfo').textContent = `LLM ${s.llm_provider} · speech-to-text ${s.asr_provider} · voice ${s.tts_provider}` +
    `${s.memory ? ' · memory on' : ''}`;
  state.audio?.setMuted(s.muted);
}

async function loadHistory() {
  state.historyLoaded = true;
  try {
    const r = await fetch(withToken('/api/history?limit=30'));
    const { messages } = await r.json();
    if (!messages.length) {
      const p = document.createElement('p');
      p.className = 'empty';
      p.textContent = `Say hello to ${state.settings.character_name || 'Riko'}, or type below.`;
      $('logPanel').appendChild(p);
    }
    for (const m of messages) {
      const el = addLine(m.role === 'user' ? 'you' : 'riko', m.text);
      if (m.interrupted) el.classList.add('interrupted');
    }
    if (messages.length) note('Earlier conversation');
  } catch { /* history is optional */ }
}

// ------------------------------------------------------------------ memory panel
function addMemory(m, cls, prepend = false) {
  const list = $('memList');
  let li = list.querySelector(`[data-id="${m.id}"]`);
  if (!li) {
    li = document.createElement('li');
    li.dataset.id = m.id;
    li.innerHTML = `<span class="content"></span><span class="kind"></span><span class="strength"></span><button>Forget</button>`;
    li.querySelector('.content').textContent = m.content;
    li.querySelector('.kind').textContent = `${LAYER_LABEL[m.layer] || m.layer} · ${m.category}`;
    li.querySelector('.strength').style.width = `${Math.round(100 * (m.strength ?? m.importance ?? 0.5))}%`;
    li.querySelector('button').onclick = () => forgetMemory(m.id, li);
    prepend ? list.prepend(li) : list.append(li);
    state.memIds.add(m.id);
  }
  li.classList.remove('fresh', 'recalled');
  void li.offsetWidth;
  li.classList.add(cls);
  $('memCount').textContent = state.memIds.size ? String(state.memIds.size) : '';
}
async function loadMemories() {
  try {
    const r = await fetch(withToken('/api/memories?limit=200'));
    const { memories } = await r.json();
    $('memList').replaceChildren();
    state.memIds.clear();
    memories.forEach((m) => addMemory(m, ''));
    if (!memories.length) $('memList').innerHTML = '<li class="empty">Nothing yet. Tell Riko about yourself.</li>';
  } catch { /* ignore */ }
}
async function forgetMemory(id, li) {
  const r = await fetch(withToken(`/api/memories/${encodeURIComponent(id)}`), { method: 'DELETE' });
  if (r.ok) { li.remove(); state.memIds.delete(id); $('memCount').textContent = state.memIds.size || ''; }
}

// ------------------------------------------------------------------ controls
async function start() {
  $('startBtn').disabled = true;
  state.audio = new AudioEngine({
    onMicFrame: (buf) => state.ws?.readyState === 1 && state.ws.send(buf),
    onPlaybackState: (p) => { send({ type: 'playback', state: p ? 'playing' : 'idle' }); if (!p) subtitle('', 'riko'); },
    onSentenceStart: (turn, idx) => { const t = state.sentences.get(`${turn}:${idx}`); if (t) subtitle(t, 'riko'); },
  });
  try {
    await state.audio.start();
  } catch (e) {
    console.warn(e);
    await state.audio.startOutputOnly();
    state.audio.setMicEnabled(false);
    send({ type: 'settings', mic: false });
    note('Microphone is blocked, so voice input is off. You can still type, or allow the microphone in your browser\'s site settings and reload.', 'error');
    $('micBtn').setAttribute('aria-pressed', 'false');
    $('micBtn').textContent = 'Mic off';
  }
  state.audio.setActiveTurn(state.activeTurn);
  state.audio.setMuted(!!state.settings.muted);
  $('startGate').remove();
  $('controls').hidden = false;
  state.started = true;
}

function pttDown() {
  if (state.pttHeld || !state.started) return;
  state.pttHeld = true;
  $('talkBtn').dataset.active = 'true';
  $('talkBtn').textContent = 'Release to send';
  state.audio?.flush();
  send({ type: 'ptt', action: 'start' });
}
function pttUp() {
  if (!state.pttHeld) return;
  state.pttHeld = false;
  $('talkBtn').dataset.active = 'false';
  $('talkBtn').textContent = state.settings.input_mode === 'ptt' ? 'Hold to talk' : 'Hold to talk over';
  send({ type: 'ptt', action: 'stop' });
}

function bindUI() {
  $('startBtn').onclick = start;
  const talk = $('talkBtn');
  talk.addEventListener('pointerdown', (e) => { talk.setPointerCapture(e.pointerId); pttDown(); });
  talk.addEventListener('pointerup', pttUp);
  talk.addEventListener('pointercancel', pttUp);
  const typing = () => ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  window.addEventListener('keydown', (e) => {
    if (typing() || $('settings').open) return;
    if (e.code === 'Space' && !e.repeat) { e.preventDefault(); pttDown(); }
    if (e.key === 'Escape') { send({ type: 'interrupt' }); state.audio?.flush(); }
    if (e.key === 'm' || e.key === 'M') $('micBtn').click();
  });
  window.addEventListener('keyup', (e) => { if (e.code === 'Space') pttUp(); });
  $('micBtn').onclick = () => {
    const on = $('micBtn').getAttribute('aria-pressed') !== 'true';
    $('micBtn').setAttribute('aria-pressed', String(on));
    $('micBtn').textContent = on ? 'Mic on' : 'Mic off';
    state.audio?.setMicEnabled(on);
    send({ type: 'settings', mic: on });
  };
  $('muteBtn').onclick = () => send({ type: 'settings', muted: !state.settings.muted });
  $('textForm').onsubmit = (e) => {
    e.preventDefault();
    const t = $('textInput').value.trim();
    if (!t) return;
    if (!state.audio && !state.started) {
      // typing works before audio is started; replies show as text and captions
    }
    state.audio?.flush();
    send({ type: 'text', text: t });
    $('textInput').value = '';
  };
  $('presetSelect').onchange = (e) => send({ type: 'settings', preset: e.target.value });
  $('characterSelect').onchange = (e) => send({ type: 'settings', character: e.target.value });
  $('voiceSelect').onchange = (e) => send({ type: 'settings', voice: e.target.value });
  $('modelSelect').onchange = (e) => send({ type: 'settings', model: e.target.value });
  document.querySelectorAll('input[name=inputMode]').forEach((r) =>
    r.addEventListener('change', () => send({ type: 'settings', input_mode: r.value })));
  $('settingsBtn').onclick = async () => {
    $('settings').showModal();
    try {
      const { models, current } = await (await fetch(withToken('/api/models'))).json();
      fillSelect($('modelSelect'), models, current);
    } catch { /* keep current */ }
  };
  $('vrmFile').onchange = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    state.avatar?.dispose();
    state.avatar = await createAvatar($('avatar'), { vrmUrl: URL.createObjectURL(f) });
  };
  const tabs = [['tabLog', 'logPanel'], ['tabMem', 'memPanel']];
  tabs.forEach(([tab, panel]) => ($(tab).onclick = () => {
    tabs.forEach(([t, p]) => { $(t).setAttribute('aria-selected', String(t === tab)); $(p).hidden = t !== tab; });
    if (tab === 'tabMem') loadMemories();
  }));
}

// ------------------------------------------------------------------ render loop
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  const lv = state.audio?.levels() ?? { riko: 0, you: 0 };
  const rikoTalking = !!state.audio?.playing;
  const speaker = rikoTalking ? 'riko' : (state.userSpeaking || state.pttHeld) ? 'you' : 'none';
  if (speaker !== state.speaker) { state.speaker = speaker; document.body.dataset.speaker = speaker; }
  const level = speaker === 'riko' ? lv.riko : speaker === 'you' ? lv.you : 0;
  $('halo').style.setProperty('--level', level.toFixed(3));
  state.avatar?.update(dt, { mouth: lv.riko, speaking: rikoTalking, listening: speaker === 'you' });
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ boot
(async function boot() {
  bindUI();
  connect();
  // Avatar config arrives with the hello event; start with whatever is configured server-side.
  const waitHello = () => new Promise((res) => {
    const t = setInterval(() => { if (state.settings.avatar) { clearInterval(t); res(); } }, 50);
    setTimeout(() => { clearInterval(t); res(); }, 4000);
  });
  await waitHello();
  const av = state.settings.avatar || { enabled: true, vrm_url: '' };
  state.avatar = await createAvatar($('avatar'), { vrmUrl: av.vrm_url, enabled: av.enabled });
  requestAnimationFrame(frame);
})();
