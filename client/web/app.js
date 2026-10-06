import { AudioEngine } from './audio.js';
import { createAvatar } from './avatar.js';

const $ = (id) => document.getElementById(id);
const token = new URLSearchParams(location.search).get('token');
const withToken = (url) => (token ? `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}` : url);
const api = (url, opts = {}) => fetch(withToken(url), opts).then(async (r) => {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || body.error || `HTTP ${r.status}`);
  return body;
});
const jsonBody = (method, obj) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });

// Emotion tags are stripped client-side too (token stream); the list comes from the server, never duplicated here.
let EMOTIONS = new Set();
const TAG_RE = /\[\s*([A-Za-z]+)\s*(:\s*\d*\.?\d+)?\s*\]\s*/g;
const stripTags = (s) => s.replace(TAG_RE, (m, name) => (EMOTIONS.has(name.toLowerCase()) ? '' : m))
  .replace(/(^|[.!?]\s+)\[\s*[a-z]+\s*(:\s*\d*\.?\d+)?\s*\]\s*/g, '$1')   // invented tags like [hello]
  .replace(/<think>[\s\S]*?(<\/think>|$)/g, '');
const LAYER_LABEL = { semantic: 'Fact', episodic: 'Event', relationship: 'About us', task: 'Goal', working: 'Right now' };
const HINTS = {
  offline: 'Reconnecting to Riko…',
  listening: 'Just talk. Hold Space to talk over her, Esc to stop her.',
  transcribing: 'Catching that…',
  thinking: 'Thinking…',
  speaking: 'Start talking to interrupt.',
};
const TOOL_LABEL = {
  get_datetime: 'checking the time', calculator: 'calculating', search_memory: 'searching memory',
  remember: 'saving a memory', forget_memory: 'forgetting', add_note: 'saving a note', search_notes: 'looking at notes',
  add_task: 'adding a task', list_tasks: 'checking tasks', complete_task: 'updating a task',
  look_at_image: 'looking at the image', get_weather: 'checking the weather', web_search: 'searching the web',
  read_file: 'reading a file', search_files: 'searching files',
};

const state = {
  ws: null, settings: {}, activeTurn: 0, rikoLine: null, tokenText: '', sentences: new Map(),
  started: false, retry: 0, avatar: null, audio: null, speaker: 'none', userSpeaking: false,
  pttHeld: false, historyLoaded: false, memIds: new Set(), pendingImage: null, thinking: false,
  mood: null, lastTyping: 0,
};

// ------------------------------------------------------------------ connection
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(withToken(`${proto}://${location.host}/ws`));
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  setConn(null, 'Connecting');
  ws.onopen = () => {
    state.retry = 0;
    setConn(true, 'Connected');
    if (state.started) send({ type: 'ready' });   // reconnect: audio was already started by a gesture
  };
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
      if (ev.mood) setMood(ev.mood);
      if (!state.historyLoaded) loadHistory();
      break;
    case 'state':
      document.body.dataset.state = ev.state;
      $('hint').textContent = HINTS[ev.state] || '';
      if (ev.state !== 'thinking') setActivity('');
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
      state.thinking = false;
      state.rikoLine = addLine('riko', '');
      if (ev.source && ev.source !== 'text' && ev.source !== 'voice') state.rikoLine.dataset.source = ev.source;
      break;
    case 'assistant_thinking':   // no text: reasoning never leaves the server
      state.thinking = true;
      setActivity('thinking…');
      break;
    case 'tool_started':
      setActivity(`${TOOL_LABEL[ev.tool] || ev.tool.replaceAll('_', ' ')}…`);
      break;
    case 'tool_finished':
      toolNote(ev);
      if (['remember', 'forget_memory'].includes(ev.tool) && !$('memPanel').hidden) loadMemories();
      if (['add_task', 'complete_task'].includes(ev.tool)) loadTasks();
      break;
    case 'image_analyzed':
      note(`Looked at your image with ${ev.model} (${(ev.ms / 1000).toFixed(1)} s).`);
      break;
    case 'llm_token':
      if (ev.turn_id !== state.activeTurn || !state.rikoLine) break;
      setActivity('');
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
      setActivity('');
      if (ev.reason !== 'completed' && ev.reason !== 'error') {
        state.audio?.flush();
        if (ev.turn_id === state.activeTurn) state.rikoLine?.classList.add('interrupted');
        subtitle('', 'riko');
      }
      if (state.rikoLine && !state.rikoLine.querySelector('.text').textContent) state.rikoLine.remove();
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
    case 'mood':
      setMood(ev);
      break;
    case 'memory_created':
      note(`Remembered: ${ev.content}`, 'memory');
      addMemory(ev, 'fresh', true);
      break;
    case 'memory_retrieved':
      for (const m of ev.items || []) addMemory(m, 'recalled');
      break;
    case 'metrics':
      showMetrics(ev);
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

function showMetrics(ev) {
  const f = (k) => (ev[k] ?? '–');
  const parts = [`first word ${f('llm_first_content_ms')} ms`, `first sentence ${f('llm_first_deliverable_ms')} ms`];
  if (ev.first_answer_audio_server_ms) parts.push(`first audio ready ${ev.first_answer_audio_server_ms} ms`);
  if (ev.asr_finalize_ms) parts.push(`speech-to-text ${ev.asr_finalize_ms} ms`);
  if (ev.prompt_tokens) parts.push(`prompt ${ev.prompt_tokens} tok (${ev.cached_tokens ?? 0} cached)`);
  if (ev.tools) parts.push(`tools: ${ev.tools.join(', ')}`);
  if (ev.fallback) parts.push(`fallback: ${ev.fallback}`);
  $('latencyInfo').textContent = `Last reply (server clock, from your finished message): ${parts.join(' · ')}.`;
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
function toolNote(ev) {
  const el = document.createElement('p');
  el.className = `note tool ${ev.ok ? '' : 'failed'}`;
  el.textContent = ev.summary || ev.tool;
  el.title = `${ev.tool} · ${ev.ms} ms`;
  (state.rikoLine && state.rikoLine.isConnected ? state.rikoLine.before(el) : $('logPanel').appendChild(el));
  scrollLog();
}
const scrollLog = () => { const l = $('logPanel'); l.scrollTop = l.scrollHeight; };
const setActivity = (t) => { $('activity').textContent = t; };

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
function setMood(m) {
  state.mood = m;
  $('emotion').title = `Overall mood: ${m.description}. It drifts back to normal on its own.`;
  state.avatar?.setMood?.(m);
}

function fillSelect(sel, items, current, label = (x) => x) {
  sel.replaceChildren(...items.map((v) => { const o = new Option(label(v), v); o.selected = v === current; return o; }));
}

function applySettings(s) {
  state.settings = s;
  EMOTIONS = new Set(s.emotions || []);
  $('charName').textContent = s.character_name;
  document.title = `${s.character_name} · Riko Next`;
  fillSelect($('presetSelect'), s.presets, s.preset, (p) => p[0].toUpperCase() + p.slice(1));
  fillSelect($('characterSelect'), s.characters, s.character);
  fillSelect($('voiceSelect'), s.voices, s.voice);
  if (!$('modelSelect').options.length) fillSelect($('modelSelect'), [s.model], s.model);
  document.querySelectorAll('input[name=inputMode]').forEach((r) => (r.checked = r.value === s.input_mode));
  $('proactiveSelect').value = s.proactive_level;
  $('memEnabled').checked = !!s.memory_enabled;
  $('memEnabled').disabled = !s.memory;
  $('familiar').textContent = s.familiarity || '';
  $('talkBtn').textContent = s.input_mode === 'ptt' ? 'Hold to talk' : 'Hold to talk over';
  $('muteBtn').setAttribute('aria-pressed', String(!s.muted));
  $('muteBtn').textContent = s.muted ? 'Voice off' : 'Voice on';
  $('engineInfo').textContent = `Brain ${s.model} (${s.llm_provider}${s.reasoning_effort ? `, ${s.reasoning_effort} effort` : ''})`
    + ` · vision ${s.vision_model || 'off'} · speech-to-text ${s.asr_provider} · voice ${s.tts_provider}`
    + ` · memory ${s.memory_enabled ? 'on' : 'off'} · tools ${s.tools ? 'on' : 'off'}`;
  $('attachBtn').hidden = $('cameraBtn').hidden = false;
  state.audio?.setMuted(s.muted);
}

async function loadHistory() {
  state.historyLoaded = true;
  try {
    const { messages } = await api('/api/history?limit=30');
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
  loadTasks();
}

// ------------------------------------------------------------------ memory panel
function memoryItem(m) {
  const li = document.createElement('li');
  li.dataset.id = m.id;
  li.innerHTML = `<span class="content" tabindex="0" role="button" title="Click to correct"></span>
    <span class="kind"></span><span class="strength"></span>
    <span class="acts"><button class="pin" type="button"></button><button class="forget" type="button">Forget</button></span>`;
  const c = li.querySelector('.content');
  c.textContent = m.content;
  if (m.evidence) c.title = `From: "${m.evidence}" (click to correct)`;
  const conf = m.confidence != null && m.confidence < 0.6 ? ' · unsure' : '';
  li.querySelector('.kind').textContent = `${LAYER_LABEL[m.layer] || m.layer} · ${m.category}${conf}${m.pinned ? ' · pinned' : ''}`;
  li.querySelector('.strength').style.width = `${Math.round(100 * (m.strength ?? m.importance ?? 0.5))}%`;
  const pin = li.querySelector('.pin');
  pin.textContent = m.pinned ? 'Unpin' : 'Pin';
  pin.onclick = () => editMemory(m.id, { pinned: !m.pinned });
  li.querySelector('.forget').onclick = () => forgetMemory(m.id, li);
  const startEdit = () => {
    const input = document.createElement('input');
    input.value = m.content;
    input.maxLength = 500;
    input.className = 'edit';
    c.replaceWith(input);
    input.focus();
    const done = async (save) => {
      if (save && input.value.trim() && input.value.trim() !== m.content) await editMemory(m.id, { content: input.value.trim() });
      else input.replaceWith(c);
    };
    input.onkeydown = (e) => { if (e.key === 'Enter') done(true); if (e.key === 'Escape') done(false); };
    input.onblur = () => done(true);
  };
  c.onclick = startEdit;
  c.onkeydown = (e) => { if (e.key === 'Enter') startEdit(); };
  return li;
}
function addMemory(m, cls, prepend = false) {
  const list = $('memList');
  list.querySelector('.empty')?.remove();
  let li = list.querySelector(`[data-id="${CSS.escape(m.id)}"]`);
  if (!li) {
    li = memoryItem(m);
    prepend ? list.prepend(li) : list.append(li);
    state.memIds.add(m.id);
  }
  li.classList.remove('fresh', 'recalled');
  void li.offsetWidth;
  if (cls) li.classList.add(cls);
  $('memCount').textContent = state.memIds.size ? String(state.memIds.size) : '';
}
async function loadMemories() {
  try {
    const layer = $('memLayer').value;
    const { memories } = await api(`/api/memories?limit=300${layer ? `&layer=${layer}` : ''}`);
    $('memList').replaceChildren();
    state.memIds.clear();
    memories.forEach((m) => addMemory(m, ''));
    if (!memories.length) $('memList').innerHTML = '<li class="empty">Nothing yet. Tell Riko about yourself.</li>';
  } catch (e) { $('memList').innerHTML = `<li class="empty"></li>`; $('memList').firstChild.textContent = e.message; }
}
async function editMemory(id, patch) {
  try { await api(`/api/memories/${encodeURIComponent(id)}`, jsonBody('PATCH', patch)); } catch (e) { note(e.message, 'error'); }
  loadMemories();
}
async function forgetMemory(id, li) {
  try {
    await api(`/api/memories/${encodeURIComponent(id)}`, { method: 'DELETE' });
    li.remove(); state.memIds.delete(id); $('memCount').textContent = state.memIds.size || '';
  } catch (e) { note(e.message, 'error'); }
}

// ------------------------------------------------------------------ tasks panel
async function loadTasks() {
  try {
    const { tasks } = await api('/api/tasks?status=open');
    $('taskCount').textContent = tasks.length || '';
    const list = $('taskList');
    list.replaceChildren();
    if (!tasks.length) { list.innerHTML = '<li class="empty">No open tasks. Try "remind me to stretch in 20 minutes".</li>'; return; }
    for (const t of tasks) {
      const li = document.createElement('li');
      li.innerHTML = `<span class="content"></span><span class="kind"></span><span class="acts"><button type="button" class="done">Done</button><button type="button" class="forget">Delete</button></span>`;
      li.querySelector('.content').textContent = t.title;
      li.querySelector('.kind').textContent = t.due_at
        ? `reminder ${new Date(t.due_at * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}${t.delivered_at ? ' · reminded' : ''}`
        : 'no reminder';
      li.querySelector('.done').onclick = async () => { await api(`/api/tasks/${t.id}`, jsonBody('PATCH', { status: 'done' })); loadTasks(); };
      li.querySelector('.forget').onclick = async () => { await api(`/api/tasks/${t.id}`, { method: 'DELETE' }); loadTasks(); };
      list.append(li);
    }
  } catch { /* memory disabled */ }
}

// ------------------------------------------------------------------ images
async function attachImage(blob) {
  if (!blob) return;
  if (blob.size > 8 * 1024 * 1024) { note('That image is larger than 8 MB.', 'error'); return; }
  $('attachText').textContent = 'Uploading…';
  $('attachImg').src = URL.createObjectURL(blob);
  $('attachPreview').hidden = false;
  try {
    const r = await api('/api/images', { method: 'POST', headers: { 'Content-Type': blob.type || 'application/octet-stream' }, body: blob });
    state.pendingImage = r.id;
    $('attachText').textContent = r.vision_model ? `Ready (seen by ${r.vision_model})` : 'No vision model configured';
  } catch (e) {
    clearAttachment();
    note(`Image upload failed: ${e.message}`, 'error');
  }
}
function clearAttachment() {
  state.pendingImage = null;
  $('attachPreview').hidden = true;
  $('attachFile').value = '';
}
async function openCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    $('cameraVideo').srcObject = stream;
    $('camera').showModal();
    const stop = () => { stream.getTracks().forEach((t) => t.stop()); $('camera').close(); };
    $('cameraCancel').onclick = stop;
    $('camera').oncancel = stop;
    $('cameraSnap').onclick = () => {
      const v = $('cameraVideo');
      const c = document.createElement('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0);
      c.toBlob((b) => { attachImage(b); stop(); }, 'image/jpeg', 0.88);
    };
  } catch (e) { note(`Camera unavailable: ${e.message}`, 'error'); }
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
  send({ type: 'ready' });   // a real user gesture happened: greetings/reminders may now play
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

async function loadStatus() {
  try {
    const s = await api('/api/status');
    const llm = s.readiness?.llm || {};
    const lines = [
      `model ${s.llm.model} · effort ${s.llm.reasoning_effort ?? 'n/a'} · context ${s.llm.num_ctx} · budget ${s.llm.max_tokens}`,
      `roles fast=${s.roles.fast} smart=${s.roles.smart} vision=${s.roles.vision ?? 'off'}`,
      `LLM ${llm.state ?? 'unknown'}${llm.warmup_ms ? ` (warm-up ${llm.warmup_ms} ms)` : ''} · ASR ${s.readiness?.asr?.state ?? 'unknown'}`,
      `background jobs: ${s.background.pending} pending, ${s.background.completed} done, ${s.background.preempted} preempted`,
    ];
    if (s.ollama) lines.push(`ollama ${s.ollama.version ?? '?'} · loaded: ${s.ollama.running.map((m) => `${m.name} ctx=${m.context_length}`).join(', ') || 'none'}`);
    if (s.memory) lines.push(`memory ${s.memory.enabled ? 'on' : 'off'} · ${JSON.stringify(s.memory.memories)} · rejected ${JSON.stringify(s.memory.rejections)}`);
    $('statusInfo').textContent = lines.join('\n');
  } catch (e) { $('statusInfo').textContent = e.message; }
}

function bindUI() {
  $('startBtn').onclick = start;
  const talk = $('talkBtn');
  talk.addEventListener('pointerdown', (e) => { talk.setPointerCapture(e.pointerId); pttDown(); });
  talk.addEventListener('pointerup', pttUp);
  talk.addEventListener('pointercancel', pttUp);
  const typing = () => ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  window.addEventListener('keydown', (e) => {
    if (typing() || $('settings').open || $('camera').open) return;
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
  $('textInput').addEventListener('input', () => {
    const now = Date.now();
    if (now - state.lastTyping > 2000) { state.lastTyping = now; send({ type: 'typing' }); }
  });
  $('textForm').onsubmit = (e) => {
    e.preventDefault();
    const t = $('textInput').value.trim();
    if (!t && !state.pendingImage) return;
    state.audio?.flush();
    send({ type: 'text', text: t, ...(state.pendingImage ? { image_id: state.pendingImage } : {}) });
    $('textInput').value = '';
    clearAttachment();
  };
  $('attachBtn').onclick = () => $('attachFile').click();
  $('attachFile').onchange = (e) => attachImage(e.target.files?.[0]);
  $('attachClear').onclick = clearAttachment;
  $('cameraBtn').onclick = openCamera;
  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) { e.preventDefault(); attachImage(item.getAsFile()); }
  });
  $('stage').addEventListener('dragover', (e) => e.preventDefault());
  $('stage').addEventListener('drop', (e) => {
    e.preventDefault();
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/'));
    if (f) attachImage(f);
  });
  $('presetSelect').onchange = (e) => send({ type: 'settings', preset: e.target.value });
  $('characterSelect').onchange = (e) => send({ type: 'settings', character: e.target.value });
  $('voiceSelect').onchange = (e) => send({ type: 'settings', voice: e.target.value });
  $('modelSelect').onchange = (e) => send({ type: 'settings', model: e.target.value });
  $('proactiveSelect').onchange = (e) => send({ type: 'settings', proactive_level: e.target.value });
  document.querySelectorAll('input[name=inputMode]').forEach((r) =>
    r.addEventListener('change', () => send({ type: 'settings', input_mode: r.value })));
  $('settingsBtn').onclick = async () => {
    $('settings').showModal();
    try {
      const { models, current } = await api('/api/models');
      fillSelect($('modelSelect'), models, current);
    } catch { /* keep current */ }
  };
  document.querySelector('details.dev').addEventListener('toggle', (e) => { if (e.target.open) loadStatus(); });
  $('vrmFile').onchange = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    state.avatar?.dispose();
    state.avatar = await createAvatar($('avatar'), { vrmUrl: URL.createObjectURL(f) });
  };
  // memory controls
  $('memEnabled').onchange = (e) => send({ type: 'settings', memory_enabled: e.target.checked });
  $('memLayer').onchange = loadMemories;
  $('memExport').onclick = async (e) => {
    e.preventDefault();
    const data = await api('/api/memories/export');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    a.download = 'riko-memory-export.json';
    a.click();
  };
  $('memClear').onclick = async () => {
    if (!confirm('Forget every memory Riko has about you? Conversation history stays unless you also clear it below.')) return;
    const history = confirm('Also delete the conversation history?');
    try { await api(`/api/memories?confirm=true&history=${history}`, { method: 'DELETE' }); } catch (e) { note(e.message, 'error'); }
    loadMemories();
  };
  $('memAdd').onsubmit = async (e) => {
    e.preventDefault();
    const v = $('memAddInput').value.trim();
    if (!v) return;
    try { await api('/api/memories', jsonBody('POST', { content: v })); $('memAddInput').value = ''; } catch (err) { note(err.message, 'error'); }
    loadMemories();
  };
  const tabs = [['tabLog', 'logPanel'], ['tabMem', 'memPanel'], ['tabTasks', 'taskPanel']];
  tabs.forEach(([tab, panel]) => ($(tab).onclick = () => {
    tabs.forEach(([t, p]) => { $(t).setAttribute('aria-selected', String(t === tab)); $(p).hidden = t !== tab; });
    if (tab === 'tabMem') loadMemories();
    if (tab === 'tabTasks') loadTasks();
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
  state.avatar?.update(dt, {
    mouth: lv.riko, speaking: rikoTalking, listening: speaker === 'you',
    thinking: document.body.dataset.state === 'thinking', arousal: state.mood?.arousal ?? 0.45,
  });
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
  if (state.mood) state.avatar?.setMood?.(state.mood);
  requestAnimationFrame(frame);
})();
