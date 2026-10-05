// Microphone capture + gapless streamed playback with instant flush, ducking,
// level metering (for the halo and lip sync) and playback-synced subtitles.
export class AudioEngine {
  constructor({ onMicFrame, onPlaybackState, onSentenceStart }) {
    this.onMicFrame = onMicFrame;
    this.onPlaybackState = onPlaybackState;
    this.onSentenceStart = onSentenceStart;
    this.ctx = null;
    this.activeTurn = 0;
    this.nextTime = 0;
    this.sources = new Set();
    this.timers = new Set();
    this.lastSentence = null;
    this.playing = false;
    this.muted = false;
    this.duckLevel = 1;
    this.micEnabled = true;
    this.speechQueue = 0;   // browser TTS fallback
    this.fakeLevel = 0;
  }

  async start() {
    this.ctx = new AudioContext({ latencyHint: 'interactive' });
    await this.ctx.resume();
    this.out = this.ctx.createGain();
    this.outAnalyser = this.ctx.createAnalyser();
    this.outAnalyser.fftSize = 512;
    this.out.connect(this.outAnalyser).connect(this.ctx.destination);
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    await this.ctx.audioWorklet.addModule('mic-worklet.js');
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.micAnalyser = this.ctx.createAnalyser();
    this.micAnalyser.fftSize = 512;
    const node = new AudioWorkletNode(this.ctx, 'riko-mic');
    node.port.onmessage = (e) => { if (this.micEnabled) this.onMicFrame(e.data); };
    const sink = this.ctx.createGain(); sink.gain.value = 0;   // keeps the worklet pulled
    src.connect(this.micAnalyser);
    src.connect(node).connect(sink).connect(this.ctx.destination);
  }

  async startOutputOnly() {
    this.ctx = new AudioContext({ latencyHint: 'interactive' });
    await this.ctx.resume();
    this.out = this.ctx.createGain();
    this.outAnalyser = this.ctx.createAnalyser();
    this.outAnalyser.fftSize = 512;
    this.out.connect(this.outAnalyser).connect(this.ctx.destination);
  }

  setMicEnabled(on) {
    this.micEnabled = on;
    this.stream?.getAudioTracks().forEach((t) => (t.enabled = on));
  }

  setMuted(m) { this.muted = m; this._applyGain(); if (m) window.speechSynthesis?.cancel(); }
  duck(level) { this.duckLevel = level; this._applyGain(); }
  _applyGain() {
    if (!this.out) return;
    const v = this.muted ? 0 : this.duckLevel;
    this.out.gain.setTargetAtTime(v, this.ctx.currentTime, 0.03);
  }

  setActiveTurn(turnId) {
    if (turnId !== this.activeTurn) { this.flush(); this.activeTurn = turnId; }
  }

  playChunk(turnId, sentenceIndex, sampleRate, pcm) {
    if (!this.ctx || turnId !== this.activeTurn || pcm.length === 0) return;   // stale audio from a cancelled turn
    const f = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
    const buf = this.ctx.createBuffer(1, f.length, sampleRate);
    buf.copyToChannel(f, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.out);
    const now = this.ctx.currentTime;
    if (this.nextTime < now + 0.02) this.nextTime = now + 0.06;   // small jitter buffer on (re)start
    src.start(this.nextTime);
    if (sentenceIndex !== this.lastSentence) {
      this.lastSentence = sentenceIndex;
      const t = setTimeout(() => { this.timers.delete(t); this.onSentenceStart(turnId, sentenceIndex); },
        Math.max(0, (this.nextTime - now) * 1000));
      this.timers.add(t);
    }
    this.nextTime += buf.duration;
    this.sources.add(src);
    src.onended = () => { this.sources.delete(src); if (!this.sources.size && !this.speechQueue) this._setPlaying(false); };
    this._setPlaying(true);
  }

  // Browser speechSynthesis fallback (tts.provider: browser)
  speak(text, { speed = 1, onStart } = {}) {
    if (!window.speechSynthesis || this.muted) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = Math.max(0.5, Math.min(2, speed));
    u.pitch = 1.25;
    this.speechQueue++;
    u.onstart = () => { this._setPlaying(true); onStart?.(); };
    u.onboundary = () => { this.fakeLevel = 0.9; };
    u.onend = u.onerror = () => { this.speechQueue = Math.max(0, this.speechQueue - 1); if (!this.speechQueue && !this.sources.size) this._setPlaying(false); };
    window.speechSynthesis.speak(u);
  }

  flush() {
    for (const s of this.sources) { try { s.onended = null; s.stop(); } catch { /* already stopped */ } }
    this.sources.clear();
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
    window.speechSynthesis?.cancel();
    this.speechQueue = 0;
    this.nextTime = 0;
    this.lastSentence = null;
    this.duck(1);
    this._setPlaying(false);
  }

  _setPlaying(p) {
    if (p !== this.playing) { this.playing = p; this.onPlaybackState(p); }
  }

  _rms(an) {
    if (!an) return 0;
    const a = new Float32Array(an.fftSize);
    an.getFloatTimeDomainData(a);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * a[i];
    return Math.sqrt(s / a.length);
  }

  levels() {
    this.fakeLevel *= 0.85;
    const speechActive = this.speechQueue > 0 ? 0.25 + 0.5 * Math.abs(Math.sin(performance.now() / 90)) * (0.4 + this.fakeLevel) : 0;
    return {
      riko: Math.min(1, Math.max(this._rms(this.outAnalyser) * 5, speechActive)),
      you: this.micEnabled ? Math.min(1, this._rms(this.micAnalyser) * 7) : 0,
    };
  }
}
