// Captures microphone audio, downsamples to 16 kHz mono int16, posts 512-sample frames (32 ms).
class RikoMic extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0; this.sum = 0; this.cnt = 0;
    this.buf = new Int16Array(512); this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      // box-filter decimation: average the input samples that fall into each output sample
      this.sum += ch[i]; this.cnt++; this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.sum / this.cnt));
        this.sum = 0; this.cnt = 0;
        this.buf[this.n++] = v * 32767;
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
          this.buf = new Int16Array(512); this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('riko-mic', RikoMic);
