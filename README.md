# Riko Next

A modular, low-latency, local-first AI voice companion, inspired by
[Project Riko](https://github.com/rayenfeng/riko_project) by Rayen Feng (Just Rayen).

You talk, and Riko starts answering in her own voice before she has finished thinking. You can interrupt her
mid-sentence, she remembers what matters about you across days, and she can show her mood through a VRM avatar.
Everything can run on your own machine: Ollama or LM Studio, Faster-Whisper and GPT-SoVITS.

> This is a re-engineered successor, not a fork. It keeps Project Riko's stack and spirit (the snarky anime
> companion who calls you "senpai"), and rebuilds the pipeline around streaming, interruption and real memory.
> See [docs/RESEARCH.md](docs/RESEARCH.md) for what the original does, what Rayen showed later, what forks added,
> and what is new here.

## Features

- **Streaming voice pipeline.** Each sentence goes to TTS as soon as the LLM finishes writing it, and audio
  streams to the browser chunk by chunk.
- **Barge-in.** Start talking and Riko's voice ducks immediately. Keep talking and she stops: generation is
  cancelled, and only what you actually heard is saved to history. Coughs and short noises don't interrupt her.
- **Voice detection or push-to-talk.** Silero VAD (bundled with faster-whisper, no torch needed) or hold Space.
- **Any LLM, chosen in config.** Ollama (native API with `keep_alive`), LM Studio, OpenAI, OpenRouter, Groq,
  llama.cpp, vLLM, or anything OpenAI-compatible. `<think>` reasoning blocks are never spoken.
- **Speech recognition.** Local Faster-Whisper with GPU auto-detect and automatic CPU fallback, or Groq /
  OpenAI-compatible remote Whisper.
- **Speech synthesis.** GPT-SoVITS (`api_v2`, streaming), OpenAI-compatible TTS (OpenAI, Kokoro-FastAPI), the
  browser's built-in voice, or text only.
- **Layered memory.** Conversation window plus rolling summary, then episodic, semantic and working memory.
  Memories are ranked by relevance, importance and recency, decay over time, get stronger when recalled, merge
  duplicates and consolidate old events.
- **Personality files.** YAML characters with presets (`default`, `friendly`, `sarcastic`, `professional`),
  switchable live.
- **Emotion.** The LLM tags its own mood at zero extra latency, with a lexicon fallback. Emotion drives avatar
  expressions and TTS speed, or a different reference clip per emotion.
- **Web UI.** A voice halo, anime-style subtitles synced to playback, conversation history, a memory panel you
  can edit, and settings.
- **Optional VRM avatar.** Idle breathing, blinking, lip sync, expressions and head motion. A 2D face is used if
  no VRM model is set.
- **Terminal mode.** `riko chat` works like the original `main_chat.py`, but streaming and interruptible.
- **Security and diagnostics.**
  - Secrets come only from `.env`, and the config is validated.
  - The server listens on localhost by default and checks the token and origin.
  - Logs are structured with secrets redacted, and `riko check` diagnoses your setup.

## Architecture

```
 Browser (mic, AEC) ──int16 16 kHz frames──▶ WebSocket ──▶ ConversationSession
                                                           │
   VAD (Silero / energy) ─▶ ASR (faster-whisper / Groq) ───┤
                                                           ▼
   Memory retrieval (SQLite + vectors) ─▶ PersonalityEngine prompt
                                                           ▼
   LLM token stream ─▶ ThinkFilter ─▶ SentenceChunker ─▶ EmotionAnalyzer
                                                           ▼ (per sentence, while the LLM keeps writing)
   TTS worker (GPT-SoVITS stream) ──binary audio frames──▶ Browser playback ─▶ halo, subtitles, VRM lip sync
                                                           ▼ (after the turn, once idle)
   Memory extraction ─▶ dedup ─▶ store ─▶ periodic decay / consolidation
```

`ConversationSession` (`server/riko/core/session.py`) is transport-agnostic. The WebSocket server and the
terminal client both drive it. Each user turn is one asyncio task, so interrupting cancels the LLM HTTP stream
and the TTS worker together.

## Screenshots

_Placeholder: add `docs/screenshot.png` after your first run._ The layout puts the stage and avatar on the
left, with a halo that glows pink when Riko speaks and cyan when you do, and subtitles at the bottom. The
conversation and memory tabs are on the right.

## Requirements

- Python 3.10+ (tested with 3.12)
- An LLM: [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) locally, or a cloud API key
- Optional voice: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Without it, use `tts.provider: browser`.
- A modern browser (Chrome, Edge, Firefox) for the web UI
- Optional NVIDIA GPU for faster ASR and TTS

## Installation

### Windows

1. Install Python 3.10+ from python.org and tick **Add python.exe to PATH**.
2. Download or clone this repository.
3. Double-click **`start.bat`**. The first run:
   - creates `.venv`
   - installs everything
   - copies `config.example.yaml` to `config.yaml` and `.env.example` to `.env`
   - opens http://localhost:8765

### Linux / macOS

```bash
sudo apt install libportaudio2      # only for terminal mode (riko chat); Debian/Ubuntu
./start.sh
```

### Manual install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all]"            # [asr] = faster-whisper, [local-audio] = sounddevice, [dev] = pytest
cp config.example.yaml config.yaml && cp .env.example .env
riko check                         # diagnose everything
riko serve --open
```

### GPU (NVIDIA) for Faster-Whisper

`device: auto` uses CUDA when available and **falls back to CPU automatically** if CUDA libraries are missing,
logging a warning. To make GPU work, follow
[faster-whisper's GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu): it needs cuBLAS for CUDA 12 and
cuDNN 9.
- **Linux:** `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*`, then add their `lib` folders to
  `LD_LIBRARY_PATH`. The faster-whisper README has the one-liner.
- **Windows:** use the CUDA/cuDNN DLLs linked from that README, placed on `PATH`.

Then set `asr.model: small` or `distil-large-v3`.

### CPU only

Use `asr.model: base.en` (default) or `tiny.en`, with `compute_type: auto` (which is int8 on CPU), and keep
`beam_size: 1`. For voice, either use `tts.provider: browser`, or run GPT-SoVITS on CPU, which is slower but
works.

## Configuration

Everything lives in `config.yaml`; see `config.example.yaml` for every option with comments. Rules:

- **Secrets go in `.env`.** Config refers to them by name (`api_key_env: OPENAI_API_KEY`). An inline
  `api_key: sk-...` is rejected at startup.
- **Paths are relative** to the config file. No machine-specific paths.
- **`${VAR}` / `${VAR:-default}`** are expanded from the environment.
- **Unknown keys are errors**, so typos can't silently do nothing.

## LLM setup

| Provider | config | notes |
|---|---|---|
| Ollama | `provider: ollama`, `model: llama3.1:8b` | `ollama pull llama3.1:8b`. `ollama_keep_alive` keeps it loaded. |
| LM Studio | `provider: lmstudio`, `model: <id shown in LM Studio>` | Start the local server in LM Studio (port 1234). |
| OpenAI | `provider: openai`, `model: gpt-4.1-mini` | Set `OPENAI_API_KEY` in `.env`. |
| OpenRouter / Groq | `provider: openrouter` / `groq` | Set `OPENROUTER_API_KEY` / `GROQ_API_KEY`. |
| Anything else | `provider: openai_compatible`, `base_url: http://host:port/v1` | llama.cpp `llama-server`, vLLM, etc. |

You can also switch models live from the settings dialog.

## ASR setup

- **Local:** `asr.provider: faster_whisper`. Models download automatically on first use from Hugging Face and
  are cached. The model loads at server start so your first sentence isn't slow.
- **Remote:** `asr.provider: groq` (needs `GROQ_API_KEY`), or `openai_compatible` with a `base_url` that serves
  `/audio/transcriptions`.

## TTS setup (GPT-SoVITS)

1. Install GPT-SoVITS following [its README](https://github.com/RVC-Boss/GPT-SoVITS). On Windows, the
   integrated package is easiest.
2. Start its API from the GPT-SoVITS folder:
   ```
   python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
   ```
   (On the Windows package, use `runtime\python.exe api_v2.py ...`.) Alternatively, set `GPT_SOVITS_DIR` in
   `.env` and the start scripts launch it for you.
3. The default voice is the original Project Riko sample (`characters/voices/riko_sample.wav`). For your own
   voice:
   - use a clean 3–10 s clip;
   - set `prompt_text` to exactly the words in that clip.
4. **Emotion voices (optional):** record one clip per emotion and map them under
   `tts.voices.riko.emotions.<emotion>.ref_audio`.

`streaming_mode`: `2` is the balanced default, `3` is the fastest, and `0` disables streaming.

Other engines:
- `openai_compatible` (OpenAI `tts-1`, or a local Kokoro-FastAPI at `http://127.0.0.1:8880/v1`)
- `browser` (the system voice; zero setup)
- `none` (text and subtitles only)

## Memory system

| Layer | Holds | Default half-life |
|---|---|---|
| Short-term | the last `short_term_messages` messages; older ones are folded into a rolling summary | — |
| Episodic | events: "User passed their driving test" | 21 days |
| Semantic | facts and preferences: "User's name is Alex" | 180 days |
| Working | current goals: "User is preparing a talk" | 2 days (expires) |

Every memory stores timestamp, importance, category, access count and last access. Retrieval scores
`0.6·relevance + 0.25·strength + 0.15·recency`. Strength is importance decayed by age, then boosted by how often
the memory has been recalled. Recalling a memory reinforces it.

Pipeline:
- After each turn, once the conversation is idle, the LLM extracts up to 4 memories as JSON. Regex heuristics
  take over if that fails.
- Near-duplicate facts update the existing memory instead of adding a new one.
- A maintenance task:
  - deletes weak, unpinned memories and expired goals;
  - merges old fading events into summaries.

**Embeddings:** the default `hash` embedder works offline but only matches words. For real semantic recall,
`ollama pull nomic-embed-text` and set `memory.embeddings.provider: ollama`, then run `riko memory reindex`.

Inspect memories in the UI's Memory tab (with Forget buttons) or with `riko memory list`. Everything is in one
SQLite file, `data/riko.db`.

## WebSocket API

Connect to `ws://localhost:8765/ws` (add `?token=...` if `RIKO_AUTH_TOKEN` is set).

**Client → server:**
- JSON messages:
  - `{"type":"text","text":"..."}`
  - `{"type":"ptt","action":"start"|"stop"}`
  - `{"type":"interrupt"}`
  - `{"type":"playback","state":"playing"|"idle"}`
  - `{"type":"settings", "input_mode"?, "muted"?, "mic"?, "character"?, "preset"?, "voice"?, "model"?}`
  - `{"type":"ping"}`
- Binary frames: int16 mono PCM at 16 kHz. All input is validated, and invalid input returns an `error` event.

**Server → client:** JSON events, each with `type` and `ts`:

| Event | Fields |
|---|---|
| `hello`, `settings` | `settings` (no secrets) |
| `state` | `listening` / `transcribing` / `thinking` / `speaking` |
| `user_speech_started`, `user_speech_stopped` | `mode`, `duration_s` |
| `transcription` | `text`, `turn_id`, `empty`/`reason` |
| `llm_started`, `llm_token`, `llm_sentence`, `llm_finished` | `turn_id`, `text`, `index` |
| `tts_started`, `tts_finished`, `tts_speak` | `turn_id`, `index` (`tts_speak` = browser voice) |
| `assistant_speaking`, `assistant_stopped` | `turn_id`, `reason` (completed, barge_in, push_to_talk, new_turn, user, error) |
| `assistant_duck`, `assistant_unduck` | `volume` |
| `emotion_changed` | `emotion`, `intensity`, `style`, `expression`, `speed_factor` |
| `memory_retrieved`, `memory_created` | memory items |
| `metrics` | `llm_first_token_ms`, `first_audio_ms`, `total_ms`, ... |
| `error` | `component` (llm, tts, asr, protocol, settings, pipeline), `message`, `recoverable` |

**Binary audio frames** have a 16-byte header: `b"RKA1"`, uint32 `turn_id`, uint32 `sentence_index`, uint32
`sample_rate` (little endian), followed by int16 mono PCM. Clients drop frames from stale turns and use
`sentence_index` to sync subtitles.

**REST:**
- `GET /api/health`
- `GET /api/models`
- `GET /api/memories?layer=&limit=`
- `DELETE /api/memories/{id}`
- `GET /api/history`

## Frontend

The client (`client/web/`) is plain HTML/CSS/JS with no build step, served by the backend.

| Action | Control |
|---|---|
| Talk over Riko | Hold **Space** or the talk button |
| Stop Riko | **Esc** |
| Toggle the mic | **M** |

Typing works too. The settings dialog switches input mode, character, preset, model and voice, and shows
latency for the last reply.

## VRM integration

Put a `.vrm` file in `client/web/avatars/` and set `frontend.avatar.vrm_url: avatars/your-model.vrm`, or load
one from your computer in Settings. VRM 0.x and 1.0 both work: `three-vrm` normalises orientation and
expression names. Riko maps her emotions to the standard expressions (`happy`, `angry`, `sad`, `relaxed`,
`surprised`) and uses `aa`/`oh` for lip sync and `blink` for blinking.

three.js and three-vrm load from the jsDelivr CDN only when a VRM is used. Offline, Riko falls back to the 2D
face. The avatar is entirely optional (`frontend.avatar.enabled: false`).

## Troubleshooting

Run **`riko check`** first. It tests config, LLM reachability and a real completion, TTS, ASR model loading,
VAD, audio devices and memory.

| Symptom | Fix |
|---|---|
| "cannot connect to ... 11434" | Start Ollama (`ollama serve`) and `ollama pull <model>`. |
| "model ... is not pulled" | `ollama pull <model>`, or pick an installed model in Settings. |
| "cannot connect to GPT-SoVITS" | Start `api_v2.py` (see TTS setup). Riko keeps replying in text meanwhile. |
| GPT-SoVITS HTTP 400 | Reference clip must be 3–10 s, and `prompt_text` must match it exactly. |
| "CUDA load failed ... falling back to CPU" | Install cuBLAS/cuDNN for CUDA 12 (see GPU section), or ignore. |
| Riko interrupts herself | Use headphones, raise `audio.barge_in.min_speech_ms`, or disable barge-in. |
| Riko cuts you off mid-sentence | Raise `audio.vad.silence_ms` (for example to 900). |
| Never detects speech | Lower `audio.vad.threshold` (0.35), check the mic toggle and browser permission. |
| "Access denied (check ?token=)" | The server has `RIKO_AUTH_TOKEN` set; open the UI with `?token=...`. |
| Terminal mode: PortAudio error | `sudo apt install libportaudio2` (Linux), or use the web UI. |
| Sees `<think>` text | Shouldn't be spoken. If shown, your model uses unusual tags; open an issue. |

Logs are structured. `--debug` adds detail, and `logging.format: json` gives machine-readable logs. Secrets
are redacted.

## Performance tuning

Each turn reports `llm_first_token_ms` and `first_audio_ms` (UI settings and logs). Typical levers:

- **LLM**
  - A smaller or quantized model is the single biggest win.
  - Keep `ollama_keep_alive` long so the model stays loaded.
  - Keep `max_tokens` modest, since the persona is designed for short spoken replies.
- **TTS:** `gpt_sovits.streaming_mode: 3` gives the fastest first audio. Running GPT-SoVITS on GPU matters most.
- **ASR**
  - `base.en`/`small.en` with `beam_size: 1`. Setting `language` explicitly skips detection.
  - `audio.vad.silence_ms` trades responsiveness against cutting you off.
- **Memory:** retrieval has a timeout (`retrieval_timeout_s`). Extraction runs only when idle, so it never
  delays a reply.

## Development

```bash
pip install -e ".[all,dev]"
pytest                 # 62 tests
riko chat --text       # keyboard chat against your configured LLM
```

**Tests**
- Providers run against mock HTTP servers that replicate the real wire formats:
  - OpenAI SSE and Ollama NDJSON
  - GPT-SoVITS streaming WAV, including mid-sample chunk splits
  - OpenAI PCM and multipart transcription
- The session tests cover streaming order, barge-in, PTT, failures and memory.
- End-to-end smoke tests drive the real FastAPI app over WebSockets with both text and synthetic microphone audio.

**Adding a provider:** implement the base class (`LLMProvider.stream_chat`, `ASRProvider.transcribe`,
`TTSProvider.synthesize`, or `Embedder.embed`), then add one branch to the matching `create_*` factory.

**What was not verified in the build environment:**
- live Ollama/LM Studio/OpenAI servers
- a live GPT-SoVITS server
- downloading Whisper weights (Hugging Face was not reachable there)
- the browser UI and VRM rendering

The wire formats were verified against upstream source code, and the server was started and exercised over a
real WebSocket. Please report anything that misbehaves on real hardware.

## Project structure

```
riko-next/
├── characters/riko.yaml          personality + presets ({user} placeholder)
├── characters/voices/            reference clips (original Riko sample)
├── client/web/                   index.html, styles.css, app.js, audio.js, mic-worklet.js, avatar.js, avatars/
├── server/riko/
│   ├── core/       config.py, logging.py, events.py, chunker.py, session.py (the pipeline)
│   ├── llm/        base, openai_compat (OpenAI/LM Studio/OpenRouter/Groq), ollama
│   ├── asr/        faster-whisper, OpenAI-compatible/Groq
│   ├── tts/        gpt_sovits, openai_compat, browser/none
│   ├── audio/      vad.py (Silero/energy + segmenter), utils.py, local_io.py (terminal mode)
│   ├── memory/     models, store (SQLite + VectorIndex), embeddings, manager
│   ├── personality/  emotion/   api/ (protocol.py, server.py)
│   └── __main__.py  CLI: serve | chat | check | memory
├── tests/                        62 tests incl. WebSocket end-to-end smoke tests
├── docs/RESEARCH.md              research report and comparison
├── config.example.yaml  .env.example  pyproject.toml  start.bat  start.sh
```

## Roadmap

- Partial (streaming) transcription for even earlier turn-taking; semantic end-of-turn detection
- Mixamo/VRMA animation layers (as in Rayen's 2026 build) and gesture triggers from `*actions*`
- Self-reflection: periodic LLM review of memories ("what do I know about senpai?")
- Discord voice channel client (the session engine is transport-agnostic, so this is a new sink)
- Optional ChromaDB/Qdrant `VectorIndex` for very large memory stores
- Live2D avatar option; packaged desktop app

## License

MIT. See [LICENSE](LICENSE).

## Credits

- **[Project Riko](https://github.com/rayenfeng/riko_project)** by Rayen Feng (Just Rayen), MIT.
  - **Reused directly:** the default system prompt (verbatim in `characters/riko.yaml`), the sample voice clip
    and its transcript.
  - **Ideas:** the overall stack, YAML presets, persistent conversation, and the VRM/streaming/Groq-ASR direction
    shown in his later releases.
  - No Python code from Project Riko was reused; the pipeline is a rewrite.
- **Community forks and PRs** inspired several features (ideas only, no code copied):
  - provider switching (PR #8 by t9sy)
  - TTS failure handling and Groq (PR #12)
  - Ollama support (PR #16)
  - RAG and fact memory (NimuthuGanegoda fork)
  - reference-clip guidance (shikosu fork)
- **[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)** (MIT): TTS engine; the client follows its `api_v2.py`
  protocol.
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** (MIT): ASR, plus the bundled
  [Silero VAD](https://github.com/snakers4/silero-vad) model (MIT). The streaming VAD wrapper reproduces the
  input handling of `faster_whisper/vad.py`.
- **[three.js](https://threejs.org)** and **[@pixiv/three-vrm](https://github.com/pixiv/three-vrm)** (MIT):
  avatar rendering.
- **Retrieval scoring** follows Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (2023).
- **[Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)**: a related project that shaped
  expectations for hands-free interruption.
