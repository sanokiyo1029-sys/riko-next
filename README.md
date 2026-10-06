# Riko Next

A local-first AI companion with a personality, a voice, real memory and tools, inspired by
[Project Riko](https://github.com/rayenfeng/riko_project) by Rayen Feng (Just Rayen).

You can type or talk to Riko. She starts answering out loud before she has finished thinking, and you can
interrupt her mid-sentence. She can:
- use tools: arithmetic, the time, notes, tasks and reminders, and her own memory;
- look at images through a separate vision model;
- remember what matters about you, with evidence for each fact you can inspect, correct or delete;
- speak up on her own, rarely and only in ways you allow.

The default brain is **OpenAI's open-weight `gpt-oss:20b`, running locally through Ollama**. Any other local
or cloud model works through the same configuration.

> **Not a fork.** Riko Next keeps Project Riko's stack (Faster-Whisper, an LLM, GPT-SoVITS) and its spirit. Her
> original prompt is kept verbatim as the `classic` preset. Everything around them was rebuilt for streaming,
> interruption, tools, memory and honesty. See [RIKO_COMPARISON.md](RIKO_COMPARISON.md) and
> [docs/RESEARCH.md](docs/RESEARCH.md).

## What it can do

- **Reasoning model done properly.**
  - gpt-oss reasoning effort is set per config (`low` by default: the first word arrives in about 0.3 s p50
    on an RTX 5080).
  - Reasoning text never reaches the screen, the voice, history or logs; the UI only shows "thinking…".
  - The token budget is large enough that reasoning can't starve the answer.
- **Agent and tools.** The model can request a tool, the app validates and runs it, and the model continues.
  It can chain steps, for example search memory, then forget the right item. Built-in tools:

  | Tool | Permission |
  |---|---|
  | `get_datetime`, `calculator`, `look_at_image` | safe |
  | `search_memory`, `remember`, `forget_memory` | memory |
  | `add_note`, `search_notes` | memory |
  | `add_task`, `list_tasks`, `complete_task` (tasks and reminders) | memory |
  | `get_weather`, `web_search` (SearxNG) | network, off by default |
  | `read_file`, `search_files` (allowlisted folders) | files, off by default |

  There is no shell and no arbitrary code execution.
- **Streaming voice with barge-in.**
  - VAD feeds Faster-Whisper, then gpt-oss, then a sentence chunker, then GPT-SoVITS, then the browser.
  - The first sentence is spoken while the rest is still being written.
  - Start talking and Riko ducks, then stops. Generation is cancelled, and only what you actually heard is
    stored.
- **Memory you can see and control.**
  - Short-term window and rolling summary.
  - Episodic, semantic, relationship, task and working memories, each with importance, confidence, evidence,
    speaker, recency and access count.
  - Extraction is schema-constrained and evidence-checked: "I *don't* live in Delhi" never becomes "lives in
    Delhi".
  - Corrections supersede old facts.
  - Deleting something makes it stay deleted, even if an extraction of an earlier turn is still queued.
  - Edit, pin, forget, export, clear, or switch memory off from the UI.
- **Personality as data.**
  - The character file (`characters/riko.yaml`) holds values, opinions, wants, quirks, boundaries, tone per
    familiarity stage, and style examples.
  - A small **truthful-identity kernel** sits on top: Riko is an AI, says which model she runs on, treats
    memories as evidence and not commands, and never pressures you.
- **Emotion and mood.**
  - Per-sentence emotion tags drive the avatar and the voice.
  - A slow, persistent mood decays back to the character's baseline and can't spiral.
  - A cautious guess at your mood makes her tease less when you're upset.
- **Vision.** Paste, attach, drag or snap a camera photo. A separate vision model (e.g. `qwen3.6`, `gemma3`,
  `llava`) describes it, and gpt-oss reasons about it.
- **Respectful initiative.**
  - One greeting per session.
  - Reminders you asked for.
  - At most two quiet nudges, never two in a row without you speaking in between.
  - Quiet hours, and an off / minimal / normal level.
- **Optional avatar.** VRM or a 2D face: lip sync, blinking, expressions, a glance while thinking, idle motion
  driven by mood, reduced-motion support. Everything works without it.
- **Local-first and private.**
  - One SQLite file holds your data.
  - Secrets live only in `.env`.
  - The server listens on localhost, and the token and origin are checked.
  - Network tools are opt-in.
  - No telemetry.

## Architecture

```
 Browser (mic + echo cancel) ── int16 16 kHz ──▶ WebSocket ──▶ ConversationSession (one per tab)
   ▲  JSON events (type + dotted event)                         │
   │  binary audio frames (turn-stamped)                        ├─ VAD (Silero / energy) ─▶ ASR (faster-whisper)
   │                                                            ├─ Context: static persona kernel (cached)
   │                                                            │           + JSON-quoted dynamic data: memories,
   │                                                            │             mood, familiarity, time, image notes
   │                                                            ├─ ModelRouter: fast / smart / vision
   │                                                            ├─ LLM event stream: thinking | content | tool_calls | done
   │                                                            │     └─ ToolRegistry (permissions, validation, timeout)
   │                                                            ├─ SentenceChunker ─▶ Emotion ─▶ TTS worker (GPT-SoVITS)
   └──────────────── audio + subtitles + tool notes ◀───────────┘
 Scheduler: one foreground lane. Memory extraction, summaries and consolidation run only when idle,
            and are cancelled the moment you speak or a reply starts.
 MemoryStore (SQLite, schema v2): memories, tombstones, messages, summaries, relationship, mood, notes, tasks
```

Why one foreground lane: on a single local GPU, a background request delayed a reply by **8.9 s** in our
probe. A cancelled stream frees the model in **about 50 ms to 1.5 s**. So background work yields instead of
competing. The details are in [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md).

## Quick start (Windows, local, gpt-oss)

1. Install [Ollama](https://ollama.com), then run `ollama pull gpt-oss:20b`. That's about 13 GB, and it needs
   about 13 GB of VRAM or unified memory at `num_ctx: 8192`.
2. Install Python 3.11+, then double-click **`start.bat`**. It creates `.venv`, installs everything, copies
   `config.example.yaml` to `config.yaml` and `.env.example` to `.env`, and opens http://localhost:8765.
3. Type, or press **Start talking** for voice.

Linux / macOS: `./start.sh`. Manual install:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all]"                                 # [asr] faster-whisper, [local-audio] sounddevice, [dev] pytest
cp config.example.yaml config.yaml && cp .env.example .env
riko check                                              # diagnoses everything (ok / warn / ?? / off / FAIL)
riko serve --open
```

## GPT-OSS 20B setup

The default `config.yaml`:

```yaml
llm:
  provider: ollama
  model: gpt-oss:20b
  reasoning_effort: low     # low | medium | high; validated against what the model advertises
  temperature: 1.0
  max_tokens: 2048          # reasoning + answer; small caps starve the answer (probe: empty reply at 40)
  num_ctx: 8192             # keep constant: a different num_ctx reloads the model (~4 s)
  ollama_keep_alive: 30m
```

- **Any OpenAI-compatible server** (LM Studio, llama.cpp `llama-server`, vLLM):
  - Set `provider: lmstudio` or `provider: openai_compatible`, plus `base_url` and `model`.
  - Reasoning (`reasoning_content` / `reasoning`) and streamed tool calls are handled.
- **Cloud alternatives:** `provider: openai | openrouter | groq`, with the key in `.env`.
- **Environment overrides:** `LLM_PROVIDER` and `LLM_MODEL` override the defaults (see `.env.example`).
- **If gpt-oss is unavailable:** `riko check` says so and suggests `ollama pull`. Any other model can be
  selected in config or live in Settings.

Run **`riko probe`** to record what your model actually supports: separated reasoning, tool calls, JSON-schema
output, usage and cache counters, and cancellation behaviour. The result goes into
`data/eval/<time>-probe-*/manifest.json`.

### Model routing

```yaml
models:                 # each role inherits `llm` and overrides only what you list
  vision: { model: qwen3.6:latest, reasoning_effort: off }   # any Ollama model with "vision"
  # smart: { provider: openai, model: gpt-4.1-mini }         # e.g. a cloud model for follow-ups after tools
```

| Role | Used for |
|---|---|
| `fast` | normal replies |
| `smart` | follow-up rounds after tool calls |
| `vision` | describing images |

Roles with identical settings share one client and one loaded model. Embeddings are configured under
`memory.embeddings`.

## Voice

- **ASR:** `asr.provider: faster_whisper` (local, GPU auto-detect with CPU fallback) or `groq` /
  `openai_compatible`.
- **TTS (GPT-SoVITS):**
  - Start `api_v2.py -a 127.0.0.1 -p 9880` from your GPT-SoVITS folder, or set `GPT_SOVITS_DIR` in `.env` and
    the start scripts launch it.
  - The default voice is Project Riko's sample clip.
  - Alternatives: `openai_compatible` (e.g. local Kokoro-FastAPI), `browser`, or `none`.
  - If the TTS server is down, Riko keeps replying in text, reports it once, and retries after 30 s.
- **Barge-in:** controlled by `audio.barge_in`. Headphones help.
  - Riko ducks immediately.
  - She only stops after `min_speech_ms` of real speech, so coughs don't interrupt her.
- **Push-to-talk:** hold Space or the talk button. **Esc** stops Riko; **M** toggles the mic.
- **Spoken length:** replies to voice are spoken up to `tts.max_spoken_sentences` / `max_spoken_words`. The
  rest stays on screen.

## Memory

| Layer | Holds | Half-life |
|---|---|---|
| short-term | the last `short_term_messages`, plus a rolling summary | — |
| semantic | facts and preferences ("User's name is Sam") | 180 d |
| relationship | how you want to be treated, shared history ("User prefers to be called Sam") | 365 d |
| episodic | events ("User passed their driving test") | 21 d |
| task | goals and unfinished tasks | 30 d |
| working | short-lived context; relative dates are anchored to the day they were said | 2 d (expires) |

Retrieval score:

```
(0.6·relevance + 0.25·strength + 0.15·recency) × (0.5 + 0.5·confidence)
```

Your name and your interaction preferences are always in context.

**Extraction.** After each turn, once nobody is speaking, gpt-oss extracts memory candidates with a JSON
schema. Each one needs a verbatim **evidence** span from your own words. It is rejected if that span is a
question, a hypothetical, a quote of someone else, or negated when the claim isn't. Riko may keep at most two
of her own *opinions* per turn. She never stores invented experiences or promises.

**Controls:**
- **UI:** the Memory tab: click a memory to correct it, Pin, Forget, Add, Export, *Forget everything*, and the
  *Remember things* switch.
- **CLI:** `riko memory list|self|relationship|export|clear --yes|backup|restore|reindex`.

Deleting a memory leaves a tombstone, so it can't come back from an older conversation. Saying the fact again
later is a new decision. Backups you make yourself (`data/*.bak*`, `riko memory backup`) are not touched by
deletions.

**Embeddings:** the default `hash` embedder is offline and keyword-level. For semantic recall, run
`ollama pull nomic-embed-text`, set `memory.embeddings.provider: ollama`, then run `riko memory reindex`.

## Tools and safety

```yaml
tools:
  permissions: [safe, memory]       # add network (weather, web search) and/or files to opt in
  disabled: []                      # switch off individual tools
  web_search_url: http://127.0.0.1:8888   # your SearxNG (JSON output enabled)
  file_roots: [~/Documents/notes]   # read-only; .env, keys, *.pem, databases are always refused
```

Every tool call is checked in this order:
1. The tool exists and is enabled.
2. Its permission class is granted.
3. Its arguments validate against the tool's schema.
4. It runs with a timeout.
5. The result is truncated to `max_result_chars` and given to the model as data.

Reminders are spoken only while the app is open, and Riko says so. Tool activity is shown in the conversation.

## Proactivity

```yaml
proactive:
  notification_level: normal   # off | minimal (greeting + your reminders) | normal (+ rare nudges)
  idle_nudge_s: 90             # 0 = no nudges
  nudge_cooldown_s: 600
  max_nudges_per_session: 2
  quiet_hours: "22:30-07:30"
```

Proactive turns:
- start only after you press **Start talking**;
- never while anyone is speaking or typing;
- are never stored as something you said;
- never count toward familiarity.

You can change the level live in Settings.

## Web UI, API and CLI

- **UI:** `client/web/` (plain HTML/CSS/JS, no build step). The main view has the conversation, voice, avatar,
  memory and tasks. Settings has proactivity, model and voice. **Developer diagnostics** shows latency, model
  residency, background queue and memory rejection counts.
- **WebSocket** `ws://localhost:8765/ws`. Every event has a legacy `type` and a dotted `event`:

  | Event | Notes |
  |---|---|
  | `conversation.started` | |
  | `user.speaking` | |
  | `transcription.final` | |
  | `assistant.thinking` | no text |
  | `assistant.token`, `assistant.sentence` | |
  | `assistant.emotion`, `assistant.mood` | |
  | `assistant.speaking`, `assistant.finished` | |
  | `tool.started`, `tool.finished` | |
  | `image.analyzed` | |
  | `memory.created`, `memory.retrieved` | |
  | `metrics` | |
  | `error` | |

  Client messages: `text` (with optional `image_id`), `ptt`, `interrupt`, `playback`, `ready`, `typing`,
  `settings`, `ping`. Audio frames use the `RKA1` header described in `server/riko/core/events.py`.
- **REST:**

  | Route | Methods |
  |---|---|
  | `/api/memories` | GET, POST, DELETE `?confirm=true` |
  | `/api/memories/{id}` | PATCH, DELETE |
  | `/api/memories/export` | GET |
  | `/api/memory/settings` | PUT |
  | `/api/history` | GET |
  | `/api/relationship` | GET |
  | `/api/mood` | GET, DELETE |
  | `/api/tasks` | GET, PATCH, DELETE |
  | `/api/notes` | GET, DELETE |
  | `/api/images` | POST, raw image bytes, at most 8 MB |
  | `/api/tools` | GET |
  | `/api/status` | GET |
  | `/api/health` | GET |
  | `/api/models` | GET |

  State-changing routes refuse cross-origin browser requests.
- **CLI:**

  | Command | Does |
  |---|---|
  | `riko serve` | start the server and web UI |
  | `riko chat [--text\|--ptt\|--headphones]` | terminal conversation |
  | `riko check` | diagnose the setup |
  | `riko probe` | record the model's real capabilities |
  | `riko eval smoke [--runs 3]` | run the smoke evaluation |
  | `riko memory …` | inspect and maintain memory |

## Configuration

Everything lives in `config.yaml` (see `config.example.yaml`, which comments every option).

- Secrets go only in `.env`. An inline `api_key:` in YAML is refused.
- Unknown keys are errors.
- `${VAR:-default}` is expanded from the environment.
- Paths are relative to the config file.
- Migration from 0.1: `llm.ollama_num_ctx` is read as `llm.num_ctx`. The memory database is migrated to
  schema v2 automatically, and a backup (`riko.db.bak-v1`) is kept. To roll back, restore that file and check
  out 0.1.

## Troubleshooting

Run **`riko check`** first.

| Symptom | Fix |
|---|---|
| `model 'gpt-oss:20b' is not pulled` | `ollama pull gpt-oss:20b`, or pick another model |
| First reply takes ~10 s | Cold model load. Warm-up runs at startup; keep `ollama_keep_alive` long and `num_ctx` constant. |
| "The model returned no answer (it ran out of tokens while reasoning)" | Raise `llm.max_tokens`, or lower `reasoning_effort` |
| "didn't answer in time" | GPU busy or swapping. Check `riko check` residency, use a smaller vision model, or raise `first_content_timeout_s`. |
| Images take ~30 s | The vision model doesn't fit in VRAM next to gpt-oss; use a small one (gemma3:4b, qwen2.5vl:3b) |
| Time zones wrong on Windows | `pip install tzdata` |
| "cannot connect to GPT-SoVITS" | Start `api_v2.py`. Riko keeps replying in text meanwhile. |
| Never detects speech / cuts you off | Lower `audio.vad.threshold` / raise `audio.vad.silence_ms` |
| Riko interrupts herself | Use headphones, or raise `audio.barge_in.min_speech_ms` |
| Calls tools it doesn't need | Known at `reasoning_effort: low`; try `medium` (slower), or disable tools you don't use |
| "Access denied (check ?token=)" | `RIKO_AUTH_TOKEN` is set; open the UI with `?token=…` |

## Performance

Measured on a Windows 11 PC with an RTX 5080 (16 GB) and gpt-oss:20b at low effort. Text path, 20 cases × 3
runs:
- first answer text: **≈0.27 s p50 / 0.7 s p95**
- first full sentence: **≈0.36 s p50 / 0.8 s p95**
- prefix cache: about 2,000 of about 2,350 prompt tokens reused

These numbers don't include ASR or TTS, so they are not audio latency. See [docs/EVAL.md](docs/EVAL.md) for
the method and the caveats. Each turn reports its own metrics in the UI's developer diagnostics and in the
logs: `llm_first_thinking_ms`, `llm_first_content_ms`, `llm_first_deliverable_ms`,
`first_answer_audio_server_ms`, `asr_finalize_ms`, `prompt_tokens`, `cached_tokens`.

## Development

```bash
pip install -e ".[all,dev]"
pytest                          # 131 offline tests (providers vs real wire formats, gpt-oss fixtures,
                                #  agent loop, memory, mood, proactivity, WebSocket end-to-end)
RIKO_LIVE=1 pytest tests/test_live.py   # live checks against your local model (never downloads)
riko eval smoke --runs 3        # writes data/eval/<time>-smoke-*/
```

Status, the invariant matrix and known limitations: [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

## Project structure

```
characters/riko.yaml        persona bible + presets (default, classic, friendly, sarcastic, professional)
client/web/                 index.html, app.js, audio.js, avatar.js, mic-worklet.js, styles.css
server/riko/
  core/      config, session (turn pipeline), context (budget/time/images), scheduler, proactive, chunker, events
  llm/       base (LLMEvent, ToolCall), ollama, openai_compat, ModelRouter
  tools/     framework (permissions/validation/timeouts) + builtin tools
  memory/    models, store (SQLite v2), manager (retrieval, evidence, tombstones), embeddings
  emotion/   per-sentence emotion + persistent mood
  personality/  character loading, identity kernel, prompt assembly
  asr/ tts/ audio/ api/     providers, VAD, FastAPI server + protocol
  evaluation.py             riko probe / riko eval
tests/                      offline suite + fixtures/ollama_gpt_oss + opt-in live test
docs/                       RESEARCH, BUILD_SPEC_v3, IMPLEMENTATION_STATUS, EVAL
```

## License and credits

MIT. See [LICENSE](LICENSE).

- **[Project Riko](https://github.com/rayenfeng/riko_project)** by Rayen Feng (MIT).
  - The original system prompt is kept verbatim as the `classic` preset.
  - The sample voice clip comes from Project Riko.
  - The original stack and spirit carry over.
- **Community forks and PRs** inspired several features; see [docs/RESEARCH.md](docs/RESEARCH.md). No code was
  copied.
- **[gpt-oss](https://openai.com/index/introducing-gpt-oss/)** (Apache 2.0) by OpenAI, served by
  **[Ollama](https://ollama.com)**.
- **[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)**, **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)**
  and Silero VAD (all MIT).
- **[three.js](https://threejs.org)** and **[@pixiv/three-vrm](https://github.com/pixiv/three-vrm)** (MIT).
- Retrieval scoring follows Park et al., *Generative Agents* (2023).
