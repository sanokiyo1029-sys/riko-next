# Research & architecture report

This report was written before and during implementation. It separates four kinds of claims:

- **A. Original code**: what `rayenfeng/riko_project` (commit `d44f7b6`) actually does, from reading every file.
- **B. Rayen's demonstrated vision**: what Rayen describes in later builds (Patreon release notes, his site, channel description).
- **C. Forks / PRs**: what community derivatives added (diffs inspected where available).
- **D. Riko Next**: what this project newly designs.

**Research limitation:** YouTube pages and transcripts could not be fetched from the build environment. Section B
therefore relies on his channel description ("I'm building my AI anime waifu here"), rayenai.com, and the public
Patreon release notes for v1.1 (2025-11-06) and 2026-02-13. Videos may show details not captured here.

## A. The original Project Riko

Four Python files (~200 lines), MIT licensed.

| File | What it does |
|---|---|
| `server/main_chat.py` | Infinite loop: record -> transcribe -> LLM -> TTS -> play -> delete WAVs. Loads `WhisperModel("base.en", device="cpu", compute_type="float32")`. |
| `process/asr_func/asr_push_to_talk.py` | Press ENTER, `sd.rec` up to 60 s at 44.1 kHz, ENTER again, write WAV, `model.transcribe(file)`. |
| `process/llm_funcs/llm_scr.py` | OpenAI **Responses API**, `stream=False`, `temperature=1`. Loads the whole `chat_history.json`, appends, saves it all back. Config loaded at import time. |
| `process/tts_func/sovits_ping.py` | `POST http://127.0.0.1:9880/tts` with the full reply, writes the WAV, `sd.play` + `sd.wait()`. |
| `character_config.yaml` | API key, model, history file, one `default` preset with a 3-line system prompt, GPT-SoVITS settings with an absolute `D:\...` path. |

Limitations observed in code:
- **Latency.** Nothing streams: the full LLM reply is synthesized in one TTS request, then played to completion.
- **Memory.** Unbounded JSON history, sent in full every turn. There is no retrieval, summarization or decay.
- **Robustness.**
  - If TTS fails, `sovits_gen` returns `None`, but `play_audio` is still called on the missing file (fixed upstream in PR #12).
  - There is no reconnection or error reporting.
- **Security and portability.**
  - The API key lives in YAML, and the reference-audio path is an absolute Windows path.
  - `requirements.txt` is really GPT-SoVITS's dependency list.
- **Interaction.**
  - The keyboard ENTER push-to-talk blocks the loop.
  - Riko can't be interrupted, and there's no UI or avatar (`client/` is empty).
- **Personality.** Only `presets.default` is ever read.

Retained in Riko Next:
- the stack choice (Faster-Whisper + GPT-SoVITS + an LLM)
- YAML personality presets
- persistent conversation across restarts
- the original prompt, kept verbatim as the default
- the sample voice

## B. Rayen's later direction (from release notes)

- **v1.1 (Nov 2025, Windows, Patreon):**
  - OpenAI-compatible **streaming**
  - **Groq ASR** as a free, fast alternative
  - "persistent conversation memory with context tracking"
  - a **VRM animated avatar** using Three-VRM
- **2026-02-13:**
  - improved animation states, layered via endpoints, with Mixamo animations blended in
  - VRM 1.0 support
- **Public README TODO list:** GUI/web interface, live microphone input, emotion/tone control in speech synthesis, VRM frontend.
- **His site:** describes a "multistage LLM stack" and real-time VRM expressions.

## C. Forks and pull requests

| Source | Change |
|---|---|
| PR #2 | Configurable OpenAI base URL |
| PR #8 | `active_provider` switch for openai / openrouter / ollama, setup and start scripts |
| PR #12 | Graceful TTS failure, Groq via litellm, relative ref-audio path |
| PR #16 | Ollama `/api/chat` (replaced OpenAI rather than adding a switch) |
| NimuthuGanegoda fork | ChromaDB RAG, "Mem0-style" fact manager, wake-word VAD, tool actions, model switching in a web UI, alpha VRM with lip sync (README claims) |
| shikosu fork | Gemini + fallback models, `max_history_turns`, Godot avatar with Rhubarb lip sync; documents GPT-SoVITS's 3-10 s reference clip limit |
| Open-LLM-VTuber (related project) | Many ASR/TTS backends, hands-free voice interruption, Live2D |

## D. Best-practice findings that shaped the design

**Streaming TTS by sentence**
- Sending the first sentence to TTS while the LLM continues is the dominant perceived-latency win.
- GPT-SoVITS `api_v2` supports `streaming_mode` 0-3.
- Verified in its source: streaming responses are a 0-frame WAV header followed by raw int16 PCM, and HTTP chunks can split samples.

**VAD**
- faster-whisper (1.2.1) bundles `silero_vad_v6.onnx`, whose `input` axis is time.
- A stateful frame-by-frame wrapper therefore gives exactly the library's batched result. Verified numerically: max difference 0.0.
- This means no torch dependency.

**Barge-in**
- Browsers provide echo cancellation in `getUserMedia`. A terminal does not, so terminal barge-in is opt-in (`--headphones`).
- Ducking, followed by a confirmation threshold on *voiced* duration, avoids false interruptions from coughs.

**Memory**
- Generative-Agents-style scoring (relevance + importance + recency) with per-layer decay.
- A single user's memory set is small, so SQLite + brute-force numpy cosine is faster and simpler than a vector database. It sits behind a `VectorIndex` seam.

**Local LLMs**
- Ollama's native API exposes `keep_alive`, which avoids model reloads between turns.
- Reasoning models emit `<think>` blocks that must never be spoken.
- Background memory extraction must wait until the conversation is idle, or a single-GPU local server delays the next reply.

## Feature comparison

| Feature | Original Riko (A) | Rayen's vision (B) | Forks (C) | Riko Next (D) |
|---|---|---|---|---|
| LLM | OpenAI Responses, no streaming | OpenAI-compatible streaming | OpenRouter, Ollama, Groq, Gemini (one at a time) | Ollama (native), LM Studio, OpenAI, OpenRouter, Groq, any compatible server; streaming; set in config |
| ASR | Faster-Whisper base.en, CPU, temp file | + Groq ASR | — | Faster-Whisper (GPU auto, CPU fallback, in-memory, hallucination filter) + Groq/OpenAI-compatible |
| Input | ENTER push-to-talk | "live mic" (TODO) | wake word + VAD | Silero VAD auto mode + push-to-talk |
| Interruption | No | not documented | not documented | Barge-in with ducking, cancels LLM + TTS, stores only what was heard |
| TTS | Whole reply, blocking | GPT-SoVITS | crash fix | Sentence-streamed GPT-SoVITS, OpenAI-compatible TTS (e.g. Kokoro), browser fallback |
| Emotion | No (TODO) | VRM expressions | — | LLM emotion tags (zero latency) + lexicon fallback -> avatar expression + TTS speed / per-emotion reference clip |
| Memory | Unbounded JSON | "context tracking" | ChromaDB RAG, fact manager | Short-term window + rolling summary; episodic, semantic, working layers; importance, decay, reinforcement, dedup, consolidation, cleanup |
| Personality | One preset used | Simple configs | — | Character files, 4 presets, `{user}` placeholder, live switching |
| UI | None | VRM app | Web UI (Nimuthu) | Web UI: captions, history, memory panel, settings |
| Avatar | None | three-vrm, animation layers | Godot, alpha VRM | Optional three-vrm (idle, blink, lip sync, expressions) + 2D fallback |
| API | None | endpoints for animations | — | WebSocket events + binary audio, REST for memories/history/health |
| Config / security | Key in YAML, absolute paths | — | Keys still in YAML | Keys only via env, validated config, origin + token checks, localhost by default |

## Decisions

| Area | Decision | Why |
|---|---|---|
| Pipeline | Redesigned as an async, cancellable, transport-agnostic session | Streaming, barge-in, and two front ends (web, terminal) share one engine |
| LLM SDK | Replaced `openai`/`litellm` with `httpx` | Fewer dependencies, exact cancellation, one client covers all compatible servers |
| Memory store | SQLite + numpy | Zero setup, one file, plenty fast; swappable via `VectorIndex` |
| Not built | Tool/OS actions, screen capture, wake word, Discord, Live2D | Out of scope for a reliable core; see roadmap |
