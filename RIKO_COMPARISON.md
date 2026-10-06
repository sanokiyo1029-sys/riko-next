# Original Riko vs Riko Next 0.1 vs Riko Next 0.2

- **Original:** [`rayenfeng/riko_project`](https://github.com/rayenfeng/riko_project) at commit `d44f7b6`. See
  [docs/RESEARCH.md](docs/RESEARCH.md) for the file-by-file reading of it.
- **Riko Next 0.1:** this repository as it was imported.
- **Riko Next 0.2:** the version after this round of work.

Claims about 0.2 are backed by tests or by the live measurements in [docs/EVAL.md](docs/EVAL.md).

## Core

### Brain

- **Original Riko:** OpenAI Responses API; no streaming; temperature 1.
- **Riko Next 0.1:**
  - Ollama, LM Studio, OpenAI, OpenRouter, Groq or any OpenAI-compatible server.
  - Streaming.
- **Riko Next 0.2:** everything in 0.1, plus:
  - **gpt-oss:20b is the local default**, with reasoning effort validated against what the model supports.
  - A 2,048-token reasoning-plus-answer budget, with model warm-up.
  - Optional fast/smart/vision roles.

### Reasoning models

- **Original Riko:** n/a.
- **Riko Next 0.1:** `<think>` tags are filtered out; Ollama's separate thinking stream is silently dropped.
- **Riko Next 0.2:**
  - Reasoning becomes text-less `assistant.thinking` events, so it can never reach the UI, TTS or history.
  - Thinking, content, tool calls and usage are separated across all providers.

### Tools and agent

- **Original Riko:** none.
- **Riko Next 0.1:** none.
- **Riko Next 0.2:** **native tool calling with multi-step loops.**
  - Tools: time, calculator, memory search/remember/forget, notes, tasks and reminders, image follow-up
    questions, plus opt-in weather, web search and file reading.
  - Each tool has permissions, validation, a timeout and bounded results.
  - There is no shell or code execution.

### Images

- **Original Riko:** none.
- **Riko Next 0.1:** none.
- **Riko Next 0.2:** attach, paste, drag or camera snapshot.
  - A separate vision model describes the image for the main model.
  - The vision model can be asked follow-up questions about the same image.

## Conversation

### Speech recognition

- **Original Riko:** Faster-Whisper `base.en` on CPU, using a temporary file.
- **Riko Next 0.1:**
  - Faster-Whisper with GPU auto-detect and CPU fallback, in-memory.
  - Hallucination filter.
  - Groq and OpenAI-compatible remote ASR.
- **Riko Next 0.2:** same as 0.1.

### Speech output

- **Original Riko:** the whole reply in one GPT-SoVITS request, played with blocking.
- **Riko Next 0.1:** each sentence is streamed to GPT-SoVITS or Kokoro, with a browser fallback.
- **Riko Next 0.2:** same as 0.1, plus:
  - Spoken-length limits for voice turns.
  - Mood-based speed (at most ±3%).
  - Outage backoff.
  - Numbers and times like "406,150.5" and "12:30" are never split by the chunker.

### Interruption

- **Original Riko:** none.
- **Riko Next 0.1:** barge-in with ducking. The turn is cancelled and only what was heard is stored.
- **Riko Next 0.2:** same as 0.1, plus speech also preempts background LLM work.

## Memory and companionship

### Memory

- **Original Riko:** an unbounded JSON history, sent in full every turn.
- **Riko Next 0.1:**
  - Episodic, semantic and working layers, with decay.
  - LLM extraction without validation.
- **Riko Next 0.2:**
  - Six layers, including relationship and task, with confidence, evidence and speaker for each memory.
  - **Evidence-validated extraction:** negations, questions, hypotheticals and quotes are rejected.
  - Corrections supersede old facts.
  - Tombstones make deletion stick, even against extraction that hasn't run yet.
  - Profile isolation.
  - Schema migration with backup, export and restore.

### Memory controls

- **Original Riko:** none.
- **Riko Next 0.1:** a list with Forget buttons.
- **Riko Next 0.2:**
  - Edit or correct, pin, forget, add, filter, export, clear all.
  - A switch to turn memory off.
  - `riko memory self` and `riko memory relationship`.

### Personality

- **Original Riko:** a 3-line prompt, with only the default preset ever used.
- **Riko Next 0.1:** YAML characters with presets.
- **Riko Next 0.2:**
  - **Truthful identity kernel:** Riko says what model she runs on and never claims to be ChatGPT.
  - A character bible: values, opinions, wants, boundaries, examples.
  - The original prompt is kept as the `classic` preset.
  - Nicknames are optional.

### Emotion

- **Original Riko:** none.
- **Riko Next 0.1:** per-sentence tags, with a lexicon fallback.
- **Riko Next 0.2:**
  - 16 emotions.
  - **Persistent mood** (valence/arousal) that decays back to the character's baseline and is protected
    against spirals.
  - A deliberately uncertain read of the user's emotional state, which softens teasing.

### Initiative

- **Original Riko:** none.
- **Riko Next 0.1:** none.
- **Riko Next 0.2:**
  - Greetings, reminders the user asked for, and rare idle nudges.
  - Off, minimal or normal levels, plus quiet hours.
  - Never stored as user messages, and never counted toward familiarity.

### Familiarity

- **Original Riko:** n/a.
- **Riko Next 0.1:** n/a.
- **Riko Next 0.2:** a three-stage familiarity based only on time and completed sessions, never on disclosures.

## Interface and avatar

### UI

- **Original Riko:** none.
- **Riko Next 0.1:** voice halo, captions, history, memory panel.
- **Riko Next 0.2:** same as 0.1, plus:
  - A thinking indicator and tool activity.
  - Image and camera input.
  - A tasks tab.
  - Proactivity settings.
  - A separate developer diagnostics section.

### Avatar

- **Original Riko:** none.
- **Riko Next 0.1:** optional VRM, with a 2D fallback.
- **Riko Next 0.2:** same as 0.1, plus:
  - A glance while thinking.
  - Idle motion and blink rate driven by mood.
  - Respects the OS reduced-motion setting.

## Operations

### Safety

- **Original Riko:** API key in YAML.
- **Riko Next 0.1:** secrets only in env, localhost binding, token and origin checks.
- **Riko Next 0.2:** same as 0.1, plus:
  - Cross-origin writes are refused.
  - Tool permission classes.
  - File roots are allowlisted and secret files refused.
  - Uploaded images are kept in memory only.

### Evaluation

- **Original Riko:** none.
- **Riko Next 0.1:** 61 offline tests.
- **Riko Next 0.2:**
  - 131 offline tests.
  - Recorded gpt-oss fixtures.
  - `riko probe` (capability manifest).
  - `riko eval smoke` (20 cases, deterministic checks, latency percentiles).
  - Opt-in live tests.

## What Riko Next still doesn't do

- Partial or streaming transcription and adaptive endpointing. These are deferred until they can be measured.
- Fillers and backchannels.
- Safe code execution.
- A blinded persona evaluation with a judge.
- Live2D and VRMA animation layers.
- Wake word.
- A Discord client.
