# Riko Next — architecture audit

Audit of the repository as imported on 2026-10-06 (commit `da57d31`, "Initial import"), before any change in
this round. It was written after reading every server module, the web client and the tests, running the
suite, and probing the local runtime. The requirements come from the chat build prompt and
`docs/BUILD_SPEC_v3.pdf` (v3.0).

## 1. Baseline facts (measured, not assumed)

| Item | Result |
|---|---|
| Tests | `pytest -q`: **61 passed, 1 skipped** (Silero VAD test: faster-whisper not installed in the venv) |
| Python | 3.13.7 (venv `.venv`, `pip install -e ".[dev]"`) |
| Ollama | 0.35.1 at `127.0.0.1:11434` |
| gpt-oss:20b | digest `17052f91a42e`, MXFP4, 20.9B params, 131072 max ctx, capabilities `completion, tools, thinking` |
| Other local model | `qwen3.6:latest` (35.5B MoE, Q4_K_M, capabilities incl. `vision`) |
| GPU | NVIDIA RTX 5080, 16 GB; gpt-oss:20b at num_ctx 8192 is fully resident (`size_vram` 12.7 GB) |
| GPT-SoVITS / LM Studio | not running during the audit, so TTS was verified only against mock servers |
| faster-whisper | not installed (no downloads were permitted), so ASR was verified only against fakes |

### gpt-oss:20b live probe (Ollama native `/api/chat`)

The probe sent synthetic inputs only. Fixtures are in `tests/fixtures/ollama_gpt_oss/`.

- **Thinking.**
  - `/api/show` reports `thinking: {values: [low, medium, high], default: medium}`.
  - `think: false` is ignored and the model still reasons, so effort must be validated against these values.
  - Reasoning streams in `message.thinking`, separately from `message.content`. No chunk mixed the two, but
    the adapter handles mixed chunks anyway.
- **Latency.** On a trivial prompt, the default medium effort produced its first content at about 2.3 s, while
  `think: "low"` took about 0.3–0.4 s. Low is the right default for conversation.
- **Starvation.** `num_predict: 40` gave `done_reason: "length"` with an **empty answer**: the reasoning ate
  the budget. The old `max_tokens: 400` was a real failure risk for reasoning models.
- **Tool calls.**
  - They arrive as one chunk: `message.tool_calls: [{id, function: {index, name, arguments: {..object..}}}]`.
  - Results go back as `{"role": "tool", "tool_name": ..., "content": ...}`.
  - The template reserves the names `python`, `browser.search`, `browser.open` and `browser.find` for built-in
    tools, so our tools must not use them.
- **Structured output.** `format: <JSON schema>` was honoured.
- **Usage counters.** The final chunk has `prompt_eval_count`, `prompt_eval_cached_count` (cache counter
  present), `eval_count`, durations and `done_reason`.
- **Template.**
  - The template hard-codes `You are ChatGPT, a large language model trained by OpenAI` in the system turn.
  - All `system` messages, wherever they sit in the list, are merged into the *developer* "# Instructions"
    block.
  - Without an identity rule, the model answered "I am ChatGPT…". The persona must state the truthful
    identity explicitly.
- **Reloads.** A request with a different `num_ctx` forces a model reload (about 3.9 s), so every request
  must send the same `num_ctx`.
- **Cancellation.**
  - Closing the client stream stops generation. The next foreground request saw its first delta after
    52–1512 ms, against 47–50 ms warm.
  - A long request running *concurrently* delayed a foreground request by **8.9 s**, because Ollama serves
    one request at a time here.
  - So background LLM work must be preempted (cancelled), not merely deprioritised.
- **Preload.** An empty `messages: []` request with `keep_alive` returned `done_reason: "load"` (cold load
  11.6 s).

## 2. What already works (keep it)

- **Session pipeline** (`core/session.py`).
  - It is transport-agnostic, and each turn runs as one cancellable asyncio task.
  - The token stream passes through ThinkFilter and SentenceChunker before a per-sentence TTS queue, so
    audio starts before the LLM finishes.
  - It handles barge-in with ducking and a confirmation threshold, and push-to-talk.
  - When interrupted, it stores only the sentences that started playing.
  - LLM, TTS and ASR failures are reported and are not fatal.
- **Providers.**
  - Ollama is native NDJSON with `keep_alive`.
  - OpenAI-compatible SSE covers OpenAI, LM Studio, OpenRouter, Groq and llama.cpp.
  - GPT-SoVITS `api_v2` streams WAV.
  - OpenAI-compatible TTS (Kokoro).
  - ASR is faster-whisper (with CPU fallback) or Groq/OpenAI.
  - Everything uses `httpx` with connection reuse, which allows exact cancellation.
- **VAD.** Silero ONNX streams frame by frame, with an energy fallback, pre-roll and hysteresis.
- **Memory.**
  - SQLite and a numpy cosine index, with Generative-Agents scoring (relevance, strength, recency).
  - Per-layer decay, reinforcement on recall, dedup, consolidation, and a rolling summary.
  - LLM extraction with a heuristic fallback.
- **Personality.** YAML character files with presets, validated with pydantic.
- **Emotion.** Inline `[tag:0.7]` with zero latency, plus a lexicon fallback; it drives TTS speed and the
  avatar expression.
- **Security.**
  - Secrets come only from env and `.env`, and inline secrets in YAML are rejected.
  - Config uses `extra=forbid`.
  - The server binds to localhost, checks the token and origin, validates WS messages with pydantic, and
    redacts secrets in logs.
- **Web client.**
  - Plain JS with no build step: a worklet for the mic, gapless playback with flush, captions, a memory
    panel, settings.
  - Optional VRM with a 2D fallback.
- **Tests.** The tests use realistic wire-format mocks, cover the session behaviour, and include WS
  end-to-end smoke tests.

## 3. What is broken or incomplete

| # | Problem | Evidence | Severity |
|---|---|---|---|
| B1 | Token budget too small for reasoning models. `max_tokens: 400` and no `think` control, so gpt-oss can return an empty answer | probe: `done_reason=length`, content empty | high |
| B2 | No `think` effort control. The default medium effort adds about 2 s to first content | probe | high |
| B3 | Ollama reasoning (`message.thinking`) is silently dropped. That is safe, but there is no "thinking" state, no metric, and no OpenAI `reasoning_content` handling | `llm/ollama.py` | medium |
| B4 | **Background LLM work (memory extraction, summary, consolidation) is not preemptable.** It waits for idle only once, then runs. If the user speaks during it, the reply queues behind it, adding about 9 s | probe E, `_post_turn_memory` | high |
| B5 | Truthful identity missing. The template says "You are ChatGPT" and the persona never states the real model | probe C | high |
| B6 | Retrieved memories are interpolated into the *instruction* text, which contradicts I04 ("Things you remember…") | `personality.build_system_prompt` | medium |
| B7 | No context accounting. Nothing guarantees `prompt + max_tokens <= num_ctx`, and `num_ctx` is unset by default (Ollama uses 4096), so long histories truncate silently | `ollama.py` | medium |
| B8 | Memory extraction accepts any JSON item. There is no evidence check, so a negation ("I do not live in Delhi") can become a fact. No confidence, provenance, status or supersession | `manager.extract_from_turn` | medium |
| B9 | Deleting a memory doesn't invalidate in-flight extraction, so a forgotten fact can be re-added a second later | `manager.forget` | medium |
| B10 | No per-profile scoping (I05). Memory is scoped only by character | `store.py` | low (single user) |
| B11 | Metrics are named `llm_first_token_ms` but measure the first delta of any kind; there is no thinking/content split and no usage | `session._run_turn` | low |
| B12 | `riko check` has only ok/FAIL, with no warn/unknown/disabled | `__main__._check` | low |

## 4. Missing features (against both specs)

- **Tool calling and agent loop.** Providers only stream text. There are no tool schemas, no execution, no
  permissions, and no multi-step loop.
- **Model routing.** There is one global `llm`. No fast, smart or vision roles.
- **Vision.** No image input path.
- **Memory types.** There are no relationship or task layers. There is no edit, correct, clear, export or
  disable from the UI or API, and no familiarity or stage.
- **Persistent mood.** There is no valence/arousal with decay, and no user-affect signal.
- **Proactivity.** There are no greetings, nudges, due follow-ups or quiet hours.
- **Event names.** The new dotted names (`assistant.token`, `tool.started`, …) and a no-text
  `assistant.thinking` state are missing.
- **Delivery bounds.** There is no content or overall deadline, and no spoken-length limit.
- **Warm-up.** There is no LLM warm-up, only ASR.
- **Evaluation.** There is no capability manifest and no eval harness (`data/eval/`).

## 5. Architecture weaknesses and technical debt

- **`ConversationSession` is a 500-line class** that mixes audio, turns, TTS and memory scheduling. It is
  acceptable, but new features should go in separate modules (tools, scheduler, proactive, mood). Don't make
  the class bigger.
- **The `Services` dataclass builds the memory "completer"** as a closure around the chat LLM. This is the
  natural place for model routing.
- **Background tasks are spawned per session** (`_spawn`), but the LLM is a shared, single-lane resource.
  Scheduling must happen at the `Services` level.
- **History role mapping.** History stores tag-stripped text, but `context_messages` replays raw content that
  still contains `[happy]` tags. That's harmless and useful, because it shows the model the tag format.
- **Duplicated emotion list.** The frontend has its own hard-coded emotion regex, so adding an emotion
  needs edits in two places.
- **Duplicated config.** `config.yaml` is a copy of `config.example.yaml`, which is fine because it's
  git-ignored.

## 6. Performance notes

- The dominant latency is LLM time to first content, which depends on effort. TTS first audio comes next
  (GPT-SoVITS is not measured here).
- `num_ctx` must be constant across *all* calls, including memory jobs, or Ollama reloads the model.
- Retrieval loads every memory row per query (`store.list`). That's fine for thousands of rows; mark it as a
  ceiling.
- Prompt-cache reuse can be measured from `prompt_eval_cached_count`. Ollama merges system messages into
  the developer block, so the cached prefix ends where dynamic context begins.

## 7. Decisions for this round (why, briefly)

1. **Extend, don't replace.** Every working module stays. New code goes into `llm/` (structured stream
   events, tools), `tools/`, `core/scheduler.py`, `core/proactive.py`, `memory/` (migrations, new layers,
   evidence), `personality/` (bible fields, identity kernel), and `emotion/` (mood).
2. **Structured LLM stream.**
   - `stream_events()` yields `Thinking` (no text leaves the provider layer), `Content`, `ToolCalls` and
     `Done(usage)`.
   - `stream_chat()` stays, as a text-only wrapper, for compatibility.
3. **Model routing by role** (`chat`, `agent`, `memory`, `vision`). Each role is configured under `models:`,
   and anything left unset falls back to `llm:`. When the configs are the same, one provider instance is
   shared, so there is no second model load.
4. **Tools on by default** (user decision). Every default tool is local and side-effect-free, or limited to
   the app's own database:
   - time, calculator, memory search/remember/forget, notes, tasks, reminders.
   - Network tools (weather, web search) and file tools are off by default and need explicit allowlists
     (I11).
   - There is no shell or code execution. Tool calls stay inside the normal streamed turn, with a bounded
     iteration count.
5. **One foreground lane.**
   - A `Scheduler` counts foreground turns. Background memory jobs run only when the system is idle, and
     they are **cancelled** when a turn starts, then retried with a bound.
   - This is justified by probe E.
6. **Dynamic context as a separate, JSON-quoted `system` message** marked as untrusted data, kept apart from
   the static persona block (I04).
7. **gpt-oss defaults:**

   | Setting | Value |
   |---|---|
   | `think` | `low` |
   | `num_ctx` | `8192` |
   | Combined generation budget | `2048` |
   | Safety margin | `512` |
   | `temperature` | `1.0` |
   | `keep_alive` | `30m` |
