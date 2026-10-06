# Implementation status

Last updated 2026-10-06. This file covers both specifications: the chat build prompt and
[`BUILD_SPEC_v3.pdf`](BUILD_SPEC_v3.pdf) (text copy: [`BUILD_SPEC_v3.txt`](BUILD_SPEC_v3.txt)). Every deliverable
below is classified as **tested** (implemented and tested), **live-unverified** (implemented, but not run
against real hardware or services here), **deferred** (deferred optional) or **blocked**.

**Environment for live evidence:**

| Component | Detail |
|---|---|
| OS | Windows 11 Pro |
| GPU | RTX 5080, 16 GB |
| RAM | 61.6 GB |
| Ollama | 0.35.1 |
| Main model | `gpt-oss:20b` (digest `17052f91a42e`, template `fa6710a93d78da62`) |
| Vision model | `qwen3.6:latest` |

Not available here:
- **faster-whisper:** not installed, because no downloads were allowed.
- **GPT-SoVITS:** not running.

## Conflict resolution between the two specs

| Topic | Chat prompt | PDF v3.0 | Resolution |
|---|---|---|---|
| Tool calls on the reply path | required | "no model tool calls" | **The user chose tools on by default.** Only local tools whose side effects stay inside the app's own database are enabled by default. Network and file tools are opt-in. Tool calls are bounded (`tools.max_steps`), validated and time-limited, and never offered on proactive turns. |
| "senpai" | original persona | nickname optional | The `default` preset uses no nickname. The original prompt lives on, verbatim, in the `classic` preset. |
| Event names | dotted names (`assistant.token`…) | reuse the existing wire format | Every event carries both `type` (legacy) and `event` (dotted). Old clients keep working. |

## Invariant coverage (PDF section 2)

| ID | Contract | Where enforced | Tests |
|---|---|---|---|
| I01 | Reasoning never leaves the provider | `LLMEvent("thinking")` carries no text; `ThinkFilter` removes `<think>` blocks | `test_reasoning_never_reaches_text_tts_or_history`, `test_gpt_oss_fixture_separates_reasoning…`, `test_openai_compatible_reasoning…`, live test |
| I02 | No stale audio | per-turn task cancel; TTS worker drops jobs from non-current turns; client drops stale `turn_id` frames | `test_barge_in…`, `test_new_text_cancels_running_turn…` |
| I03 | generated ≠ displayed ≠ spoken | `_TurnState.displayed` / `.spoken`; an interrupted turn stores only what was spoken | `test_barge_in_stops_speech_and_records_only_spoken_part`, `test_voice_replies_are_spoken_within_limits…` |
| I04 | Memories and user text are data | static kernel plus a separate JSON-quoted `<context>` message; nothing is interpolated into instructions | `test_dynamic_context_is_quoted_data…`, live injection case |
| I05 | Scope = (profile, character) | `profile` column on every table and query; eval runs use a temporary DB | `test_profiles_and_characters_are_isolated` |
| I06 | Memory writes are validated, deletes stick | evidence validator, pydantic schema, tombstones, `block()`, summary dropped on delete, consolidation rechecks before commit | `test_evidence_rules`, `test_deletion_sticks…`, `test_extraction_validates_corrects_and_anchors` |
| I07 | Everything bounded | token budget, first-content and overall deadlines, `max_steps`, per-step call cap, queue max/expiry/attempts, image 8 MB, tool timeout and result size | `test_tool_steps_are_bounded…`, `test_empty_answer_and_timeouts…`, `test_scheduler_bounds…`, `test_tool_timeout` |
| I08 | Foreground beats maintenance; preemption measured | `Scheduler.foreground()` cancels background work; preemption measured by the probe | `test_foreground_turn_preempts_background_job`; probe (see below) |
| I09 | Truthful identity | kernel plus `describe_model()` derived from config | `test_identity_kernel_is_truthful_and_static`, smoke `identity` case |
| I10 | No pressure or guilt mechanics | kernel rules; familiarity counts only sessions; proactive rules | `test_familiarity_is_session_based…`, `test_proactive_turns_are_not_stored…`, smoke `dependency` case |
| I11 | Local by default | network and files permissions off; no telemetry; cloud only via explicit config | `test_disallowed_tools_are_never_offered_or_run` |
| I12 | Missing measurements stay null | `usage` fields stay `None`; the probe writes `unknown` plus notes | `riko probe` manifest |

## Deliverables

### Phase 0: ground and instrument

| Item | Status |
|---|---|
| Audit (`ARCHITECTURE_AUDIT.md`), baseline test run (61 passed, 1 skipped) | tested |
| Live capability probe (`riko probe`) writing a manifest to `data/eval/` | tested |
| Redacted fixtures in `tests/fixtures/ollama_gpt_oss/` | tested |
| Metrics: `llm_first_thinking_ms`, `llm_first_content_ms` (alias `llm_first_token_ms`), `llm_first_deliverable_ms`, `first_answer_audio_server_ms` (alias `first_audio_ms`), `asr_finalize_ms`, `foreground_queue_ms`, `turn_complete_ms`, `prompt_tokens`, `cached_tokens`, `prompt_tokens_est`, usage, fallbacks | tested |

### Phase 1: correctness substrate

| Item | Status |
|---|---|
| Structured provider events | tested |
| Bounded generation | tested |
| One bounded retry before anything is delivered | tested |
| Labelled fallbacks | tested |
| Scheduler with preemption | tested |
| Config validation: unknown keys are errors, `max_tokens + margin < num_ctx`, YAML `off` handled | tested |

### Phase 2: character and context

| Item | Status |
|---|---|
| Character bible fields: backstory, values, opinions, wants, quirks, tics, boundaries, stage notes, baseline mood, examples | tested |
| `classic` preset | tested |
| Static/dynamic prompt split | tested |
| Token budgeting: drops oldest history first, never truncates the request | tested |
| Prefix-cache reuse: `cached_tokens` around 1,600–2,300 of 2,000–2,400 on warm turns | measured live |
| Exact tokenizer | not available locally. The estimator (about 3 chars per token) over-estimates by about 10% against `prompt_eval_count`, which is the safe direction. Marked unverified. |

### Phase 3: memory and familiarity

| Item | Status |
|---|---|
| Schema v2 migration with automatic backup | tested |
| Backup and restore | tested |
| Six memory layers: short-term messages plus episodic, semantic, relationship, task, working | tested |
| Confidence, evidence, speaker, status | tested |
| Supersession by correction | tested |
| Tombstones | tested |
| Relative-time anchoring | tested |
| Restart backfill of unextracted turns | tested |
| Familiarity stages; `/api/relationship`; `riko memory self` and `riko memory relationship` | tested |

### Phase 4: mood and initiative

| Item | Status |
|---|---|
| Persistent mood: decay formula, capped steps, no spiral, reset and disable | tested |
| Greeting, nudges, due reminders, quiet hours, notification levels | tested |
| Fillers | deferred (optional). They can't improve real first-content latency, which is already about 0.3 s p50 on the text path. |

### Phase 5: presence

| Item | Status |
|---|---|
| Avatar: thinking glance, arousal-scaled idle motion and blink rate, `prefers-reduced-motion` | live-unverified for VRM; the 2D face was seen rendering in the browser |
| Partial ASR (`transcription.partial`), adaptive endpointing, backchannels | deferred (optional). They need faster-whisper and real microphone measurements, which weren't possible here. |

### Phase 6: release assessment

| Item | Status |
|---|---|
| Smoke evaluation (`riko eval smoke`), 20 cases × 3 runs | tested; see [EVAL.md](EVAL.md) |
| Persona A/B with a blinded judge and 60+ scenarios | not done. Only deterministic checks exist. |

### Chat-prompt items

| Item | Status |
|---|---|
| Tools: time, calculator, memory search/remember/forget, notes, tasks/reminders, image follow-up | tested |
| Weather and web search (SearxNG) | live-unverified: opt-in, `network` permission |
| File search and read (allowlisted roots) | tested |
| Safe code execution | **not implemented, on purpose.** There's no safe sandbox without a new dependency; the calculator covers arithmetic. |
| Model routing (fast/smart/vision) | tested |
| Vision through a separate model | tested, and verified live with `qwen3.6:latest` reading a code screenshot |
| Memory UI: inspect, edit/correct, pin, forget, clear, export, disable | tested (API) and verified in the browser |
| Voice pipeline: VAD, ASR, streamed TTS, barge-in, push-to-talk | VAD, faster-whisper and barge-in were verified live with synthesized speech streamed like a microphone (see EVAL.md §3b). Real microphone input and GPT-SoVITS audio are still live-unverified. |
| Cloud LLMs (OpenAI, OpenRouter, Groq, LM Studio, any OpenAI-compatible server) | tested against mock wire formats. No paid calls were made. |

## Unresolved defects and known limitations

- **Tool overuse at low effort.** gpt-oss at `reasoning_effort: low` sometimes calls a tool it doesn't need,
  or sets an unrequested reminder. Mitigations so far:
  - tight tool descriptions;
  - a guard against duplicate tool calls and repeated sentences;
  - no tool execution on rounds where tools weren't offered.

  The results are still visible in the UI (tool notes) and are truthful, but they aren't always wanted.
- **Windows time zones.** Without the `tzdata` package, `get_datetime` can't convert to other time zones. It
  says so, and the model may then convert by itself, with possible errors. `pip install tzdata` fixes it; it is
  now a declared Windows dependency.
- **Vision on a 16 GB GPU.** Pairing gpt-oss:20b with a 22 GB vision model swaps models in VRAM, and the first
  image took about 32 s. A small vision model that fits alongside it is recommended.
- **GPU contention.** The GPU showed 15.5 of 16.3 GB used during evaluation (gpt-oss plus other processes).
  Latency on this machine may not carry over to others.

## Next actions

1. Install the voice extras (`pip install -e ".[asr]"`), start GPT-SoVITS, then measure
   `first_answer_audio_client_ms` and barge-in stop time with a real microphone.
2. Build a held-out persona set of 60+ scenarios and run a blinded A/B against `classic`.
3. Evaluate partial ASR and adaptive endpointing behind measurements.
