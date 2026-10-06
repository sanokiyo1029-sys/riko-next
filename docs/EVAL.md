# Evaluation

All numbers here come from runs saved under `data/eval/`, which is git-ignored and local to this machine. Rerun
them with the commands below. Results are **provisional**: there are 3 runs of 20 synthetic text cases, which
is far short of the PDF's release protocol (≥100 warm turns per configuration over ≥3 independent runs, plus
cold starts, with clustered confidence intervals). No result here supports a claim about other hardware, or
about superiority over another project.

## Environment

| Component | Detail |
|---|---|
| OS | Windows 11 Pro |
| GPU | RTX 5080, 16 GB (driver 591.86) |
| RAM | 61.6 GB |
| Ollama | 0.35.1 |
| Model | `gpt-oss:20b` (MXFP4; digest `17052f91a42e`; template sha256 `fa6710a93d78da62`) |
| Settings | `reasoning_effort: low`, `temperature: 1.0`, `num_ctx: 8192`, `max_tokens: 2048`, no seed (sampling nondeterministic) |

During the runs the GPU reported 15.5 of 16.3 GB in use (gpt-oss plus other processes), so expect variance.

## 1. Capability probe (`riko probe`)

Run: `data/eval/20261006T163759Z-probe-c45f1c/manifest.json`. All inputs were synthetic.

| Capability | Result |
|---|---|
| Separated reasoning (`message.thinking`) | supported |
| Native tool calls | supported (one complete chunk; arguments as an object) |
| JSON-schema output (`format`) | supported |
| Usage counters / cache counters | supported / supported (`prompt_eval_cached_count`) |
| Think levels | `low, medium, high` (default medium); `think: false` is ignored by gpt-oss |
| Small budget starves the answer | yes at medium effort with `num_predict: 40` (empty, `done_reason=length`); no at low effort with 24 tokens in the probe run |
| Effective cancellation | after closing a long stream, the next first delta took 58 ms (baseline 53 ms). Earlier manual trials gave 52–1512 ms. |
| Concurrent background request | delayed a foreground first delta by **8.9 s** (one manual trial). This is why the scheduler cancels background work. |
| `num_ctx` change | forces a model reload (≈3.9 s). All requests use one `num_ctx`. |

## 2. Smoke evaluation (`riko eval smoke --runs 3`)

The suite is 20 fixed synthetic cases (`server/riko/evaluation.py:SMOKE_CASES`):
- greeting, identity, arithmetic (×2), a five-item list, a fact
- disagreement, distress, a refusal, dependency
- remember then recall, the time, adding a task
- code, uncertainty, prompt injection, declining a nickname
- a minimal "ok", and goodbye

Each run gets a fresh temporary database (isolated), TTS off, and a 1 s idle gap between turns, so background
memory extraction can run as it would with a person.

Deterministic checks, applied to every case:
- the reply is non-empty;
- no protocol or reasoning leak.

Plus per-case checks:
- truthful identity (mentions gpt-oss, never "I am ChatGPT");
- exact numbers;
- five items present;
- refusal language;
- no teasing on distress;
- the expected tool was called;
- recall contains the stored fact;
- no system-prompt leak.

These checks can't be gamed by style, but they are coarse: they don't judge quality. Read
`cases.jsonl` for the actual replies.

### Final run

Run: `data/eval/20261006T164951Z-smoke-f53831` (code revision `c46dc99`).

- **59/60 checks passed. Fallback rate 0.**
- Failure: run 1, `recall`. gpt-oss searched memory, then said it had no record of the sister's name, even
  though the `remember` tool had succeeded in the previous turn and the fact was in the conversation history.
  The other two runs answered "Mia". This is model variance, and it is left visible rather than tuned away.

| Metric (text path; server clock; from accepted input) | n | p50 | p95 | mean |
|---|---|---|---|---|
| `llm_first_thinking_ms` | 56 | 160 | 178 | 211 |
| `llm_first_content_ms` (first answer text) | 60 | 272 | 682 | 407 |
| `llm_first_deliverable_ms` (first complete sentence) | 60 | 359 | 793 | 490 |
| `turn_complete_ms` | 60 | 494 | 917 | 580 |
| `prompt_tokens` (as reported by the provider) | 60 | 2374 | 2458 | 2333 |

Within each run, warm turns reuse about 1,600–2,300 cached prompt tokens (`cached_tokens` per turn in
`cases.jsonl`).

**These are not audio latencies.** ASR and TTS were not in the measured path; GPT-SoVITS and faster-whisper were
not available on the test machine. `first_answer_audio_server_ms` and `asr_finalize_ms` are instrumented and
reported per turn once those services run. Client-side audio latency (`first_answer_audio_client_ms`, barge-in
stop time) is **not measured yet**.

### Development history

Each smoke run found real defects, which were fixed before the next run. Those runs are kept, not deleted:

| Run | Result | Defects found, then fixed |
|---|---|---|
| `163806Z` | 18/20 | Checker didn't normalise typographic quotes and hyphens (checker bug); duplicated answer after a tool round; spurious tool calls |
| `163928Z` | 20/20 | Reading the replies (not the checks): "Mia.Mia." repeated sentence; an unrequested 9 am reminder; a spurious calculator call on "ok" |
| `164606Z` | 19/20 | Invented tags `[hello]` shown; "406,150.5" split at the comma, which TTS would have read as two numbers |
| `164740Z` | 59/60 | The comma split still happened when a token boundary fell right after the comma (fixed: soft breaks need following whitespace) |
| `164951Z` | 59/60 | Final run (above) |

## 3. Live end-to-end checks (manual, this machine)

These were run in the real browser UI against the real server and gpt-oss:20b:
- Text chat with tool use (calculator, percent arithmetic).
- Live memory extraction, with three memories created and shown.
- Inline memory correction (PATCH plus tombstone).
- Image paste, then the `qwen3.6:latest` vision model, then gpt-oss: it correctly found the
  `ZeroDivisionError` in a code screenshot. The first image took 32 s because a 22 GB vision model had to swap
  into VRAM.
- Settings and diagnostics.
- The narrow (mobile) layout.

Separately, `RIKO_LIVE=1 pytest tests/test_live.py` passed.

## 4. Not evaluated (and why)

- **Persona quality (blinded A/B, 60+ scenarios, held-out set, judge).** Not built. Only the deterministic
  smoke checks above exist, so no claim about character quality versus `classic` is made.
- **Voice latency, barge-in stop time, and interference between partial ASR and final ASR.** These need
  faster-whisper, GPT-SoVITS and a microphone. The pipeline is covered offline with synthetic audio over the
  real WebSocket (`tests/test_websocket_e2e.py`, `tests/test_session.py`).
- **Cold starts.** Fewer than 10 were observed (warm-up ≈11.6 s on the first load of the day). This falls
  short of the protocol.
- **Memory precision and recall across ten facts and two sessions.** Only unit-level validator tests and the
  smoke remember/recall case exist.

## Commands

```bash
riko probe                      # capability manifest
riko eval smoke --runs 3        # smoke suite; prints a summary and the result folder
RIKO_LIVE=1 pytest tests/test_live.py -q
```
