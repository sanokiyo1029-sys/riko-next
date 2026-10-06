"""Command line entry point.

    riko serve            start the WebSocket server + web UI (default)
    riko chat             talk in the terminal (local mic/speakers), like the original main_chat.py
    riko chat --text      keyboard-only chat (no audio devices needed)
    riko check            diagnose configuration, LLM, TTS, ASR, audio devices, memory
    riko probe            record what the configured model really supports (data/eval/...)
    riko eval smoke       run the synthetic smoke evaluation (data/eval/...)
    riko memory list|stats|forget ID|maintain|reindex|self|relationship|export|clear|backup|restore
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import webbrowser
from typing import Any, Optional

from . import __version__
from .core.config import ConfigError, RikoConfig, get_secret, load_config
from .core.logging import get_logger, setup_logging

log = get_logger("cli")


def _setup(args: argparse.Namespace) -> RikoConfig:
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        raise SystemExit(2)
    level = "DEBUG" if getattr(args, "debug", False) else cfg.logging.level
    secrets = [get_secret(n) for n in (cfg.llm.api_key_env, cfg.asr.api_key_env,
                                       cfg.tts.openai_compatible.api_key_env,
                                       cfg.memory.embeddings.api_key_env, cfg.server.auth_token_env)]
    setup_logging(level, cfg.logging.format, cfg.resolve_path(cfg.logging.file) if cfg.logging.file else None, secrets)
    return cfg


# ------------------------------------------------------------------ serve
def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api.server import check_bind_security, create_app

    cfg = _setup(args)
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    check_bind_security(cfg)
    url = f"http://{'localhost' if cfg.server.host in ('127.0.0.1', '0.0.0.0') else cfg.server.host}:{cfg.server.port}"
    print(f"\n  Riko Next {__version__} -> open {url}\n")
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level="warning", ws_max_size=1 << 20)


# ------------------------------------------------------------------ chat (terminal)
class TerminalSink:
    def __init__(self, player: Any = None) -> None:
        self.player = player
        self.active_turn = 0

    async def send_event(self, ev: dict[str, Any]) -> None:
        t = ev["type"]
        if t == "transcription" and ev.get("text"):
            print(f"\n  you  > {ev['text']}")
        elif t == "llm_started":
            self.active_turn = ev["turn_id"]
        elif t == "llm_sentence":
            print(f"  riko > {ev['text']}")
        elif t == "emotion_changed":
            print(f"         ({ev['emotion']} {ev['intensity']})")
        elif t == "memory_created":
            print(f"         [remembered: {ev['content']}]")
        elif t == "assistant_stopped" and ev.get("reason") not in ("completed",):
            if self.player:
                self.player.flush()
            print(f"         [stopped: {ev['reason']}]")
        elif t == "error":
            print(f"  ! {ev.get('component')}: {ev.get('message')}")
        elif t == "user_speech_started":
            print("  ... listening", end="\r")

    async def send_audio(self, turn_id: int, sentence_index: int, sample_rate: int, pcm: bytes) -> None:
        if self.player and turn_id == self.active_turn:
            self.player.play(pcm, sample_rate)


async def _chat(cfg: RikoConfig, text_only: bool, ptt: bool, headphones: bool) -> None:
    from .core.session import ConversationSession, Services

    services = Services.from_config(cfg)
    if not headphones:
        cfg.audio.barge_in.enabled = False
    loop = asyncio.get_running_loop()
    player = mic = None
    sink = TerminalSink()
    session = ConversationSession(services, sink)
    if not text_only:
        from .audio.local_io import MicStream, Player
        print("Loading speech recognition...")
        await services.asr.warmup()
        player = Player(lambda playing: loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(session.set_playback(playing))), cfg.audio.local.output_device)
        sink.player = player
        mic = MicStream(loop, lambda b: asyncio.ensure_future(session.feed_audio(b)), cfg.audio.local.input_device)
        mic.start()
        session.input_mode = "ptt" if ptt else "auto"
    else:
        session.mic_enabled = False
    name = services.personality.current.name
    if text_only:
        print(f"\nChatting with {name} (text). Type a message, '/quit' to exit.\n")
    elif ptt:
        print(f"\nPush-to-talk: press ENTER to start talking, ENTER again to send. '/quit' + ENTER to exit.\n")
    else:
        print(f"\nJust talk - {name} is listening (voice detection). Type text + ENTER to send text, '/quit' to exit.\n")
    recording = False
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line or line.strip() in ("/quit", "/exit"):
                break
            line = line.strip()
            if not text_only and ptt and not line:
                if not recording:
                    await session.ptt_start()
                    print("  * recording... press ENTER to send")
                else:
                    await session.ptt_stop()
                recording = not recording
            elif line:
                await session.submit_text(line)
            if session._turn_task:
                await asyncio.sleep(0)
    finally:
        await session.close()
        if mic:
            mic.close()
        if player:
            player.close()
        for t in list(session._bg):
            try:
                await asyncio.wait_for(t, 20)
            except Exception:  # noqa: BLE001
                pass
        await services.aclose()


def cmd_chat(args: argparse.Namespace) -> None:
    cfg = _setup(args)
    try:
        asyncio.run(_chat(cfg, args.text, args.ptt, args.headphones))
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


# ------------------------------------------------------------------ check
async def _check(cfg: RikoConfig, skip_asr: bool) -> int:
    """Distinguishes errors (FAIL), warnings (warn), unknowns (??) and disabled features (off)."""
    from .core.session import Services

    failures = 0

    def report(level: str, label: str, detail: str) -> None:
        nonlocal failures
        failures += level == "FAIL"
        print(f"  [{level:<4}] {label:<14} {detail}")

    print(f"\nRiko Next {__version__} diagnostics (config root: {cfg.root})\n")
    try:
        svc = Services.from_config(cfg)
    except Exception as e:  # noqa: BLE001
        report("FAIL", "startup", str(e))
        return 1
    report("ok", "config", f"llm={cfg.llm.provider}:{cfg.llm.model} asr={cfg.asr.provider} tts={cfg.tts.provider}")
    report("ok", "personality", f"{svc.personality.current.name} presets={','.join(svc.personality.presets())}")
    ok, msg = await svc.llm.health()
    report("ok" if ok else "FAIL", "llm", msg)
    if ok:
        caps = await svc.llm.capabilities()
        levels = caps.get("think_levels")
        eff = cfg.llm.reasoning_effort
        if eff and levels and eff not in levels and eff != "off":
            report("warn", "reasoning", f"effort '{eff}' not in model's levels {levels}; not sent")
        else:
            report("ok" if caps.get("thinking") is not None else "??", "reasoning",
                   f"thinking={caps.get('thinking')} levels={levels} effort={eff}")
        report("ok" if caps.get("tools") else "??" if caps.get("tools") is None else "warn", "tool calls",
               "model advertises tools" if caps.get("tools") else
               "unknown for this provider (run riko probe)" if caps.get("tools") is None else
               "model does not advertise tool support; set tools.enabled: false")
        if caps.get("context_length") and caps["context_length"] < cfg.llm.num_ctx:
            report("warn", "context", f"num_ctx {cfg.llm.num_ctx} > model maximum {caps['context_length']}")
        try:
            reply = await asyncio.wait_for(svc.llm.complete([{"role": "user", "content": "Say hi in three words."}]), 120)
            report("ok" if reply.strip() else "FAIL", "llm stream",
                   repr(reply.strip()[:60]) if reply.strip() else "empty answer (raise llm.max_tokens?)")
        except Exception as e:  # noqa: BLE001
            report("FAIL", "llm stream", str(e))
        if hasattr(svc.llm, "running"):
            loaded = await svc.llm.running()  # type: ignore[attr-defined]
            report("ok" if loaded else "??", "residency",
                   ", ".join(f"{m.get('name')} vram={round((m.get('size_vram') or 0) / 2**30, 1)}GB "
                             f"ctx={m.get('context_length')}" for m in loaded) or "no model loaded right now")
    vision = cfg.model_for("vision")
    if vision is None:
        report("off", "vision", "models.vision not set: images are not understood")
    else:
        vp = svc.router.get("vision")
        vok, vmsg = await vp.health()
        vcaps = await vp.capabilities() if vok else {}
        report("ok" if vok and vcaps.get("vision") is not False else "FAIL" if not vok else "warn", "vision",
               f"{vision.model}: {vmsg}" + (" (model does not advertise vision)" if vcaps.get("vision") is False else ""))
    report("ok" if cfg.tools.enabled else "off", "tools",
           f"{len(svc.tools.active(__import__('riko.tools', fromlist=['ToolContext']).ToolContext(cfg, svc.memory)))} "
           f"active, permissions={cfg.tools.permissions}" if cfg.tools.enabled else "tools.enabled: false")
    if svc.tts.produces_audio:
        ok, msg = await svc.tts.health()
        report("ok" if ok else "warn", "tts", msg + ("" if ok else " (replies continue as text)"))
        voice = cfg.tts.voices[cfg.tts.voice]
        if cfg.tts.provider == "gpt_sovits" and voice.ref_audio:
            p = cfg.resolve_path(voice.ref_audio)
            report("ok" if p.is_file() else "FAIL", "ref audio", str(p))
    else:
        report("off" if cfg.tts.provider == "none" else "ok", "tts", f"{cfg.tts.provider} (no server audio)")
    if not skip_asr:
        try:
            await svc.asr.warmup()
            report("ok", "asr", f"{cfg.asr.provider} {getattr(svc.asr, 'device', '') or ''}".strip())
        except Exception as e:  # noqa: BLE001
            report("warn", "asr", f"{e} (text chat still works)")
    from .audio.vad import EnergyVAD, create_vad_engine
    eng = create_vad_engine(cfg.audio.vad)
    fallback = isinstance(eng, EnergyVAD) and cfg.audio.vad.engine != "energy"
    report("warn" if fallback else "ok", "vad",
           eng.__class__.__name__ + (" (fallback - install faster-whisper for Silero)" if fallback else ""))
    try:
        import sounddevice as sd  # type: ignore
        report("ok", "audio devices", f"in={sd.query_devices(kind='input')['name']} | "
                                      f"out={sd.query_devices(kind='output')['name']}")
    except Exception as e:  # noqa: BLE001
        report("off", "audio devices", f"terminal mode unavailable ({e.__class__.__name__}); web UI unaffected")
    if svc.memory:
        try:
            vec = await svc.memory.embedder.embed(["test"])
            report("ok", "memory", f"{cfg.resolve_path(cfg.memory.db_path)} embed={svc.memory.embedder.model_id} "
                                   f"dim={vec.shape[1]} enabled={svc.memory.enabled} "
                                   f"{svc.memory.store.stats(svc.memory.character, profile=svc.memory.profile)}")
        except Exception as e:  # noqa: BLE001
            report("FAIL", "memory", str(e))
    else:
        report("off", "memory", "memory.enabled: false")
    await svc.aclose()
    print(f"\n{'All checks passed.' if not failures else f'{failures} check(s) failed.'}\n")
    return 1 if failures else 0


def cmd_check(args: argparse.Namespace) -> None:
    cfg = _setup(args)
    raise SystemExit(asyncio.run(_check(cfg, args.skip_asr)))


# ------------------------------------------------------------------ memory
def cmd_memory(args: argparse.Namespace) -> None:
    from .core.session import Services
    from .memory import Layer, MemoryStore

    cfg = _setup(args)
    if args.action == "restore":
        if not args.id:
            raise SystemExit("usage: riko memory restore BACKUP_FILE  (stop the server first)")
        MemoryStore.restore(args.id, cfg.resolve_path(cfg.memory.db_path))
        print(f"restored {cfg.resolve_path(cfg.memory.db_path)} from {args.id}")
        return

    async def run() -> None:
        svc = Services.from_config(cfg)
        mem = svc.memory
        if not mem:
            print("memory is disabled in config")
            return
        if args.action == "list":
            for m in mem.store.list(mem.character, Layer(args.layer) if args.layer else None, profile=mem.profile):
                print(f"{m.id}  {m.layer.value:<12} imp={m.importance:.2f} conf={m.confidence:.2f} "
                      f"str={mem.strength(m):.2f}  {m.content}")
        elif args.action == "self":
            for m in mem.store.list(mem.character, profile=mem.profile):
                if m.speaker == "assistant":
                    print(f"{m.id}  {m.content}   (from: \"{m.evidence[:80]}\")")
        elif args.action == "relationship":
            from .core.context import familiarity_stage
            from .personality import STAGE_LABELS
            rel = mem.store.get_relationship(mem.character, profile=mem.profile)
            stage = familiarity_stage(rel, cfg.memory.familiarity, __import__("time").time())
            print(f"stage {stage} ({STAGE_LABELS[stage]}), meaningful sessions={rel['sessions']}, "
                  f"completed user turns={rel['user_turns']}")
            for m in mem.store.list(mem.character, Layer.RELATIONSHIP, profile=mem.profile):
                print(f"{m.id}  {m.content}")
        elif args.action == "export":
            import json
            data = json.dumps(mem.store.export(mem.character, profile=mem.profile), indent=2, default=str)
            if args.id:
                open(args.id, "w", encoding="utf-8").write(data)
                print(f"exported to {args.id}")
            else:
                print(data)
        elif args.action == "clear":
            if not args.yes:
                print("refusing without --yes (deletes every memory for this character)")
            else:
                print(f"deleted {mem.clear()} memories")
        elif args.action == "backup":
            dest = args.id or str(cfg.resolve_path(cfg.memory.db_path)) + f".backup-{__import__('time').strftime('%Y%m%d-%H%M%S')}"
            print(f"backup written to {mem.store.backup(dest)}")
        elif args.action == "stats":
            print(mem.store.stats(mem.character))
        elif args.action == "forget":
            print("deleted" if args.id and mem.forget(args.id) else "not found")
        elif args.action == "maintain":
            print(f"cleaned {mem.decay_and_cleanup()}, consolidated {await mem.consolidate()}")
        elif args.action == "reindex":
            n = 0
            for m in mem.store.list(mem.character):
                if m.embed_model != mem.embedder.model_id:
                    m.embedding = (await mem.embedder.embed([m.content]))[0]
                    m.embed_model = mem.embedder.model_id
                    mem.store.upsert(m)
                    n += 1
            print(f"re-embedded {n} memories with {mem.embedder.model_id}")
        await svc.aclose()

    asyncio.run(run())


def cmd_probe(args: argparse.Namespace) -> None:
    import json

    from .evaluation import probe
    cfg = _setup(args)
    m = asyncio.run(probe(cfg))
    print(json.dumps({k: m.get(k) for k in ("health", "ollama_version", "digest", "template_sha256", "warmup",
                                            "capabilities", "notes", "path")}, indent=2, default=str))


def cmd_eval(args: argparse.Namespace) -> None:
    import json

    from .evaluation import smoke
    cfg = _setup(args)
    m = asyncio.run(smoke(cfg, runs=args.runs, limit=args.limit))
    print(json.dumps({k: m.get(k) for k in ("passed", "total", "failed_cases", "fallback_rate", "latency_ms",
                                            "path_measured", "status", "path")}, indent=2, default=str))


def main(argv: Optional[list[str]] = None) -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    ap = argparse.ArgumentParser(prog="riko", description="Riko Next AI voice companion")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", help="path to config.yaml (default: ./config.yaml or $RIKO_CONFIG)")
        p.add_argument("--debug", action="store_true", help="verbose logging")

    p = sub.add_parser("serve", help="start server + web UI")
    common(p)
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--open", action="store_true", help="open the browser")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("chat", help="terminal conversation")
    common(p)
    p.add_argument("--text", action="store_true", help="keyboard only, no audio devices")
    p.add_argument("--ptt", action="store_true", help="push-to-talk with ENTER (default: voice detection)")
    p.add_argument("--headphones", action="store_true", help="allow barge-in (needs headphones: no echo cancellation)")
    p.set_defaults(fn=cmd_chat)
    p = sub.add_parser("check", help="diagnose setup")
    common(p)
    p.add_argument("--skip-asr", action="store_true")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("memory", help="inspect or maintain memory")
    common(p)
    p.add_argument("action", choices=["list", "stats", "forget", "maintain", "reindex", "self", "relationship",
                                      "export", "clear", "backup", "restore"])
    p.add_argument("id", nargs="?", help="memory id (forget) or file path (export/backup/restore)")
    p.add_argument("--layer", choices=["episodic", "semantic", "relationship", "task", "working"])
    p.add_argument("--yes", action="store_true", help="confirm destructive actions")
    p.set_defaults(fn=cmd_memory)
    p = sub.add_parser("probe", help="probe the configured model's real capabilities")
    common(p)
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("eval", help="run an evaluation suite")
    common(p)
    p.add_argument("suite", choices=["smoke"])
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--limit", type=int, help="only the first N cases")
    p.set_defaults(fn=cmd_eval)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        args = ap.parse_args(["serve", *(argv or sys.argv[1:])])
    args.fn(args)


if __name__ == "__main__":
    main()
