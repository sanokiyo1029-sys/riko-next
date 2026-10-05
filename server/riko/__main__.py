"""Command line entry point.

    riko serve            start the WebSocket server + web UI (default)
    riko chat             talk in the terminal (local mic/speakers), like the original main_chat.py
    riko chat --text      keyboard-only chat (no audio devices needed)
    riko check            diagnose configuration, LLM, TTS, ASR, audio devices, memory
    riko memory list|stats|forget ID|maintain
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
    from .core.session import Services

    failures = 0

    def report(ok: bool, label: str, detail: str) -> None:
        nonlocal failures
        failures += 0 if ok else 1
        print(f"  [{'ok' if ok else 'FAIL'}] {label:<14} {detail}")

    print(f"\nRiko Next {__version__} diagnostics (config root: {cfg.root})\n")
    try:
        svc = Services.from_config(cfg)
    except Exception as e:  # noqa: BLE001
        report(False, "startup", str(e))
        return 1
    report(True, "config", f"llm={cfg.llm.provider}:{cfg.llm.model} asr={cfg.asr.provider} tts={cfg.tts.provider}")
    report(True, "personality", f"{svc.personality.current.name} presets={','.join(svc.personality.presets())}")
    ok, msg = await svc.llm.health()
    report(ok, "llm", msg)
    if ok:
        try:
            reply = await asyncio.wait_for(svc.llm.complete([{"role": "user", "content": "Say hi in three words."}],
                                                            max_tokens=20), 60)
            report(bool(reply.strip()), "llm stream", repr(reply.strip()[:60]))
        except Exception as e:  # noqa: BLE001
            report(False, "llm stream", str(e))
    if svc.tts.produces_audio:
        ok, msg = await svc.tts.health()
        report(ok, "tts", msg)
        voice = cfg.tts.voices[cfg.tts.voice]
        if cfg.tts.provider == "gpt_sovits" and voice.ref_audio:
            p = cfg.resolve_path(voice.ref_audio)
            report(p.is_file(), "ref audio", str(p))
    else:
        report(True, "tts", f"{cfg.tts.provider} (no server audio)")
    if not skip_asr:
        try:
            await svc.asr.warmup()
            report(True, "asr", f"{cfg.asr.provider} {getattr(svc.asr, 'device', '') or ''}".strip())
        except Exception as e:  # noqa: BLE001
            report(False, "asr", str(e))
    from .audio.vad import EnergyVAD, create_vad_engine
    eng = create_vad_engine(cfg.audio.vad)
    report(True, "vad", eng.__class__.__name__ + ("" if not isinstance(eng, EnergyVAD) or cfg.audio.vad.engine == "energy"
                                                  else " (fallback - install faster-whisper for Silero)"))
    try:
        import sounddevice as sd  # type: ignore
        report(True, "audio devices", f"in={sd.query_devices(kind='input')['name']} | "
                                      f"out={sd.query_devices(kind='output')['name']}")
    except Exception as e:  # noqa: BLE001
        print(f"  [--] audio devices  not available for terminal mode ({e.__class__.__name__}); web UI unaffected")
    if svc.memory:
        try:
            vec = await svc.memory.embedder.embed(["test"])
            report(True, "memory", f"{cfg.resolve_path(cfg.memory.db_path)} embed={svc.memory.embedder.model_id} "
                                   f"dim={vec.shape[1]} {svc.memory.store.stats(svc.memory.character)}")
        except Exception as e:  # noqa: BLE001
            report(False, "memory", str(e))
    await svc.aclose()
    print(f"\n{'All checks passed.' if not failures else f'{failures} check(s) failed.'}\n")
    return 1 if failures else 0


def cmd_check(args: argparse.Namespace) -> None:
    cfg = _setup(args)
    raise SystemExit(asyncio.run(_check(cfg, args.skip_asr)))


# ------------------------------------------------------------------ memory
def cmd_memory(args: argparse.Namespace) -> None:
    from .core.session import Services
    from .memory import Layer

    cfg = _setup(args)

    async def run() -> None:
        svc = Services.from_config(cfg)
        mem = svc.memory
        if not mem:
            print("memory is disabled in config")
            return
        if args.action == "list":
            for m in mem.store.list(mem.character, Layer(args.layer) if args.layer else None):
                print(f"{m.id}  {m.layer.value:<8} imp={m.importance:.2f} str={mem.strength(m):.2f}  {m.content}")
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
    p.add_argument("action", choices=["list", "stats", "forget", "maintain", "reindex"])
    p.add_argument("id", nargs="?")
    p.add_argument("--layer", choices=["episodic", "semantic", "working"])
    p.set_defaults(fn=cmd_memory)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        args = ap.parse_args(["serve", *(argv or sys.argv[1:])])
    args.fn(args)


if __name__ == "__main__":
    main()
