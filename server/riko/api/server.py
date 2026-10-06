"""FastAPI app: WebSocket endpoint, REST API (memory, tasks, notes, images, status), and the static web client.

Security: binds to localhost by default; optional shared token (``RIKO_AUTH_TOKEN``) checked with a
constant-time compare on every route and the socket; browser requests from foreign origins are refused
on the socket and on every state-changing route. No secret ever appears in a response.
"""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .. import __version__
from ..core.config import RikoConfig, get_secret
from ..core.context import MAX_IMAGE_BYTES
from ..core.events import Ev, encode_audio_frame
from ..core.logging import get_logger
from ..core.session import ConversationSession, Services
from ..emotion import Mood, MoodModel, strip_tags
from ..llm import OllamaLLM
from ..memory import Layer
from ..tools import ToolContext
from .protocol import (MAX_AUDIO_FRAME_BYTES, MemoryCreate, MemoryPatch, MemorySettings, TaskPatch,
                       client_message)

log = get_logger("api")
WEB_DIR = Path(__file__).resolve().parents[3] / "client" / "web"


class WebSocketSink:
    """Serialises concurrent sends from the pipeline onto one socket."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._lock = asyncio.Lock()

    async def send_event(self, event: dict[str, Any]) -> None:
        async with self._lock:
            await self.ws.send_text(json.dumps(event, default=str))

    async def send_audio(self, turn_id: int, sentence_index: int, sample_rate: int, pcm: bytes) -> None:
        async with self._lock:
            await self.ws.send_bytes(encode_audio_frame(turn_id, sentence_index, sample_rate, pcm))


def _is_loopback(host: str) -> bool:
    if host in ("localhost", "testserver"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind_security(cfg: RikoConfig) -> None:
    """Refuse to expose an unauthenticated server beyond localhost."""
    if not _is_loopback(cfg.server.host) and not get_secret(cfg.server.auth_token_env) \
            and not cfg.server.allow_insecure_remote:
        raise SystemExit(f"Refusing to listen on {cfg.server.host} without an auth token. Set "
                         f"{cfg.server.auth_token_env} in .env (or server.allow_insecure_remote: true).")


async def _warm_llm(svc: Services) -> None:
    """Bounded, non-blocking preload. Readiness is reported, never faked."""
    t = time.perf_counter()
    svc.status["llm"] = {"state": "loading"}
    try:
        caps = await asyncio.wait_for(svc.llm.capabilities(), 15)
        res = await asyncio.wait_for(svc.llm.warmup(), 300)
    except Exception as e:  # noqa: BLE001
        svc.status["llm"] = {"state": "error", "error": e.__class__.__name__}
        return
    svc.status["llm"] = {"state": "loaded" if res.get("loaded") else "unknown" if res.get("loaded") is None
                         else "error", **res, "capabilities": caps, "warmup_ms": round((time.perf_counter() - t) * 1000)}
    log.info("LLM warm-up", extra={"model": svc.llm.model, **{k: v for k, v in svc.status["llm"].items()
                                                              if k != "capabilities"}})


def create_app(cfg: RikoConfig, services: Optional[Services] = None, *, warmup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = services or Services.from_config(cfg)
        app.state.services = svc
        tasks: list[asyncio.Task[Any]] = []
        if warmup:
            async def _warm_asr() -> None:
                svc.status["asr"] = {"state": "loading"}
                try:
                    await svc.asr.warmup()
                    svc.status["asr"] = {"state": "ready", "device": getattr(svc.asr, "device", None)}
                except Exception as e:  # noqa: BLE001
                    svc.status["asr"] = {"state": "error", "error": str(e)[:200]}
                    log.error("ASR warm-up failed - voice input will not work until fixed", extra={"error": str(e)})
            tasks.append(asyncio.create_task(_warm_asr()))

            async def _probe_tts() -> None:
                if svc.tts.produces_audio:
                    ok, detail = await svc.tts.health()
                    svc.status["tts"] = {"state": "ready" if ok else "unreachable", "detail": detail}
                    if not ok:
                        svc.status["tts_down_until"] = time.monotonic() + 30.0
                        log.warning("TTS server unreachable at startup; using fallback voice until it answers",
                                    extra={"fallback": cfg.tts.fallback})
            tasks.append(asyncio.create_task(_probe_tts()))
            tasks.append(asyncio.create_task(_warm_llm(svc)))
        if svc.memory:
            mem = svc.memory
            # Restart backfill: turns that ended before extraction ran are re-queued (bounded).
            for u, a in mem.store.unextracted_pairs(mem.character, cfg.memory.backfill_turns, profile=mem.profile):
                async def job(u: dict = u, a: dict = a) -> None:
                    await mem.extract_from_turn(u["content"], a["content"], source_ts=u["created_at"],
                                                turn_id=u["turn_id"])
                    mem.store.mark_extracted([u["id"], a["id"]])
                svc.scheduler.submit(f"extract:{mem.profile}:{mem.character}:{u['id']}", job)

            async def _maintain() -> None:
                while True:
                    async def job() -> None:
                        mem.decay_and_cleanup()
                        merged = await mem.consolidate()
                        if merged:
                            log.info("memories consolidated", extra={"merged": merged})
                    svc.scheduler.submit("maintenance", job)
                    await asyncio.sleep(cfg.memory.maintenance_interval_hours * 3600)
            tasks.append(asyncio.create_task(_maintain()))
        log.info("Riko Next ready", extra={"llm": f"{cfg.llm.provider}:{cfg.llm.model}", "asr": cfg.asr.provider,
                                           "tts": cfg.tts.provider, "memory": cfg.memory.enabled,
                                           "tools": cfg.tools.enabled})
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if services is None:
                await svc.aclose()
            else:
                await svc.scheduler.aclose()

    app = FastAPI(title="Riko Next", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)

    def _svc(request: Request) -> Services:
        return request.app.state.services

    def _origin_ok(origin: Optional[str], host_header: Optional[str]) -> bool:
        if not origin:  # non-browser clients (CLI, tests) send no Origin
            return True
        if origin in cfg.server.allowed_origins:
            return True
        o = urlparse(origin)
        return bool(host_header) and o.netloc == host_header

    def _guard(request: Request, token: Optional[str], mutating: bool = False) -> None:
        expected = get_secret(cfg.server.auth_token_env)  # read per request: rotating the token needs no restart
        if expected and not (token and hmac.compare_digest(token, expected)):
            raise HTTPException(status_code=401, detail="missing or invalid token")
        if mutating and not _origin_ok(request.headers.get("origin"), request.headers.get("host")):
            raise HTTPException(status_code=403, detail="cross-origin request refused")

    def _mem(request: Request) -> Any:
        mem = _svc(request).memory
        if not mem:
            raise HTTPException(404, "memory is disabled in config")
        return mem

    # ------------------------------------------------------------------ health & status
    @app.get("/api/health")
    async def health(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        svc = _svc(request)
        llm_ok, llm_msg = await svc.llm.health()
        tts_ok, tts_msg = await svc.tts.health()
        return {"version": __version__, "llm": {"ok": llm_ok, "detail": llm_msg},
                "tts": {"ok": tts_ok, "detail": tts_msg}, "memory": {"ok": svc.memory is not None}}

    @app.get("/api/status")
    async def status(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        """Developer diagnostics: readiness, capabilities, residency, background backlog. No secrets."""
        _guard(request, token)
        svc = _svc(request)
        out: dict[str, Any] = {"version": __version__, "readiness": svc.status,
                               "llm": {"provider": cfg.llm.provider, "model": svc.llm.model,
                                       "reasoning_effort": cfg.llm.reasoning_effort, "num_ctx": cfg.llm.num_ctx,
                                       "max_tokens": cfg.llm.max_tokens},
                               "roles": {r: (c.model if c else None) for r in ("fast", "smart", "vision")
                                         for c in [svc.router.config_for(r)]},
                               "background": svc.scheduler.backlog()}
        if isinstance(svc.llm, OllamaLLM):
            out["ollama"] = {"version": await svc.llm.version(),
                             "running": [{k: m.get(k) for k in ("name", "digest", "size_vram", "context_length",
                                                                 "expires_at")} for m in await svc.llm.running()]}
        if svc.memory:
            out["memory"] = {"enabled": svc.memory.enabled, "rejections": svc.memory.rejections,
                             **svc.memory.store.stats(svc.memory.character, profile=svc.memory.profile)}
        return out

    @app.get("/api/models")
    async def models(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        svc = _svc(request)
        try:
            names = await asyncio.wait_for(svc.llm.list_models(), 5)
        except Exception as e:  # noqa: BLE001
            return {"current": svc.llm.model, "models": [svc.llm.model], "error": str(e)}
        return {"current": svc.llm.model, "models": names if svc.llm.model in names else [svc.llm.model, *names]}

    @app.get("/api/tools")
    async def tools(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        svc = _svc(request)
        ctx = ToolContext(cfg, svc.memory, (lambda q: asyncio.sleep(0)) if svc.router.config_for("vision") else None,
                          http=svc.http)
        return {"enabled": cfg.tools.enabled, "permissions": cfg.tools.permissions, "tools": svc.tools.describe(ctx)}

    # ------------------------------------------------------------------ memory
    @app.get("/api/memories")
    async def list_memories(request: Request, layer: Optional[str] = None, limit: int = Query(100, ge=1, le=1000),
                            include_superseded: bool = False, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        mem = _svc(request).memory
        if not mem:
            return {"memories": [], "stats": {}, "enabled": False}
        try:
            lay = Layer(layer) if layer else None
        except ValueError:
            raise HTTPException(400, "layer must be one of: " + ", ".join(x.value for x in Layer))
        items = mem.store.list(mem.character, lay, limit, profile=mem.profile, include_inactive=include_superseded)
        return {"memories": [{**m.public(), "strength": round(mem.strength(m), 3)} for m in items],
                "stats": mem.store.stats(mem.character, profile=mem.profile), "enabled": mem.enabled}

    @app.post("/api/memories")
    async def add_memory(body: MemoryCreate, request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        mem = _mem(request)
        m, created = await mem.remember(body.content, Layer(body.layer), 0.8, body.category, body.pinned,
                                        source="user_edit", confidence=1.0)
        if m is None:
            raise HTTPException(409, "could not store memory")
        return {"memory": m.public(), "created": created}

    @app.patch("/api/memories/{memory_id}")
    async def edit_memory(memory_id: str, body: MemoryPatch, request: Request,
                          token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        mem = _mem(request)
        if not memory_id.isalnum():
            raise HTTPException(404, "memory not found")
        m = await mem.correct(memory_id, body.content, importance=body.importance, pinned=body.pinned,
                              layer=body.layer, status=body.status)
        if m is None:
            raise HTTPException(404, "memory not found")
        return {"memory": m.public()}

    @app.delete("/api/memories/{memory_id}")
    async def forget(memory_id: str, request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        mem = _svc(request).memory
        if not mem or not memory_id.isalnum() or not mem.forget(memory_id):
            raise HTTPException(404, "memory not found")
        return {"deleted": memory_id}

    @app.delete("/api/memories")
    async def clear_memories(request: Request, confirm: bool = False, history: bool = False,
                             token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        if not confirm:
            raise HTTPException(400, "add ?confirm=true to delete every memory")
        mem = _mem(request)
        n = mem.clear()
        if history:
            mem.store.clear_messages(mem.character, profile=mem.profile)
        return {"deleted": n, "history_cleared": history,
                "note": "Backups you made yourself (data/*.bak*) are not touched."}

    @app.get("/api/memories/export")
    async def export_memories(request: Request, token: Optional[str] = Query(None)) -> JSONResponse:
        _guard(request, token)
        mem = _mem(request)
        return JSONResponse(mem.store.export(mem.character, profile=mem.profile),
                            headers={"Content-Disposition": 'attachment; filename="riko-memory-export.json"'})

    @app.put("/api/memory/settings")
    async def memory_settings(body: MemorySettings, request: Request,
                              token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        mem = _mem(request)
        mem.set_enabled(body.enabled)
        return {"enabled": mem.enabled}

    @app.get("/api/history")
    async def history(request: Request, limit: int = Query(50, ge=1, le=500),
                      token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        mem = _svc(request).memory
        if not mem:
            return {"messages": []}
        msgs = mem.store.recent_messages(mem.character, limit, profile=mem.profile)
        return {"messages": [{"role": m["role"], "text": strip_tags(m["content"]), "ts": m["created_at"],
                              "interrupted": bool(m["interrupted"]), "source": m["source"]} for m in msgs]}

    # ------------------------------------------------------------------ relationship & mood
    @app.get("/api/relationship")
    async def relationship(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        from ..core.context import familiarity_stage
        from ..personality import STAGE_LABELS
        mem = _mem(request)
        rel = mem.store.get_relationship(mem.character, profile=mem.profile)
        stage = familiarity_stage(rel, cfg.memory.familiarity, time.time())
        return {"stage": stage, "label": STAGE_LABELS[stage], "meaningful_sessions": rel["sessions"],
                "completed_user_turns": rel["user_turns"], "first_seen": rel["first_seen"],
                "last_seen": rel["last_seen"],
                "explanation": "Familiarity grows only with time and completed conversations. It is not a score "
                               "of affection, and nothing you share or withhold changes it."}

    @app.get("/api/mood")
    async def mood(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        svc = _svc(request)
        mem = _mem(request)
        row = mem.store.get_mood(mem.character, profile=mem.profile)
        m = svc.mood_model.decayed(Mood(*row) if row else None, time.time())
        return {"enabled": cfg.emotion.mood_enabled, **MoodModel.to_dict(m)}

    @app.delete("/api/mood")
    async def reset_mood(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        mem = _mem(request)
        mem.store.reset_mood(mem.character, profile=mem.profile)
        return {"reset": True}

    # ------------------------------------------------------------------ tasks & notes
    @app.get("/api/tasks")
    async def tasks(request: Request, status: Optional[str] = Query("open", pattern="^(open|done|all)$"),
                    token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        mem = _mem(request)
        return {"tasks": mem.store.list_tasks(mem.character, None if status == "all" else status, 200,
                                              profile=mem.profile)}

    @app.patch("/api/tasks/{task_id}")
    async def edit_task(task_id: str, body: TaskPatch, request: Request,
                        token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        mem = _mem(request)
        if not mem.store.update_task(task_id, profile=mem.profile, **body.model_dump(exclude_none=True)):
            raise HTTPException(404, "task not found")
        return {"updated": task_id}

    @app.delete("/api/tasks/{task_id}")
    async def delete_task(task_id: str, request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        mem = _mem(request)
        if not mem.store.delete_task(task_id, profile=mem.profile):
            raise HTTPException(404, "task not found")
        return {"deleted": task_id}

    @app.get("/api/notes")
    async def notes(request: Request, q: str = Query("", max_length=200),
                    token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token)
        mem = _mem(request)
        return {"notes": mem.store.list_notes(mem.character, q, 200, profile=mem.profile)}

    @app.delete("/api/notes/{note_id}")
    async def delete_note(note_id: str, request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _guard(request, token, mutating=True)
        mem = _mem(request)
        if not mem.store.delete_note(note_id, profile=mem.profile):
            raise HTTPException(404, "note not found")
        return {"deleted": note_id}

    # ------------------------------------------------------------------ images
    @app.post("/api/images")
    async def upload_image(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        """Raw image bytes in the body (PNG/JPEG/GIF/WebP, <= 8 MB). Kept in memory for 30 minutes only."""
        _guard(request, token, mutating=True)
        if int(request.headers.get("content-length") or 0) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "image too large")
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > MAX_IMAGE_BYTES:
                raise HTTPException(413, "image too large")
        try:
            iid = _svc(request).images.put(bytes(data))
        except ValueError as e:
            raise HTTPException(415, str(e))
        vision = _svc(request).router.config_for("vision")
        return {"id": iid, "vision_model": vision.model if vision else None}

    # ------------------------------------------------------------------ websocket
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket, token: Optional[str] = Query(None)) -> None:
        if not _origin_ok(ws.headers.get("origin"), ws.headers.get("host")):
            log.warning("websocket rejected: origin", extra={"origin": ws.headers.get("origin")})
            await ws.close(code=1008)
            return
        expected = get_secret(cfg.server.auth_token_env)
        if expected and not (token and hmac.compare_digest(token, expected)):
            await ws.close(code=1008)
            return
        await ws.accept()
        svc: Services = ws.app.state.services
        session = ConversationSession(svc, WebSocketSink(ws))
        log.info("client connected", extra={"session": session.id})
        await session.start()
        bad = 0
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if (data := msg.get("bytes")) is not None:
                    if len(data) > MAX_AUDIO_FRAME_BYTES or len(data) % 2:
                        bad += 1
                        await session._emit(Ev.ERROR, component="protocol", message="invalid audio frame")
                    else:
                        await session.feed_audio(data)
                    continue
                text = msg.get("text") or ""
                if len(text) > cfg.server.max_ws_message_bytes:
                    bad += 1
                    await session._emit(Ev.ERROR, component="protocol", message="message too large")
                    continue
                try:
                    m = client_message.validate_json(text)
                except ValidationError as e:
                    bad += 1
                    first = e.errors()[0] if e.errors() else {}
                    await session._emit(Ev.ERROR, component="protocol",
                                        message=f"invalid message: {first.get('loc', '')} {first.get('msg', '')}")
                    if bad > 50:
                        await ws.close(code=1008)
                        break
                    continue
                await _dispatch(session, m)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.exception("websocket loop failed", extra={"session": session.id})
        finally:
            await session.close()
            log.info("client disconnected", extra={"session": session.id})

    async def _dispatch(session: ConversationSession, m: Any) -> None:
        t = m.type
        if t == "text":
            await session.submit_text(m.text, source="text", image_id=m.image_id)
        elif t == "ptt":
            await (session.ptt_start() if m.action == "start" else session.ptt_stop())
        elif t == "interrupt":
            await session.interrupt("user")
        elif t == "playback":
            await session.set_playback(m.state == "playing")
        elif t == "ready":
            await session.set_client_ready()
        elif t == "typing":
            session.proactive.user_typing()
            session.note_activity()
        elif t == "settings":
            await session.apply_settings(**m.model_dump(exclude={"type"}, exclude_none=True))
        elif t == "ping":
            await session._emit(Ev.PONG)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled HTTP error", extra={"path": request.url.path})
        return JSONResponse({"error": "internal error"}, status_code=500)

    @app.middleware("http")
    async def _no_stale_client(request: Request, call_next: Any) -> Any:
        # Browsers otherwise keep running an old app.js/audio.js after an update; files are tiny and local,
        # so always revalidate (ETag -> 304 when unchanged).
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
