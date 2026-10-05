"""FastAPI app: WebSocket endpoint, small REST API, and the static web client."""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
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
from ..core.events import Ev, encode_audio_frame
from ..core.logging import get_logger
from ..core.session import ConversationSession, Services
from ..memory import Layer
from .protocol import MAX_AUDIO_FRAME_BYTES, client_message

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


def create_app(cfg: RikoConfig, services: Optional[Services] = None, *, warmup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = services or Services.from_config(cfg)
        app.state.services = svc
        tasks: list[asyncio.Task[Any]] = []
        if warmup:
            async def _warm() -> None:
                try:
                    await svc.asr.warmup()
                except Exception as e:  # noqa: BLE001
                    log.error("ASR warm-up failed - voice input will not work until fixed",
                              extra={"error": str(e)})
            tasks.append(asyncio.create_task(_warm()))
        if svc.memory:
            async def _maintain() -> None:
                while True:
                    try:
                        svc.memory.decay_and_cleanup()
                        merged = await svc.memory.consolidate()
                        if merged:
                            log.info("memories consolidated", extra={"merged": merged})
                    except Exception:  # noqa: BLE001
                        log.exception("memory maintenance failed")
                    await asyncio.sleep(cfg.memory.maintenance_interval_hours * 3600)
            tasks.append(asyncio.create_task(_maintain()))
        log.info("Riko Next ready", extra={"llm": f"{cfg.llm.provider}:{cfg.llm.model}", "asr": cfg.asr.provider,
                                           "tts": cfg.tts.provider, "memory": cfg.memory.enabled})
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if services is None:
                await svc.aclose()

    app = FastAPI(title="Riko Next", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)

    def _svc(request: Request) -> Services:
        return request.app.state.services

    def _require_token(t: Optional[str]) -> None:
        expected = get_secret(cfg.server.auth_token_env)  # read per request: rotating the token needs no restart
        if expected and not (t and hmac.compare_digest(t, expected)):
            raise HTTPException(status_code=401, detail="missing or invalid token")

    def _origin_ok(origin: Optional[str], host_header: Optional[str]) -> bool:
        if not origin:  # non-browser clients (CLI, tests) send no Origin
            return True
        if origin in cfg.server.allowed_origins:
            return True
        o = urlparse(origin)
        return bool(host_header) and o.netloc == host_header

    @app.get("/api/health")
    async def health(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _require_token(token)
        svc = _svc(request)
        llm_ok, llm_msg = await svc.llm.health()
        tts_ok, tts_msg = await svc.tts.health()
        return {"version": __version__, "llm": {"ok": llm_ok, "detail": llm_msg},
                "tts": {"ok": tts_ok, "detail": tts_msg}, "memory": {"ok": svc.memory is not None}}

    @app.get("/api/models")
    async def models(request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _require_token(token)
        svc = _svc(request)
        try:
            names = await asyncio.wait_for(svc.llm.list_models(), 5)
        except Exception as e:  # noqa: BLE001
            return {"current": svc.llm.model, "models": [svc.llm.model], "error": str(e)}
        return {"current": svc.llm.model, "models": names if svc.llm.model in names else [svc.llm.model, *names]}

    @app.get("/api/memories")
    async def list_memories(request: Request, layer: Optional[str] = None, limit: int = Query(100, ge=1, le=1000),
                            token: Optional[str] = Query(None)) -> dict[str, Any]:
        _require_token(token)
        mem = _svc(request).memory
        if not mem:
            return {"memories": [], "stats": {}}
        try:
            lay = Layer(layer) if layer else None
        except ValueError:
            raise HTTPException(400, "layer must be episodic, semantic or working")
        items = mem.store.list(mem.character, lay, limit)
        return {"memories": [{**m.public(), "strength": round(mem.strength(m), 3)} for m in items],
                "stats": mem.store.stats(mem.character)}

    @app.delete("/api/memories/{memory_id}")
    async def forget(memory_id: str, request: Request, token: Optional[str] = Query(None)) -> dict[str, Any]:
        _require_token(token)
        mem = _svc(request).memory
        if not mem or not memory_id.isalnum() or not mem.forget(memory_id):
            raise HTTPException(404, "memory not found")
        return {"deleted": memory_id}

    @app.get("/api/history")
    async def history(request: Request, limit: int = Query(50, ge=1, le=500),
                      token: Optional[str] = Query(None)) -> dict[str, Any]:
        _require_token(token)
        mem = _svc(request).memory
        if not mem:
            return {"messages": []}
        from ..emotion import strip_tags
        msgs = mem.store.recent_messages(mem.character, limit)
        return {"messages": [{"role": m["role"], "text": strip_tags(m["content"]), "ts": m["created_at"],
                              "interrupted": bool(m["interrupted"])} for m in msgs]}

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
            await session.submit_text(m.text, source="text")
        elif t == "ptt":
            await (session.ptt_start() if m.action == "start" else session.ptt_stop())
        elif t == "interrupt":
            await session.interrupt("user")
        elif t == "playback":
            await session.set_playback(m.state == "playing")
        elif t == "settings":
            await session.apply_settings(**m.model_dump(exclude={"type"}, exclude_none=True))
        elif t == "ping":
            await session._emit(Ev.PONG)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled HTTP error", extra={"path": request.url.path})
        return JSONResponse({"error": "internal error"}, status_code=500)

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
