"""Tool framework: schema, permission, validation, execution, bounded results.

The model never executes anything itself. It *requests* a tool; the registry checks that the tool exists,
is enabled, and that its permission class is granted in config (``tools.permissions``); validates the
arguments against the tool's pydantic model; runs it with a timeout; and returns a size-bounded result
to the model as data. There is no shell, no arbitrary code execution, and no filesystem access outside
explicitly configured roots.

Permission classes:
    safe     pure computation / local read-only (time, calculator, image questions)
    memory   reads or writes the app's own database (memories, notes, tasks/reminders)
    network  contacts the internet (weather, web search) - off by default
    files    reads files under configured roots - off by default
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from ..core.config import RikoConfig, ToolsConfig
from ..core.logging import get_logger
from ..llm import ToolCall

log = get_logger("tools")


class ToolError(Exception):
    """An expected failure with a message that is safe to show the model and the user."""


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    cfg: RikoConfig
    memory: Any = None                                           # MemoryManager | None
    describe_image: Optional[Callable[[str], Awaitable[str]]] = None
    turn_id: Optional[int] = None
    http: Any = None                                             # shared httpx.AsyncClient for network tools


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: str            # what the model sees (JSON text, bounded)
    summary: str            # short, safe line for the UI
    ms: int = 0
    data: Any = field(default=None, repr=False)


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    permission: ClassVar[str] = "safe"
    Args: ClassVar[type[BaseModel]] = Args

    def available(self, ctx: ToolContext) -> bool:
        return True

    async def run(self, args: Any, ctx: ToolContext) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def summarize(self, args: Any, result: Any) -> str:
        return self.name.replace("_", " ")

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": _clean_schema(self.Args.model_json_schema())}}


def _clean_schema(s: Any) -> Any:
    """Flatten pydantic's JSON schema into the simple subset every tool-calling template understands."""
    if isinstance(s, dict):
        # "title" is pydantic metadata here; property *names* (which may be "title") live under "properties"
        s = {k: v for k, v in s.items() if k not in ("title", "additionalProperties")}
        if "anyOf" in s:  # Optional[X] -> X (absence means null)
            non_null = [x for x in s.pop("anyOf") if x.get("type") != "null"]
            if len(non_null) == 1:
                s = {**non_null[0], **{k: v for k, v in s.items() if k != "default"}}
        return {k: ({name: _clean_schema(p) for name, p in v.items()} if k == "properties" else _clean_schema(v))
                for k, v in s.items()}
    if isinstance(s, list):
        return [_clean_schema(x) for x in s]
    return s


class ToolRegistry:
    def __init__(self, cfg: ToolsConfig, tools: list[Tool]) -> None:
        self.cfg = cfg
        self._tools = {t.name: t for t in tools}

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def allowed(self, tool: Tool, ctx: ToolContext) -> bool:
        return (self.cfg.enabled and tool.name not in self.cfg.disabled
                and tool.permission in self.cfg.permissions and tool.available(ctx))

    def active(self, ctx: ToolContext) -> list[Tool]:
        return [t for t in self._tools.values() if self.allowed(t, ctx)]

    def schemas(self, ctx: ToolContext) -> list[dict[str, Any]]:
        return [t.schema() for t in self.active(ctx)]

    def describe(self, ctx: ToolContext) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "permission": t.permission,
                 "enabled": self.allowed(t, ctx),
                 "reason": None if self.allowed(t, ctx) else self._why_not(t, ctx)} for t in self._tools.values()]

    def _why_not(self, t: Tool, ctx: ToolContext) -> str:
        if not self.cfg.enabled:
            return "tools disabled"
        if t.name in self.cfg.disabled:
            return "disabled in config"
        if t.permission not in self.cfg.permissions:
            return f"permission '{t.permission}' not granted"
        return "not configured"

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()
        tool = self._tools.get(call.name)

        def fail(msg: str, summary: str) -> ToolResult:
            return ToolResult(call.id, call.name, False, json.dumps({"error": msg}), summary,
                              round((time.perf_counter() - t0) * 1000))

        if tool is None or not self.allowed(tool, ctx):
            return fail(f"tool '{call.name}' is not available", f"unavailable tool: {call.name[:40]}")
        try:
            args = tool.Args.model_validate(call.arguments)
        except ValidationError as e:
            first = e.errors()[0] if e.errors() else {}
            return fail(f"invalid arguments: {first.get('loc', '')} {first.get('msg', '')}",
                        f"{tool.name}: invalid arguments")
        try:
            out = await asyncio.wait_for(tool.run(args, ctx), self.cfg.timeout_s)
        except asyncio.TimeoutError:
            return fail("the tool timed out", f"{tool.name}: timed out")
        except ToolError as e:
            return fail(str(e), f"{tool.name}: {str(e)[:80]}")
        except Exception as e:  # noqa: BLE001 - a broken tool must not break the turn
            log.exception("tool failed", extra={"tool": tool.name})
            return fail(f"the tool failed ({e.__class__.__name__})", f"{tool.name}: failed")
        content = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
        if len(content) > self.cfg.max_result_chars:
            content = content[: self.cfg.max_result_chars] + " ...[truncated]"
        return ToolResult(call.id, tool.name, True, content, tool.summarize(args, out)[:160],
                          round((time.perf_counter() - t0) * 1000), out)


def default_registry(cfg: RikoConfig) -> ToolRegistry:
    from .builtin import BUILTIN_TOOLS

    return ToolRegistry(cfg.tools, [cls() for cls in BUILTIN_TOOLS])


__all__ = ["Args", "Tool", "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "default_registry"]
