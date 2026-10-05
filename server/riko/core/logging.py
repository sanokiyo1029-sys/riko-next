"""Structured logging.

Every log call can carry structured fields via ``extra={...}``; they are
rendered as ``key=value`` (pretty) or merged into a JSON object (json).
A redaction filter scrubs anything that looks like an API key, bearer token,
or the literal value of any registered secret environment variable.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

_STD_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}

_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(gsk_[A-Za-z0-9]{8,})"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{6,}"),
]
_secret_values: set[str] = set()


def register_secret(value: Optional[str]) -> None:
    """Make sure an exact secret value never appears in logs."""
    if value and len(value) >= 6:
        _secret_values.add(value)


def redact(text: str) -> str:
    for v in _secret_values:
        if v in text:
            text = text.replace(v, "***")
    for pat in _PATTERNS:
        if pat.groups >= 1 and pat.pattern.startswith("(?i)("):
            text = pat.sub(lambda m: m.group(1) + "***", text)
        else:
            text = pat.sub("***", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        record.msg = redact(msg)
        record.args = ()
        for k, v in list(record.__dict__.items()):
            if k not in _STD_ATTRS and isinstance(v, str):
                record.__dict__[k] = redact(v)
        return True


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


class PrettyFormatter(logging.Formatter):
    COLORS = {"DEBUG": "\033[90m", "INFO": "\033[36m", "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[41m"}

    def __init__(self, color: bool) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(record.created))
        lvl = record.levelname
        name = record.name.removeprefix("riko.")
        extras = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        line = f"{t} {lvl:<7} {name:<12} {record.getMessage()}" + (f"  {extras}" if extras else "")
        if record.exc_info:
            line += "\n" + redact(self.formatException(record.exc_info))
        if self.color:
            line = f"{self.COLORS.get(lvl, '')}{line}\033[0m"
        return line


def setup_logging(level: str = "INFO", fmt: str = "pretty", file: Optional[Path] = None,
                  secrets: Iterable[Optional[str]] = ()) -> None:
    for s in secrets:
        register_secret(s)
    root = logging.getLogger("riko")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonFormatter() if fmt == "json" else PrettyFormatter(color=sys.stderr.isatty()))
    stream.addFilter(RedactingFilter())
    root.addHandler(stream)
    if file:
        file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        fh.addFilter(RedactingFilter())
        root.addHandler(fh)
    # Keep noisy libraries quiet unless debugging.
    for noisy in ("httpx", "httpcore", "faster_whisper"):
        logging.getLogger(noisy).setLevel("DEBUG" if level == "DEBUG" else "WARNING")


def get_logger(component: str) -> logging.Logger:
    lg = logging.getLogger(f"riko.{component}")
    if not any(isinstance(f, RedactingFilter) for f in lg.filters):
        lg.addFilter(RedactingFilter())
    return lg
