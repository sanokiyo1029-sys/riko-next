"""One foreground inference lane with a bounded, preemptible background queue (I07, I08).

Why: a local model server usually serves one request at a time. Measured on Ollama 0.35.1 + gpt-oss:20b,
a background generation running when the user speaks delayed the reply's first token by ~8.9 s, while a
*cancelled* stream released the model within 50-1500 ms. So background work (memory extraction, summaries,
consolidation) runs only when no conversation turn is active and nobody is speaking, and is cancelled -
then retried later - the moment a foreground turn starts.

Bounds: at most ``max_pending`` jobs (oldest dropped and counted), coalescing by key, expiry, a retry
limit with backoff. Backlog age and counters are exposed for diagnostics.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from .logging import get_logger

log = get_logger("scheduler")


@dataclass
class Job:
    key: str
    factory: Callable[[], Awaitable[Any]]
    created: float = field(default_factory=time.monotonic)
    attempts: int = 0
    not_before: float = 0.0
    preempted: bool = False


class Scheduler:
    def __init__(self, max_pending: int = 100, expiry_s: float = 3600.0, max_attempts: int = 3,
                 idle_s: float = 0.8) -> None:
        self.max_pending = max_pending
        self.expiry_s = expiry_s
        self.max_attempts = max_attempts
        self.idle_s = idle_s
        self._fg = 0
        self._last_fg_end = 0.0
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._running: Optional[tuple[Job, asyncio.Task[Any]]] = None
        self._worker: Optional[asyncio.Task[Any]] = None
        self._wake = asyncio.Event()
        self.busy_checks: dict[str, Callable[[], bool]] = {}  # e.g. per session: user speaking / audio playing
        self.stats = {"completed": 0, "preempted": 0, "expired": 0, "dropped": 0, "failed": 0, "coalesced": 0}

    # ------------------------------------------------------------------ foreground
    @asynccontextmanager
    async def foreground(self) -> AsyncIterator[None]:
        self._fg += 1
        self.preempt()
        try:
            yield
        finally:
            self._fg -= 1
            self._last_fg_end = time.monotonic()
            self._wake.set()

    @property
    def foreground_active(self) -> bool:
        return self._fg > 0

    def preempt(self) -> None:
        if self._running and not self._running[1].done():
            job, task = self._running
            job.preempted = True
            task.cancel()

    # ------------------------------------------------------------------ background
    def submit(self, key: str, factory: Callable[[], Awaitable[Any]]) -> None:
        if key in self._jobs:  # coalesce: newest evidence wins, keep the original queue age
            self._jobs[key].factory = factory
            self.stats["coalesced"] += 1
        else:
            while len(self._jobs) >= self.max_pending:
                dropped, _ = self._jobs.popitem(last=False)
                self.stats["dropped"] += 1
                log.warning("background queue full; dropped oldest job", extra={"key": dropped})
            self._jobs[key] = Job(key, factory)
        self._wake.set()
        if self._worker is None or self._worker.done():  # self-starting: works in the app, CLI and tests
            self._worker = asyncio.get_running_loop().create_task(self.run())

    async def aclose(self) -> None:
        self.preempt()
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    def idle(self) -> bool:
        if self._fg or time.monotonic() - self._last_fg_end < self.idle_s:
            return False
        return not any(check() for check in list(self.busy_checks.values()))

    def backlog(self) -> dict[str, Any]:
        now = time.monotonic()
        oldest = min((j.created for j in self._jobs.values()), default=None)
        return {"pending": len(self._jobs), "running": self._running[0].key if self._running else None,
                "oldest_age_s": round(now - oldest, 1) if oldest else None, **self.stats}

    async def run(self) -> None:
        """Worker loop; run it as one task for the app's lifetime."""
        while True:
            if not self._jobs:
                self._wake.clear()
                await self._wake.wait()
                continue
            if not self.idle():
                await asyncio.sleep(0.2)
                continue
            now = time.monotonic()
            job = next((j for j in self._jobs.values() if j.not_before <= now), None)
            if job is None:
                await asyncio.sleep(0.2)
                continue
            del self._jobs[job.key]
            if now - job.created > self.expiry_s:
                self.stats["expired"] += 1
                continue
            job.preempted = False
            job.attempts += 1
            task = asyncio.create_task(job.factory())
            self._running = (job, task)
            try:
                await asyncio.wait({task})
            finally:
                self._running = None
            if task.cancelled():
                if not job.preempted:  # shutdown
                    return
                self.stats["preempted"] += 1
                self._requeue(job, backoff=0.0, count_attempt=False)
            elif task.exception() is not None:
                self.stats["failed"] += 1
                log.warning("background job failed", extra={"key": job.key,
                                                            "error": task.exception().__class__.__name__})
                self._requeue(job, backoff=5.0 * 2 ** job.attempts)
            else:
                self.stats["completed"] += 1

    def _requeue(self, job: Job, backoff: float, count_attempt: bool = True) -> None:
        if not count_attempt:
            job.attempts -= 1  # being preempted isn't the job's fault
        if job.attempts >= self.max_attempts or job.key in self._jobs:
            return
        job.not_before = time.monotonic() + backoff
        self._jobs[job.key] = job
        self._jobs.move_to_end(job.key, last=False)
