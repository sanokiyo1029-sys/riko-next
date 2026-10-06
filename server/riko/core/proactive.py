"""Respectful initiative: greetings, reminders the user asked for, and rare idle nudges.

Rules (all re-checked right before a proactive turn starts):
* nothing happens until the client signals it is ready (after the user's audio gesture);
* ``notification_level``: off = never speak first; minimal = greeting + requested reminders;
  normal = + idle nudges;
* quiet hours suppress greetings and nudges (reminders the user explicitly set still fire);
* never while a reply is pending, audio is playing, or the user is speaking/typing;
* at most one greeting per logical session, and only if the user hasn't spoken first;
* nudges: after ``idle_nudge_s`` of quiet, ``nudge_cooldown_s`` apart, at most ``max_nudges_per_session``,
  and never two without a user turn in between;
* proactive output is a typed turn (greeting/nudge/reminder) - never a stored fictitious user message,
  and it never counts toward familiarity.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from .context import in_quiet_hours, local_now
from .logging import get_logger

if TYPE_CHECKING:
    from .session import ConversationSession

log = get_logger("proactive")

GREETING = ("The user just opened the app. Greet them in one short, natural sentence. Use 'last_talked' and "
            "'familiarity' from the context. If this is your first meeting, introduce yourself briefly. You may "
            "mention at most one current goal if it is clearly relevant. Do not invent shared history, and "
            "don't ask whether they missed you.")
NUDGE = ("The user has been quiet for a while. Only if there is something genuinely useful (an open task or "
         "goal from the context, or a follow-up to what you were just discussing), mention it in one short, "
         "low-pressure sentence. Otherwise make one light remark they can ignore. Never ask whether they are "
         "still there, never guilt them, and never claim to have seen what they are doing.")
REMINDER = ("A reminder the user asked for is due now: {titles}. Tell them briefly and naturally, in one or two "
            "sentences.")


class ProactiveController:
    def __init__(self, session: "ConversationSession") -> None:
        self.s = session
        self._task: Optional[asyncio.Task[Any]] = None
        self.greeting_eligible = True
        self.nudges = 0
        self.last_nudge = 0.0
        self.user_turn_since_nudge = True
        self.last_typing = 0.0

    # ------------------------------------------------------------------ signals from the session
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def user_spoke(self) -> None:
        self.greeting_eligible = False  # user speech/typing first cancels the greeting

    def user_typing(self) -> None:
        self.last_typing = time.monotonic()

    def user_turn_completed(self) -> None:
        self.user_turn_since_nudge = True

    # ------------------------------------------------------------------ policy
    @property
    def cfg(self) -> Any:
        return self.s.s.config.proactive

    def _blocked(self) -> bool:
        s = self.s
        return (s._closed or not s.client_ready or s.assistant_active or s._ptt_active or s.segmenter.in_speech
                or time.monotonic() - self.last_typing < 5.0)

    def _quiet(self, now: Optional[datetime] = None) -> bool:
        return in_quiet_hours(self.cfg.quiet_hours, now or local_now(self.cfg.timezone))

    def greeting_due(self) -> bool:
        return (self.greeting_eligible and self.cfg.greeting and self.s.proactive_level != "off"
                and self.s.new_logical_session and not self._quiet())

    def nudge_due(self, now_mono: float) -> bool:
        c = self.cfg
        if self.s.proactive_level != "normal" or c.idle_nudge_s <= 0 or self._quiet():
            return False
        return (self.nudges < c.max_nudges_per_session and self.user_turn_since_nudge
                and now_mono - self.s.last_activity >= c.idle_nudge_s
                and (self.last_nudge == 0.0 or now_mono - self.last_nudge >= c.nudge_cooldown_s)
                and self.s.user_turns > 0)

    async def tick(self) -> Optional[str]:
        """One policy evaluation. Returns the kind of turn started (for tests), or None."""
        s = self.s
        if s.proactive_level == "off" or self._blocked():
            return None
        mem = s.memory
        if mem is not None:  # reminders the user explicitly asked for
            due = mem.store.due_tasks(mem.character, time.time(), profile=mem.profile)
            claimed = [t for t in due if mem.store.claim_task_delivery(t["id"], profile=mem.profile)]
            if claimed:
                titles = "; ".join(t["title"] for t in claimed[:3])
                await s.proactive_turn("reminder", REMINDER.format(titles=titles),
                                       {"due_reminders": [t["title"] for t in claimed[:3]]})
                return "reminder"
        if self.greeting_due():
            self.greeting_eligible = False
            await s.proactive_turn("greeting", GREETING)
            return "greeting"
        now = time.monotonic()
        if self.nudge_due(now):
            self.nudges += 1
            self.last_nudge = now
            self.user_turn_since_nudge = False
            await s.proactive_turn("nudge", NUDGE)
            return "nudge"
        return None

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                try:
                    await self.tick()
                except Exception:  # noqa: BLE001 - initiative is optional; never kill the session
                    log.exception("proactive tick failed")
        except asyncio.CancelledError:
            pass
