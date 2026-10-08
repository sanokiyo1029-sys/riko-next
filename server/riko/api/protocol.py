"""Client -> server WebSocket messages, validated with pydantic.

Text frames are JSON objects with a ``type``; binary frames are int16 mono PCM at 16 kHz. Anything else is
rejected with an ``error`` event.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

_NAME = r"^[A-Za-z0-9_\-]{1,64}$"
_MODEL = r"^[A-Za-z0-9_.:/@\-]{1,128}$"
_ID = r"^[a-f0-9]{8,32}$"


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextMsg(_Msg):
    type: Literal["text"]
    text: str = Field("", max_length=20000)
    image_id: Optional[str] = Field(None, pattern=_ID)  # from POST /api/images

    @model_validator(mode="after")
    def _something(self) -> "TextMsg":
        if not self.text.strip() and not self.image_id:
            raise ValueError("text or image_id is required")
        return self


class PTTMsg(_Msg):
    type: Literal["ptt"]
    action: Literal["start", "stop"]


class InterruptMsg(_Msg):
    type: Literal["interrupt"]


class PlaybackMsg(_Msg):
    type: Literal["playback"]
    state: Literal["playing", "idle"]


class ReadyMsg(_Msg):
    """Sent once the user has started audio (a real gesture) - enables greetings and reminders."""
    type: Literal["ready"]


class TypingMsg(_Msg):
    type: Literal["typing"]


class SettingsMsg(_Msg):
    type: Literal["settings"]
    input_mode: Optional[Literal["auto", "ptt"]] = None
    muted: Optional[bool] = None
    mic: Optional[bool] = None
    character: Optional[str] = Field(None, pattern=_NAME)
    preset: Optional[str] = Field(None, pattern=_NAME)
    voice: Optional[str] = Field(None, pattern=_NAME)
    model: Optional[str] = Field(None, pattern=_MODEL)
    memory_enabled: Optional[bool] = None
    proactive_level: Optional[Literal["off", "minimal", "normal"]] = None


class AvatarStateMsg(_Msg):
    """What the browser avatar is doing right now; relayed to VSeeFace when frontend.vtuber is on."""
    type: Literal["avatar_state"]
    mouth: float = Field(0.0, ge=0.0, le=1.0)
    expression: Literal["neutral", "happy", "angry", "sad", "relaxed", "surprised"] = "neutral"
    intensity: float = Field(0.0, ge=0.0, le=1.0)
    blink: float = Field(0.0, ge=0.0, le=1.0)
    speaking: bool = False


class PingMsg(_Msg):
    type: Literal["ping"]


ClientMessage = Annotated[Union[TextMsg, PTTMsg, InterruptMsg, PlaybackMsg, ReadyMsg, TypingMsg, SettingsMsg,
                                AvatarStateMsg, PingMsg], Field(discriminator="type")]
client_message = TypeAdapter(ClientMessage)

MAX_AUDIO_FRAME_BYTES = 64000  # 2 s of 16 kHz int16; clients send ~20-64 ms frames


# ------------------------------------------------------------------ REST bodies
class MemoryCreate(_Msg):
    content: str = Field(min_length=3, max_length=500)
    layer: Literal["semantic", "episodic", "relationship", "task", "working"] = "semantic"
    category: str = Field("general", pattern=r"^[a-z_]{1,32}$")
    pinned: bool = False


class MemoryPatch(_Msg):
    content: Optional[str] = Field(None, min_length=3, max_length=500)
    importance: Optional[float] = Field(None, ge=0, le=1)
    pinned: Optional[bool] = None
    layer: Optional[Literal["semantic", "episodic", "relationship", "task", "working"]] = None
    status: Optional[Literal["active", "disputed"]] = None


class MemorySettings(_Msg):
    enabled: bool


class TaskPatch(_Msg):
    title: Optional[str] = Field(None, min_length=2, max_length=200)
    status: Optional[Literal["open", "done"]] = None
