"""Client -> server WebSocket messages, validated with pydantic.

Text frames are JSON objects with a ``type``; binary frames are int16 mono
PCM at 16 kHz. Anything else is rejected with an ``error`` event.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

_NAME = r"^[A-Za-z0-9_\-]{1,64}$"
_MODEL = r"^[A-Za-z0-9_.:/@\-]{1,128}$"


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextMsg(_Msg):
    type: Literal["text"]
    text: str = Field(min_length=1, max_length=20000)


class PTTMsg(_Msg):
    type: Literal["ptt"]
    action: Literal["start", "stop"]


class InterruptMsg(_Msg):
    type: Literal["interrupt"]


class PlaybackMsg(_Msg):
    type: Literal["playback"]
    state: Literal["playing", "idle"]


class SettingsMsg(_Msg):
    type: Literal["settings"]
    input_mode: Optional[Literal["auto", "ptt"]] = None
    muted: Optional[bool] = None
    mic: Optional[bool] = None
    character: Optional[str] = Field(None, pattern=_NAME)
    preset: Optional[str] = Field(None, pattern=_NAME)
    voice: Optional[str] = Field(None, pattern=_NAME)
    model: Optional[str] = Field(None, pattern=_MODEL)


class PingMsg(_Msg):
    type: Literal["ping"]


ClientMessage = Annotated[Union[TextMsg, PTTMsg, InterruptMsg, PlaybackMsg, SettingsMsg, PingMsg],
                          Field(discriminator="type")]
client_message = TypeAdapter(ClientMessage)

MAX_AUDIO_FRAME_BYTES = 64000  # 2 s of 16 kHz int16; clients send ~20-64 ms frames
