"""VTuber output: drive the avatar in VSeeFace (or any VMC receiver) over the VMC protocol.

The original Riko video shows this layer but its code was never published (see
docs/PROJECT_RIKO_REPORT.md, section C). This follows the VMC spec the report checked:

* VMC runs over OSC/UDP. Expressions are ``/VMC/Ext/Blend/Val`` (name, value) per shape followed by one
  ``/VMC/Ext/Blend/Apply``. There is no ``/VMC/Ext/Blend/Expression`` address (the video's failed attempt).
* Shape names are VRM 0.x preset names and are case-sensitive: ``A I U E O Blink Joy Angry Sorrow Fun
  Surprised``. VSeeFace only loads VRM 0.x models, so export from VRoid as VRM0 for it.
* Bones are ``/VMC/Ext/Bone/Pos`` with all 7 numbers after the name: position x, y, z, then a quaternion
  x, y, z, w.
* VSeeFace's VMC receiver listens on UDP 39539 by default. We only send, and only to loopback unless
  configured otherwise.

OSC encoding is tiny, so it is done here with the standard library instead of adding python-osc.
"""
from __future__ import annotations

import math
import socket
import struct
import time
from typing import Optional, Union

from .core.logging import get_logger

log = get_logger("vmc")

OscArg = Union[str, float, int]

# Our (VRM 1.0 style) expression names -> VRM 0.x preset names used by VMC receivers.
EXPRESSION_TO_VRM0 = {"happy": "Joy", "angry": "Angry", "sad": "Sorrow", "relaxed": "Fun", "surprised": "Surprised"}
VISEMES = ("A", "I", "U", "E", "O")


def _pad(b: bytes) -> bytes:
    return b + b"\x00" * (4 - len(b) % 4)  # always at least one NUL terminator, 4-byte aligned


def osc_message(address: str, *args: OscArg) -> bytes:
    """Encode one OSC 1.0 message (strings, float32, int32)."""
    tags, payload = ",", b""
    for a in args:
        if isinstance(a, str):
            tags += "s"
            payload += _pad(a.encode("utf-8"))
        elif isinstance(a, bool) or isinstance(a, int):
            tags += "i"
            payload += struct.pack(">i", int(a))
        else:
            tags += "f"
            payload += struct.pack(">f", float(a))
    return _pad(address.encode("ascii")) + _pad(tags.encode("ascii")) + payload


def osc_bundle(*messages: bytes) -> bytes:
    """An OSC bundle (timetag "immediately") so a frame's shapes and the Apply arrive together."""
    out = _pad(b"#bundle") + struct.pack(">Q", 1)
    for m in messages:
        out += struct.pack(">i", len(m)) + m
    return out


def _quat_from_euler(pitch: float, yaw: float, roll: float) -> tuple[float, float, float, float]:
    """Radians -> quaternion (x, y, z, w) in Unity's left-handed convention (ZXY order)."""
    cx, sx = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cz, sz = math.cos(roll / 2), math.sin(roll / 2)
    return (cy * sx * cz + sy * cx * sz, sy * cx * cz - cy * sx * sz, cy * cx * sz - sy * sx * cz,
            cy * cx * cz + sy * sx * sz)


class VMCSender:
    """Turns the avatar state the web client reports (mouth level, expression, blink) into VMC packets."""

    def __init__(self, host: str = "127.0.0.1", port: int = 39539, *, rate_hz: float = 30.0,
                 send_bones: bool = False) -> None:
        self.addr = (host, port)
        self.min_interval = 1.0 / max(1.0, rate_hz)
        self.send_bones = send_bones
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._last = 0.0
        self._t0 = time.monotonic()
        self.sent = 0
        self.errors = 0

    def frame(self, *, mouth: float = 0.0, expression: str = "neutral", intensity: float = 0.0,
              blink: float = 0.0, speaking: bool = False, now: Optional[float] = None) -> Optional[bytes]:
        """Build (and send) one frame. Returns the packet, or None if rate-limited."""
        now = time.monotonic() if now is None else now
        if now - self._last < self.min_interval:
            return None
        self._last = now
        mouth = max(0.0, min(1.0, mouth))
        msgs = []
        # Lip sync without a virtual audio cable: drive the A/I/U/E/O shapes directly (report, step 4).
        # A carries the level; a slow wobble between O/U/E keeps it from looking like a single shape.
        t = now - self._t0
        wobble = (math.sin(t * 9.0) + 1) / 2
        vis = {"A": mouth * 0.85, "O": mouth * 0.35 * wobble, "U": mouth * 0.15 * (1 - wobble),
               "E": mouth * 0.2 * (1 - wobble), "I": 0.0}
        for name in VISEMES:
            msgs.append(osc_message("/VMC/Ext/Blend/Val", name, round(vis[name], 4)))
        target = EXPRESSION_TO_VRM0.get(expression)
        level = max(0.0, min(1.0, 0.35 + intensity * 0.6)) if target else 0.0
        for name in EXPRESSION_TO_VRM0.values():
            msgs.append(osc_message("/VMC/Ext/Blend/Val", name, round(level if name == target else 0.0, 4)))
        msgs.append(osc_message("/VMC/Ext/Blend/Val", "Blink", round(max(0.0, min(1.0, blink)), 4)))
        msgs.append(osc_message("/VMC/Ext/Blend/Apply"))
        if self.send_bones:
            # Scripted head motion (like the video): small idle sway, a nod while talking.
            pitch = (math.sin(t * 7) * mouth * 0.08) if speaking else math.sin(t * 0.9) * 0.02
            qx, qy, qz, qw = _quat_from_euler(pitch, math.sin(t * 0.45) * 0.06, math.sin(t * 0.33) * 0.03)
            msgs.append(osc_message("/VMC/Ext/Bone/Pos", "Head", 0.0, 0.0, 0.0, qx, qy, qz, qw))
        packet = osc_bundle(*msgs)
        try:
            self._sock.sendto(packet, self.addr)
            self.sent += 1
        except OSError as e:  # receiver not running is normal for UDP; just count it
            self.errors += 1
            if self.errors == 1:
                log.warning("VMC send failed", extra={"error": e.__class__.__name__, "target": f"{self.addr}"})
        return packet

    def close(self) -> None:
        self._sock.close()


def decode_osc(packet: bytes) -> list[tuple[str, list[OscArg]]]:
    """Minimal decoder for our own packets (tests and `riko vmc-listen`)."""
    def read_str(b: bytes, i: int) -> tuple[str, int]:
        end = b.index(b"\x00", i)
        s = b[i:end].decode("utf-8")
        return s, (end // 4 + 1) * 4

    def message(b: bytes) -> tuple[str, list[OscArg]]:
        addr, i = read_str(b, 0)
        tags, i = read_str(b, i)
        args: list[OscArg] = []
        for tag in tags[1:]:
            if tag == "s":
                v, i = read_str(b, i)
                args.append(v)
            elif tag == "f":
                args.append(struct.unpack(">f", b[i:i + 4])[0])
                i += 4
            elif tag == "i":
                args.append(struct.unpack(">i", b[i:i + 4])[0])
                i += 4
        return addr, args

    if not packet.startswith(b"#bundle\x00"):
        return [message(packet)]
    out, i = [], 16
    while i < len(packet):
        (n,) = struct.unpack(">i", packet[i:i + 4])
        out.append(message(packet[i + 4:i + 4 + n]))
        i += 4 + n
    return out
