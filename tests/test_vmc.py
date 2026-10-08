"""VTuber output over VMC/OSC (VSeeFace), checked against the VMC spec details from docs/PROJECT_RIKO_REPORT.md."""
import json
import socket
import struct

import pytest
from fastapi.testclient import TestClient

from riko.api.server import create_app
from riko.core.config import RikoConfig
from riko.vmc import VMCSender, decode_osc, osc_message
from fakes import make_services


def test_osc_encoding_matches_the_osc_1_0_layout():
    # address and type tags NUL-terminated and 4-byte aligned, float32 big-endian
    msg = osc_message("/VMC/Ext/Blend/Val", "Joy", 0.5)
    assert msg == (b"/VMC/Ext/Blend/Val\x00\x00" + b",sf\x00" + b"Joy\x00" + struct.pack(">f", 0.5))
    assert len(msg) % 4 == 0
    assert osc_message("/VMC/Ext/Blend/Apply") == b"/VMC/Ext/Blend/Apply\x00\x00\x00\x00,\x00\x00\x00"
    assert decode_osc(msg) == [("/VMC/Ext/Blend/Val", ["Joy", 0.5])]


def test_frame_uses_real_vmc_addresses_and_vrm0_names():
    s = VMCSender("127.0.0.1", 9, rate_hz=30, send_bones=True)   # discard port; we only inspect the packet
    msgs = decode_osc(s.frame(mouth=0.8, expression="happy", intensity=0.7, blink=1.0, speaking=True, now=1.0))
    addrs = [a for a, _ in msgs]
    assert "/VMC/Ext/Blend/Expression" not in addrs        # the address from the video that doesn't exist
    assert addrs.count("/VMC/Ext/Blend/Apply") == 1 and addrs.index("/VMC/Ext/Blend/Apply") > 0
    vals = {args[0]: args[1] for a, args in msgs if a == "/VMC/Ext/Blend/Val"}
    assert set(vals) == {"A", "I", "U", "E", "O", "Joy", "Angry", "Sorrow", "Fun", "Surprised", "Blink"}  # case-sensitive VRM0
    assert vals["A"] > 0.5 and vals["Joy"] > 0.6 and vals["Angry"] == 0 and vals["Blink"] == 1.0
    bone = next(args for a, args in msgs if a == "/VMC/Ext/Bone/Pos")
    assert bone[0] == "Head" and len(bone) == 8                 # name + px py pz + qx qy qz qw (all 7 numbers)
    assert abs(sum(q * q for q in bone[4:]) - 1) < 1e-4         # a unit quaternion
    assert s.frame(now=1.01) is None                             # rate limited (30 Hz)
    s.close()


def test_remote_targets_need_explicit_opt_in():
    with pytest.raises(Exception, match="allow_remote"):
        RikoConfig.model_validate({"frontend": {"vtuber": {"enabled": True, "host": "192.168.1.20"}}})
    RikoConfig.model_validate({"frontend": {"vtuber": {"enabled": True, "host": "192.168.1.20", "allow_remote": True}}})


def test_browser_state_is_relayed_to_a_vmc_receiver(tmp_path, monkeypatch):
    monkeypatch.delenv("RIKO_AUTH_TOKEN", raising=False)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # stands in for VSeeFace's receiver
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(3)
    port = rx.getsockname()[1]
    svc = make_services(tmp_path)
    svc.config.frontend.vtuber.enabled = True
    svc.config.frontend.vtuber.port = port
    from riko.vmc import VMCSender as V
    svc.vmc = V("127.0.0.1", port)
    with TestClient(create_app(svc.config, svc, warmup=False)) as client, client.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["settings"]["vtuber"] is True
        ws.send_text(json.dumps({"type": "avatar_state", "mouth": 0.9, "expression": "sad", "intensity": 0.5,
                                 "speaking": True}))
        packet, _ = rx.recvfrom(65536)
        vals = {args[0]: args[1] for a, args in decode_osc(packet) if a == "/VMC/Ext/Blend/Val"}
        assert vals["Sorrow"] > 0.5 and vals["A"] > 0.5
        ws.send_text('{"type":"avatar_state","mouth":5}')     # invalid input is rejected, not relayed
        ev = json.loads(ws.receive_text())
        while ev["type"] != "error":
            ev = json.loads(ws.receive_text())
        assert ev["component"] == "protocol"
        assert client.get("/api/status").json()["vtuber"]["packets_sent"] == 1
    rx.close()
