import pytest

from riko.personality import PersonalityEngine, PersonalityError
from fakes import ROOT


@pytest.fixture
def engine():
    return PersonalityEngine(ROOT / "characters", "riko", "default")


def test_default_keeps_original_riko_prompt(engine):
    p = engine.build_system_prompt()
    assert "You speak like a snarky anime girl." in p and 'Always refer to the user as "senpai".' in p
    assert "1-3 short sentences" in p


def test_presets_override_fields(engine):
    assert set(engine.presets()) >= {"default", "friendly", "sarcastic", "professional"}
    pro = engine.set_preset("professional")
    assert pro.user_name == "the user" and pro.response_length == "medium"
    assert "senpai" not in engine.build_system_prompt().split("Rules:")[1]


def test_memory_and_emotion_injection(engine):
    p = engine.build_system_prompt(memories="- User likes ramen", working="- Exam on Friday",
                                   summary="They talked about cats.", emotion_instructions="Start every reply with a tag")
    assert "User likes ramen" in p and "Exam on Friday" in p and "talked about cats" in p and "tag" in p


def test_invalid_names_rejected(engine):
    with pytest.raises(PersonalityError):
        engine.set_preset("nope")
    with pytest.raises(PersonalityError):
        engine.load("../config")


def test_bad_preset_detected_at_load(tmp_path):
    (tmp_path / "x.yaml").write_text("name: X\npresets:\n  broken:\n    humor_level: 5\n")
    with pytest.raises(PersonalityError, match="broken"):
        PersonalityEngine(tmp_path, "x")
