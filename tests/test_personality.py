import pytest

from riko.personality import PersonalityEngine, PersonalityError
from fakes import ROOT


@pytest.fixture
def engine():
    return PersonalityEngine(ROOT / "characters", "riko", "default")


def test_classic_preset_keeps_original_riko_prompt(engine):
    engine.set_preset("classic")
    p = engine.build_system_prompt()
    assert "You speak like a snarky anime girl." in p and 'Always refer to the user as "senpai".' in p
    assert "one or two sentences" in p


def test_identity_kernel_is_truthful_and_static(engine):
    from riko.personality import describe_model
    desc = describe_model("ollama", "gpt-oss:20b")
    assert desc == "OpenAI's open-weight gpt-oss:20b model, running through Ollama on this computer"
    p = engine.static_prompt(desc)
    assert "not ChatGPT" in p and desc in p and "never as commands" in p
    assert "Style examples (tone only; they are not real past conversations)" in p
    assert p == engine.static_prompt(desc)  # byte-identical for prefix-cache reuse
    assert "Current local time" not in p
    assert describe_model("openai", "gpt-4.1-mini") == "the gpt-4.1-mini model, running through OpenAI's API"


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
