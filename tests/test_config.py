import os

import pytest

from riko.core.config import ConfigError, RikoConfig, load_config, load_dotenv
from fakes import ROOT


def test_example_config_loads():
    cfg = load_config(ROOT / "config.example.yaml", env_file=False)
    assert cfg.llm.provider == "ollama" and cfg.llm.base_url == "http://127.0.0.1:11434"
    assert cfg.resolve_path(cfg.tts.voices["riko"].ref_audio).is_file()
    assert cfg.personality.preset == "default"


def test_provider_defaults():
    lm = RikoConfig.model_validate({"llm": {"provider": "lmstudio", "model": "x"}}).llm
    assert lm.base_url == "http://127.0.0.1:1234/v1" and lm.api_key_env is None
    oa = RikoConfig.model_validate({"llm": {"provider": "openai", "model": "gpt-4.1-mini"}}).llm
    assert oa.api_key_env == "OPENAI_API_KEY"
    with pytest.raises(Exception):
        RikoConfig.model_validate({"llm": {"provider": "openai_compatible"}})  # needs base_url


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_inline_secret_rejected(tmp_path):
    with pytest.raises(ConfigError, match="inline secret"):
        load_config(_write(tmp_path, "llm:\n  provider: openai\n  api_key: sk-abc123456\n"), env_file=False)


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "llm:\n  modle: oops\n"), env_file=False)


def test_env_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("RIKO_TEST_MODEL", "qwen2.5:7b")
    cfg = load_config(_write(tmp_path, "llm:\n  model: ${RIKO_TEST_MODEL}\nserver:\n  port: ${RIKO_PORT_X:-9999}\n"),
                      env_file=False)
    assert cfg.llm.model == "qwen2.5:7b" and cfg.server.port == 9999
    with pytest.raises(ConfigError, match="not set"):
        load_config(_write(tmp_path, "llm:\n  model: ${RIKO_DEFINITELY_UNSET}\n"), env_file=False)


def test_relative_paths_resolve_to_config_dir(tmp_path):
    cfg = load_config(_write(tmp_path, "memory:\n  db_path: data/x.db\n"), env_file=False)
    assert cfg.resolve_path(cfg.memory.db_path) == (tmp_path / "data" / "x.db").resolve()


def test_tts_voice_must_exist():
    with pytest.raises(Exception, match="not defined"):
        RikoConfig.model_validate({"tts": {"voice": "ghost"}})


def test_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("RIKO_KEEP", "original")
    (tmp_path / ".env").write_text('# c\nRIKO_NEW="hello"\nRIKO_KEEP=changed\nbad line\n')
    load_dotenv(tmp_path / ".env")
    assert os.environ["RIKO_NEW"] == "hello" and os.environ["RIKO_KEEP"] == "original"
    monkeypatch.delenv("RIKO_NEW")


def test_logging_redacts_secrets():
    from riko.core.logging import redact, register_secret
    register_secret("supersecretvalue99")
    out = redact("key sk-abcdefghijklmnop and Bearer abcdefghijkl and supersecretvalue99")
    assert "sk-abcdef" not in out and "abcdefghijkl" not in out and "supersecretvalue99" not in out
