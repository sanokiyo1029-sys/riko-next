"""Configuration loading and validation.

Design rules (fixing problems in the original Project Riko config):
  * Secrets never live in YAML. Config names the *environment variable* that
    holds a key (``api_key_env: OPENAI_API_KEY``); the value comes from the
    environment or a local ``.env`` file.
  * Paths are relative to the project root, never absolute machine paths.
  * Unknown keys are rejected (``extra="forbid"``) so typos fail loudly at
    startup instead of being silently ignored.
  * ``${VAR}`` / ``${VAR:-default}`` interpolation works in any string value.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.:/@\-]{1,128}$")


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- LLM
LLMProviderName = Literal["openai", "lmstudio", "ollama", "openrouter", "groq", "openai_compatible"]

# provider -> (default base URL, default env var holding the API key)
LLM_PROVIDER_DEFAULTS: dict[str, tuple[Optional[str], Optional[str]]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "lmstudio": ("http://127.0.0.1:1234/v1", None),
    "ollama": ("http://127.0.0.1:11434", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai_compatible": (None, None),
}


class LLMConfig(_Strict):
    provider: LLMProviderName = "ollama"
    model: str = "llama3.1:8b"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = Field(0.85, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(400, ge=16, le=16384)
    timeout_s: float = Field(60.0, gt=0)
    ollama_keep_alive: str = "30m"
    ollama_num_ctx: Optional[int] = Field(None, ge=512)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_defaults(self) -> "LLMConfig":
        url, key_env = LLM_PROVIDER_DEFAULTS[self.provider]
        if self.base_url is None:
            if url is None:
                raise ValueError("llm.base_url is required for provider 'openai_compatible'")
            object.__setattr__(self, "base_url", url)
        if self.api_key_env is None and key_env:
            object.__setattr__(self, "api_key_env", key_env)
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        return self


# --------------------------------------------------------------------------- ASR
class ASRConfig(_Strict):
    provider: Literal["faster_whisper", "openai_compatible", "groq"] = "faster_whisper"
    model: str = "base.en"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "auto"
    language: Optional[str] = "en"
    beam_size: int = Field(1, ge=1, le=10)
    cpu_threads: int = Field(0, ge=0)
    initial_prompt: Optional[str] = None
    # Remote (Groq / OpenAI / speaches / any OpenAI-compatible transcription server)
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    remote_model: str = "whisper-large-v3-turbo"
    timeout_s: float = 30.0

    @model_validator(mode="after")
    def _remote_defaults(self) -> "ASRConfig":
        if self.provider == "groq":
            self.base_url = self.base_url or "https://api.groq.com/openai/v1"
            self.api_key_env = self.api_key_env or "GROQ_API_KEY"
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError("asr.base_url is required for provider 'openai_compatible'")
        return self


# --------------------------------------------------------------------------- Audio / VAD
class VADConfig(_Strict):
    engine: Literal["silero", "energy"] = "silero"
    threshold: float = Field(0.5, gt=0.0, lt=1.0)
    min_speech_ms: int = Field(200, ge=32)
    silence_ms: int = Field(650, ge=100)
    preroll_ms: int = Field(300, ge=0)
    max_utterance_s: float = Field(30.0, ge=1.0)
    energy_margin_db: float = 12.0


class BargeInConfig(_Strict):
    enabled: bool = True
    min_speech_ms: int = Field(350, ge=64)
    duck_volume: float = Field(0.25, ge=0.0, le=1.0)


class LocalAudioConfig(_Strict):
    input_device: Optional[str] = None
    output_device: Optional[str] = None


class AudioConfig(_Strict):
    input_mode: Literal["auto", "ptt"] = "auto"
    sample_rate: Literal[16000] = 16000
    vad: VADConfig = Field(default_factory=VADConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)
    local: LocalAudioConfig = Field(default_factory=LocalAudioConfig)


# --------------------------------------------------------------------------- TTS
class EmotionVoiceOverride(_Strict):
    ref_audio: Optional[str] = None
    prompt_text: Optional[str] = None
    speed_factor: Optional[float] = Field(None, gt=0.3, lt=3.0)


class VoiceConfig(_Strict):
    # GPT-SoVITS fields
    ref_audio: Optional[str] = None
    prompt_text: str = ""
    prompt_lang: str = "en"
    aux_ref_audio: list[str] = Field(default_factory=list)
    speed_factor: float = Field(1.0, gt=0.3, lt=3.0)
    emotions: dict[str, EmotionVoiceOverride] = Field(default_factory=dict)
    # OpenAI-compatible TTS fields (OpenAI, Kokoro-FastAPI, etc.)
    voice_id: Optional[str] = None


class GPTSoVITSConfig(_Strict):
    base_url: str = "http://127.0.0.1:9880"
    text_lang: str = "en"
    streaming_mode: int = Field(2, ge=0, le=3)
    text_split_method: str = "cut0"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    timeout_s: float = 60.0
    # When GPT-SoVITS runs on another machine, map our local paths to its paths.
    remote_path_prefix: Optional[str] = None


class OpenAITTSConfig(_Strict):
    base_url: str = "http://127.0.0.1:8880/v1"
    api_key_env: Optional[str] = None
    model: str = "kokoro"
    sample_rate: int = 24000
    timeout_s: float = 60.0


class TTSConfig(_Strict):
    provider: Literal["gpt_sovits", "openai_compatible", "browser", "none"] = "gpt_sovits"
    voice: str = "riko"
    voices: dict[str, VoiceConfig] = Field(default_factory=lambda: {"riko": VoiceConfig()})
    strip_actions: bool = True
    gpt_sovits: GPTSoVITSConfig = Field(default_factory=GPTSoVITSConfig)
    openai_compatible: OpenAITTSConfig = Field(default_factory=OpenAITTSConfig)

    @model_validator(mode="after")
    def _voice_exists(self) -> "TTSConfig":
        if self.voice not in self.voices:
            raise ValueError(f"tts.voice '{self.voice}' is not defined in tts.voices")
        return self


# --------------------------------------------------------------------------- Memory
class EmbeddingConfig(_Strict):
    provider: Literal["ollama", "openai_compatible", "hash"] = "hash"
    model: str = "nomic-embed-text"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    dim: int = Field(384, ge=32, le=8192)
    timeout_s: float = 15.0


class RetrievalWeights(_Strict):
    relevance: float = 0.6
    importance: float = 0.25
    recency: float = 0.15


class MemoryConfig(_Strict):
    enabled: bool = True
    db_path: str = "data/riko.db"
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    top_k: int = Field(6, ge=0, le=50)
    min_relevance: float = Field(0.2, ge=0.0, le=1.0)
    retrieval_timeout_s: float = Field(1.5, gt=0)
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    half_life_days: dict[str, float] = Field(
        default_factory=lambda: {"episodic": 21.0, "semantic": 180.0, "working": 2.0}
    )
    recency_half_life_hours: float = 72.0
    dedup_similarity: float = Field(0.9, ge=0.5, le=1.0)
    cleanup_threshold: float = Field(0.05, ge=0.0, le=1.0)
    extraction: Literal["llm", "heuristic", "off"] = "llm"
    short_term_messages: int = Field(16, ge=2, le=200)
    summarize_batch: int = Field(12, ge=2, le=200)
    consolidate_after_days: float = 7.0
    maintenance_interval_hours: float = 6.0


# --------------------------------------------------------------------------- Personality / emotion
class PersonalityConfig(_Strict):
    character: str = "riko"
    preset: str = "default"
    characters_dir: str = "characters"

    @field_validator("character", "preset")
    @classmethod
    def _safe(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", v):
            raise ValueError("character/preset names may only contain letters, digits, '_' and '-'")
        return v


class EmotionConfig(_Strict):
    enabled: bool = True
    mode: Literal["tags", "lexicon"] = "tags"
    min_change: float = Field(0.2, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- Server / client
class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(8765, ge=1, le=65535)
    auth_token_env: Optional[str] = "RIKO_AUTH_TOKEN"
    allowed_origins: list[str] = Field(default_factory=list)
    allow_insecure_remote: bool = False
    max_text_chars: int = Field(2000, ge=1, le=20000)
    max_ws_message_bytes: int = Field(65536, ge=1024)


class AvatarConfig(_Strict):
    enabled: bool = True
    vrm_url: str = ""  # e.g. "avatars/riko.vrm" (served from client/web)
    camera_distance: float = 1.6


class FrontendConfig(_Strict):
    show_token_stream: bool = True
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["pretty", "json"] = "pretty"
    file: Optional[str] = None


class RikoConfig(_Strict):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Not part of the YAML; set by the loader.
    root: Path = Field(default=PROJECT_ROOT, exclude=True)

    def resolve_path(self, value: str | os.PathLike[str]) -> Path:
        """Resolve a config path relative to the project root (absolute paths kept)."""
        p = Path(os.path.expanduser(str(value)))
        return p if p.is_absolute() else (self.root / p).resolve()


# --------------------------------------------------------------------------- helpers
def load_dotenv(path: Path, override: bool = False) -> int:
    """Minimal .env loader (KEY=VALUE, '#' comments, optional quotes). Returns count loaded."""
    if not path.is_file():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def _interpolate(obj: Any) -> Any:
    if isinstance(obj, str):
        def repl(m: re.Match[str]) -> str:
            val = os.environ.get(m.group(1))
            if val is None:
                if m.group(2) is not None:
                    return m.group(2)
                raise ConfigError(f"environment variable '{m.group(1)}' is referenced in config but not set")
            return val
        return _ENV_PATTERN.sub(repl, obj)
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v) for v in obj]
    return obj


def get_secret(env_name: Optional[str]) -> Optional[str]:
    """Return a secret from the environment, or None. Never read secrets from YAML."""
    if not env_name:
        return None
    val = os.environ.get(env_name, "").strip()
    return val or None


def is_safe_name(value: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(value))


def find_config_file(explicit: Optional[str | os.PathLike[str]] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        return p.resolve()
    env = os.environ.get("RIKO_CONFIG")
    for cand in [env, PROJECT_ROOT / "config.yaml", PROJECT_ROOT / "config.example.yaml"]:
        if cand and Path(cand).is_file():
            return Path(cand).resolve()
    return None


def load_config(path: Optional[str | os.PathLike[str]] = None, *, env_file: bool = True) -> RikoConfig:
    """Load, interpolate and validate configuration.

    Root for relative paths is the directory that contains the config file
    (normally the project root).
    """
    cfg_path = find_config_file(path)
    root = cfg_path.parent if cfg_path else PROJECT_ROOT
    if env_file:
        load_dotenv(root / ".env")
    data: dict[str, Any] = {}
    if cfg_path:
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {cfg_path}: {e}") from e
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{cfg_path} must contain a mapping at the top level")
        data = loaded
    data = _interpolate(data)
    _reject_inline_secrets(data)
    try:
        cfg = RikoConfig.model_validate(data)
    except Exception as e:  # pydantic.ValidationError
        raise ConfigError(f"invalid configuration ({cfg_path}):\n{e}") from e
    cfg.root = root
    return cfg


_SECRET_KEYS = re.compile(r"(api[_-]?key|secret|password|token)$", re.I)


def _reject_inline_secrets(data: Any, path: str = "") -> None:
    """The original project stored OPENAI_API_KEY in YAML; we refuse that."""
    if isinstance(data, dict):
        for k, v in data.items():
            kp = f"{path}.{k}" if path else str(k)
            if isinstance(k, str) and _SECRET_KEYS.search(k) and not k.endswith("_env") and isinstance(v, str) and v:
                raise ConfigError(
                    f"'{kp}' looks like an inline secret. Put the value in .env and reference it with "
                    f"'{k}_env: YOUR_ENV_VAR_NAME' instead."
                )
            _reject_inline_secrets(v, kp)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            _reject_inline_secrets(v, f"{path}[{i}]")
