"""Configuration-driven personality.

A character file (``characters/<name>.yaml``) defines the base persona plus
``presets`` that override any field. Switching preset or character never
requires touching Python.
"""
from __future__ import annotations

import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

LENGTH_GUIDE = {
    "short": "Keep replies to 1-3 short sentences unless asked for more. You are speaking out loud.",
    "medium": "Keep replies to about 2-5 sentences. You are speaking out loud.",
    "long": "You may give longer answers when useful, but stay conversational. You are speaking out loud.",
}


class PersonalityError(Exception):
    pass


class Character(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    user_name: str = "the user"
    system_prompt: str = ""
    personality: list[str] = Field(default_factory=list)
    speaking_style: str = ""
    emotional_tendencies: str = ""
    relationship_style: str = ""
    humor_level: float = Field(0.5, ge=0.0, le=1.0)
    response_length: Literal["short", "medium", "long"] = "short"
    allowed_behaviors: list[str] = Field(default_factory=list)
    forbidden_behaviors: list[str] = Field(default_factory=list)
    memory_rules: list[str] = Field(default_factory=list)
    voice: Optional[str] = None
    greeting: Optional[str] = None
    extra_instructions: str = ""


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _fill_user(c: "Character") -> "Character":
    """Replace the ``{user}`` placeholder so presets can rename the user consistently."""
    who = c.user_name

    def f(v: Any) -> Any:
        if isinstance(v, str):
            return v.replace("{user}", who)
        if isinstance(v, list):
            return [f(x) for x in v]
        return v
    return c.model_copy(update={k: f(getattr(c, k)) for k in type(c).model_fields if k != "user_name"})


class PersonalityEngine:
    def __init__(self, characters_dir: Path, character: str = "riko", preset: str = "default") -> None:
        self.dir = Path(characters_dir)
        self._raw: dict[str, Any] = {}
        self.character_id = ""
        self.preset = preset
        self.load(character, preset)

    def available_characters(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.yaml") if _NAME.fullmatch(p.stem))

    def load(self, character: str, preset: Optional[str] = None) -> None:
        if not _NAME.fullmatch(character):
            raise PersonalityError(f"invalid character name: {character!r}")
        path = (self.dir / f"{character}.yaml").resolve()
        if path.parent != self.dir.resolve() or not path.is_file():
            raise PersonalityError(f"character file not found: {character}.yaml in {self.dir}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise PersonalityError(f"invalid YAML in {path.name}: {e}") from e
        if not isinstance(raw, dict) or "name" not in raw:
            raise PersonalityError(f"{path.name} must be a mapping with at least a 'name'")
        presets = raw.get("presets") or {}
        if not isinstance(presets, dict):
            raise PersonalityError("'presets' must be a mapping")
        presets.setdefault("default", {})
        raw["presets"] = presets
        base = {k: v for k, v in raw.items() if k != "presets"}
        for name, over in presets.items():  # validate all presets at load time
            if not _NAME.fullmatch(str(name)):
                raise PersonalityError(f"invalid preset name {name!r}")
            try:
                Character.model_validate(_deep_merge(base, over or {}))
            except Exception as e:
                raise PersonalityError(f"preset '{name}' in {path.name} is invalid: {e}") from e
        self._raw = raw
        self.character_id = character
        self.set_preset(preset if preset in presets else "default")

    def presets(self) -> list[str]:
        return list(self._raw.get("presets", {}).keys())

    def set_preset(self, preset: str) -> "Character":
        if preset not in self._raw["presets"]:
            raise PersonalityError(f"unknown preset '{preset}'. Available: {', '.join(self.presets())}")
        self.preset = preset
        return self.current

    @property
    def current(self) -> Character:
        base = {k: v for k, v in self._raw.items() if k != "presets"}
        char = Character.model_validate(_deep_merge(base, self._raw["presets"].get(self.preset) or {}))
        return _fill_user(char)

    def build_system_prompt(self, *, memories: str = "", working: str = "", summary: str = "",
                            emotion_instructions: str = "", now: Optional[datetime] = None) -> str:
        c = self.current
        now = now or datetime.now()
        parts: list[str] = [c.system_prompt.strip() or f"You are {c.name}. {c.description}".strip()]
        traits = []
        if c.personality:
            traits.append("Personality: " + ", ".join(c.personality) + ".")
        if c.speaking_style:
            traits.append("Speaking style: " + c.speaking_style.strip())
        if c.emotional_tendencies:
            traits.append("Emotional tendencies: " + c.emotional_tendencies.strip())
        if c.relationship_style:
            traits.append(f"Relationship with {c.user_name}: " + c.relationship_style.strip())
        humor = ("rarely joke" if c.humor_level < 0.25 else "joke sometimes" if c.humor_level < 0.6
                 else "be playful and witty often")
        traits.append(f"Humor: {humor}.")
        parts.append("\n".join(traits))
        rules = [LENGTH_GUIDE[c.response_length],
                 "Write plain spoken dialogue: no markdown, lists, emojis, or stage directions in asterisks."]
        if c.user_name and c.user_name != "the user":
            rules.append(f'Address the user as "{c.user_name}".')
        rules += [f"You may: {b}" for b in c.allowed_behaviors]
        rules += [f"Never: {b}" for b in c.forbidden_behaviors]
        parts.append("Rules:\n" + "\n".join(f"- {r}" for r in rules))
        if emotion_instructions:
            parts.append(emotion_instructions)
        if c.memory_rules:
            parts.append("How to use memories:\n" + "\n".join(f"- {r}" for r in c.memory_rules))
        if summary:
            parts.append("Summary of the earlier conversation:\n" + summary.strip())
        if working:
            parts.append("Current context and goals:\n" + working.strip())
        if memories:
            parts.append("Things you remember (use naturally, don't recite them):\n" + memories.strip())
        if c.extra_instructions:
            parts.append(c.extra_instructions.strip())
        parts.append(f"Current local time: {now.strftime('%A %d %B %Y, %H:%M')}.")
        return "\n\n".join(parts)
