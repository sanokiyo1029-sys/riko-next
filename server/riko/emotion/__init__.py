"""Emotion / style analysis for each spoken sentence.

Primary method ("tags"): the LLM is asked to prefix sentences with a tag such
as ``[happy]`` or ``[smug:0.7]``. This costs zero extra latency because the
tag arrives in the same token stream. If the model ignores the instruction,
a small lexicon analyzer is the fallback, so the feature never blocks speech.
The resulting state is sent to the avatar (expression) and to TTS (speed /
per-emotion reference clip).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class EmotionStyle:
    expression: str  # VRM expression preset: neutral/happy/angry/sad/relaxed/surprised
    style: str
    speed: float
    valence: float   # -1..1, feeds the persistent mood
    arousal: float   # 0..1


EMOTIONS: dict[str, EmotionStyle] = {
    "neutral": EmotionStyle("neutral", "calm", 1.0, 0.0, 0.35),
    "calm": EmotionStyle("relaxed", "calm", 0.98, 0.2, 0.2),
    "happy": EmotionStyle("happy", "cheerful", 1.04, 0.7, 0.55),
    "excited": EmotionStyle("happy", "energetic", 1.1, 0.8, 0.9),
    "playful": EmotionStyle("happy", "teasing", 1.05, 0.6, 0.65),
    "smug": EmotionStyle("relaxed", "teasing", 1.0, 0.4, 0.5),
    "curious": EmotionStyle("surprised", "inquisitive", 1.02, 0.35, 0.55),
    "serious": EmotionStyle("neutral", "steady", 0.97, 0.0, 0.4),
    "thinking": EmotionStyle("neutral", "thoughtful", 0.96, 0.05, 0.35),
    "empathetic": EmotionStyle("relaxed", "warm", 0.95, 0.2, 0.3),
    "concerned": EmotionStyle("sad", "gentle", 0.95, -0.2, 0.4),
    "confused": EmotionStyle("surprised", "hesitant", 0.97, -0.1, 0.45),
    "embarrassed": EmotionStyle("happy", "flustered", 1.06, 0.2, 0.7),
    "surprised": EmotionStyle("surprised", "lively", 1.08, 0.2, 0.8),
    "sad": EmotionStyle("sad", "soft", 0.92, -0.6, 0.25),
    "angry": EmotionStyle("angry", "sharp", 1.05, -0.6, 0.8),
}

_TAG = re.compile(r"\[\s*([A-Za-z]+)\s*(?::\s*(\d*\.?\d+))?\s*\]\s*")

_LEXICON: dict[str, tuple[str, ...]] = {
    "happy": ("haha", "hehe", "yay", "glad", "love", "nice", "great", "awesome", "fun", "cute", "happy"),
    "excited": ("wow", "amazing", "can't wait", "incredible", "let's go"),
    "smug": ("hmph", "obviously", "of course", "told you", "as expected", "naturally", "duh"),
    "embarrassed": ("b-baka", "baka", "it's not like", "don't get the wrong idea", "um,", "flustered"),
    "sad": ("sorry", "sad", "miss", "unfortunately", "lonely", "hurts"),
    "angry": ("ugh", "annoying", "seriously", "how dare", "idiot", "stop it"),
    "surprised": ("what?", "huh?", "really?", "no way", "whoa", "eh?"),
    "thinking": ("hmm", "let me think", "maybe", "i wonder", "perhaps"),
    "concerned": ("are you okay", "be careful", "worried", "take care", "rest"),
    "curious": ("tell me more", "how come", "what made you", "i'm curious", "interesting"),
    "empathetic": ("that sounds hard", "that sounds rough", "i'm here", "makes sense that you", "understandable"),
    "confused": ("i'm not sure", "wait, what", "i don't follow", "confusing"),
    "playful": ("just kidding", "gotcha", "nice try"),
}


@dataclass
class EmotionState:
    emotion: str
    intensity: float
    style: str
    expression: str
    speed_factor: float

    def to_dict(self) -> dict:
        return asdict(self)


def make_state(emotion: str, intensity: float) -> EmotionState:
    st = EMOTIONS.get(emotion, EMOTIONS["neutral"])
    intensity = max(0.0, min(1.0, intensity))
    speed = 1.0 + (st.speed - 1.0) * (0.5 + intensity)  # stronger emotion -> stronger speed shift
    return EmotionState(emotion, round(intensity, 2), st.style, st.expression, round(speed, 3))


# A lowercase one-word bracket at the start of a sentence is the model attempting an emotion tag it
# invented ("[hello]", "[calculating]"); it is never meant to be read. Other brackets are content.
_LEADING_TAG = re.compile(r"(^|(?<=[.!?])\s+)\[\s*[a-z]+\s*(?::\s*\d*\.?\d+)?\s*\]\s*")


def strip_tags(text: str) -> str:
    """Remove recognised emotion tags anywhere, and invented tag-like words at sentence starts."""
    text = _TAG.sub(lambda m: "" if m.group(1).lower() in EMOTIONS else m.group(0), text)
    return _LEADING_TAG.sub(lambda m: m.group(1), text).strip()


class EmotionAnalyzer:
    def __init__(self, enabled: bool = True, mode: str = "tags", min_change: float = 0.2) -> None:
        self.enabled = enabled
        self.mode = mode
        self.min_change = min_change
        self.current = make_state("neutral", 0.3)

    def instructions(self) -> str:
        if not (self.enabled and self.mode == "tags"):
            return ""
        names = ", ".join(EMOTIONS)
        return ("Start every reply with an emotion tag in square brackets showing how you feel, optionally with "
                f"an intensity from 0 to 1, e.g. [smug:0.7] or [happy]. Allowed emotions: {names}. "
                "If your mood changes mid-reply, put a new tag at the start of that sentence. "
                "Tags are silent; never mention them.")

    def analyze(self, sentence: str) -> tuple[str, Optional[EmotionState]]:
        """Return (sentence without tags, detected state or None)."""
        found: Optional[EmotionState] = None
        for m in _TAG.finditer(sentence):
            name = m.group(1).lower()
            if name in EMOTIONS:
                val = float(m.group(2)) if m.group(2) else 0.6
                found = make_state(name, val)
                break
        clean = strip_tags(sentence)
        if not self.enabled:
            return clean, None
        if found is None:
            found = self._lexicon(clean)
        return clean, found

    def _lexicon(self, text: str) -> Optional[EmotionState]:
        low = f" {text.lower()} "
        scores = {emo: sum(low.count(w) for w in words) for emo, words in _LEXICON.items()}
        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return None
        intensity = min(1.0, 0.35 + 0.15 * scores[best] + 0.1 * text.count("!"))
        return make_state(best, intensity)

    def update(self, state: Optional[EmotionState]) -> Optional[EmotionState]:
        """Adopt ``state`` and return it if it is a meaningful change, else None."""
        if state is None:
            return None
        if state.emotion == self.current.emotion and abs(state.intensity - self.current.intensity) < self.min_change:
            return None
        self.current = state
        return state


from .mood import Mood, MoodModel, UserAffect, classify_user_affect  # noqa: E402
