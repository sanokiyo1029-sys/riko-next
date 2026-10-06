"""Persistent mood (valence/arousal) and a deliberately uncertain user-affect signal."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import EmotionState

_USER_AFFECT: dict[str, tuple[str, ...]] = {
    "sad": ("sad", "depressed", "lonely", "miserable", "crying", "heartbroken", "feel awful", "feeling down",
            "lost my", "passed away", "grief", "hopeless"),
    "frustrated": ("annoyed", "frustrated", "angry", "pissed", "sick of", "fed up", "this is stupid",
                   "doesn't work", "not working", "ugh", "hate this", "wtf"),
    "anxious": ("anxious", "nervous", "worried", "scared", "stressed", "panic", "overwhelmed", "afraid"),
    "positive": ("happy", "great", "awesome", "excited", "glad", "love it", "amazing", "yay", "thank you",
                 "thanks", "good news", "got the job"),
}
_NEGATOR = re.compile(r"\b(not|never|no|hardly)\s+(?:\w+\s+){0,2}$|n't\s+(?:\w+\s+){0,2}$", re.I)


@dataclass
class UserAffect:
    label: str          # sad | frustrated | anxious | positive | unknown
    confidence: float   # 0..1 - weak lexical evidence, never a diagnosis


def classify_user_affect(text: str) -> UserAffect:
    """Cheap lexicon guess with negation handling. Says 'unknown' (e.g. other languages) instead of guessing."""
    low = f" {text.lower()} "
    scores: dict[str, float] = {}
    for label, words in _USER_AFFECT.items():
        for w in words:
            for m in re.finditer(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low):
                if _NEGATOR.search(low[max(0, m.start() - 30):m.start()]):
                    continue  # "not happy" is not positive
                scores[label] = scores.get(label, 0.0) + 1.0
    if not scores:
        return UserAffect("unknown", 0.0)
    best = max(scores, key=lambda k: scores[k])
    conf = min(0.8, 0.3 + 0.15 * scores[best] + 0.05 * min(3, text.count("!")))
    return UserAffect(best, round(conf, 2))


@dataclass
class Mood:
    valence: float      # -1..1
    arousal: float      # 0..1
    updated_at: float   # unix seconds (UTC)


class MoodModel:
    """Mood decays toward a baseline with a half-life and moves a bounded step per turn:

        value(t) = baseline + (value(t0) - baseline) * 2 ** (-(t - t0) / half_life)

    Safeguards: capped change per turn; Riko's own negative tags count half and stop pushing once
    mood is already low (no self-reinforcing spiral); user frustration lowers arousal, it never makes
    Riko hostile; the user never has to "be nice" for mood to recover - time alone restores baseline.
    """

    def __init__(self, baseline_valence: float = 0.25, baseline_arousal: float = 0.45,
                 half_life_minutes: float = 20.0, max_step: float = 0.15) -> None:
        self.bv = max(-1.0, min(1.0, baseline_valence))
        self.ba = max(0.0, min(1.0, baseline_arousal))
        self.half_life_s = max(1.0, half_life_minutes * 60)
        self.max_step = max_step

    def decayed(self, mood: Optional[Mood], now: float) -> Mood:
        if mood is None:
            return Mood(self.bv, self.ba, now)
        elapsed = max(0.0, now - mood.updated_at)  # a clock jumping backwards never amplifies mood
        f = 2 ** (-elapsed / self.half_life_s)
        return Mood(self.bv + (mood.valence - self.bv) * f, self.ba + (mood.arousal - self.ba) * f,
                    max(now, mood.updated_at))

    def update(self, mood: Optional[Mood], now: float, expressed: list["EmotionState"],
               user: Optional[UserAffect] = None) -> Mood:
        from . import EMOTIONS

        cur = self.decayed(mood, now)
        dv = da = 0.0
        if expressed:
            weights = [max(0.1, e.intensity) for e in expressed]
            styles = [EMOTIONS.get(e.emotion, EMOTIONS["neutral"]) for e in expressed]
            tv = sum(s.valence * w for s, w in zip(styles, weights)) / sum(weights)
            ta = sum(s.arousal * w for s, w in zip(styles, weights)) / sum(weights)
            dv, da = (tv - cur.valence) * 0.3, (ta - cur.arousal) * 0.3
            if dv < 0:
                dv *= 0.5 if cur.valence > -0.3 else 0.0
        if user and user.confidence > 0:
            if user.label in ("sad", "anxious"):
                if cur.valence > -0.2:                             # sympathetic, never mirroring despair
                    dv -= 0.05 * user.confidence
                da += (0.3 - cur.arousal) * 0.3 * user.confidence
            elif user.label == "frustrated":
                da += (0.3 - cur.arousal) * 0.3 * user.confidence  # calmer, never hostile
            elif user.label == "positive":
                dv += 0.08 * user.confidence

        def step(x: float) -> float:
            return max(-self.max_step, min(self.max_step, x))
        return Mood(max(-1.0, min(1.0, cur.valence + step(dv))), max(0.0, min(1.0, cur.arousal + step(da))),
                    cur.updated_at)

    @staticmethod
    def describe(m: Mood) -> str:
        """Short, tentative words for the prompt - never numbers or psychological claims."""
        v = ("upbeat" if m.valence > 0.45 else "in a good mood" if m.valence > 0.15 else
             "a bit subdued" if m.valence < -0.35 else "a little flat" if m.valence < -0.1 else "even-keeled")
        a = "energetic" if m.arousal > 0.7 else "relaxed" if m.arousal < 0.3 else ""
        return f"{v}, {a}" if a else v

    @staticmethod
    def speed_factor(m: Mood) -> float:
        """Restrained TTS speed shift, at most +/-3%."""
        return round(1.0 + max(-0.03, min(0.03, (m.arousal - 0.45) * 0.08)), 3)

    @staticmethod
    def to_dict(m: Mood) -> dict:
        return {"valence": round(m.valence, 3), "arousal": round(m.arousal, 3), "updated_at": m.updated_at,
                "description": MoodModel.describe(m)}
