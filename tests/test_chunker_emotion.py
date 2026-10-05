from riko.core.chunker import SentenceChunker, ThinkFilter, clean_for_tts
from riko.emotion import EmotionAnalyzer, strip_tags


def stream(text, ch=None):
    ch = ch or SentenceChunker()
    out = []
    for tok in text.split(" "):
        out += ch.feed(tok + " ")
    return out + ch.flush()


def test_first_chunk_is_eager_and_abbrev_safe():
    out = stream("Hmph. Dr. Tanaka said pi is 3.14 today! Well, fine.")
    assert out[0] == "Hmph."
    assert out[1] == "Dr. Tanaka said pi is 3.14 today!"


def test_long_first_sentence_splits_at_comma():
    out = stream("Well senpai, if you insist on asking such obvious questions all day long then I suppose I can help.")
    assert out[0].endswith(",") and len(out) == 2


def test_max_chars_forces_split():
    out = stream(("word " * 120).strip(), SentenceChunker(max_chars=100))
    assert all(len(c) <= 100 for c in out)


def test_cjk_sentences():
    ch = SentenceChunker()
    assert ch.feed("こんにちは。元気ですか？はい") == ["こんにちは。", "元気ですか？"]


def test_think_filter_split_tags():
    f = ThinkFilter()
    out = "".join(f.feed(x) for x in ["Hi <thi", "nk>hidden plan</th", "ink> there"]) + f.flush()
    assert out == "Hi  there"


def test_clean_for_tts():
    assert clean_for_tts("*rolls eyes* Oh **really**? 😏 https://x.y ok") == "Oh really? ok"
    assert clean_for_tts("...!") == ""


def test_emotion_tags_and_fallback():
    a = EmotionAnalyzer()
    text, st = a.analyze("[smug:0.8] Obviously, senpai.")
    assert text == "Obviously, senpai." and st.emotion == "smug" and st.intensity == 0.8
    assert a.update(st) is not None and a.update(st) is None  # no duplicate change events
    _, st2 = a.analyze("Ugh, seriously? That's so annoying!")
    assert st2.emotion == "angry"
    assert strip_tags("[happy] Hi [note] there") == "Hi [note] there"
    assert "smug" in a.instructions()
