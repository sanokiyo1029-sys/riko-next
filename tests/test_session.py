"""Pipeline behaviour: streaming order, emotion, barge-in, interruption history, PTT, failures."""
import asyncio

import numpy as np

from riko.core.session import ConversationSession
from riko.llm import LLMError, LLMProvider
from riko.tts import TTSError
from fakes import FakeASR, FakeLLM, FakeTTS, RecordingSink, make_services, speech_audio, to_frames


async def wait_for(pred, timeout=5.0):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if pred():
            return True
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


async def test_text_turn_streams_sentences_to_tts_before_llm_finishes(tmp_path):
    tts = FakeTTS()
    llm = FakeLLM("[happy] Hello senpai! It's nice to see you again. How was your day?", delay=0.02)
    svc = make_services(tmp_path, llm=llm, tts=tts)
    sink = RecordingSink()
    s = ConversationSession(svc, sink)
    await s.start()
    await s.submit_text("hi")
    await wait_for(lambda: "assistant_stopped" in sink.types())
    t = sink.types()
    # first audio is produced before the LLM has finished the reply
    assert t.index("assistant_speaking") < t.index("llm_finished")
    assert t.index("transcription") < t.index("llm_started") < t.index("llm_sentence")
    assert [e["text"] for e in sink.of("llm_sentence")] == ["Hello senpai!", "It's nice to see you again.",
                                                             "How was your day?"]
    assert sink.of("emotion_changed")[0]["emotion"] == "happy"
    assert tts.texts == ["Hello senpai!", "It's nice to see you again.", "How was your day?"]
    assert {a[0] for a in sink.audio} == {1} and sink.audio[0][2] == 24000
    assert sink.of("assistant_stopped")[0]["reason"] == "completed"
    assert "first_audio_ms" in sink.of("metrics")[0]
    # system prompt carries personality + emotion-tag instruction
    assert "snarky anime girl" in llm.calls[0][0]["content"] and "[smug:0.7]" in llm.calls[0][0]["content"]


async def test_memory_created_after_turn_and_used_next_turn(tmp_path):
    llm = FakeLLM("[happy] Nice to meet you, Alex.")
    svc = make_services(tmp_path, llm=llm)
    sink = RecordingSink()
    s = ConversationSession(svc, sink)
    await s.submit_text("Hi, my name is Alex")
    await wait_for(lambda: sink.of("memory_created"), timeout=8)
    assert sink.of("memory_created")[0]["content"] == "User's name is Alex"
    await s.submit_text("what is my name, Alex?")
    await wait_for(lambda: len(sink.of("assistant_stopped")) == 2)
    assert "User's name is Alex" in llm.calls[-1][0]["content"]
    assert sink.of("memory_retrieved")
    # short-term history was persisted and replayed
    assert any(m["content"] == "Hi, my name is Alex" for m in llm.calls[-1][1:])


async def test_barge_in_stops_speech_and_records_only_spoken_part(tmp_path):
    llm = FakeLLM("[smug] First sentence is here. " + " ".join(f"Sentence number {i} goes on." for i in range(40)),
                  delay=0.01)
    svc = make_services(tmp_path, llm=llm, tts=FakeTTS(chunk_delay=0.03))
    sink = RecordingSink()
    s = ConversationSession(svc, sink)
    s.input_mode = "auto"
    await s.submit_text("tell me a long story")
    await wait_for(lambda: "assistant_speaking" in sink.types())
    for f in to_frames(speech_audio(speech_s=0.8, lead_s=0.3, tail_s=0.1)):
        await s.feed_audio(f)
    await wait_for(lambda: sink.of("assistant_stopped"))
    assert "assistant_duck" in sink.types()
    assert sink.of("assistant_stopped")[0]["reason"] == "barge_in"
    msgs = svc.memory.store.recent_messages("riko", 10)
    assistant = [m for m in msgs if m["role"] == "assistant"][0]
    assert assistant["interrupted"] == 1 and "Sentence number 39" not in assistant["content"]
    assert "First sentence is here." in assistant["content"]


async def test_short_noise_while_speaking_only_ducks(tmp_path):
    svc = make_services(tmp_path, llm=FakeLLM("A reply that keeps going for a while. " * 5, delay=0.01),
                        tts=FakeTTS(chunk_delay=0.02))
    svc.config.audio.barge_in.min_speech_ms = 800
    sink = RecordingSink()
    s = ConversationSession(svc, sink)
    await s.submit_text("hi")
    await wait_for(lambda: "assistant_speaking" in sink.types())
    for f in to_frames(speech_audio(speech_s=0.35, lead_s=0.3, tail_s=0.9)):
        await s.feed_audio(f)
    await wait_for(lambda: "assistant_unduck" in sink.types())
    await wait_for(lambda: sink.of("assistant_stopped"))
    assert sink.of("assistant_stopped")[0]["reason"] == "completed"


async def test_voice_detection_and_push_to_talk(tmp_path):
    asr = FakeASR("Hello from my voice")
    svc = make_services(tmp_path, asr=asr)
    sink = RecordingSink()
    s = ConversationSession(svc, sink)
    for f in to_frames(speech_audio()):
        await s.feed_audio(f)
    await wait_for(lambda: sink.of("assistant_stopped"))
    assert sink.of("transcription")[0]["text"] == "Hello from my voice" and asr.calls == 1
    # push-to-talk ignores VAD and sends exactly what was held
    s.input_mode = "ptt"
    await s.ptt_start()
    for f in to_frames(np.zeros(16000, dtype=np.float32)):
        await s.feed_audio(f)
    await s.ptt_stop()
    await wait_for(lambda: len(sink.of("assistant_stopped")) == 2)
    assert asr.calls == 2
    await s.ptt_start()
    await s.ptt_stop()  # too short
    assert sink.of("transcription")[-1].get("reason") == "too_short"


class BrokenLLM(LLMProvider):
    name = "broken"

    async def stream_chat(self, messages, **kw):
        raise LLMError("ollama: cannot connect")
        yield ""  # pragma: no cover

    async def list_models(self):
        return []


class BrokenTTS(FakeTTS):
    async def synthesize(self, req):
        raise TTSError("cannot connect to GPT-SoVITS", unavailable=True)
        yield  # pragma: no cover


async def test_failures_are_reported_not_fatal(tmp_path):
    sink = RecordingSink()
    s = ConversationSession(make_services(tmp_path, llm=BrokenLLM("m")), sink)
    await s.submit_text("hi")
    await wait_for(lambda: sink.of("assistant_stopped"))
    assert sink.of("error")[0]["component"] == "llm" and sink.of("assistant_stopped")[0]["reason"] == "error"

    sink2 = RecordingSink()
    tts = BrokenTTS()
    s2 = ConversationSession(make_services(tmp_path / "b", tts=tts), sink2)
    await s2.submit_text("hi")
    await wait_for(lambda: sink2.of("assistant_stopped"))
    assert len(sink2.of("error")) == 1  # reported once, not per sentence
    assert len(sink2.of("llm_sentence")) == 3  # text still delivered


async def test_new_text_cancels_running_turn_and_settings(tmp_path):
    svc = make_services(tmp_path, llm=FakeLLM("word " * 200 + ".", delay=0.01))
    sink = RecordingSink()
    s = ConversationSession(svc, sink)
    await s.submit_text("first")
    await wait_for(lambda: "llm_token" in sink.types())
    await s.submit_text("second")
    await wait_for(lambda: len(sink.of("assistant_stopped")) == 2, timeout=10)
    assert [e["reason"] for e in sink.of("assistant_stopped")] == ["new_turn", "completed"]
    await s.apply_settings(preset="sarcastic", model="other-model", muted=True)
    assert svc.personality.preset == "sarcastic" and svc.llm.model == "other-model" and s.muted
    await s.apply_settings(preset="does-not-exist")
    assert sink.of("error")[-1]["component"] == "settings"
