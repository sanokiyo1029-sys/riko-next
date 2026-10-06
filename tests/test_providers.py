"""LLM / ASR / TTS providers against mock servers that replicate the real wire formats."""
import io
import json
import wave

import httpx
import numpy as np
import pytest

from riko.asr import ASRError, FasterWhisperASR, OpenAICompatibleASR
from riko.core.config import ASRConfig, GPTSoVITSConfig, LLMConfig, OpenAITTSConfig, VoiceConfig, EmotionVoiceOverride
from riko.llm import LLMError, OllamaLLM, OpenAICompatibleLLM, create_llm
from riko.tts import GPTSoVITSTTS, OpenAICompatibleTTS, SpeechRequest, TTSError
from fakes import ROOT

MSGS = [{"role": "user", "content": "hi"}]


async def collect(agen):
    return [x async for x in agen]


async def chunked(parts):
    for p in parts:
        yield p


# ------------------------------------------------------------------ LLM
async def test_openai_sse_stream_and_auth():
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        sse = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
               b': keep-alive\n\n'
               b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
               b'data: {"choices":[{"delta":{"content":"lo!"}}]}\n\n'
               b'data: [DONE]\n\n')
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    llm = OpenAICompatibleLLM(name="openai", base_url="https://x/v1", model="m", api_key="sk-test1234567",
                              transport=httpx.MockTransport(handler))
    assert await collect(llm.stream_chat(MSGS)) == ["Hel", "lo!"]
    assert seen["auth"] == "Bearer sk-test1234567" and seen["body"]["stream"] is True and seen["body"]["model"] == "m"


async def test_openai_error_is_readable_and_keyless():
    llm = OpenAICompatibleLLM(name="openai", base_url="https://x/v1", model="m", api_key="sk-secret999999",
                              transport=httpx.MockTransport(lambda r: httpx.Response(401, text='{"error":"bad key"}')))
    with pytest.raises(LLMError) as e:
        await collect(llm.stream_chat(MSGS))
    assert "authentication failed" in str(e.value) and "sk-secret" not in str(e.value)


async def test_ollama_ndjson_stream():
    def handler(req):
        body = json.loads(req.content)
        assert body["keep_alive"] == "30m" and body["options"]["num_predict"] == 2048
        lines = [{"message": {"content": "Hi "}, "done": False}, {"message": {"content": "senpai"}, "done": False},
                 {"message": {"content": ""}, "done": True}]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
    llm = OllamaLLM(base_url="http://o", model="llama3.1:8b", transport=httpx.MockTransport(handler))
    assert "".join(await collect(llm.stream_chat(MSGS))) == "Hi senpai"


async def test_ollama_missing_model_hint():
    llm = OllamaLLM(base_url="http://o", model="nope", transport=httpx.MockTransport(
        lambda r: httpx.Response(404, json={"error": "model 'nope' not found"}) if r.url.path == "/api/chat"
        else httpx.Response(200, json={"models": [{"name": "llama3.1:latest"}]})))
    with pytest.raises(LLMError, match="ollama pull nope"):
        await collect(llm.stream_chat(MSGS))
    ok, msg = await llm.health()
    assert not ok and "ollama pull" in msg
    llm.model = "llama3.1"
    assert (await llm.health())[0]


async def test_connection_refused_is_llm_error():
    def boom(req):
        raise httpx.ConnectError("refused")
    llm = OllamaLLM(base_url="http://o", model="m", transport=httpx.MockTransport(boom))
    with pytest.raises(LLMError, match="ollama serve"):
        await collect(llm.stream_chat(MSGS))


def test_factory_reads_key_from_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        create_llm(LLMConfig(provider="openai", model="gpt-4.1-mini"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fromenv12345")
    llm = create_llm(LLMConfig(provider="openai", model="gpt-4.1-mini"))
    assert llm._client.headers["authorization"] == "Bearer sk-fromenv12345"
    assert isinstance(create_llm(LLMConfig(provider="lmstudio", model="x")), OpenAICompatibleLLM)
    assert isinstance(create_llm(LLMConfig(provider="ollama", model="x")), OllamaLLM)


async def test_list_models_openai():
    llm = OpenAICompatibleLLM(name="lmstudio", base_url="http://l/v1", model="b", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"data": [{"id": "b"}, {"id": "a"}]})))
    assert await llm.list_models() == ["a", "b"]


# ------------------------------------------------------------------ TTS
def _streaming_wav_parts(sr=32000, n=3000):
    """Exactly what GPT-SoVITS api_v2 streams: a 0-frame WAV header, then raw int16 PCM."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(b"")
    pcm = (np.sin(np.arange(n) / 5) * 9000).astype("<i2").tobytes()
    stream = buf.getvalue() + pcm
    sizes = [7, 30, 13, 101, 999, 3]  # boundaries in the middle of the header and of samples
    parts, i, k = [], 0, 0
    while i < len(stream):
        s = sizes[k % len(sizes)]
        parts.append(stream[i:i + s]); i += s; k += 1
    return parts, pcm


def _sovits(handler, **kw):
    return GPTSoVITSTTS(GPTSoVITSConfig(**kw), lambda p: (ROOT / p).resolve(), transport=httpx.MockTransport(handler))


VOICE = VoiceConfig(ref_audio="characters/voices/riko_sample.wav", prompt_text="sample", prompt_lang="en",
                    emotions={"sad": EmotionVoiceOverride(speed_factor=0.9)})


async def test_gpt_sovits_streaming_reassembles_pcm():
    parts, pcm = _streaming_wav_parts()
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, content=chunked(parts))

    chunks = await collect(_sovits(handler).synthesize(SpeechRequest("Hello.", VOICE, "sad", 0.7, 1.0)))
    assert b"".join(c.pcm for c in chunks) == pcm
    assert all(c.sample_rate == 32000 and len(c.pcm) % 2 == 0 for c in chunks)
    assert seen["ref_audio_path"] == str((ROOT / "characters/voices/riko_sample.wav").resolve())
    assert seen["speed_factor"] == 0.9 and seen["media_type"] == "wav" and seen["streaming_mode"] == 2


async def test_gpt_sovits_full_wav_and_errors():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x01\x00" * 100)
    chunks = await collect(_sovits(lambda r: httpx.Response(200, content=buf.getvalue()), streaming_mode=0)
                           .synthesize(SpeechRequest("Hi.", VOICE)))
    assert sum(len(c.pcm) for c in chunks) == 200 and chunks[0].sample_rate == 24000

    err = _sovits(lambda r: httpx.Response(400, json={"message": "tts failed", "Exception": "ref too long"}))
    with pytest.raises(TTSError, match="ref too long"):
        await collect(err.synthesize(SpeechRequest("Hi.", VOICE)))

    def boom(r):
        raise httpx.ConnectError("refused")
    with pytest.raises(TTSError) as e:
        await collect(_sovits(boom).synthesize(SpeechRequest("Hi.", VOICE)))
    assert e.value.unavailable

    with pytest.raises(TTSError, match="not found"):
        await collect(_sovits(lambda r: httpx.Response(200)).synthesize(
            SpeechRequest("Hi.", VoiceConfig(ref_audio="missing.wav"))))


async def test_openai_tts_pcm_stream():
    def handler(req):
        body = json.loads(req.content)
        assert body["response_format"] == "pcm" and body["voice"] == "af_heart"
        return httpx.Response(200, content=chunked([b"\x01", b"\x00\x02\x00\x03", b"\x00"]))
    tts = OpenAICompatibleTTS(OpenAITTSConfig(), transport=httpx.MockTransport(handler))
    chunks = await collect(tts.synthesize(SpeechRequest("Hi", VoiceConfig(voice_id="af_heart"))))
    assert b"".join(c.pcm for c in chunks) == b"\x01\x00\x02\x00\x03\x00"


# ------------------------------------------------------------------ ASR
async def test_openai_compatible_asr_multipart():
    def handler(req):
        body = req.content
        assert b'name="model"' in body and b"whisper-large-v3-turbo" in body and b"RIFF" in body
        return httpx.Response(200, json={"text": " hello there "})
    asr = OpenAICompatibleASR(ASRConfig(provider="openai_compatible", base_url="http://a/v1"),
                              transport=httpx.MockTransport(handler))
    tr = await asr.transcribe(np.zeros(8000, dtype=np.float32))
    assert tr.text == "hello there" and tr.audio_s == 0.5


def test_groq_asr_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ASRError, match="GROQ_API_KEY"):
        OpenAICompatibleASR(ASRConfig(provider="groq"))


class _Seg:
    def __init__(self, text, nsp=0.0, lp=-0.2):
        self.text, self.no_speech_prob, self.avg_logprob = text, nsp, lp


class _FakeWhisper:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kw):
        self.kwargs = kw
        return iter([_Seg(" Hi Riko."), _Seg(" Thanks for watching!", nsp=0.9, lp=-1.5)]), type("I", (), {"language": "en"})()


async def test_faster_whisper_wrapper_filters_hallucinations():
    fake = _FakeWhisper()
    asr = FasterWhisperASR(ASRConfig(beam_size=1), model=fake)
    tr = await asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert tr.text == "Hi Riko." and fake.kwargs["beam_size"] == 1 and fake.kwargs["vad_filter"] is False
