import numpy as np
import pytest

from riko.audio.utils import encode_wav, parse_wav_header, resample_linear
from riko.audio.vad import FRAME, EnergyVAD, SileroVAD, SpeechEnd, SpeechStart, VADSegmenter, find_silero_model
from riko.core.events import decode_audio_frame, encode_audio_frame
from fakes import speech_audio


def run(seg, audio, block=1000):
    evs = []
    for i in range(0, audio.size, block):
        evs += seg.process(audio[i:i + block])
    return evs


def test_energy_segmenter_finds_one_utterance_with_preroll():
    seg = VADSegmenter(EnergyVAD(12.0), min_speech_ms=200, silence_ms=500, preroll_ms=300)
    evs = run(seg, speech_audio(speech_s=1.0))
    starts = [e for e in evs if isinstance(e, SpeechStart)]
    ends = [e for e in evs if isinstance(e, SpeechEnd)]
    assert len(starts) == 1 and len(ends) == 1
    assert 1.1 <= ends[0].duration_s <= 1.8  # speech + pre-roll + kept tail


def test_short_blip_is_not_speech():
    seg = VADSegmenter(EnergyVAD(12.0), min_speech_ms=200, silence_ms=500)
    assert not [e for e in run(seg, speech_audio(speech_s=0.1)) if isinstance(e, SpeechStart)]


def test_max_utterance_forces_end():
    seg = VADSegmenter(EnergyVAD(12.0), min_speech_ms=100, silence_ms=500, max_utterance_s=1.0)
    ends = [e for e in run(seg, speech_audio(speech_s=2.5)) if isinstance(e, SpeechEnd)]
    assert ends and ends[0].forced


@pytest.mark.skipif(find_silero_model() is None, reason="faster-whisper not installed")
def test_streaming_silero_matches_reference_batch():
    from faster_whisper.vad import SileroVADModel
    audio = speech_audio(speech_s=1.0)
    audio = audio[: audio.size // FRAME * FRAME]
    ref = SileroVADModel(str(find_silero_model()))(audio).reshape(-1)
    v = SileroVAD(find_silero_model())
    mine = np.array([v.speech_prob(audio[i * FRAME:(i + 1) * FRAME]) for i in range(audio.size // FRAME)])
    assert np.allclose(ref, mine, atol=1e-5)


def test_wav_and_frames_roundtrip():
    wav = encode_wav(b"\x00\x01" * 10, 22050)
    fmt = parse_wav_header(wav)
    assert fmt.sample_rate == 22050 and wav[fmt.data_offset:] == b"\x00\x01" * 10
    assert parse_wav_header(wav[:20]) is None
    assert decode_audio_frame(encode_audio_frame(7, 3, 32000, b"ab")) == (7, 3, 32000, b"ab")
    assert resample_linear(np.ones(32000, dtype=np.float32), 32000, 16000).size == 16000
