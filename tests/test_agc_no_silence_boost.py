import torch

from callclarity.methods.leveler.speech_aware_agc import SpeechAwareAgc
from callclarity.types import AudioChunk


def test_agc_does_not_keep_rising_during_silence():
    agc = SpeechAwareAgc(
        {
            "target_speech_rms_dbfs": -20.0,
            "max_boost_db": 15.0,
            "attack_db_per_sec": 20.0,
            "release_db_per_sec": 2.0,
            "vad_required": True,
            "vad_threshold": 0.55,
            "freeze_gain_on_silence": True,
            "compressor": {"enabled": False},
            "limiter": {"enabled": False},
        }
    )
    speech = AudioChunk(torch.ones(1, 160) * 0.01, 16000, 0.0, metadata={"speech_prob": 1.0})
    for idx in range(20):
        agc.process(AudioChunk(speech.samples, 16000, idx * 0.01, metadata={"speech_prob": 1.0}))
    gain_before = agc.current_gain_db
    silence = torch.zeros(1, 160)
    for idx in range(300):
        agc.process(AudioChunk(silence, 16000, idx * 0.01, metadata={"speech_prob": 0.0}))
    assert agc.current_gain_db == gain_before
