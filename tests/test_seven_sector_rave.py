import numpy as np

from truevision_runtime.rendering.seven_sector_rave import build_envelope, map_stem_name_to_role, normalize_audio


def test_map_stem_name_to_role_prefers_explicit_names():
    assert map_stem_name_to_role("Lead Vocals.wav") == "vocal"
    assert map_stem_name_to_role("Drums.wav") == "drums"
    assert map_stem_name_to_role("Bass.wav") == "bass"
    assert map_stem_name_to_role("Synth.wav") == "synth"
    assert map_stem_name_to_role("Guitar.wav") == "guitar"
    assert map_stem_name_to_role("Keyboard.wav") == "keys"
    assert map_stem_name_to_role("Other.wav") == "other"


def test_normalize_audio_handles_silence_and_peak():
    silent = normalize_audio([0.0, 0.0, 0.0])
    assert silent == [0.0, 0.0, 0.0]
    normalized = normalize_audio([0.0, 2.0, -1.0])
    assert normalized == [0.0, 1.0, -0.5]


def test_build_envelope_returns_one_value_per_frame():
    samples = np.ones(48000, dtype=np.float32) * 0.25
    envelope = build_envelope(samples, sample_rate=48000, fps=30, duration_seconds=1.0)
    assert len(envelope) == 30
    assert all(0.99 <= value <= 1.0 for value in envelope)


def test_build_envelope_tracks_quiet_and_loud_regions():
    samples = np.concatenate(
        [
            np.zeros(24000, dtype=np.float32),
            np.ones(24000, dtype=np.float32),
        ]
    )
    envelope = build_envelope(samples, sample_rate=48000, fps=10, duration_seconds=1.0)
    assert max(envelope[:5]) == 0.0
    assert min(envelope[5:]) > 0.9
