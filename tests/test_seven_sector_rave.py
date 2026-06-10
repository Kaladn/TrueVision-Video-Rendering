from truevision_runtime.rendering.seven_sector_rave import map_stem_name_to_role, normalize_audio


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
