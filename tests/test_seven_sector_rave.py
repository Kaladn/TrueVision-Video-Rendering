from pathlib import Path
import zipfile

import numpy as np
import pytest

import truevision_runtime.rendering.seven_sector_rave as rave
from truevision_runtime.rendering.seven_sector_rave import (
    assign_stem_paths,
    build_envelope,
    build_manifest,
    build_sector_states,
    extract_stems,
    map_stem_name_to_role,
    normalize_audio,
    render_frame,
)


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


def test_assign_stem_paths_records_fallbacks():
    paths = [
        Path("Lead Vocals.wav"),
        Path("Drums.wav"),
        Path("Bass.wav"),
        Path("Mystery.wav"),
    ]
    mapping, fallbacks = assign_stem_paths(paths)
    assert mapping["vocal"].name == "Lead Vocals.wav"
    assert mapping["drums"].name == "Drums.wav"
    assert mapping["bass"].name == "Bass.wav"
    assert mapping["other"].name == "Mystery.wav"
    assert "synth" in fallbacks
    assert "guitar" in fallbacks
    assert "keys" in fallbacks


def test_extract_stems_rejects_oversized_zip_members(tmp_path, monkeypatch):
    stems_zip = tmp_path / "stems.zip"
    with zipfile.ZipFile(stems_zip, "w") as archive:
        archive.writestr("Lead Vocals.wav", b"abcd")

    monkeypatch.setattr(rave, "MAX_STEM_MEMBER_BYTES", 3, raising=False)

    with pytest.raises(ValueError, match="exceeds"):
        extract_stems(stems_zip, tmp_path / "work", seconds=1.0)


def test_extract_stems_uses_unique_names_and_decodes_wavs(tmp_path, monkeypatch):
    stems_zip = tmp_path / "stems.zip"
    with zipfile.ZipFile(stems_zip, "w") as archive:
        archive.writestr("first/Lead Vocals.wav", b"first")
        archive.writestr("second/Lead Vocals.wav", b"second")

    commands = []

    def fake_run(command, check, capture_output, text):
        commands.append(command)
        Path(command[-1]).write_bytes(b"decoded")

    monkeypatch.setattr(rave.subprocess, "run", fake_run)

    decoded = extract_stems(stems_zip, tmp_path / "work", seconds=2.5)

    assert [path.name for path in decoded] == ["000_Lead_Vocals.wav", "001_Lead_Vocals.wav"]
    assert [command[3] for command in commands] == ["2.500", "2.500"]
    assert all(command[4] == "-i" for command in commands)

def test_build_sector_states_contains_all_roles():
    envelopes = {role: [0.0, 0.5, 1.0] for role in ["vocal", "drums", "bass", "synth", "guitar", "keys", "other"]}
    states = build_sector_states(envelopes, fps=3, duration_seconds=1.0)
    assert len(states) == 3
    assert set(states[0]["sectors"]) == {"vocal", "drums", "bass", "synth", "guitar", "keys", "other"}
    assert states[2]["sectors"]["vocal"]["energy"] == 1.0


def test_render_frame_is_nonblank():
    state = {
        "frame": 0,
        "time": 0.0,
        "sectors": {role: {"energy": 0.8, "transient": 0.3, "phase": 0.2} for role in ["vocal", "drums", "bass", "synth", "guitar", "keys", "other"]},
    }
    frame = render_frame(state, width=640, height=360, waveform=[0.0, 0.5, -0.5, 0.0])
    assert frame.shape == (360, 640, 3)
    assert int(frame.max()) > 20


def test_build_manifest_records_contract():
    manifest = build_manifest(
        run_id="demo",
        audio_path="song.wav",
        stems_zip="stems.zip",
        output_video="out.mp4",
        fps=30,
        seconds=30.0,
        width=1280,
        height=720,
        stem_mapping={"vocal": "Lead Vocals.wav"},
        fallbacks=["synth"],
    )
    assert manifest["kind"] == "truevision_seven_sector_rave_reactor_manifest"
    assert manifest["run_id"] == "demo"
    assert manifest["sector_law"]["center"] == "vocal exact waveform"
    assert manifest["fallbacks"] == ["synth"]
