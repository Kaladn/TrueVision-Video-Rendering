import json
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
    render_seven_sector_rave,
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
        sector_drivers={
            "vocal": {"source_type": "stem", "source_path": "Lead Vocals.wav"},
            "synth": {"source_type": "master_mix_fallback", "source_path": "song.wav"},
        },
    )
    assert manifest["kind"] == "truevision_seven_sector_rave_reactor_manifest"
    assert manifest["run_id"] == "demo"
    assert manifest["sector_law"]["center"] == "vocal exact waveform"
    assert manifest["fallbacks"] == ["synth"]
    assert manifest["sector_drivers"]["synth"]["source_type"] == "master_mix_fallback"
    assert manifest["kaleidoscope"]["palette_size"] == 6
    assert manifest["kaleidoscope"]["intensity"] == "low"
    assert manifest["kaleidoscope"]["intensity_gain"] == 1.15
    assert manifest["center_sector"]["role"] == "vocal"
    assert manifest["center_sector"]["palette_name"] == "heat_orange_red"


def test_kaleidoscope_palette_is_six_low_intensity_colors():
    assert len(rave.KALEIDOSCOPE_COLORS) == 6
    assert all(max(color) <= 210 for color in rave.KALEIDOSCOPE_COLORS)
    assert rave.KALEIDOSCOPE_INTENSITY_GAIN == 1.15


def test_center_vocal_palette_is_heat_colored_without_flame_effect():
    assert rave.CENTER_VOCAL_COLORS["core"] == (255, 185, 55)
    assert rave.CENTER_VOCAL_COLORS["hot"] == (255, 78, 42)
    assert rave.CENTER_VOCAL_COLORS["effect"] == "reactor_waveform_not_flame"


def test_render_seven_sector_rave_writes_truthful_receipts(tmp_path, monkeypatch):
    master_path = tmp_path / "master.wav"
    stems_zip = tmp_path / "stems.zip"
    master_path.write_bytes(b"master")
    stems_zip.write_bytes(b"stems")
    vocal_path = tmp_path / "Lead Vocals.wav"
    vocal_path.write_bytes(b"vocal")

    monkeypatch.setattr(rave, "extract_stems", lambda stems, work, seconds: [vocal_path])

    def fake_read_wav_mono(path):
        if Path(path) == vocal_path:
            return np.linspace(-0.5, 0.5, 20, dtype=np.float32), 10
        return np.ones(20, dtype=np.float32) * 0.25, 10

    monkeypatch.setattr(rave, "read_wav_mono", fake_read_wav_mono)

    class FakeVideoWriter:
        def __init__(self, *args):
            self.frames = []

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            pass

    monkeypatch.setattr(rave.cv2, "VideoWriter", FakeVideoWriter)
    monkeypatch.setattr(rave.subprocess, "run", lambda *args, **kwargs: None)

    manifest = render_seven_sector_rave(
        audio_path=master_path,
        stems_zip=stems_zip,
        output_root=tmp_path / "out",
        run_id="demo",
        seconds=1.0,
        fps=2,
        width=64,
        height=36,
    )

    manifest_path = Path(manifest["manifest_path"])
    state_trace_path = Path(manifest["state_trace_path"])
    assert manifest["frame_count"] == 2
    assert manifest["sector_drivers"]["vocal"]["source_type"] == "stem"
    assert manifest["sector_drivers"]["synth"]["source_type"] == "master_mix_fallback"
    assert manifest_path.exists()
    assert state_trace_path.exists()

    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_lines = [json.loads(line) for line in state_trace_path.read_text(encoding="utf-8").splitlines()]
    assert written_manifest["state_trace_path"] == str(state_trace_path)
    assert [line["frame"] for line in trace_lines] == [0, 1]
    assert all("vocal" in line["sectors"] for line in trace_lines)
