import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_runtime.learning_intake.meter_grid import (
    build_meter_grid_profile_from_native_capture,
    build_metered_section_selection_plan,
    write_meter_grid_from_capture,
)


FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "delta_luma_abs",
    "edge_density",
    "texture_energy",
    "motion_energy",
]


def _write_meter_capture(root: Path, *, run_id: str, kind: str) -> Path:
    run_dir = root / run_id
    cell_dir = run_dir / "cell_state_native"
    cell_dir.mkdir(parents=True)
    frames = 36
    rows = 8
    cols = 8
    cells = np.zeros((frames, rows, cols, len(FEATURE_NAMES)), dtype="<f4")
    yy, xx = np.mgrid[0:rows, 0:cols]
    base = np.full((rows, cols), 0.08, dtype=np.float32)
    previous = base.copy()
    for frame_index in range(frames):
        luma = base.copy()
        edge = np.full((rows, cols), 0.08, dtype=np.float32)
        texture = np.full((rows, cols), 0.06, dtype=np.float32)
        motion = np.full((rows, cols), 0.03, dtype=np.float32)
        saturation = np.full((rows, cols), 0.14, dtype=np.float32)
        if kind == "lightning":
            branch = ((np.abs(xx - 4) <= 1) & (yy >= 1) & (yy <= 5)) | ((yy == 3) & (xx >= 2) & (xx <= 6))
            distance = np.sqrt((xx - 4) ** 2 + (yy - 3) ** 2)
            if frame_index == 10:
                luma += np.where(branch, 0.88, 0.0)
                luma += np.clip(0.48 - distance * 0.08, 0.0, 0.48)
            elif 11 <= frame_index <= 18:
                decay = 0.58 * np.exp(-(frame_index - 11) / 3.0)
                luma += np.clip(decay - distance * 0.04, 0.0, decay)
            edge += np.where(branch, 0.66, 0.0)
            texture += np.where(branch, 0.32, 0.0)
            motion += np.where(branch, 0.30 if 10 <= frame_index <= 12 else 0.04, 0.0)
            saturation += np.where(branch, 0.06, 0.0)
        elif kind == "persistent_reflection":
            band = (yy >= 3) & (yy <= 4)
            shimmer = 0.04 * np.sin(frame_index / 4.0)
            luma += np.where(band, 0.62 + shimmer, 0.0)
            edge += np.where(band, 0.10, 0.0)
            texture += np.where(band, 0.12, 0.0)
            motion += np.where(band, 0.08, 0.0)
        elif kind == "static_line":
            line = xx == 4
            luma += np.where(line, 0.74, 0.0)
            edge += np.where(line, 0.70, 0.0)
            texture += np.where(line, 0.28, 0.0)
        else:
            raise ValueError(kind)
        luma = np.clip(luma, 0.0, 1.0)
        delta = np.abs(luma - previous)
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_r")] = luma
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_g")] = np.clip(luma * 0.96, 0.0, 1.0)
        cells[frame_index, :, :, FEATURE_NAMES.index("rgb_mean_b")] = np.clip(luma * 1.04, 0.0, 1.0)
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_mean")] = luma
        cells[frame_index, :, :, FEATURE_NAMES.index("luma_std")] = np.clip(luma * 0.20 + edge * 0.05, 0.0, 1.0)
        cells[frame_index, :, :, FEATURE_NAMES.index("saturation_mean")] = saturation
        cells[frame_index, :, :, FEATURE_NAMES.index("delta_luma_abs")] = delta
        cells[frame_index, :, :, FEATURE_NAMES.index("edge_density")] = np.clip(edge, 0.0, 1.0)
        cells[frame_index, :, :, FEATURE_NAMES.index("texture_energy")] = np.clip(texture, 0.0, 1.0)
        cells[frame_index, :, :, FEATURE_NAMES.index("motion_energy")] = np.clip(motion, 0.0, 1.0)
        previous = luma
    chunk = cell_dir / f"{run_id}_cells_0000.tvcells"
    chunk.write_bytes(cells.tobytes())
    records = run_dir / f"{run_id}_records.jsonl"
    with records.open("w", encoding="utf-8") as handle:
        for frame_index in range(frames):
            handle.write(json.dumps({"frame_number": frame_index + 1, "elapsed_seconds": frame_index / 30.0}) + "\n")
    manifest = {
        "schema_version": 1,
        "record_kind": "truevision_native_rs_frame_state",
        "run_id": run_id,
        "records_jsonl": str(records),
        "config": {
            "duration_seconds": frames / 30.0,
            "capture_fps": 30,
            "capture_resolution": [cols * 8, rows * 8],
            "grid_size_xy": [cols, rows],
            "capture_region": [0, 0, cols * 8, rows * 8],
            "cell_chunk_frames": frames,
        },
        "summary": {"frame_count": frames, "duration_seconds": frames / 30.0},
        "cell_state": {
            "enabled": True,
            "format": "tvcells_f32le_v1",
            "feature_names": FEATURE_NAMES,
            "chunks": [
                {
                    "chunk_id": 0,
                    "path": str(chunk),
                    "format": "tvcells_f32le_v1",
                    "frames": frames,
                    "grid_shape": [rows, cols],
                    "feature_count": len(FEATURE_NAMES),
                }
            ],
        },
        "boundary": {"raw_frame_saved": False},
    }
    manifest_path = run_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class MeterGridTests(unittest.TestCase):
    def test_meter_grid_supports_sudden_lightning_and_writes_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_meter_capture(root, run_id="meter_lightning", kind="lightning")
            result = write_meter_grid_from_capture(
                {
                    "manifest": str(manifest),
                    "run_id": "meter_lightning",
                    "section_id": "yt_lightning_003_t10_18",
                    "event_type_candidate": "candidate_lightning",
                },
                storage_root=root / "storage",
            )
            profile = json.loads(Path(result["profile_json"]).read_text(encoding="utf-8"))
            receipt = json.loads(Path(result["receipt_json"]).read_text(encoding="utf-8"))
            graph_headers = {name: Path(path).read_bytes()[:8] for name, path in result["graphs"].items()}

        self.assertEqual(profile["schema_version"], "truevision_meter_grid_profile_v0")
        self.assertIn("luma_mean", profile["meter_names"])
        self.assertIn("bloom_pressure", profile["meter_names"])
        self.assertEqual(profile["event_profiles"][0]["status"], "visually_supported")
        self.assertEqual(profile["event_profiles"][0]["event_type_candidate"], "candidate_lightning")
        self.assertGreater(profile["event_profiles"][0]["support"]["luma_delta"], 0.70)
        self.assertLessEqual(profile["event_profiles"][0]["support"]["rise_time_frames"], 2)
        self.assertGreater(profile["event_profiles"][0]["support"]["bloom_radius_cells"], 1)
        self.assertEqual(receipt["profile_sha256"], profile["profile_sha256"])
        for header in graph_headers.values():
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")

    def test_meter_grid_rejects_persistent_reflection_and_static_line_as_lightning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reflection_manifest = _write_meter_capture(
                root,
                run_id="meter_reflection",
                kind="persistent_reflection",
            )
            line_manifest = _write_meter_capture(root, run_id="meter_static_line", kind="static_line")

            reflection = build_meter_grid_profile_from_native_capture(
                reflection_manifest,
                section_id="persistent_ocean_reflection",
                event_type_candidate="candidate_lightning",
            )
            static_line = build_meter_grid_profile_from_native_capture(
                line_manifest,
                section_id="static_white_line",
                event_type_candidate="candidate_lightning",
            )

        self.assertEqual(reflection["event_profiles"][0]["status"], "rejected")
        self.assertIn("persistent_bright_region", reflection["event_profiles"][0]["rejection_reasons"])
        self.assertEqual(static_line["event_profiles"][0]["status"], "rejected")
        self.assertIn("no_temporal_flash", static_line["event_profiles"][0]["rejection_reasons"])

    def test_metered_section_selector_ranks_long_video_probes_by_target_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes = []
            for kind in ["persistent_reflection", "static_line", "lightning"]:
                manifest = _write_meter_capture(root, run_id=f"probe_{kind}", kind=kind)
                probes.append(
                    build_meter_grid_profile_from_native_capture(
                        manifest,
                        section_id=f"probe_{kind}",
                        event_type_candidate="candidate_lightning",
                    )
                )

            plan = build_metered_section_selection_plan(
                probes,
                target_signature="candidate_lightning",
                controller_id="lee_operator_agent",
            )

        self.assertEqual(plan["schema_version"], "truevision_metered_section_selection_plan_v0")
        self.assertEqual(plan["controller"]["controller_id"], "lee_operator_agent")
        self.assertEqual(plan["ranked_sections"][0]["section_id"], "probe_lightning")
        self.assertEqual(plan["ranked_sections"][0]["recommended_action"], "capture_full_section")
        self.assertGreater(plan["ranked_sections"][0]["score"], plan["ranked_sections"][1]["score"])
        self.assertTrue(plan["boundary"]["agent_controls_navigation_only_after_meter_goal"])


if __name__ == "__main__":
    unittest.main()
