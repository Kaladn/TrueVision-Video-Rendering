import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from truevision_runtime.geometry_generation import (
    KNOWN_LOGGER_LANES,
    build_big_shape_library,
    build_geometry_scene_from_logger_bundle,
    render_geometry_overlay_frame,
    write_geometry_generation_run,
)


def _sample_meter_profile(status: str = "visually_supported", rejection_reasons: list[str] | None = None) -> dict:
    return {
        "schema_version": "truevision_meter_grid_profile_v0",
        "profile_sha256": "sha256:meter",
        "source": {
            "manifest_json": "capture/lightning_manifest.json",
            "records_jsonl": "capture/lightning_records.jsonl",
            "teacher_chunks": [{"path": "capture/cell_state_native/lightning_cells_0000.tvcells"}],
        },
        "section_id": "yt_lightning_t10_18",
        "frame_summaries": [
            {"frame_index": 9, "time_sec": 0.3, "luma_peak": 0.12},
            {"frame_index": 10, "time_sec": 0.333333, "luma_peak": 0.98},
        ],
        "event_profiles": [
            {
                "event_type_candidate": "candidate_lightning",
                "status": status,
                "frame_start": 9,
                "frame_peak": 10,
                "frame_end": 18,
                "cell_bounds": [3, 1, 4, 6],
                "support": {
                    "luma_delta": 0.86,
                    "flash_peak_luma": 0.98,
                    "rise_time_frames": 1,
                    "falloff_frames": 8,
                    "bloom_radius_cells": 4.5,
                    "branching_score": 0.77,
                    "surrounding_exposure_lift": 0.42,
                },
                "rejection_reasons": rejection_reasons or [],
            }
        ],
    }


def _sample_angular_profile() -> dict:
    return {
        "schema_version": "truevision_angular_seismic_profile_v0",
        "profile_sha256": "sha256:angular",
        "source": {"source_video": "local/road.mp4", "source_sha256": "sha256:road"},
        "angular_signature": {
            "dominant_direction": "southeast",
            "dominant_angle_degrees": 45.0,
            "field_coherence_mean": 0.41,
        },
        "seismic_trace": {"impulse_peak": 0.31},
        "candidate_profiles": {
            "glass_reflections": {"peak": 0.22},
            "walking_camera_relation": {"motion_mean": 0.11, "softness_mean": 0.54},
        },
    }


def _sample_focus_profile() -> dict:
    return {
        "schema_version": "truevision_lightfield_focus_profile_v1",
        "profile_sha256": "sha256:focus",
        "source": {"manifest_json": "capture/focus_manifest.json"},
        "active_bounds": {
            "grid_xywh": [2, 1, 10, 6],
            "normalized_xywh": [0.1, 0.1, 0.5, 0.6],
            "orientation": "horizontal_wide",
        },
        "focus_planes": [{"focus_depth": 0.5, "focus_score": 8.2}],
    }


def _sample_element_profile() -> dict:
    return {
        "schema_version": "truevision_element_creation_profile_v1",
        "element_id": "lightning_arc_bloom",
        "profile_sha256": "sha256:element",
        "source": {
            "manifest_json": "capture/lightning_manifest.json",
            "records_jsonl": "capture/lightning_records.jsonl",
            "teacher_chunks": [{"path": "capture/cell_state_native/lightning_cells_0000.tvcells"}],
        },
        "creation_signature": {
            "density_opacity": {"mean": 0.19, "maximum": 0.70},
            "bloom_intensity": {"mean": 0.21, "maximum": 0.99},
            "edge_softness": {"mean": 0.91},
            "growth_decay": {"growth_max": 0.70, "decay_max": 0.42},
            "renderer_binding": {
                "element_id": "lightning_arc_bloom",
                "drive_channels": "density_opacity edge_softness bloom_intensity",
            },
        },
        "boundary": {
            "learned_from_state": True,
            "state_creation_not_replay": True,
            "generated_media_is_evidence": False,
        },
    }


class GeometryGenerationEngineTests(unittest.TestCase):
    def test_big_shape_library_contains_required_state_forms(self):
        library = build_big_shape_library()

        self.assertEqual(library["schema_version"], "truevision_big_shape_library_v1")
        names = {shape["shape_type"] for shape in library["shapes"]}
        for shape_type in [
            "road_plane",
            "horizon_band",
            "vanishing_corridor",
            "fog_bank",
            "depth_wall",
            "light_cone",
            "occlusion_slab",
            "motion_stream",
            "reflection_vector_field",
            "atmosphere_volume",
            "lightning_branch",
        ]:
            self.assertIn(shape_type, names)
        for shape in library["shapes"]:
            self.assertIn("raw_data_refs", shape["required_data_slots"])
            self.assertIn("true_local_metrics", shape["required_data_slots"])
            self.assertIn("filtered_metrics", shape["required_data_slots"])

    def test_lightning_shape_carries_raw_source_truth_separate_from_filters(self):
        scene = build_geometry_scene_from_logger_bundle(
            {
                "meter_grid_profile": _sample_meter_profile(),
                "angular_seismic_profile": _sample_angular_profile(),
                "state_focus_profile": _sample_focus_profile(),
                "element_creation_profile": _sample_element_profile(),
            },
            run_id="geometry_lightning_truth",
        )

        self.assertEqual(scene["schema_version"], "truevision_geometry_scene_v1")
        lightning = next(shape for shape in scene["shape_units"] if shape["shape_type"] == "lightning_branch")
        self.assertEqual(lightning["schema_version"], "truevision_geometry_shape_unit_v1")
        self.assertTrue(lightning["raw_data_refs"])
        self.assertEqual(lightning["source_region"]["cell_bounds"], [3, 1, 4, 6])
        self.assertEqual(lightning["true_local_metrics"]["luma_delta"], 0.86)
        self.assertEqual(lightning["true_local_metrics"]["flash_peak_luma"], 0.98)
        self.assertEqual(lightning["filtered_metrics"]["event_type_candidate"], "candidate_lightning")
        self.assertEqual(lightning["filtered_metrics"]["recognizer"], "meter_grid_filter")
        self.assertTrue(lightning["evidence_boundary"]["raw_state_owns_evidence"])
        self.assertFalse(lightning["evidence_boundary"]["filtered_metrics_are_truth"])
        self.assertFalse(lightning["evidence_boundary"]["generated_media_is_evidence"])

    def test_rejected_candidate_shape_still_carries_source_truth(self):
        scene = build_geometry_scene_from_logger_bundle(
            {
                "meter_grid_profile": _sample_meter_profile("rejected", ["slow_rise_not_lightning"]),
                "element_creation_profile": _sample_element_profile(),
            },
            run_id="geometry_rejected_lightning_test",
        )

        lightning = next(shape for shape in scene["shape_units"] if shape["shape_type"] == "lightning_branch")

        self.assertEqual(lightning["filtered_metrics"]["status"], "rejected")
        self.assertIn("slow_rise_not_lightning", lightning["filtered_metrics"]["rejection_reasons"])
        self.assertGreater(len(lightning["raw_data_refs"]), 0)
        self.assertLess(lightning["confidence"], 0.5)
        self.assertFalse(lightning["evidence_boundary"]["object_truth_promoted"])

    def test_logger_lane_plan_exposes_every_known_lane_for_next_video(self):
        scene = build_geometry_scene_from_logger_bundle(
            {
                "meter_grid_profile": _sample_meter_profile(),
                "angular_seismic_profile": _sample_angular_profile(),
                "state_focus_profile": _sample_focus_profile(),
                "element_creation_profile": _sample_element_profile(),
            },
            run_id="geometry_logger_lanes",
        )

        lane_ids = {lane["lane_id"] for lane in scene["logger_lane_visibility_plan"]}
        self.assertEqual(lane_ids, set(KNOWN_LOGGER_LANES))
        for lane in scene["logger_lane_visibility_plan"]:
            self.assertIn(lane["visible_as"], {"overlay", "meter_graph", "geometry_marks", "state_panel", "receipt_ref"})

    def test_write_geometry_generation_run_writes_manifest_receipt_and_preview_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_geometry_generation_run(
                {
                    "run_id": "geometry_write_test",
                    "meter_grid_profile": _sample_meter_profile(),
                    "angular_seismic_profile": _sample_angular_profile(),
                    "state_focus_profile": _sample_focus_profile(),
                    "element_creation_profile": _sample_element_profile(),
                    "render_preview": True,
                    "duration": 0.2,
                    "fps": 10,
                    "width": 160,
                    "height": 90,
                    "output_root": str(root / "outputs"),
                },
                storage_root=root / "storage",
            )

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            receipt = json.loads(Path(result["receipt_json"]).read_text(encoding="utf-8"))
            scene = json.loads(Path(result["scene_json"]).read_text(encoding="utf-8"))
            self.assertTrue(Path(result["scene_json"]).exists())
            self.assertTrue(Path(result["shape_library_json"]).exists())
            self.assertTrue(Path(result["preview_video"]).exists())
            self.assertEqual(manifest["schema_version"], "truevision_geometry_generation_manifest_v1")
            self.assertEqual(receipt["schema_version"], "truevision_geometry_generation_receipt_v1")
            self.assertEqual(manifest["logger_lane_count"], len(KNOWN_LOGGER_LANES))
            self.assertEqual(receipt["scene_sha256"], scene["scene_sha256"])
            self.assertFalse(receipt["boundary"]["object_truth_promoted"])

    def test_render_geometry_overlay_frame_returns_rgb_pixels(self):
        scene = build_geometry_scene_from_logger_bundle(
            {
                "meter_grid_profile": _sample_meter_profile(),
                "angular_seismic_profile": _sample_angular_profile(),
                "state_focus_profile": _sample_focus_profile(),
                "element_creation_profile": _sample_element_profile(),
            },
            run_id="geometry_frame_test",
        )

        frame = render_geometry_overlay_frame(scene, frame_index=0, total_frames=30, width=320, height=180)

        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(frame.mean(), 3.0)


if __name__ == "__main__":
    unittest.main()
