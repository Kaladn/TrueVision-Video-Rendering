import tempfile
import unittest
from pathlib import Path

from truevision_runtime.av_tools.av_tool_registry import list_av_tools
from truevision_runtime.av_tools.av_tool_runner import run_av_tool_call
from truevision_runtime.studio.studio_tooling import (
    get_render_preset,
    list_render_presets,
    list_studio_tools,
    preset_to_template,
)


class TrueVisionStudioToolingTests(unittest.TestCase):
    def test_studio_tools_cover_requested_control_surface(self):
        tools = {tool["tool_id"]: tool for tool in list_studio_tools()}

        self.assertIn("source_snap_tool", tools)
        self.assertIn("existing_state_animator", tools)
        self.assertIn("electric_glow_intensity_animator", tools)
        self.assertIn("spectrum_audio_reactive_city", tools)
        self.assertIn("frame_diff_replay_accuracy", tools)
        self.assertIn("manifest_browser", tools)
        self.assertIn("render_preset_library", tools)

    def test_av_registry_exposes_studio_tools_as_audio_video_only(self):
        tools = {tool["name"]: tool for tool in list_av_tools()}

        for name in [
            "source_snap_tool",
            "existing_state_animator",
            "electric_glow_intensity_animator",
            "spectrum_audio_reactive_city",
            "frame_diff_replay_accuracy",
            "manifest_browser",
            "render_preset_library",
        ]:
            self.assertIn(name, tools)
            self.assertEqual(tools[name]["domain"], "audio_video")

    def test_house_remix_preset_is_promotable_to_template(self):
        preset = get_render_preset("house_remix_audio_city")
        template = preset_to_template(
            preset,
            name="House Remix Test",
            prompt="dance city visual",
            audio_path="D:/music/house.wav",
            duration_seconds=12.5,
            fps=30,
        )

        self.assertEqual(template["renderer"], "truevision_weird_occlusion_rs")
        self.assertEqual(template["visual_mode"], "house_remix_city_glow")
        self.assertEqual(template["timeline"]["frame_count"], 375)
        self.assertIn("bottom_up_city_spectrum", template["visual_parameters"]["state_layers"])
        self.assertFalse(template["boundary"]["evidence"])

    def test_dead_memory_vice_chamber_preset_is_registered_for_native_render(self):
        preset = get_render_preset("dead_memory_vice_chamber")
        template = preset_to_template(
            preset,
            name="Dead Memory Vice Chamber Test",
            prompt="heartbreak and deception as mechanical pressure",
            audio_path="D:/music/dead_memory_bleed.mp4",
            duration_seconds=239.616,
            fps=30,
        )

        self.assertEqual(template["renderer"], "truevision_weird_occlusion_rs")
        self.assertEqual(template["visual_mode"], "dead_memory_vice_chamber")
        self.assertEqual(template["timeline"]["frame_count"], 7188)
        self.assertIn("black_iron_vice_jaws", template["visual_parameters"]["state_layers"])
        self.assertIn("thin_gold_white_survival_fracture", template["visual_parameters"]["state_layers"])
        self.assertTrue(template["boundary"]["no_literal_gore"])
        self.assertTrue(template["boundary"]["no_cartoon_devil"])

    def test_daughter_star_locket_sea_preset_is_registered_for_native_render(self):
        preset = get_render_preset("daughter_star_locket_sea")
        template = preset_to_template(
            preset,
            name="What Did They Daughter Star Test",
            prompt="heartbroken father to daughter as star, cracked heart locket, night water, and hope",
            audio_path="D:/music/what_did_they.mp4",
            duration_seconds=240.0,
            fps=30,
        )

        self.assertEqual(template["renderer"], "truevision_weird_occlusion_rs")
        self.assertEqual(template["visual_mode"], "daughter_star_locket_sea")
        self.assertEqual(template["timeline"]["frame_count"], 7200)
        self.assertIn("daughter_star_glow", template["visual_parameters"]["state_layers"])
        self.assertIn("cracked_father_heart_locket", template["visual_parameters"]["state_layers"])
        self.assertIn("perspective_depth_plane", template["visual_parameters"]["state_layers"])
        self.assertIn("dimensional_heart_locket_shading", template["visual_parameters"]["state_layers"])
        self.assertIn("controlled_roiling_fog_field", template["visual_parameters"]["state_layers"])
        self.assertIn("geometric_state_transform_switching", template["visual_parameters"]["state_layers"])
        self.assertIn("plane_depth_pulse", template["visual_parameters"]["state_layers"])
        self.assertIn("fade_shimmer_gate", template["visual_parameters"]["state_layers"])
        self.assertIn("soft_distortion_haze", template["visual_parameters"]["state_layers"])
        self.assertTrue(template["boundary"]["no_external_visual_assets"])
        self.assertTrue(template["boundary"]["no_literal_faces"])
        self.assertTrue(template["boundary"]["fog_is_effect_not_story"])
        self.assertTrue(template["boundary"]["arc_solver_operator_discipline"])

    def test_edge_nightmare_world_preset_is_registered_for_native_render(self):
        preset = get_render_preset("edge_nightmare_world")
        template = preset_to_template(
            preset,
            name="Edge Nightmare World Test",
            prompt="nightmare cliff world with full POV motion and human silhouettes",
            audio_path="D:/music/edge_nightmare.wav",
            duration_seconds=278.32,
            fps=60,
        )

        self.assertEqual(template["renderer"], "truevision_weird_occlusion_rs")
        self.assertEqual(template["visual_mode"], "edge_nightmare_world")
        self.assertEqual(template["timeline"]["frame_count"], 16699)
        self.assertIn("nightmare_cliff_rim", template["visual_parameters"]["state_layers"])
        self.assertIn("top_down_abyss_view", template["visual_parameters"]["state_layers"])
        self.assertIn("falling_camera_spiral", template["visual_parameters"]["state_layers"])
        self.assertIn("human_silhouette_motion", template["visual_parameters"]["state_layers"])
        self.assertIn("arc_learning_transform_mix", template["visual_parameters"]["state_layers"])
        self.assertTrue(template["boundary"]["full_motion_camera_pov"])
        self.assertTrue(template["boundary"]["arc_solver_operator_discipline"])

    def test_state_presentation_preset_and_native_lane_are_registered(self):
        preset = get_render_preset("state_presentation_truevision_labs")
        template = preset_to_template(
            preset,
            name="State Presentation Test",
            prompt="calm systems reveal",
            audio_path="D:/voice/state_presentation.wav",
            duration_seconds=240,
            fps=30,
        )

        self.assertEqual(template["renderer"], "truevision_weird_occlusion_rs")
        self.assertEqual(template["visual_mode"], "state_presentation")
        self.assertEqual(template["timeline"]["frame_count"], 7200)
        self.assertIn("validated_state_packets", template["visual_parameters"]["state_layers"])
        self.assertIn("OpenAI Codex Workspace Agent", preset["credits"])
        self.assertEqual(preset["presentation_outline"]["slide_count"], 12)
        self.assertIn("When Media Becomes State", preset["presentation_outline"]["slides"][0]["title"])
        self.assertIn("Record state", preset["presentation_outline"]["slides"][2]["body"])

        rust_source = Path("native/truevision_capture_rs/src/bin/truevision_weird_occlusion_rs.rs").read_text(
            encoding="utf-8"
        )
        self.assertIn("state_presentation", rust_source)
        self.assertIn("TRUEVISION LABS STATE PRESENTATION", rust_source)
        self.assertIn("WHEN MEDIA BECOMES STATE", rust_source)
        self.assertIn("RECORD STATE PLAN STATE TRANSFORM STATE", rust_source)

    def test_boardroom_presentation_preset_uses_gothic_industrial_panel_surface(self):
        preset = get_render_preset("state_presentation_v3_boardroom")

        self.assertEqual(preset["renderer"], "edge_headless_panel_export")
        self.assertEqual(preset["visual_mode"], "gothic_industrial_systems_panels")
        self.assertEqual(preset["presentation_outline"]["slide_count"], 12)
        self.assertIn("architecture_overview", preset["panel_types"])
        self.assertIn("proof_metrics", preset["panel_types"])
        self.assertIn("trust_boundary", preset["panel_types"])

        panel_html = Path("ui/state_presentation_boardroom.html").read_text(encoding="utf-8")
        self.assertIn("Architecture Overview", panel_html)
        self.assertIn("Known State A", panel_html)
        self.assertIn("232.88s", panel_html)
        self.assertIn("Observed", panel_html)
        self.assertIn("Generated", panel_html)

        exporter = Path("scripts/render_state_presentation_boardroom_panels.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("FINAL_RECEIPT_HOLD_SECONDS = 15.0", exporter)
        self.assertIn("TRANSITION_SECONDS = 0.45", exporter)
        self.assertIn("xfade=transition=fade", exporter)
        self.assertIn("edge_audio_river_smoke", exporter)
        self.assertIn("hide_backbone", exporter)

    def test_render_preset_library_tool_lists_and_promotes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            listed = run_av_tool_call(
                {"tool": "render_preset_library", "args": {"action": "list"}},
                storage_root=storage,
            )
            self.assertTrue(listed["ok"])
            preset_ids = {preset["preset_id"] for preset in listed["result"]["presets"]}
            self.assertIn("glitch_444_alive_poster", preset_ids)
            self.assertIn("house_remix_audio_city", preset_ids)

            promoted = run_av_tool_call(
                {
                    "tool": "render_preset_library",
                    "args": {
                        "action": "promote_to_template",
                        "preset_id": "house_remix_audio_city",
                        "template_name": "house_remix_audio_city.json",
                        "duration_seconds": 8,
                        "fps": 24,
                    },
                },
                storage_root=storage,
            )

            self.assertTrue(promoted["ok"])
            self.assertEqual(promoted["result"]["template"]["timeline"]["frame_count"], 192)
            self.assertTrue(Path(promoted["result"]["path"]).exists())

    def test_studio_plan_tools_write_receipts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_av_tool_call(
                {
                    "tool": "electric_glow_intensity_animator",
                    "args": {"audio_path": "D:/music/glitch.wav"},
                },
                storage_root=Path(tmpdir),
            )

            self.assertTrue(result["ok"])
            self.assertIn("detect_existing_regions", result["result"]["plan"])
            self.assertTrue(Path(result["receipt"]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
