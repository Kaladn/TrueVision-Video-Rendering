from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StudioTool:
    tool_id: str
    label: str
    purpose: str
    writes: tuple[str, ...]
    runtime: str
    status: str = "available"


STUDIO_TOOLS: tuple[StudioTool, ...] = (
    StudioTool(
        "source_snap_tool",
        "Source Snap Tool",
        "Create still/video source-state packets for record, regen, or generation reference.",
        ("storage/artifacts", "storage/manifests"),
        "python_or_rust_capture",
    ),
    StudioTool(
        "existing_state_animator",
        "Existing-State Animator",
        "Animate only detected source-state regions without adding new composition.",
        ("storage/templates", "storage/manifests"),
        "rust_renderer_plan",
    ),
    StudioTool(
        "electric_glow_intensity_animator",
        "Electric/Glow Intensity Animator",
        "Pulse existing lightning, glow, halos, tower energy, and waveform traces by intensity only.",
        ("storage/templates", "storage/manifests"),
        "rust_renderer_plan",
    ),
    StudioTool(
        "spectrum_audio_reactive_city",
        "Spectrum/Audio-Reactive City Tool",
        "Map audio bands and beat pressure to bottom-up city windows, skyline glow, fog, and frame pressure.",
        ("storage/templates", "storage/manifests"),
        "rust_renderer_plan",
    ),
    StudioTool(
        "frame_diff_replay_accuracy",
        "Frame Diff / Replay Accuracy Tool",
        "Compare source and regenerated frames or manifests for replay drift and visual-state accuracy.",
        ("storage/reports", "storage/manifests"),
        "analysis_tool",
    ),
    StudioTool(
        "manifest_browser",
        "Manifest Browser",
        "Browse render/capture manifests without opening runtime artifact folders by hand.",
        ("storage/manifests",),
        "studio_index",
    ),
    StudioTool(
        "render_preset_library",
        "Render Preset Library",
        "List, load, save, and promote successful render lanes into reusable presets.",
        ("storage/presets", "storage/templates"),
        "studio_index",
    ),
    StudioTool(
        "local_qwen_controller",
        "Local Qwen Controller",
        "Keep Qwen as a validated AV-state planner that requests tools instead of executing directly.",
        ("storage/chats", "storage/outbox", "storage/receipts"),
        "loopback_llm_adapter",
    ),
)


BUILTIN_RENDER_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "preset_id": "state_presentation_v3_boardroom",
        "name": "State Presentation V3 Gothic Industrial",
        "renderer": "edge_headless_panel_export",
        "scene_mode": "boardroom_systems_panels",
        "visual_mode": "gothic_industrial_systems_panels",
        "status": "ready",
        "purpose": "Dark gothic-industrial systems-presentation panel deck with architecture diagrams, flow control, trust boundaries, proof metrics, and traceable-state explanation.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "runtime_defaults": {
            "panel_surface": "ui/state_presentation_boardroom.html",
            "exporter": "scripts/render_state_presentation_boardroom_panels.py",
            "raster_outputs": ["png", "bmp"],
            "video_encoder": "h264_qsv",
            "fallback_encoder": "libx264",
        },
        "panel_types": [
            "title_reveal",
            "problem_comparison",
            "architecture_overview",
            "execution_pipeline",
            "forward_reverse_boundary",
            "frame_generation_bridge",
            "log_memory",
            "trust_boundary",
            "proof_metrics",
            "system_shape",
            "not_this",
            "closing",
        ],
        "presentation_outline": {
            "slide_count": 12,
            "slides": [
                {
                    "title": "TrueVision - When Media Becomes State",
                    "body": "State-native media for machine understanding.",
                },
                {
                    "title": "The Problem",
                    "body": "Most media AI stores pixels, guesses frames, and hides process. This blurs observed, reconstructed, and generated boundaries.",
                },
                {
                    "title": "The Core Rule",
                    "body": "Record state. Plan state. Transform state. Render pixels last.",
                },
                {
                    "title": "What TrueVision Does",
                    "body": "Observed audio/video becomes structured state: audio features, grid/cell fields, temporal transitions, manifests, receipts, and frame-state logs.",
                },
                {
                    "title": "Forward / Reverse",
                    "body": "Forward records observed state. Reverse replays, regenerates, or demonstrates state. Generated media is synthetic state media, not evidence.",
                },
                {
                    "title": "Frame Generation",
                    "body": "Known State A becomes a transition field, midpoint state, recursive subdivision, and smooth playback.",
                },
                {
                    "title": "Why Logs Matter",
                    "body": "Structured logs become recoverable state memory instead of residue.",
                },
                {
                    "title": "The Trust Boundary",
                    "body": "TrueVision separates observed, reconstructed, and generated state; each run leaves receipts, manifests, reports, and state records.",
                },
                {
                    "title": "Current Proof",
                    "body": "Native full-song lane: 232.88 seconds, 6,986 frames/state records, 1280x720, 30 FPS, 32 render threads, about 1.5x realtime.",
                },
                {
                    "title": "System Shape",
                    "body": "Human direction/audio to Studio/CLI, state draft, schema validator, AV policy, renderer, FFmpeg, MP4, manifest, frame-state JSONL, and report.",
                },
                {
                    "title": "What This Is Not",
                    "body": "Not cloud video generation, prompt magic, forensic proof software, raw video storage, or uncontrolled model output.",
                },
                {
                    "title": "Closing",
                    "body": "The future of machine media is not opaque. The future is traceable state.",
                },
            ],
        },
        "state_layers": [
            "presentation_panel_raster",
            "architecture_overview",
            "execution_pipeline",
            "trust_boundary",
            "proof_metrics",
            "state_packet_flow",
            "manifest_receipt_surface",
        ],
        "credits": [
            "Lee Mercey Architect Engineer Lead Engineer",
            "OpenAI",
            "OpenAI Codex",
            "OpenAI Codex Workspace Agent",
            "Rust",
            "Python",
            "FFmpeg",
            "Microsoft Edge Headless",
            "Local Windows Fonts",
        ],
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "no_external_visual_assets": True,
            "panel_export_surface": True,
            "audio_video_only": True,
            "state_fields_first_pixels_last": True,
        },
    },
    {
        "preset_id": "state_presentation_truevision_labs",
        "name": "State Presentation By TrueVision Labs",
        "renderer": "truevision_weird_occlusion_rs",
        "scene_mode": "state_presentation",
        "visual_mode": "state_presentation",
        "status": "ready",
        "purpose": "Calm production systems reveal explaining TrueVision state media, validated packets, harness boundaries, manifests, receipts, credits, and synthetic-media limits.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "runtime_defaults": {
            "render_threads": 32,
            "video_encoder": "h264_qsv",
            "bitrate": "18M",
            "fallback_encoder": "libx264",
        },
        "audio_mapping": {
            "voice": ["caption_timing", "state_packet_pulse", "credit_reveal"],
            "rms": ["state_field_breathing", "grid_wake", "manifest_glow"],
            "bass": ["single_pulse", "temporal_bridge_weight"],
            "highs": ["particle_trace", "receipt_hash_shimmer"],
            "beat": ["section_transition_pressure"],
        },
        "presentation_outline": {
            "slide_count": 12,
            "slides": [
                {
                    "title": "TrueVision - When Media Becomes State",
                    "body": "State-native media for machine understanding.",
                },
                {
                    "title": "The Problem",
                    "body": "Most media AI stores pixels, guesses frames, and hides process. The result is weak provenance and blurred boundaries between observed and synthetic.",
                },
                {
                    "title": "The Core Rule",
                    "body": "Record state. Plan state. Transform state. Render pixels last.",
                },
                {
                    "title": "What TrueVision Does",
                    "body": "TrueVision records observed audio/video as structured state: audio features, grid/cell fields, temporal transitions, manifests, receipts, and frame-state logs.",
                },
                {
                    "title": "Forward / Reverse",
                    "body": "Forward records observed state. Reverse replays, regenerates, or demonstrates state. Generated media is synthetic state media, not evidence.",
                },
                {
                    "title": "Frame Generation",
                    "body": "Known State A becomes a transition field, midpoint state, recursive subdivision, and smooth playback. One A-to-B bridge; many frames walk the bridge.",
                },
                {
                    "title": "Why Logs Matter",
                    "body": "Structured logs become recoverable state memory instead of residue.",
                },
                {
                    "title": "The Trust Boundary",
                    "body": "TrueVision separates observed, reconstructed, and generated state. Every run leaves receipts, manifests, reports, and state records.",
                },
                {
                    "title": "Current Proof",
                    "body": "Native full-song lane: 232.88 seconds, 6,986 frames/state records, 1280x720, 30 FPS, 32 render threads, about 1.5x realtime speed.",
                },
                {
                    "title": "System Shape",
                    "body": "Human direction/audio to Studio/CLI, state draft, schema validator, AV policy, renderer, FFmpeg, MP4, manifest, frame-state JSONL, and report.",
                },
                {
                    "title": "What This Is Not",
                    "body": "Not cloud video generation, prompt magic, forensic proof software, raw video storage, or uncontrolled model output.",
                },
                {
                    "title": "Closing",
                    "body": "The future of machine media is not opaque. The future is traceable state.",
                },
            ],
        },
        "state_layers": [
            "black_field_single_pulse",
            "state_cell_grid",
            "validated_state_packets",
            "aw_sc_truevision_harness_nodes",
            "temporal_bridge_ab",
            "manifest_receipts_hashes",
            "third_party_credits",
            "system_voice_narration",
        ],
        "credits": [
            "Lee Mercey Architect Engineer Lead Engineer",
            "OpenAI",
            "OpenAI Codex",
            "OpenAI Codex Workspace Agent",
            "Rust",
            "Python",
            "FFmpeg",
            "OpenCV",
            "NumPy",
            "MSS",
        ],
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "no_external_visual_assets": True,
            "rust_hot_path": True,
            "audio_video_only": True,
            "state_fields_first_pixels_last": True,
        },
    },
    {
        "preset_id": "fade_away_memory_cathedral",
        "name": "Fade Away Memory Cathedral",
        "renderer": "truevision_weird_occlusion_rs",
        "scene_mode": "memory_cathedral",
        "visual_mode": "ambient_electronic_memory_collapse",
        "status": "ready",
        "purpose": "Ambient melancholic dream lane: near-black memory space, soft doorway depth, central human absence, paired voice lights, inward particles, dream snap, and a final heart-like fade.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "runtime_defaults": {
            "render_threads": 32,
            "video_encoder": "h264_qsv",
            "bitrate": "24M",
            "fallback_encoder": "libx264",
        },
        "audio_mapping": {
            "rms": ["memory_breathing", "doorway_depth_bloom", "veil_density"],
            "bass": ["inward_depth_pressure", "heart_sink_warmth"],
            "highs": ["memory_particles", "dream_snap_silver_edges"],
            "beat": ["dream_snap_pulse", "collapse_ring_pressure"],
            "vocal_presence": ["central_absence_rim", "paired_voice_light_fields"],
        },
        "state_layers": [
            "near_black_blue_memory_field",
            "soft_doorway_depth_windows",
            "central_human_absence_not_portrait",
            "left_right_voice_light_fields",
            "inward_memory_particles",
            "dream_snap_collapse_gate",
            "outro_heart_sink",
            "no_city",
            "no_fire",
            "no_hard_lasers",
        ],
        "boundary": {
            "synthetic_state_media": True,
            "no_external_visual_assets": True,
            "rust_hot_path": True,
            "audio_video_only": True,
            "state_fields_first_pixels_last": True,
        },
    },
    {
        "preset_id": "glitch_444_alive_poster",
        "name": "Glitch 444 Alive Poster",
        "renderer": "truevision_weird_occlusion_rs",
        "scene_mode": "spectrum_backdrop",
        "visual_mode": "alive_poster_intensity",
        "status": "proven",
        "purpose": "Keep the source poster fixed while existing electric regions, waveform, radar, and analyzer panels react to audio.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "audio_mapping": {
            "waveform": "real_audio_signal_strip",
            "spectrum": "low_mid_high_facsimile_until_fft_upgrade",
            "bass": ["tower_core_glow", "lower_waveform_pressure"],
            "mids": ["circular_halo_pulse"],
            "highs": ["lightning_flicker", "small_electric_accents"],
        },
        "state_layers": [
            "full_backdrop_letterbox_fit",
            "existing_electric_intensity_only",
            "black_replacement_analyzer_box",
            "blue_gold_schema_analyzer",
            "black_replacement_waveform_box",
            "real_audio_waveform_panel",
            "approx_headphone_soundfield_radar",
            "truevision_prototype_mark",
        ],
        "boundary": {
            "no_new_art": True,
            "no_warping": True,
            "intensity_only": True,
        },
    },
    {
        "preset_id": "house_remix_audio_city",
        "name": "House Remix Audio City",
        "renderer": "truevision_weird_occlusion_rs",
        "scene_mode": "lyric_city",
        "visual_mode": "house_remix_city_glow",
        "status": "ready",
        "purpose": "Free-reign dance visual lane: city silhouettes, bottom-up spectrum windows, glow, fog breathing, and beat-synced frame pressure.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "audio_mapping": {
            "kick": ["bottom_up_window_bloom", "street_level_glow"],
            "bass": ["fog_breathing", "building_core_pressure"],
            "mids": ["skyline_color_swell", "camera_push"],
            "highs": ["sparkle_windows", "edge_shimmer"],
            "beat": ["frame_pressure", "smooth_flash_transition"],
        },
        "state_layers": [
            "night_city_silhouettes",
            "bottom_up_city_spectrum",
            "electric_intensity_pulse",
            "fog_breathing",
            "beat_synced_frame_pressure",
            "blue_gold_house_palette",
        ],
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "audio_video_only": True,
        },
    },
    {
        "preset_id": "abstract_symphony_soft_beams",
        "name": "Abstract Symphony Soft Beams",
        "renderer": "truevision_weird_occlusion_rs",
        "scene_mode": "abstract_symphony",
        "visual_mode": "abstract_soft_field_club_beams",
        "status": "needs_rework",
        "purpose": "Abstract house-beat visual lane using dark nightclub fields, soft volumetric beams, fog pressure, soundfield rings, waveform ribbons, and no hard-edged geometry.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "audio_mapping": {
            "bass": ["soft_beam_reach", "low_frequency_glow", "wet_reflection_pressure"],
            "rms": ["fog_density", "field_saturation", "waveform_ribbon_width"],
            "highs": ["aurora_flicker", "soft_sparkle_cloud", "electric_haze"],
            "beat": ["beam_pressure", "soundfield_ring_pulse", "gold_purple_bloom"],
        },
        "state_layers": [
            "nightclub_dark_background",
            "soft_volumetric_fog_field",
            "defined_soft_club_beams",
            "audio_waveform_ribbon",
            "electric_glow_pressure",
            "soundfield_rings",
            "wet_reflection_pressure",
            "no_hard_edges",
        ],
        "boundary": {
            "synthetic_state_media": True,
            "no_external_visual_assets": True,
            "no_hard_edges": True,
            "rust_hot_path": True,
            "audio_video_only": True,
        },
    },
    {
        "preset_id": "center_warp_laserfield",
        "name": "Center Warp Laserfield",
        "renderer": "truevision_weird_occlusion_rs",
        "scene_mode": "warp_laser_field",
        "visual_mode": "pure_black_center_radial_lasers",
        "status": "draft",
        "purpose": "Pure black audio-reactive warp field with center-origin lasers, radial star streaks, tunnel rings, and no fog or background wash.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "runtime_defaults": {
            "render_threads": 32,
            "video_encoder": "h264_qsv",
            "bitrate": "24M",
            "fallback_encoder": "h264_amf",
        },
        "audio_mapping": {
            "bass": ["center_core_size", "beam_travel_speed", "tunnel_depth"],
            "rms": ["beam_brightness", "starfield_velocity"],
            "highs": ["star_streak_shimmer", "laser_color_energy"],
            "beat": ["center_flash", "radial_beam_pressure", "tunnel_ring_pulse"],
        },
        "state_layers": [
            "pure_black_background",
            "center_origin_lasers",
            "radial_warp_starfield",
            "beat_pulse_core",
            "audio_reactive_beam_pressure",
            "no_fog",
        ],
        "boundary": {
            "synthetic_state_media": True,
            "no_external_visual_assets": True,
            "pure_black_background": True,
            "no_fog": True,
            "rust_hot_path": True,
            "audio_video_only": True,
        },
    },
    {
        "preset_id": "storm_ember_city",
        "name": "Storm Ember City",
        "renderer": "template_renderer",
        "visual_mode": "storm_ember_city",
        "status": "proven",
        "purpose": "Rain, ember, burning-city glow, wet pavement, and silhouette memory lane.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "state_layers": ["rain_streaks", "ember_ash_particles", "wet_pavement_reflections", "lonely_backlit_silhouette"],
        "boundary": {"no_external_visual_assets": True, "synthetic_state_media": True},
    },
    {
        "preset_id": "mirror_maze_realism",
        "name": "Mirror Maze Realism",
        "renderer": "template_renderer",
        "visual_mode": "mirror_maze_realism",
        "status": "proven",
        "purpose": "Reflective maze, density-field smoke, occlusion, and cinematic glow.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "state_layers": ["mirror_shards", "density_field_fog", "portal_glow", "wet_reflection"],
        "boundary": {"fog_uses_density_field": True, "synthetic_state_media": True},
    },
    {
        "preset_id": "edge_audio_river",
        "name": "Edge Audio River",
        "renderer": "edge_audio_river",
        "visual_mode": "audio_reactive_river",
        "status": "proven",
        "purpose": "Letterboxed black field with a thin river of color reacting to song dynamics.",
        "default_size": [1280, 720],
        "default_fps": 30,
        "state_layers": ["audio_level_river", "letterbox_bands", "beat_color_pressure"],
        "boundary": {"synthetic_state_media": True},
    },
)


def list_studio_tools() -> list[dict[str, Any]]:
    return [asdict(tool) for tool in STUDIO_TOOLS]


def list_builtin_render_presets() -> list[dict[str, Any]]:
    return [deepcopy(preset) for preset in BUILTIN_RENDER_PRESETS]


def _read_storage_presets(storage_root: Path | None) -> list[dict[str, Any]]:
    if storage_root is None:
        return []
    preset_root = storage_root / "presets"
    if not preset_root.exists():
        return []
    presets: list[dict[str, Any]] = []
    for path in sorted(preset_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload.setdefault("preset_id", path.stem)
            payload.setdefault("name", path.stem)
            payload["source"] = "storage"
            payload["path"] = str(path)
            presets.append(payload)
    return presets


def list_render_presets(storage_root: Path | None = None) -> list[dict[str, Any]]:
    builtins = list_builtin_render_presets()
    for preset in builtins:
        preset["source"] = "builtin"
    user_presets = _read_storage_presets(storage_root)
    known = {preset["preset_id"]: preset for preset in builtins}
    for preset in user_presets:
        known[str(preset["preset_id"])] = preset
    return sorted(known.values(), key=lambda item: (item.get("status") != "proven", str(item.get("name", ""))))


def get_render_preset(preset_id: str, storage_root: Path | None = None) -> dict[str, Any]:
    for preset in list_render_presets(storage_root):
        if preset.get("preset_id") == preset_id:
            return deepcopy(preset)
    raise KeyError(f"unknown render preset: {preset_id}")


def save_render_preset(storage_root: Path, preset: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(preset, dict):
        raise TypeError("preset must be an object")
    preset_id = str(preset.get("preset_id") or preset.get("name") or "render_preset")
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in preset_id).strip("_")
    if not safe:
        raise ValueError("preset_id is required")
    payload = deepcopy(preset)
    payload["preset_id"] = safe
    payload.setdefault("source", "storage")
    path = storage_root / "presets" / f"{safe}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return {"preset_id": safe, "path": str(path), "preset": payload}


def preset_to_template(
    preset: dict[str, Any],
    *,
    name: str | None = None,
    prompt: str = "",
    audio_path: str = "",
    duration_seconds: float | None = None,
    fps: int | None = None,
) -> dict[str, Any]:
    fps_value = int(fps or preset.get("default_fps") or 30)
    duration = float(duration_seconds or 60.0)
    frame_count = max(1, int(round(duration * fps_value)))
    return {
        "schema_version": 1,
        "name": name or str(preset.get("name") or "TrueVision render preset"),
        "renderer": str(preset.get("renderer") or "state_formula"),
        "visual_mode": str(preset.get("visual_mode") or preset.get("scene_mode") or "state_formula"),
        "prompt": prompt,
        "media": {
            "audio_path": audio_path,
            "audio_duration_seconds": duration_seconds,
            "sync_to_audio": bool(audio_path),
        },
        "timeline": {
            "duration_seconds": round(duration, 6),
            "fps": fps_value,
            "frame_count": frame_count,
            "start_seconds": 0,
            "end_seconds": round(duration, 6),
        },
        "time_distance": {
            "source": "render_preset_library",
            "seconds_per_frame": round(1 / fps_value, 9),
            "frames_per_second": fps_value,
            "total_frames": frame_count,
        },
        "visual_parameters": {
            "preset_id": preset.get("preset_id"),
            "scene_mode": preset.get("scene_mode"),
            "audio_mapping": deepcopy(preset.get("audio_mapping") or {}),
            "state_layers": deepcopy(preset.get("state_layers") or []),
        },
        "state_plan": {
            "preset": deepcopy(preset),
            "tool_route": "render_preset_library",
        },
        "boundary": {
            "synthetic_state_media": True,
            "evidence": False,
            "renderer_executes_validated_state": True,
            **deepcopy(preset.get("boundary") or {}),
        },
    }


def build_studio_tool_plan(tool_id: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    args = args or {}
    tool_ids = {tool.tool_id for tool in STUDIO_TOOLS}
    if tool_id not in tool_ids:
        raise KeyError(f"unknown studio tool: {tool_id}")
    if tool_id == "source_snap_tool":
        return {
            "tool_id": tool_id,
            "mode": str(args.get("mode") or "still_or_video_source_state"),
            "outputs": ["source_state_manifest", "optional_cell_state", "source_fingerprint"],
            "rule": "record observed state; do not claim generated evidence",
        }
    if tool_id == "existing_state_animator":
        return {
            "tool_id": tool_id,
            "mode": "state_intensity_or_displacement_only",
            "allowed_changes": ["brightness", "glow", "subtle local displacement", "opacity", "timing"],
            "forbidden_changes": ["new composition", "new symbols", "semantic redraw"],
        }
    if tool_id == "electric_glow_intensity_animator":
        return {
            "tool_id": tool_id,
            "detect_existing_regions": ["lightning", "halo", "tower_core", "waveform", "spectrum_panel"],
            "animate": ["brightness_pulse", "bloom_strength", "glow_radius"],
            "audio_lanes": ["bass", "mid", "high", "beat"],
        }
    if tool_id == "spectrum_audio_reactive_city":
        return get_render_preset("house_remix_audio_city")
    if tool_id == "frame_diff_replay_accuracy":
        return {
            "tool_id": tool_id,
            "metrics": ["hash_match", "duration_match", "frame_count_match", "size_delta", "manifest_diff"],
            "note": "pixel or state-channel diff is used when source frames/state are supplied",
        }
    if tool_id == "manifest_browser":
        return {"tool_id": tool_id, "lanes": ["storage/manifests", "outputs/*/*_manifest.json"], "mode": "read_only_index"}
    if tool_id == "render_preset_library":
        return {"tool_id": tool_id, "presets": list_render_presets(), "mode": "list_load_save_promote"}
    if tool_id == "local_qwen_controller":
        return {
            "tool_id": tool_id,
            "role": "planner_only",
            "endpoint_policy": "loopback_only",
            "trust_boundary": "validated_state_json_and_av_tool_receipts",
        }
    raise AssertionError(tool_id)
