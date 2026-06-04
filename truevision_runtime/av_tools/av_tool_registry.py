from __future__ import annotations

from dataclasses import asdict, dataclass

from truevision_runtime.state_language import build_state_language


@dataclass(frozen=True)
class AVToolSpec:
    name: str
    purpose: str
    phase: int
    priority: str
    output: str
    approval_required: bool = False
    domain: str = "audio_video"
    behavior_family: str = "unclassified"
    can_witness: bool = False
    can_profile: bool = False
    can_plan: bool = False
    can_replay: bool = False
    can_surface: bool = False
    raw_media_saved: bool = False
    copies_source_media: bool = False


AV_TOOL_LANGUAGE_OVERRIDES: dict[str, dict[str, object]] = {
    "audio_probe_duration": {"behavior_family": "audio_state", "can_witness": True, "can_profile": True},
    "audio_analyze_levels": {"behavior_family": "audio_state", "can_witness": True, "can_profile": True},
    "audio_extract_features": {"behavior_family": "audio_state", "can_witness": True, "can_profile": True},
    "trueaudio_log_pre_sound": {"behavior_family": "audio_state", "can_witness": True, "can_profile": True},
    "trueaudio_log_machine_pre_sound": {"behavior_family": "audio_state", "can_witness": True, "can_profile": True},
    "trueaudio_log_file_replayable": {"behavior_family": "audio_state", "can_witness": True, "can_profile": True},
    "trueaudio_log_machine_replayable": {"behavior_family": "audio_state", "can_witness": True, "can_profile": True},
    "trueaudio_replay_state": {"behavior_family": "audio_state", "can_plan": True, "can_replay": True, "can_surface": True},
    "trueaudio_replay_replayable": {"behavior_family": "audio_state", "can_plan": True, "can_replay": True, "can_surface": True},
    "truespeech_detect_segments": {"behavior_family": "speech_state", "can_witness": True, "can_profile": True},
    "truespeech_align_lyrics_candidate": {"behavior_family": "speech_state", "can_witness": True, "can_profile": True},
    "template_from_audio_signals": {"behavior_family": "audio_state", "can_profile": True, "can_plan": True},
    "template_create": {"behavior_family": "state_planning_surface", "can_plan": True},
    "template_load": {"behavior_family": "state_planning_surface", "can_plan": True},
    "template_save": {"behavior_family": "state_planning_surface", "can_plan": True},
    "template_patch": {"behavior_family": "state_planning_surface", "can_plan": True},
    "template_create_variant": {"behavior_family": "state_planning_surface", "can_plan": True},
    "template_delete": {"behavior_family": "state_planning_surface", "can_plan": True},
    "time_marker_add": {"behavior_family": "timing_curve", "can_plan": True},
    "time_marker_list": {"behavior_family": "timing_curve", "can_profile": True},
    "recalibration_add_note": {"behavior_family": "state_compare_adjust", "can_profile": True, "can_plan": True},
    "recalibration_apply": {"behavior_family": "state_compare_adjust", "can_profile": True, "can_plan": True},
    "video_render_preview": {"behavior_family": "pixel_state_transform", "can_plan": True, "can_replay": True, "can_surface": True},
    "video_prepare_full_render": {"behavior_family": "pixel_state_transform", "can_plan": True, "can_replay": True, "can_surface": True},
    "video_execute_full_render": {"behavior_family": "pixel_state_transform", "can_plan": True, "can_replay": True, "can_surface": True},
    "manifest_generate": {"behavior_family": "state_contract", "can_plan": True},
    "receipt_create": {"behavior_family": "state_contract"},
    "learning_record_save": {"behavior_family": "state_compare_adjust", "can_profile": True},
    "storage_list_artifacts": {"behavior_family": "state_library", "can_profile": True},
    "storage_list_templates": {"behavior_family": "state_library", "can_profile": True},
    "source_snap_tool": {"behavior_family": "photo_state_transform", "can_witness": True, "can_profile": True, "can_plan": True},
    "existing_state_animator": {"behavior_family": "pixel_state_transform", "can_plan": True, "can_replay": True},
    "electric_glow_intensity_animator": {"behavior_family": "light_pressure", "can_plan": True, "can_replay": True},
    "spectrum_audio_reactive_city": {"behavior_family": "audio_state", "can_plan": True, "can_replay": True, "can_surface": True},
    "atmosphere_profile_from_capture": {"behavior_family": "fog_reveal", "can_witness": True, "can_profile": True},
    "atmosphere_toolset_create": {"behavior_family": "fog_reveal", "can_profile": True, "can_plan": True},
    "source_surface_capture_plan": {"behavior_family": "state_witness", "can_plan": True},
    "source_surface_multi_sample_plan": {"behavior_family": "state_witness", "can_plan": True},
    "source_surface_video_state_receipt": {"behavior_family": "state_witness", "can_profile": True},
    "element_creation_profile_from_capture": {"behavior_family": "transform_operator_profile", "can_witness": True, "can_profile": True},
    "meter_grid_from_capture": {"behavior_family": "meter_grid", "can_witness": True, "can_profile": True},
    "frame_diff_replay_accuracy": {"behavior_family": "state_compare_adjust", "can_profile": True},
    "manifest_browser": {"behavior_family": "state_library", "can_profile": True},
    "render_preset_library": {"behavior_family": "state_planning_surface", "can_plan": True},
}


AV_TOOLS: tuple[AVToolSpec, ...] = (
    AVToolSpec("audio_probe_duration", "Read exact media duration with local ffprobe.", 1, "high", "audio metadata"),
    AVToolSpec("audio_analyze_levels", "Use ffmpeg PCM decode to extract levels, peaks, valleys, and section energy.", 1, "high", "audio signal JSON"),
    AVToolSpec("audio_extract_features", "Extract WAV audio feature timeline for AV state templates.", 1, "high", "feature timeline"),
    AVToolSpec("trueaudio_log_pre_sound", "Log derived TrueAudio state from decoded PCM before playback/output.", 1, "high", "state JSONL + manifest + receipt"),
    AVToolSpec("trueaudio_log_machine_pre_sound", "Log derived TrueAudio state from the local machine output mix before speakers.", 1, "high", "state JSONL + manifest + receipt"),
    AVToolSpec("trueaudio_replay_state", "Replay TrueAudio state logs as bounded sonification, not recovered source audio.", 1, "medium", "WAV + manifest + receipt"),
    AVToolSpec("trueaudio_log_file_replayable", "Log replayable derived TrueAudio spectral state from a source audio file.", 1, "high", "replayable NPZ + manifest + receipt"),
    AVToolSpec("trueaudio_log_machine_replayable", "Log replayable derived TrueAudio spectral state from the machine output mix.", 1, "high", "replayable NPZ + manifest + receipt"),
    AVToolSpec("trueaudio_replay_replayable", "Replay close audio from replayable TrueAudio spectral state.", 1, "high", "WAV + manifest + receipt"),
    AVToolSpec("truespeech_detect_segments", "Detect speech/background segments from replayable TrueAudio state without transcript claims.", 1, "high", "speech segment JSON + manifest + receipt"),
    AVToolSpec("truespeech_align_lyrics_candidate", "Align provided lyrics to detected speech segments as candidate timing, not ASR truth.", 1, "high", "lyric alignment JSON + manifest + receipt"),
    AVToolSpec("template_from_audio_signals", "Create an AV template from analyzed audio signal events and state patterns.", 1, "high", "template JSON"),
    AVToolSpec("template_create", "Create an AV state template from prompt and timing inputs.", 1, "high", "template JSON"),
    AVToolSpec("template_load", "Load one flat template JSON file.", 1, "high", "template object"),
    AVToolSpec("template_save", "Save one flat template JSON file.", 1, "high", "saved template"),
    AVToolSpec("template_patch", "Patch a validated template path for recalibration.", 1, "high", "patched template"),
    AVToolSpec("template_create_variant", "Create a named variant from a source template and changes.", 1, "high", "new template"),
    AVToolSpec("template_delete", "Delete one flat template file after confirmation.", 1, "medium", "delete receipt", True),
    AVToolSpec("time_marker_add", "Add a time-based AV control marker.", 1, "high", "marker event"),
    AVToolSpec("time_marker_list", "List time markers for a template.", 1, "medium", "marker list"),
    AVToolSpec("recalibration_add_note", "Add human feedback for an AV artifact at a time marker.", 1, "high", "recalibration event"),
    AVToolSpec("recalibration_apply", "Prepare a template patch from recalibration notes.", 1, "high", "patch proposal"),
    AVToolSpec("video_render_preview", "Prepare a low-duration proof-surface job manifest.", 1, "high", "preview manifest"),
    AVToolSpec("video_prepare_full_render", "Prepare a full proof-surface job manifest.", 1, "high", "surface job manifest"),
    AVToolSpec("video_execute_full_render", "Execute an approved proof-surface job.", 1, "high", "final proof surface", True),
    AVToolSpec("manifest_generate", "Generate a deterministic manifest for an AV object.", 1, "high", "manifest"),
    AVToolSpec("receipt_create", "Create a tool receipt; normally internal.", 1, "high", "receipt"),
    AVToolSpec("learning_record_save", "Save AV render success/failure learning.", 1, "medium", "learning record"),
    AVToolSpec("storage_list_artifacts", "List media artifacts from AV storage lanes only.", 1, "medium", "media listing"),
    AVToolSpec("storage_list_templates", "List saved AV templates.", 1, "medium", "template listing"),
    AVToolSpec("source_snap_tool", "Prepare still/video source-state snap packets for witness/profile/plan reference.", 1, "high", "source snap plan"),
    AVToolSpec("existing_state_animator", "Animate only existing source-state regions without adding new composition.", 1, "high", "state animation plan"),
    AVToolSpec("electric_glow_intensity_animator", "Pulse existing electric/glow regions by intensity only.", 1, "high", "glow intensity plan"),
    AVToolSpec("spectrum_audio_reactive_city", "Create an audio-reactive city/spectrum preset or template.", 1, "high", "city spectrum preset"),
    AVToolSpec("atmosphere_profile_from_capture", "Extract fog/mist/cloud/rain-on-glass state profiles and 6-1-6 windows from native TrueVision capture state.", 1, "high", "atmosphere profile JSON"),
    AVToolSpec("atmosphere_toolset_create", "Create reusable atmosphere/weather state tools for fog, mist, clouds, and rain on glass.", 1, "high", "toolset template + manifest"),
    AVToolSpec("source_surface_capture_plan", "Prepare a deterministic approved-source capture plan that starts recording before play and stops from source video time.", 1, "high", "capture plan + receipt"),
    AVToolSpec("source_surface_multi_sample_plan", "Prepare four-section sampling plans for long approved videos so intake learns multiple sections without hoarding source state.", 1, "high", "multi-sample plan + receipt"),
    AVToolSpec("source_surface_video_state_receipt", "Verify an approved source really loaded video state before a capture can count as complete.", 1, "high", "verified video-state receipt"),
    AVToolSpec("element_creation_profile_from_capture", "Convert temporary teacher capture state into a compact creation signature, then optionally purge bulky observed state.", 1, "high", "creation profile + purge receipt"),
    AVToolSpec("meter_grid_from_capture", "Extract measured cell meters, event profiles, and tuning graphs from native TrueVision capture state.", 1, "high", "meter profile + event graphs + receipt"),
    AVToolSpec("frame_diff_replay_accuracy", "Compare source and regen manifests or state artifacts for replay drift.", 1, "medium", "accuracy report"),
    AVToolSpec("manifest_browser", "Browse AV render/capture manifests.", 1, "medium", "manifest listing"),
    AVToolSpec("render_preset_library", "List, load, save, and promote reusable render presets.", 1, "high", "preset library result"),
)


def get_av_tool(name: str) -> AVToolSpec | None:
    return {tool.name: tool for tool in AV_TOOLS}.get(name)


def list_av_tools() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tool in AV_TOOLS:
        row = asdict(tool)
        row.update(AV_TOOL_LANGUAGE_OVERRIDES.get(tool.name, {}))
        row["state_language"] = build_state_language(row)
        rows.append(row)
    return rows
