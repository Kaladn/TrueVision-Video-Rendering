from __future__ import annotations

from typing import Any

from truevision_runtime.av_tools.av_tool_receipts import stable_hash


TRUDEPTH_LAW_SCHEMA_VERSION = "truevision_trudepth_law_v1"
VOLUMETRIC_STATE_FIELD_SCHEMA_VERSION = "truevision_volumetric_state_field_contract_v1"
EFFECT_STATE_PROFILE_SCHEMA_VERSION = "truevision_effect_state_profile_contract_v1"
EFFECT_STATE_TRANSFORM_SCHEMA_VERSION = "truevision_effect_state_transform_contract_v1"
TRUDEPTH_LOGGING_ARRAY_SCHEMA_VERSION = "truevision_trudepth_logging_array_contract_v1"
TRUDEPTH_VALIDATION_SCHEMA_VERSION = "truevision_trudepth_validation_contract_v1"
TRUDEPTH_BUNDLE_SCHEMA_VERSION = "truevision_trudepth_contract_bundle_v1"


TRUDEPTH_LAWS = [
    "Copy behavior, not pixels.",
    "Transform state, not identity.",
    "Validate before render.",
    "Depth is a logged relationship between density, occlusion, reveal, motion, light, and layer separation.",
    "Fog is an effect, not the story.",
    "No meter, no claim.",
]


VOLUMETRIC_STATE_CHANNELS = [
    "density_slice",
    "depth_layer",
    "occlusion_pressure",
    "light_scatter",
    "reveal_rate",
    "edge_recovery",
    "motion_parallax",
]


EFFECT_PROFILE_METERS = [
    "density_over_depth",
    "edge_recovery_rate",
    "contrast_rise",
    "texture_birth",
    "bloom_bleed",
    "parallax_speed",
    "reveal_distance",
]


EFFECT_TRANSFORM_OPERATORS = [
    "rotate_direction",
    "deepen_density",
    "invert_depth_bias",
    "compress_reveal_window",
    "expand_reveal_window",
    "redirect_motion",
    "reweight_near_mid_far",
    "change_light_source",
]


TRUDEPTH_LOGGING_ARRAY_FIELDS = [
    "schema_version",
    "frame_index",
    "time_sec",
    "cell_x",
    "cell_y",
    "cell_id",
    "source_profile_id",
    "effect_type",
    "transform_id",
    "density_slice_near",
    "density_slice_mid",
    "density_slice_far",
    "density_delta",
    "depth_layer",
    "depth_confidence",
    "occlusion_pressure",
    "light_scatter",
    "bloom_bleed",
    "reveal_rate",
    "edge_recovery",
    "contrast_recovery",
    "texture_birth",
    "motion_parallax",
    "parallax_direction_16",
    "angular_energy_16",
    "softness",
    "persistence_frames",
    "validation_flags",
]


FOG_BELONGS_RULES = [
    "soften_edges",
    "reduce_contrast_with_distance",
    "reveal_nearer_objects_first",
    "drift_slowly",
    "occlude_without_hard_borders",
]


LIGHTNING_BELONGS_RULES = [
    "spike_fast",
    "bloom_outward",
    "lift_surrounding_exposure",
    "decay_quickly",
    "leave_short_afterglow",
]


OCEAN_BELONGS_RULES = [
    "persist_as_mass",
    "move_in_bands",
    "shimmer_locally",
    "keep_horizon_or_plane_behavior",
]


LASER_BELONGS_RULES = [
    "stay_collimated",
    "bloom_through_haze",
    "sweep_with_continuous_direction",
    "remain_high_saturation",
    "leave_short_persistence",
    "reflect_on_wet_or_glossy_surfaces",
]


def _rules_for_effect(effect_type: str) -> list[str]:
    normalized = effect_type.strip().lower()
    if normalized == "lightning":
        return LIGHTNING_BELONGS_RULES
    if normalized == "ocean":
        return OCEAN_BELONGS_RULES
    if normalized in {"laser", "laser_show", "rave_laser"}:
        return LASER_BELONGS_RULES
    return FOG_BELONGS_RULES


def build_trudepth_law() -> dict[str, Any]:
    return {
        "schema_version": TRUDEPTH_LAW_SCHEMA_VERSION,
        "name": "TruDepth Law",
        "laws": TRUDEPTH_LAWS,
        "plain_language": (
            "Meters teach the effect. ARC transforms the effect. "
            "The renderer speaks the transformed state."
        ),
        "hard_boundary": {
            "copy_source_pixels": False,
            "copy_source_frames": False,
            "copy_behavior_profile": True,
            "generated_media_is_evidence": False,
        },
    }


def build_volumetric_state_field_contract(effect_type: str = "fog") -> dict[str, Any]:
    return {
        "schema_version": VOLUMETRIC_STATE_FIELD_SCHEMA_VERSION,
        "effect_type": effect_type,
        "primitive_name": "Volumetric State Field",
        "channels": VOLUMETRIC_STATE_CHANNELS,
        "layer_model": {
            "required_depth_layers": ["near", "mid", "far"],
            "per_layer_density_required": True,
            "foreground_midground_background_separation_required": True,
        },
        "renderer_contract": {
            "input": "effect_state_profile",
            "output": "renderable_volumetric_state",
            "pixels_last": True,
        },
    }


def build_effect_state_profile_contract(effect_type: str = "fog") -> dict[str, Any]:
    return {
        "schema_version": EFFECT_STATE_PROFILE_SCHEMA_VERSION,
        "effect_type": effect_type,
        "profile_name": f"{effect_type}_effect_state_profile",
        "meters": EFFECT_PROFILE_METERS,
        "source_policy": {
            "teacher_frames_retained": False,
            "teacher_video_retained": False,
            "compact_behavior_signature_retained": True,
            "source_hash_required": True,
        },
        "required_receipt_fields": [
            "source_profile_id",
            "source_hash",
            "meter_profile_hash",
            "teacher_chunks_purged",
            "operator_approval_state",
        ],
    }


def build_effect_state_transform_contract(effect_type: str = "fog") -> dict[str, Any]:
    return {
        "schema_version": EFFECT_STATE_TRANSFORM_SCHEMA_VERSION,
        "effect_type": effect_type,
        "operators": EFFECT_TRANSFORM_OPERATORS,
        "input": "effect_state_profile",
        "output": "transformed_effect_state",
        "transformable_controls": [
            "direction",
            "density",
            "depth_bias",
            "reveal_window",
            "motion_vector",
            "light_source",
            "near_mid_far_weight",
        ],
        "forbidden_controls": [
            "source_pixel_copy",
            "source_frame_copy",
            "identity_promotion_without_validation",
        ],
    }


def build_trudepth_logging_array_contract(effect_type: str = "fog") -> dict[str, Any]:
    return {
        "schema_version": TRUDEPTH_LOGGING_ARRAY_SCHEMA_VERSION,
        "effect_type": effect_type,
        "name": "TruDepth Logging Array",
        "grain": "per_frame_per_cell",
        "fields": TRUDEPTH_LOGGING_ARRAY_FIELDS,
        "compact_summary_outputs": [
            "per_region_depth_summary",
            "effect_event_profile",
            "transform_profile",
            "validation_receipt",
        ],
        "retention": {
            "keep_raw_cell_array_by_default": False,
            "keep_compact_summaries": True,
            "keep_validation_receipts": True,
        },
    }


def build_trudepth_validation_contract(effect_type: str = "fog") -> dict[str, Any]:
    return {
        "schema_version": TRUDEPTH_VALIDATION_SCHEMA_VERSION,
        "effect_type": effect_type,
        "belongs_rules": _rules_for_effect(effect_type),
        "pass_conditions": [
            "source_profile_hash_present",
            "transform_profile_hash_present",
            "no_source_frames_used",
            "belongs_rules_checked",
            "receipt_written",
        ],
        "failure_policy": {
            "render_rule_promotion_allowed": False,
            "keep_output_as_proof": False,
            "human_review_required": True,
        },
    }


def build_trudepth_contract_bundle(effect_type: str = "fog") -> dict[str, Any]:
    bundle = {
        "schema_version": TRUDEPTH_BUNDLE_SCHEMA_VERSION,
        "effect_type": effect_type,
        "law": build_trudepth_law(),
        "volumetric_state_field": build_volumetric_state_field_contract(effect_type),
        "effect_state_profile": build_effect_state_profile_contract(effect_type),
        "effect_state_transform": build_effect_state_transform_contract(effect_type),
        "logging_array": build_trudepth_logging_array_contract(effect_type),
        "validation": build_trudepth_validation_contract(effect_type),
    }
    bundle["bundle_hash"] = stable_hash(bundle)
    return bundle
