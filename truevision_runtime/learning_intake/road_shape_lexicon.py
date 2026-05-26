from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from truevision_runtime.av_tools.av_tool_receipts import stable_hash, utc_now


LEXICON_SCHEMA_VERSION = "truevision_recognition_shape_lexicon_v1"
SYMBOL_SCHEMA_VERSION = "truevision_recognition_shape_symbol_v1"
BRIDGE_SCHEMA_VERSION = "truevision_aw_shape_bridge_v1"
MANIFEST_SCHEMA_VERSION = "truevision_recognition_shape_lexicon_manifest_v1"
RECEIPT_SCHEMA_VERSION = "truevision_recognition_shape_lexicon_receipt_v1"


ROAD_SHAPE_RECORDS = (
    ("circle", "primitive_shape", "round sign/light/reflector shape", ["traffic_light", "reflector", "wheel_hint"]),
    ("octagon", "primitive_shape", "eight-sided stop-sign-like geometry", ["stop_sign_shape"]),
    ("diamond", "primitive_shape", "warning-sign-like rotated square", ["warning_sign_shape"]),
    ("triangle", "primitive_shape", "yield/chevron/roofline geometry", ["yield_shape", "roofline"]),
    ("square", "primitive_shape", "equal-sided box geometry", ["sign_panel", "window"]),
    ("vertical_rectangle", "primitive_shape", "tall sign/building/mailbox face geometry", ["speed_limit_shape", "mailbox"]),
    ("horizontal_rectangle", "primitive_shape", "wide sign/vehicle/building face geometry", ["vehicle_box", "barn_box"]),
    ("line_segment", "primitive_shape", "single measured line segment", ["road_marking", "wire", "edge"]),
    ("vertical_line", "primitive_shape", "pole/post/trunk shape", ["utility_pole", "sign_post", "tree_trunk"]),
    ("horizontal_line", "primitive_shape", "horizon/guardrail/road edge line", ["horizon", "guardrail"]),
    ("diagonal_line", "primitive_shape", "roadside diagonal or chevron stroke", ["chevron", "crossbuck_part"]),
    ("parallel_lines", "compound_shape", "lane/road/guardrail paired line behavior", ["lane_lines", "road_edges"]),
    ("x_cross", "compound_shape", "railroad crossbuck or crossed structural shape", ["railroad_crossbuck"]),
    ("chevron_arrow", "compound_shape", "road curve chevron or directional arrow", ["curve_marker", "arrow_sign"]),
    ("circle_stack", "compound_shape", "stacked signal light circles", ["traffic_signal"]),
    ("lane_line", "road_shape", "painted lane or center road line", ["lane_line", "center_line"]),
    ("road_edge_line", "road_shape", "edge-of-road or shoulder boundary", ["road_edge", "shoulder"]),
    ("horizon_band", "scene_shape", "distant horizontal separation band", ["horizon", "skyline_base"]),
    ("utility_pole", "roadside_shape", "tall narrow pole-like vertical", ["utility_pole", "sign_post"]),
    ("power_wire_line", "roadside_shape", "overhead cable line", ["power_wire"]),
    ("mailbox_box", "roadside_shape", "small roadside rectangular box", ["mailbox"]),
    ("barn_box", "roadside_shape", "large rural building box", ["barn", "farm_building"]),
    ("vehicle_box", "roadside_shape", "moving or parked vehicle-like rectangle", ["vehicle_shape"]),
    ("tree_trunk_line", "natural_shape", "vertical trunk or tall grass stem", ["tree_trunk", "tall_grass"]),
    ("tree_canopy_mass", "natural_shape", "irregular upper tree mass", ["tree_mass"]),
    ("reflector_dot", "roadside_shape", "small bright circular reflector", ["reflector", "marker_light"]),
    ("guardrail_line", "roadside_shape", "long roadside barrier line", ["guardrail"]),
    ("bridge_rail_line", "roadside_shape", "parallel bridge/road rail structure", ["bridge_rail"]),
)


def _symbol_id(index: int) -> str:
    return f"TVG:{index:08d}"


def _binary_code(index: int) -> str:
    return format(index, "016b")


def _symbol_record(index: int, name: str, kind: str, human_label: str, contexts: list[str]) -> dict[str, Any]:
    record = {
        "schema_version": SYMBOL_SCHEMA_VERSION,
        "symbol_id": _symbol_id(index),
        "name": name,
        "domain": "high_speed_awareness_geometry",
        "kind": kind,
        "binary_code": _binary_code(index),
        "human_label": human_label,
        "allowed_contexts": contexts,
        "route_context": "northwest_ohio_country_road_to_monroe_michigan",
        "not_word_anchor": True,
        "promotion_status": "approved_shape_item",
        "truth_boundary": {
            "recognition_shape_only": True,
            "does_not_confirm_object_identity": True,
            "does_not_mutate_aw_lexicon": True,
        },
    }
    record["symbol_hash"] = stable_hash(record)
    return record


def build_road_shape_lexicon() -> dict[str, Any]:
    symbols = [
        _symbol_record(index, name, kind, label, list(contexts))
        for index, (name, kind, label, contexts) in enumerate(ROAD_SHAPE_RECORDS, start=1)
    ]
    lexicon = {
        "schema_version": LEXICON_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "name": "TrueVision Recognition Shape Lexicon",
        "authority": "truevision",
        "route_context": "northwest_ohio_country_road_to_monroe_michigan",
        "grain": "simple_visible_geometry_before_meaning",
        "symbols": symbols,
        "boundary": {
            "not_aw_word_lexicon": True,
            "recognition_items_not_truth_claims": True,
            "words_name_shapes_elsewhere": True,
            "state_symbols_drive_detection": True,
            "adapters_connect_to_aw": True,
        },
        "law": "Shapes first. Meaning later. Review before truth.",
    }
    lexicon["lexicon_hash"] = stable_hash({key: value for key, value in lexicon.items() if key != "lexicon_hash"})
    return lexicon


def lookup_shape_symbols(names: list[str]) -> dict[str, Any]:
    normalized = [str(name).strip().lower().replace(" ", "_") for name in names]
    by_name = {entry["name"]: entry for entry in build_road_shape_lexicon()["symbols"]}
    matches = [by_name[name] for name in normalized if name in by_name]
    unknown = [name for name in normalized if name not in by_name]
    result = {
        "schema_version": "truevision_recognition_shape_lookup_v1",
        "query_names": normalized,
        "matches": matches,
        "unknown_names": unknown,
        "boundary": {
            "symbols_are_recognition_shapes": True,
            "traffic_truth_promoted": False,
            "aw_lexicon_mutated": False,
        },
    }
    result["lookup_hash"] = stable_hash(result)
    return result


def build_country_road_aw_bridge() -> dict[str, Any]:
    mappings = [
        {"aw_anchor": "stop sign", "shape_names": ["octagon", "vertical_line"], "relation": "names_candidate_shape"},
        {"aw_anchor": "speed limit sign", "shape_names": ["vertical_rectangle", "vertical_line"], "relation": "names_candidate_shape"},
        {"aw_anchor": "warning sign", "shape_names": ["diamond", "vertical_line"], "relation": "names_candidate_shape"},
        {"aw_anchor": "railroad crossing", "shape_names": ["x_cross", "vertical_line"], "relation": "names_candidate_shape"},
        {"aw_anchor": "traffic light", "shape_names": ["circle_stack", "vertical_rectangle"], "relation": "names_candidate_shape"},
        {"aw_anchor": "lane line", "shape_names": ["lane_line", "parallel_lines"], "relation": "names_candidate_shape"},
        {"aw_anchor": "road edge", "shape_names": ["road_edge_line", "horizontal_line"], "relation": "names_candidate_shape"},
        {"aw_anchor": "utility pole", "shape_names": ["utility_pole", "vertical_line"], "relation": "names_candidate_shape"},
        {"aw_anchor": "mailbox", "shape_names": ["mailbox_box", "vertical_rectangle"], "relation": "names_candidate_shape"},
        {"aw_anchor": "barn", "shape_names": ["barn_box", "horizontal_rectangle", "triangle"], "relation": "names_candidate_shape"},
        {"aw_anchor": "tree", "shape_names": ["tree_trunk_line", "tree_canopy_mass"], "relation": "names_candidate_shape"},
        {"aw_anchor": "reflector", "shape_names": ["reflector_dot", "circle"], "relation": "names_candidate_shape"},
        {"aw_anchor": "guardrail", "shape_names": ["guardrail_line", "horizontal_line"], "relation": "names_candidate_shape"},
    ]
    bridge = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "bridge_name": "country_road_aw_to_truevision_shape_bridge",
        "route_context": "northwest_ohio_country_road_to_monroe_michigan",
        "ownership": {
            "aw_owns": "language_anchors",
            "truevision_owns": "recognition_shape_symbols",
            "bridge_owns": "candidate_alignment",
        },
        "mappings": [
            {
                **row,
                "tv_shape_symbols": lookup_shape_symbols(row["shape_names"])["matches"],
                "promotion_status": "candidate_mapping",
            }
            for row in mappings
        ],
        "boundary": {
            "no_aw_dependency": True,
            "no_aw_mutation": True,
            "no_object_truth_promotion": True,
        },
    }
    bridge["bridge_hash"] = stable_hash({key: value for key, value in bridge.items() if key != "bridge_hash"})
    return bridge


def write_road_shape_lexicon(storage_root: Path, *, run_id: str = "road_shape_lexicon") -> dict[str, Any]:
    storage_root = Path(storage_root)
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(run_id)).strip("_") or "road_shape_lexicon"
    artifact_root = storage_root / "artifacts" / "recognition_shape_lexicon"
    manifest_root = storage_root / "manifests" / "recognition_shape_lexicon"
    receipt_root = storage_root / "receipts" / "recognition_shape_lexicon"
    for path in (artifact_root, manifest_root, receipt_root):
        path.mkdir(parents=True, exist_ok=True)
    lexicon = build_road_shape_lexicon()
    bridge = build_country_road_aw_bridge()
    lexicon_path = artifact_root / f"{safe}_lexicon.json"
    bridge_path = artifact_root / f"{safe}_aw_bridge.json"
    lexicon_path.write_text(json.dumps(lexicon, indent=2, allow_nan=False), encoding="utf-8")
    bridge_path.write_text(json.dumps(bridge, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_id": safe,
        "lexicon_json": str(lexicon_path),
        "bridge_json": str(bridge_path),
        "lexicon_hash": lexicon["lexicon_hash"],
        "symbol_count": len(lexicon["symbols"]),
        "route_context": lexicon["route_context"],
        "boundary": lexicon["boundary"],
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path = manifest_root / f"{safe}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "tool": "truevision_recognition_shape_lexicon",
        "run_id": safe,
        "lexicon_json": str(lexicon_path),
        "manifest_json": str(manifest_path),
        "bridge_json": str(bridge_path),
        "symbol_count": len(lexicon["symbols"]),
        "boundary": {
            "source_video_retained": False,
            "aw_lexicon_mutated": False,
            "object_truth_promoted": False,
            "recognition_shapes_only": True,
        },
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    receipt_path = receipt_root / f"{safe}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_id": safe,
        "lexicon_json": str(lexicon_path),
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "bridge_json": str(bridge_path),
        "symbol_count": len(lexicon["symbols"]),
    }
