from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKER_FORGE_SCHEMA = "truevision_worker_forge_v1"
INVENTORY_SCHEMA = "truevision_worker_inventory_manifest_v1"
CHAT_FORGE_SCHEMA = "truevision_chat_forged_tool_request_v1"
CHOICE_SCHEMA = "truevision_local_worker_choice_manifest_v1"

LOCAL_WORKER_DIRS = (
    "truevision_runtime/learning_intake",
    "truevision_runtime/av_tools",
    "truevision_runtime/rendering",
    "trueaudio_runtime",
    "trueframegen",
)

LOCAL_TOOL_DIRS = ("scripts",)

REQUIRED_AGENT_FIELDS = {
    "agent_id",
    "name",
    "version",
    "runtime_language",
    "entrypoint",
    "entrypoint_hash",
    "allowed_reads",
    "allowed_writes",
    "requires_approval",
    "approval_phrase",
    "mutation_class",
    "dry_run_supported",
    "log_stream",
    "test_command",
    "risk_tier",
    "prompt_only_allowed",
    "required_params",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def safe_slug(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)).strip("_")
    return clean[:96] or "worker_forge"


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def append_hash_chained_jsonl(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = load_jsonl(path)
    previous_hash = prior[-1]["record_hash"] if prior else ""
    payload = dict(record)
    payload.setdefault("schema_version", WORKER_FORGE_SCHEMA)
    payload.setdefault("written_at_utc", utc_now())
    payload["sequence"] = len(prior) + 1
    payload["previous_hash"] = previous_hash
    payload["record_hash"] = stable_hash({k: v for k, v in payload.items() if k != "record_hash"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def write_receipt(storage_root: Path, *, action: str, status: str, result: dict[str, Any]) -> Path:
    receipt = {
        "schema_version": "truevision_worker_forge_receipt_v1",
        "written_at_utc": utc_now(),
        "action": action,
        "status": status,
        "result_hash": stable_hash(result),
        "result": result,
        "boundary": {
            "local_mini_securecore": True,
            "execution_allowed": False,
            "manifest_only": True,
            "workers_remain_local": True,
        },
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return write_json(
        storage_root / "receipts" / "worker_forge" / f"{utc_now().replace(':', '').replace('.', '_')}_{safe_slug(action)}.json",
        receipt,
    )


def infer_worker_kind(path: Path) -> str:
    text = path.as_posix().lower()
    name = path.stem.lower()
    if "audio" in text or "speech" in text:
        return "audio_worker"
    if "framegen" in text:
        return "framegen_worker"
    if "meter" in name:
        return "meter_worker"
    if "driving" in name or "road" in name:
        return "driving_worker"
    if "youtube" in name or "coordinate" in name:
        return "operator_approved_intake_worker"
    if "render" in name or "scene" in name or "template" in name:
        return "render_worker"
    if "focus" in name or "depth" in name or "trudepth" in name:
        return "depth_worker"
    if "seismic" in name or "angular" in name:
        return "motion_event_worker"
    return "local_worker"


def worker_keywords(path: Path, kind: str) -> list[str]:
    parts = set(path.stem.lower().replace("truevision_", "").split("_"))
    parts.update(kind.split("_"))
    return sorted(item for item in parts if item)


def discover_local_workers(repo_root: Path) -> list[dict[str, Any]]:
    workers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative_dir in LOCAL_WORKER_DIRS:
        directory = repo_root / relative_dir
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name == "__init__.py":
                continue
            rel = relative_path(repo_root, path)
            if rel in seen:
                continue
            seen.add(rel)
            kind = infer_worker_kind(path)
            workers.append(
                {
                    "worker_id": safe_slug(path.stem),
                    "name": path.stem,
                    "path": rel,
                    "runtime_language": "python",
                    "unit_type": "worker",
                    "kind": kind,
                    "keywords": worker_keywords(path, kind),
                    "status": "local_candidate",
                    "execution_allowed": False,
                    "manifest_only": True,
                }
            )
    return workers


def discover_local_tools(repo_root: Path) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative_dir in LOCAL_TOOL_DIRS:
        directory = repo_root / relative_dir
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name == "__init__.py":
                continue
            rel = relative_path(repo_root, path)
            if rel in seen:
                continue
            seen.add(rel)
            kind = infer_worker_kind(path)
            tools.append(
                {
                    "worker_id": safe_slug(path.stem),
                    "tool_id": safe_slug(path.stem),
                    "name": path.stem,
                    "path": rel,
                    "runtime_language": "python",
                    "unit_type": "tool",
                    "kind": kind,
                    "keywords": worker_keywords(path, kind),
                    "status": "local_candidate",
                    "execution_allowed": False,
                    "manifest_only": True,
                }
            )
    return tools


def validate_agent_candidate(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "agent_id": path.stem.replace(".agent", ""),
            "path": path.as_posix(),
            "status": "invalid_manifest",
            "errors": [f"invalid_json:{exc}"],
        }
    errors: list[str] = []
    missing = sorted(REQUIRED_AGENT_FIELDS.difference(manifest))
    errors.extend(f"missing:{field}" for field in missing)
    if manifest.get("prompt_only_allowed") is not False:
        errors.append("prompt_only_forbidden")
    if str(manifest.get("runtime_language", "")).lower() == "prompt":
        errors.append("prompt_runtime_forbidden")
    agent_id = str(manifest.get("agent_id") or path.stem.replace(".agent", ""))
    return {
        "agent_id": agent_id,
        "name": str(manifest.get("name") or agent_id),
        "path": path.as_posix(),
        "runtime_language": str(manifest.get("runtime_language") or ""),
        "mutation_class": str(manifest.get("mutation_class") or ""),
        "risk_tier": manifest.get("risk_tier"),
        "status": "valid_manifest" if not errors else "invalid_manifest",
        "errors": errors,
        "execution_allowed": False,
        "manifest_only": True,
    }


def discover_agent_candidates(agent_candidates_root: Path | None) -> list[dict[str, Any]]:
    if not agent_candidates_root:
        return []
    manifest_dir = agent_candidates_root / "AGENTS" / "agents"
    if not manifest_dir.exists():
        return []
    return [validate_agent_candidate(path) for path in sorted(manifest_dir.glob("*.agent.json"))]


def build_manifest_inventory(
    *,
    repo_root: Path,
    storage_root: Path,
    agent_candidates_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    storage_root = storage_root.resolve()
    tools = discover_local_tools(repo_root)
    workers = discover_local_workers(repo_root)
    agents = discover_agent_candidates(agent_candidates_root.resolve() if agent_candidates_root else None)
    manifest = {
        "schema_version": INVENTORY_SCHEMA,
        "written_at_utc": utc_now(),
        "repo_root": str(repo_root),
        "storage_root": str(storage_root),
        "tools": tools,
        "workers": workers,
        "agent_candidates": agents,
        "policy": {
            "manifest_only": True,
            "execution_allowed": False,
            "workers_remain_local": True,
            "agents_transfer_to_securecore_only_after_review": True,
            "no_worker_migration_to_securecore": True,
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path = storage_root / "manifests" / "worker_forge" / "local_worker_inventory_manifest.json"
    write_json(manifest_path, manifest)
    event = append_hash_chained_jsonl(
        storage_root / "events" / "worker_forge.jsonl",
        {
            "event_type": "inventory_built",
            "manifest_json": str(manifest_path),
            "tool_count": len(tools),
            "worker_count": len(workers),
            "agent_candidate_count": len(agents),
        },
    )
    receipt_path = write_receipt(
        storage_root,
        action="worker_inventory_built",
        status="manifest_written",
        result={"manifest_json": str(manifest_path), "event_hash": event["record_hash"]},
    )
    return {
        "status": "manifest_written",
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "tool_count": len(tools),
        "worker_count": len(workers),
        "agent_candidate_count": len(agents),
    }


def chat_forge_tool_request(
    *,
    storage_root: Path,
    requested_by: str,
    chat_text: str,
    tool_name: str,
    organ: str,
    purpose: str,
    input_refs: list[str],
) -> dict[str, Any]:
    storage_root = storage_root.resolve()
    run_id = f"{utc_now().replace(':', '').replace('.', '_')}_{safe_slug(tool_name)}"
    manifest = {
        "schema_version": CHAT_FORGE_SCHEMA,
        "run_id": run_id,
        "written_at_utc": utc_now(),
        "requested_by": requested_by,
        "chat_text": chat_text,
        "tool_name": tool_name,
        "organ": organ,
        "purpose": purpose,
        "input_refs": list(input_refs),
        "status": "forged_manifest_only",
        "execution_allowed": False,
        "policy": {
            "local_mini_securecore": True,
            "chat_forged": True,
            "selection_only": True,
            "no_execution": True,
            "workers_remain_local": True,
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path = storage_root / "manifests" / "worker_forge" / "chat_forged_tools" / f"{run_id}.json"
    write_json(manifest_path, manifest)
    chat_record = append_hash_chained_jsonl(
        storage_root / "chats" / "tool_forge.jsonl",
        {
            "event_type": "chat_tool_request",
            "run_id": run_id,
            "requested_by": requested_by,
            "chat_text": chat_text,
            "tool_name": tool_name,
            "manifest_json": str(manifest_path),
        },
    )
    event_record = append_hash_chained_jsonl(
        storage_root / "events" / "worker_forge.jsonl",
        {
            "event_type": "tool_manifest_forged",
            "run_id": run_id,
            "tool_name": tool_name,
            "organ": organ,
            "manifest_json": str(manifest_path),
            "chat_record_hash": chat_record["record_hash"],
        },
    )
    receipt_path = write_receipt(
        storage_root,
        action="chat_forged_tool_request",
        status="forged_manifest_only",
        result={
            "manifest_json": str(manifest_path),
            "chat_record_hash": chat_record["record_hash"],
            "event_record_hash": event_record["record_hash"],
        },
    )
    return {
        "status": "forged_manifest_only",
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "chat_record_hash": chat_record["record_hash"],
        "event_record_hash": event_record["record_hash"],
    }


def score_worker(worker: dict[str, Any], request_text: str) -> tuple[int, list[str]]:
    request = request_text.lower()
    reasons: list[str] = []
    score = 0
    for keyword in worker.get("keywords", []):
        if keyword and keyword in request:
            score += 2
            reasons.append(f"keyword:{keyword}")
    name = str(worker.get("name", "")).lower()
    for part in name.split("_"):
        if part and part in request:
            score += 1
            reasons.append(f"name_part:{part}")
    kind = str(worker.get("kind", "")).lower()
    if kind.replace("_", " ") in request:
        score += 2
        reasons.append(f"kind:{kind}")
    return score, reasons


def choose_local_worker(
    *,
    storage_root: Path,
    inventory_manifest: Path,
    request_text: str,
) -> dict[str, Any]:
    storage_root = storage_root.resolve()
    inventory = json.loads(inventory_manifest.read_text(encoding="utf-8"))
    scored: list[dict[str, Any]] = []
    for worker in list(inventory.get("tools", [])) + list(inventory.get("workers", [])):
        score, reasons = score_worker(worker, request_text)
        if score > 0:
            scored.append({"score": score, "reasons": reasons, "worker": worker})
    scored.sort(key=lambda item: (-item["score"], item["worker"].get("name", "")))
    selected = scored[0] if scored else None
    status = "candidate_selected" if selected else "no_candidate"
    manifest = {
        "schema_version": CHOICE_SCHEMA,
        "written_at_utc": utc_now(),
        "request_text": request_text,
        "inventory_manifest": str(inventory_manifest),
        "status": status,
        "execution_allowed": False,
        "selected_worker": selected["worker"] if selected else None,
        "selection_score": selected["score"] if selected else 0,
        "selection_reasons": selected["reasons"] if selected else [],
        "candidate_count": len(scored),
        "policy": {
            "local_mini_securecore": True,
            "selection_only": True,
            "no_execution": True,
            "workers_remain_local": True,
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    run_id = f"{utc_now().replace(':', '').replace('.', '_')}_worker_choice"
    manifest_path = storage_root / "manifests" / "worker_forge" / "choices" / f"{run_id}.json"
    write_json(manifest_path, manifest)
    event_record = append_hash_chained_jsonl(
        storage_root / "events" / "worker_forge.jsonl",
        {
            "event_type": "local_worker_choice",
            "status": status,
            "choice_manifest_json": str(manifest_path),
            "selected_worker": manifest["selected_worker"]["name"] if manifest["selected_worker"] else "",
        },
    )
    receipt_path = write_receipt(
        storage_root,
        action="local_worker_choice",
        status=status,
        result={"choice_manifest_json": str(manifest_path), "event_record_hash": event_record["record_hash"]},
    )
    return {
        "status": status,
        "selected_worker": manifest["selected_worker"],
        "execution_allowed": False,
        "choice_manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
    }
