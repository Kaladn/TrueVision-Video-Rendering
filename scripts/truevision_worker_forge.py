from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truevision_runtime.worker_forge import (
    build_manifest_inventory,
    forge_tool_request,
    choose_local_worker,
)


def _print_result(result: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    for key, value in result.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TrueVision local mini-SecureCore worker/tool forge. Manifest-only; no worker execution."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="Inventory local tools, workers, and staged SecureCore agent candidates.")
    inventory.add_argument("--repo-root", type=Path, default=ROOT)
    inventory.add_argument("--storage-root", type=Path, default=ROOT / "storage")
    inventory.add_argument(
        "--agent-candidates-root",
        type=Path,
        default=ROOT / "transfer_to_securecore" / "truevision_agent_candidates",
    )
    inventory.add_argument("--json", action="store_true")

    forge = sub.add_parser("forge", help="Forge a requested tool/worker manifest and append local event logs.")
    forge.add_argument("--storage-root", type=Path, default=ROOT / "storage")
    forge.add_argument("--requested-by", required=True)
    forge.add_argument("--request-text", required=True)
    forge.add_argument("--tool-name", required=True)
    forge.add_argument("--organ", required=True)
    forge.add_argument("--purpose", required=True)
    forge.add_argument("--input-ref", action="append", default=[])
    forge.add_argument("--json", action="store_true")

    choose = sub.add_parser("choose", help="Select a local worker/tool candidate from an inventory manifest.")
    choose.add_argument("--storage-root", type=Path, default=ROOT / "storage")
    choose.add_argument("--inventory-manifest", type=Path, required=True)
    choose.add_argument("--request-text", required=True)
    choose.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inventory":
        result = build_manifest_inventory(
            repo_root=args.repo_root,
            storage_root=args.storage_root,
            agent_candidates_root=args.agent_candidates_root,
        )
        _print_result(result, as_json=args.json)
        return 0

    if args.command == "forge":
        result = forge_tool_request(
            storage_root=args.storage_root,
            requested_by=args.requested_by,
            request_text=args.request_text,
            tool_name=args.tool_name,
            organ=args.organ,
            purpose=args.purpose,
            input_refs=args.input_ref,
        )
        _print_result(result, as_json=args.json)
        return 0

    if args.command == "choose":
        result = choose_local_worker(
            storage_root=args.storage_root,
            inventory_manifest=args.inventory_manifest,
            request_text=args.request_text,
        )
        _print_result(result, as_json=args.json)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
