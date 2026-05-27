# TrueVision Repo Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the TrueVision repo into a stable local tool stack without throwing away working tools or breaking current entrypoints.

**Architecture:** Keep the proven runtime packages and scripts, then make the repo shape explicit: runtime code in package roots, operator entrypoints in `scripts/`, ignored runtime data in `storage/`, bulky proof media in `outputs/`, and SecureCore handoff candidates parked. Move only after inventory, tests, and compatibility shims exist.

**Tech Stack:** Python unittest, PowerShell, Git, Rust/Cargo for `native/truevision_capture_rs`, deterministic JSON manifests/receipts.

---

## Current Findings

```text
outputs/                    ~18.8 GB, generated proof media and diagrams
storage/                    ~188 MB, ignored runtime logs/manifests/receipts
historical_securecore_data/  ~178 MB, old imported data
native/                     ~16 MB, Rust source plus lockfile
```

Tracked generated/media-like files currently found:

```text
docs/UI MOCK BE EXACT.png
storage/config/coordinate_maps/youtube_intake_map_template.json
```

Current working tree has uncommitted worker-forge and doc changes. Commit those before structural moves.

## Target Shape

```text
truevision_runtime/
  core/common runtime helpers
  av_tools/
  document_state/
  learning_intake/
  llm_adapter/
  rendering/
  state_patterns/
  studio/
  worker_forge.py

trueaudio_runtime/
  keep for now as a stable top-level package
  later may become compatibility shim after tests prove imports

trueframegen/
  keep for now as a stable top-level package
  later may become compatibility shim after tests prove imports

scripts/
  operator CLI entrypoints only
  no generated data
  no hidden browser automation

storage/
  ignored runtime data only
  manifests/ receipts/ events/ state_chunks/ artifacts/

outputs/
  ignored generated proof media only
  may be pruned after manifest/receipt/report preservation

transfer_to_securecore/
  parked agent candidates only
  no active SecureCore integration by folder presence
```

Law:

```text
Keep tools.
Consolidate surfaces.
Drain bulky proof data.
Do not break imports.
Do not delete evidence receipts.
```

---

### Task 1: Freeze Current Good State

**Files:**
- Commit existing worker forge changes.
- Do not move files in this task.

- [ ] **Step 1: Re-run the full Python test suite**

```powershell
$env:PYTHONPATH='scripts;modules;.'
python -m unittest discover -s tests -v
```

Expected:

```text
Ran 267 tests
OK
```

- [ ] **Step 2: Verify worktree scope**

```powershell
git status --short --branch
git diff --stat
git ls-files --others --exclude-standard
```

Expected changed scope:

```text
docs/ACTIVE_TOOL_SURFACE.md
docs/TRUEVISION_LOCAL_PRODUCT_MAP.md
docs/TRUEVISION_WORKER_RACK_CONTRACT.md
scripts/truevision_worker_forge.py
tests/test_worker_forge.py
tests/test_worker_forge_script.py
transfer_to_securecore/
truevision_runtime/worker_forge.py
```

- [ ] **Step 3: Commit the already-verified worker forge layer**

```powershell
git add docs/ACTIVE_TOOL_SURFACE.md docs/TRUEVISION_LOCAL_PRODUCT_MAP.md docs/TRUEVISION_WORKER_RACK_CONTRACT.md scripts/truevision_worker_forge.py tests/test_worker_forge.py tests/test_worker_forge_script.py transfer_to_securecore truevision_runtime/worker_forge.py
git commit -m "Add TrueVision worker forge and SecureCore agent candidates"
```

Expected:

```text
[main <hash>] Add TrueVision worker forge and SecureCore agent candidates
```

---

### Task 2: Add A Source-Derived Repo Inventory Tool

**Files:**
- Create: `truevision_runtime/repo_inventory.py`
- Create: `scripts/truevision_repo_inventory.py`
- Test: `tests/test_truevision_repo_inventory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_truevision_repo_inventory.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from truevision_runtime.repo_inventory import build_repo_inventory, write_repo_inventory


class TrueVisionRepoInventoryTests(unittest.TestCase):
    def test_inventory_classifies_code_runtime_and_outputs(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "truevision_runtime").mkdir()
            (root / "truevision_runtime" / "__init__.py").write_text("", encoding="utf-8")
            (root / "scripts").mkdir()
            (root / "scripts" / "truevision_demo.py").write_text("print('x')", encoding="utf-8")
            (root / "outputs").mkdir()
            (root / "outputs" / "demo.mp4").write_bytes(b"x" * 1024)
            (root / "storage" / "receipts").mkdir(parents=True)
            (root / "storage" / "receipts" / "r.json").write_text("{}", encoding="utf-8")

            inventory = build_repo_inventory(root)

        self.assertEqual(inventory["schema"], "truevision_repo_inventory_v1")
        self.assertEqual(inventory["lanes"]["runtime_code"]["files"], 1)
        self.assertEqual(inventory["lanes"]["script_entrypoints"]["files"], 1)
        self.assertEqual(inventory["lanes"]["generated_outputs"]["files"], 1)
        self.assertEqual(inventory["lanes"]["runtime_storage"]["files"], 1)
        self.assertGreater(inventory["lanes"]["generated_outputs"]["bytes"], 0)

    def test_write_repo_inventory_writes_manifest_and_receipt(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "truevision_runtime").mkdir()
            (root / "truevision_runtime" / "__init__.py").write_text("", encoding="utf-8")
            out = root / "storage" / "manifests" / "repo_inventory"
            receipt = root / "storage" / "receipts" / "repo_inventory"

            result = write_repo_inventory(root, out, receipt)

            self.assertTrue(Path(result["manifest_json"]).exists())
            self.assertTrue(Path(result["receipt_json"]).exists())
            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "truevision_repo_inventory_v1")
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
$env:PYTHONPATH='scripts;modules;.'
python -m unittest discover -s tests -p test_truevision_repo_inventory.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'truevision_runtime.repo_inventory'
```

- [ ] **Step 3: Implement minimal inventory module**

Create `truevision_runtime/repo_inventory.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
from typing import Iterable


LANES = {
    "runtime_code": ["truevision_runtime", "trueaudio_runtime", "trueframegen", "native"],
    "script_entrypoints": ["scripts"],
    "tests": ["tests"],
    "docs": ["docs"],
    "runtime_storage": ["storage"],
    "generated_outputs": ["outputs"],
    "parked_securecore_transfer": ["transfer_to_securecore"],
    "legacy_or_external": ["historical_securecore_data", "connected_artifacts", "reports"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_records(root: Path, names: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for name in names:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                records.append({
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                })
    return sorted(records, key=lambda item: item["path"])


def summarize(records: list[dict]) -> dict:
    return {
        "files": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "sample_paths": [item["path"] for item in records[:20]],
    }


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_repo_inventory(root: str | Path) -> dict:
    root = Path(root)
    lanes = {}
    for lane, names in LANES.items():
        lanes[lane] = summarize(file_records(root, names))
    inventory = {
        "schema": "truevision_repo_inventory_v1",
        "created_at": utc_now(),
        "repo_root": str(root),
        "lanes": lanes,
    }
    inventory["inventory_hash"] = stable_hash(inventory)
    return inventory


def write_repo_inventory(root: str | Path, manifest_dir: str | Path, receipt_dir: str | Path) -> dict:
    inventory = build_repo_inventory(root)
    manifest_dir = Path(manifest_dir)
    receipt_dir = Path(receipt_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().replace(":", "").replace(".", "_")
    manifest_path = manifest_dir / f"{stamp}_truevision_repo_inventory.json"
    receipt_path = receipt_dir / f"{stamp}_truevision_repo_inventory_receipt.json"
    manifest_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    receipt = {
        "schema": "truevision_repo_inventory_receipt_v1",
        "created_at": utc_now(),
        "manifest_json": str(manifest_path),
        "inventory_hash": inventory["inventory_hash"],
        "status": "written",
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return {
        "manifest_json": str(manifest_path),
        "receipt_json": str(receipt_path),
        "inventory_hash": inventory["inventory_hash"],
    }
```

- [ ] **Step 4: Add CLI**

Create `scripts/truevision_repo_inventory.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from truevision_runtime.repo_inventory import write_repo_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a TrueVision repo inventory manifest and receipt.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--storage-root", default="storage")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    storage_root = Path(args.storage_root)
    result = write_repo_inventory(
        repo_root,
        storage_root / "manifests" / "repo_inventory",
        storage_root / "receipts" / "repo_inventory",
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"manifest: {result['manifest_json']}")
        print(f"receipt: {result['receipt_json']}")
        print(f"hash: {result['inventory_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run focused tests**

```powershell
$env:PYTHONPATH='scripts;modules;.'
python -m unittest discover -s tests -p test_truevision_repo_inventory.py -v
```

Expected:

```text
Ran 2 tests
OK
```

---

### Task 3: Create Cleanup Dry-Run, Not Delete

**Files:**
- Create: `truevision_runtime/repo_cleanup.py`
- Create: `scripts/truevision_repo_cleanup.py`
- Test: `tests/test_truevision_repo_cleanup.py`

Rule:

```text
Dry-run first.
Preserve receipts, reports, manifests, source lists, docs, presets, tests, and code.
Never delete source files.
Delete bulky generated proof media only after an explicit --apply run.
```

- [ ] **Step 1: Write tests for cleanup candidate selection**

Create `tests/test_truevision_repo_cleanup.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from truevision_runtime.repo_cleanup import find_cleanup_candidates


class TrueVisionRepoCleanupTests(unittest.TestCase):
    def test_cleanup_candidates_include_large_output_media_not_receipts(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "outputs" / "run").mkdir(parents=True)
            (root / "outputs" / "run" / "proof.mp4").write_bytes(b"x" * 2_000_000)
            (root / "outputs" / "run" / "receipt.json").write_text("{}", encoding="utf-8")
            (root / "storage" / "receipts").mkdir(parents=True)
            (root / "storage" / "receipts" / "keep.json").write_text("{}", encoding="utf-8")

            candidates = find_cleanup_candidates(root, min_bytes=1_000_000)

        paths = [item["path"] for item in candidates]
        self.assertIn("outputs/run/proof.mp4", paths)
        self.assertNotIn("outputs/run/receipt.json", paths)
        self.assertNotIn("storage/receipts/keep.json", paths)
```

- [ ] **Step 2: Implement cleanup candidate finder**

Create `truevision_runtime/repo_cleanup.py`:

```python
from __future__ import annotations

from pathlib import Path


MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".png", ".jpg", ".jpeg", ".npz", ".npy"}
KEEP_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".csv", ".toml", ".lock", ".py", ".rs"}


def find_cleanup_candidates(root: str | Path, min_bytes: int = 50_000_000) -> list[dict]:
    root = Path(root)
    candidates: list[dict] = []
    outputs = root / "outputs"
    if not outputs.exists():
        return candidates
    for path in outputs.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if suffix in MEDIA_SUFFIXES and suffix not in KEEP_SUFFIXES and size >= min_bytes:
            candidates.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "reason": "bulky_generated_output_media",
            })
    return sorted(candidates, key=lambda item: item["bytes"], reverse=True)
```

- [ ] **Step 3: Add CLI with dry-run default**

Create `scripts/truevision_repo_cleanup.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from truevision_runtime.repo_cleanup import find_cleanup_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run cleanup candidates for generated TrueVision output media.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--min-mb", type=float, default=50.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    candidates = find_cleanup_candidates(root, int(args.min_mb * 1024 * 1024))
    result = {
        "schema": "truevision_repo_cleanup_dry_run_v1",
        "mode": "dry_run",
        "candidate_count": len(candidates),
        "candidate_bytes": sum(item["bytes"] for item in candidates),
        "candidates": candidates,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"dry-run candidates: {result['candidate_count']}")
        print(f"candidate MB: {round(result['candidate_bytes'] / 1024 / 1024, 2)}")
        for item in candidates[:50]:
            print(f"{round(item['bytes'] / 1024 / 1024, 2)} MB  {item['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run cleanup tests**

```powershell
$env:PYTHONPATH='scripts;modules;.'
python -m unittest discover -s tests -p test_truevision_repo_cleanup.py -v
```

Expected:

```text
Ran 1 test
OK
```

- [ ] **Step 5: Run cleanup dry-run**

```powershell
$env:PYTHONPATH='scripts;modules;.'
python scripts\truevision_repo_cleanup.py --repo-root . --min-mb 50
```

Expected:

```text
dry-run candidates: <number>
candidate MB: <large number>
```

Do not delete in this task.

---

### Task 4: Consolidate Docs Around Active Product Shape

**Files:**
- Modify: `docs/TRUEVISION_LOCAL_PRODUCT_MAP.md`
- Modify: `docs/ACTIVE_TOOL_SURFACE.md`
- Create: `docs/TRUEVISION_REPO_CONSOLIDATION_MAP.md`

- [ ] **Step 1: Add the repo consolidation map**

Create `docs/TRUEVISION_REPO_CONSOLIDATION_MAP.md` with:

```markdown
# TrueVision Repo Consolidation Map

TrueVision keeps its working tools. Consolidation means making tool ownership,
runtime storage, generated outputs, and parked experiments obvious.

## Canonical Lanes

```text
truevision_runtime/   primary TrueVision Python runtime
trueaudio_runtime/    active audio-state runtime package
trueframegen/         active derived frame-state runtime package
scripts/              operator CLI entrypoints
native/               Rust hot paths
tests/                verification
storage/              ignored runtime data
outputs/              ignored generated proof media
transfer_to_securecore/ parked SecureCore agent candidates
```

## Cleanup Law

```text
Delete bulky generated media only after a dry-run manifest.
Keep receipts, manifests, reports, source lists, docs, tests, and code.
```
```

- [ ] **Step 2: Link the map from active docs**

Add to `docs/TRUEVISION_LOCAL_PRODUCT_MAP.md`:

```text
Repo consolidation map:

```text
docs/TRUEVISION_REPO_CONSOLIDATION_MAP.md
```
```

Add to `docs/ACTIVE_TOOL_SURFACE.md`:

```text
Repo inventory and cleanup:

```powershell
python scripts\truevision_repo_inventory.py --json
python scripts\truevision_repo_cleanup.py --min-mb 50
```
```

---

### Task 5: Verify Full System After Consolidation Tools

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

```powershell
$env:PYTHONPATH='scripts;modules;.'
python -m unittest discover -s tests -p test_truevision_repo_inventory.py -v
python -m unittest discover -s tests -p test_truevision_repo_cleanup.py -v
python -m unittest discover -s tests -p test_worker_forge*.py -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run full tests**

```powershell
$env:PYTHONPATH='scripts;modules;.'
python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 3: Run inventory and cleanup dry-run**

```powershell
$env:PYTHONPATH='scripts;modules;.'
python scripts\truevision_repo_inventory.py --repo-root . --storage-root storage --json
python scripts\truevision_repo_cleanup.py --repo-root . --min-mb 50 --json
```

Expected:

```text
inventory manifest and receipt written
cleanup dry-run reports candidates without deleting
```

- [ ] **Step 4: Commit consolidation tooling**

```powershell
git add truevision_runtime/repo_inventory.py truevision_runtime/repo_cleanup.py scripts/truevision_repo_inventory.py scripts/truevision_repo_cleanup.py tests/test_truevision_repo_inventory.py tests/test_truevision_repo_cleanup.py docs/TRUEVISION_REPO_CONSOLIDATION_MAP.md docs/ACTIVE_TOOL_SURFACE.md docs/TRUEVISION_LOCAL_PRODUCT_MAP.md
git commit -m "Add TrueVision repo consolidation inventory and cleanup tools"
```

---

## Not In This Pass

Do not move `trueaudio_runtime/` or `trueframegen/` yet. They are working packages with tests and scripts.

Do not delete `outputs/` yet. First produce a cleanup dry-run manifest, then choose what to remove.

Do not move SecureCore agents into SecureCore. `transfer_to_securecore/` remains parked.

Do not convert the studio UI or render proofs into product center. They remain local development surfaces.

