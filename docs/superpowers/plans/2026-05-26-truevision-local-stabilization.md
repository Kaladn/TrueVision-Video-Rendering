# TrueVision Local Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock TrueVision as a local, repeatable state-machine tool with clear hierarchy, active tools, parked experiments, receipt rules, and a dev prerequisite checker.

**Architecture:** Stabilize in place without moving imports. Add docs that define the AW/SC/TV boundaries and a lightweight preflight script that reports local prerequisites and repo health without installing or mutating anything.

**Tech Stack:** Python 3.11+, PowerShell-friendly CLI, unittest, existing TrueVision package layout.

---

### Task 1: Local Product Map Docs

**Files:**
- Create: `docs/TRUEVISION_LOCAL_PRODUCT_MAP.md`
- Create: `docs/ACTIVE_TOOL_SURFACE.md`
- Create: `docs/PARKED_EXPERIMENTS.md`
- Create: `docs/TRUEVISION_TO_ANCHORWORKS_PACKET_CONTRACT.md`
- Create: `docs/TRUEVISION_SECURECORE_SAFETY_BOUNDARY.md`
- Create: `docs/TRUEVISION_RECEIPT_AND_MANIFEST_RULES.md`
- Modify: `README.md`
- Modify: `TODO.md`

- [ ] **Step 1: Add hierarchy docs**

Create docs that state:

```text
TrueVision records state.
AnchorWorks interprets meaning.
SecureCore proves safety.
```

- [ ] **Step 2: Add active and parked tool maps**

Separate supported local tools from parked browser/intake/video-proof experiments.

- [ ] **Step 3: Update README and TODO**

Point operators to the local map and doctor script.

### Task 2: TrueVision Preflight

**Files:**
- Create: `scripts/truevision_preflight.py`
- Create: `tests/test_truevision_preflight.py`

- [ ] **Step 1: Write failing tests**

Tests must cover:

```text
Python version check
required executable checks
required Python module checks
storage/output ignore health
JSON report shape
exit status calculation
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m unittest discover -s tests -p test_truevision_preflight.py -v
```

Expected: fail because `scripts/truevision_preflight.py` does not exist.

- [ ] **Step 3: Implement minimal preflight**

Add a non-mutating CLI that prints text by default and JSON with `--json`.

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
python -m unittest discover -s tests -p test_truevision_preflight.py -v
```

Expected: pass.

### Task 3: Verification

**Files:**
- Existing focused tests only.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_truevision_preflight.py" -v
python -m unittest tests.test_truevision_timing_audit tests.test_driving_school -v
```

- [ ] **Step 2: Show changed files**

Run:

```powershell
git status --short
git diff --stat
```

- [ ] **Step 3: Stop before commit**

Do not commit until the operator reviews the first stabilization diff.
