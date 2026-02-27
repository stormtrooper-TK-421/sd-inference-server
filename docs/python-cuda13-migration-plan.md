# CUDA 13 Python Version Alignment Plan

## Goal
Align the repository to the Python versions that are explicitly supported by the CUDA 13 (`cu130`) PyTorch stack used in this project.

## Current State (from repo)
- The runtime stack is already configured for `cu130` nightly wheels via `requirements.txt`.
- `README.md` currently documents Python `3.12.x / 3.13.x (experimental)` with `>=3.12,<3.14`.
- There is no centralized interpreter pinning file (`.python-version`, `pyproject.toml`, CI matrix) in the repo yet.

## Target Version Policy
1. **Primary (required) Python version: `3.12`**
   - Treat 3.12 as the stable/default interpreter for all development, docs, and validation.
2. **Secondary (allowed) Python version: `3.13`**
   - Keep as opt-in/experimental until CUDA 13 + PyTorch nightly behavior is consistently validated in this repo.
3. **Out of scope for CUDA 13 path: `<=3.11` and `>=3.14`**
   - Do not advertise these versions for the CUDA 13 runtime path.

## Rollout Plan

### Phase 1: Source-of-truth and guardrails
- Add a single source of truth for Python compatibility (recommended: `pyproject.toml` with `requires-python = ">=3.12,<3.14"`).
- Add `.python-version` pinned to `3.12` (or exact patch used by your runtime image).
- Add startup/runtime guard in server entry points that errors early on unsupported Python versions.

### Phase 2: Dependency and install hardening
- Keep CUDA 13 nightly index setup in `requirements.txt`.
- Split optional dependency files by intent:
  - `requirements.txt` for CUDA 13 runtime path.
  - `requirements-dev.txt` for lint/test tooling.
- Add an automated check script that validates the installed torch build metadata and Python major/minor pair.

### Phase 3: Test matrix enforcement
- Run tests on:
  - Python 3.12 (required gate)
  - Python 3.13 (non-blocking initially, then promote to required once stable)
- Keep `tests/test_dependency_compatibility.py` and extend it with an explicit Python-version assertion aligned to policy.

### Phase 4: Documentation and operator UX
- Update all install snippets to default to Python 3.12.
- Document 3.13 as experimental with known caveats.
- Add a short troubleshooting section for interpreter mismatch (e.g. creating a new venv with 3.12).

## Suggested Repo-wide Change List
1. **Interpreter declaration**
   - Add `pyproject.toml` (`requires-python`).
   - Add `.python-version` pinned to 3.12.
2. **Runtime checks**
   - Add a helper in startup path (`server.py` or bootstrap module) that validates `sys.version_info`.
3. **Tests**
   - Add/extend compatibility tests for python range and torch metadata expectations.
4. **Docs**
   - Keep README compatibility table synchronized with the policy above.

## Acceptance Criteria
- A fresh setup using Python 3.12 installs and runs with CUDA 13 nightly wheels without manual dependency surgery.
- Running tests under Python 3.12 passes.
- Python 3.13 status is explicitly tracked (pass/fail + caveats), not ambiguous.
- Unsupported Python versions fail fast with actionable error messages.

## Execution Order (low risk)
1. Land docs + compatibility declaration files.
2. Land runtime guardrail check.
3. Land CI/test matrix updates.
4. Promote/demote 3.13 support level based on test signal.
