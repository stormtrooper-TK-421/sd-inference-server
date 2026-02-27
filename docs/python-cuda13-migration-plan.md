# CUDA 13 Python 3.14 Alignment Plan

## Goal
Align the repository to Python `3.14` as the single supported interpreter track for the CUDA 13 (`cu130`) PyTorch stack used in this project.

## Current State (from repo)
- The runtime stack is configured for `cu130` nightly wheels via `requirements.txt`.
- Interpreter policy is now centralized in:
  - `pyproject.toml` with `requires-python = ">=3.14,<3.15"`
  - `.python-version` pinned to `3.14`
- CLI entry points perform a startup check and fail fast when Python `<3.14` or `>=3.15` is detected.

## Version Policy
1. **Required Python version line: `3.14.x`**
   - Development, docs, and runtime behavior target Python 3.14.
2. **Out of scope: `<=3.13` and `>=3.15`**
   - Do not advertise or validate these versions for the CUDA 13 path until policy changes.

## Rollout Checklist
1. Keep docs and metadata aligned (`README.md`, `.python-version`, `pyproject.toml`).
2. Keep runtime guardrails in entry points (`server.py`, `remote.py`, `convert.py`, `scripts/example.py`).
3. Validate dependency compatibility against Python 3.14 whenever updating `requirements.txt`.
4. Add/adjust CI test matrices to use Python 3.14 only.

## Acceptance Criteria
- A fresh setup using Python 3.14 installs with existing dependency files.
- Runtime entry points fail fast with actionable guidance when launched on unsupported versions.
- Compatibility documentation consistently references Python 3.14 as required.
