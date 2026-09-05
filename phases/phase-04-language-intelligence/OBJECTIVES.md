# Phase 4 — Language Intelligence — Objectives

Give A.S.C.S. evidence-based language and framework awareness:

- Language detection (Python, Node/TS, Rust, Go, etc.) from project evidence
  (`pyproject.toml`, `requirements.txt`, `package.json`, `Cargo.toml`,
  `go.mod`), not guesses.
- Framework/toolchain awareness: only suggest commands that exist
  (e.g. `python -m pytest` only when `pytest`/tests exist; `npm test` only
  when the script exists) via `agent/context/toolchain.py`.
- Language-specific verification: derive per-task `verification` steps from
  the detected toolchain.
- Language-aware context: symbol/import/dependency extraction and
  dependency-aware retrieval per language.

Preserve model-specific chunking (30b: 8192 / 14b: 4096). No behavior changes
to modes, shell, learning, or reliability beyond language awareness.
