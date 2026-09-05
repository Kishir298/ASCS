# Phase 0 — Acceptance Criteria

Phase 0 is complete only when all of the following hold:

- [ ] Existing repository was inspected before modification (structure, imports,
  entrypoints, tests, docs, legacy/duplicates).
- [ ] Git state was inspected (`status`, branch, log); no reset, no discarded
  user work, no unrelated files committed.
- [ ] Runtime vs development artifacts are separated: `agent/` runtime only,
  `phases/` dev docs only, `tests/` tests only, `docs/` docs only, `scripts/`
  maintenance only.
- [ ] `agent/` has clear boundaries (`core`, `planning`, `execution`, `tools`,
  `context`, `experience`, `verification`, `models`, `terminal`).
- [ ] `phases/phase-00` … `phase-06` exist, each with `OBJECTIVES.md`,
  `ACCEPTANCE.md`, `NOTES.md`.
- [ ] `tests/` has domain organization (`core`, `planning`, `execution`,
  `tools`, `context`, `experience`, `verification`, `terminal`,
  `integration`) and remains discoverable by pytest.
- [ ] `docs/architecture/` (`OVERVIEW.md`, `RUNTIME.md`, `COMPONENTS.md`,
  `DECISIONS.md`) and `docs/phases/ROADMAP.md` exist and reflect actual code
  (planned items labeled `Planned` / `Future phase`).
- [ ] No runtime code imports from `phases/` (verified by grep/script).
- [ ] No unnecessary duplicate implementations were created; moves used shims
  or package re-exports where needed to preserve public APIs.
- [ ] Package discovery still works (`agent*` in `pyproject.toml` covers new
  subpackages).
- [ ] `pytest` passes at least at the pre-migration baseline
  (baseline: `7 failed, 578 passed, 5 skipped` — all 7 failures pre-existing
  TUI shell-spec issues; no new failures).
- [ ] `python -m agent.terminal --check` exits 0 (Ollama reachable, model
  available when server is up).
- [ ] `python -m agent.terminal --list-models` lists installed models.
- [ ] Terminal entrypoint still starts (`--tui`; `--ui` remains a compat alias).
- [ ] No Phase 1+ behavior changes were made (brain, modes, PowerShell,
  language intel, learning, verification logic untouched).
- [ ] `git diff --stat` / `git diff` reviewed: no `.venv`, secrets, model
  files, temp files, or unrelated work committed.
