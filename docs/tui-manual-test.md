# TUI Manual Hardware Test (not CI)

> [!CAUTION] Status: **NOT PERFORMED** — do not claim validation until
> each step passes in a real terminal.

Perform in Windows Terminal (UTF-8) on the Windows host. Headless shells
and redirected output cannot render curses; they exit through the TTY
fallback path instead (expected, not a failure of these steps).

## 1. Prerequisites

- `.venv` with `windows-curses` (`npm run ASCS` bootstraps it).
- Ollama reachable with `qwen3-coder:30b` (or `qwen2.5-coder:14b`).

## 2. Launch

```powershell
npm run ASCS
```

- [ ] ASCII banner renders (no `UnicodeEncodeError`; on legacy
      codepages the ASCII fallback banner appears instead).
- [ ] Boot log reaches `A.S.C.S. ready.` with model + tool counts.
- [ ] Full-screen TUI takes over (header, chat pane, status footer).

## 3. Modes and input

- [ ] `TAB` cycles modes; footer reflects PLAN / BUILD / AUTO.
- [ ] Typing a question in PLAN mode produces an inspected plan
      without file edits.
- [ ] `/help` lists slash commands; `/models` lists Ollama models.

## 4. Resize

- [ ] Shrinking below 40x10 shows the too-small guard, no traceback.
- [ ] Enlarging redraws cleanly (no smeared panes).

## 5. Task run

- [ ] A small BUILD task (e.g. "add a `--verbose` flag and test it")
      completes with verified steps and an action log.

## 6. Cancellation and exit

- [ ] `Ctrl+C` mid-run cancels cleanly (no hung locks).
- [ ] Quitting the TUI returns to the shell with exit code `0`.
- [ ] A launcher-owned `ollama serve` is stopped; a pre-existing
      server keeps running (`ollama ps`).

## 7. Diagnostics (reference, runnable headless)

```powershell
npm run ascs:check
npm run ascs:doctor
```

- [ ] Both exit `0` on this machine.
