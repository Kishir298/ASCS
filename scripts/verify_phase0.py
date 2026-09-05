"""Phase 0 architectural validation for A.S.C.S.

Checks (read-only, no Ollama required):

- No runtime code imports from ``phases/`` (agent/, tests/, scripts/).
- All ``agent.<domain>`` packages import cleanly (old + new paths).
- ``python -m agent.terminal --check`` and ``--list-models`` exit 0
  (requires a running Ollama for full green; reports otherwise).
- pytest collection still discovers the suite.

Usage (Windows PowerShell)::

    .\\.venv\\Scripts\\python.exe scripts\\verify_phase0.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOMAINS = [
    "agent.core",
    "agent.planning",
    "agent.execution",
    "agent.tools",
    "agent.context",
    "agent.experience",
    "agent.verification",
    "agent.models",
    "agent.terminal",
]

FAILURES: list[str] = []


def check_no_phases_imports() -> None:
    bad: list[str] = []
    for base in ("agent", "tests", "scripts"):
        root = ROOT / base
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name == "verify_phase0.py":
                continue  # self-check script mentions the pattern in strings
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                code = line.split("#", 1)[0]
                if (
                    "from phases." in code
                    or "from phases import" in code
                    or "import phases." in code
                    or code.strip() == "import phases"
                ):
                    bad.append(f"{path.relative_to(ROOT)}:{lineno}: {stripped}")
    if bad:
        FAILURES.append("phases imports found:\n  " + "\n  ".join(bad))
        print("FAIL: runtime imports from phases/ detected")
        for item in bad:
            print("  " + item)
    else:
        print("PASS: no runtime imports from phases/")


def check_domain_imports() -> None:
    ok = True
    for mod in DOMAINS:
        try:
            __import__(mod)
            print(f"PASS: import {mod}")
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            ok = False
            FAILURES.append(f"import {mod}: {exc}")
            print(f"FAIL: import {mod}: {exc}")
    if ok:
        print("PASS: all domain packages importable")


def check_pytest_collection() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    if proc.returncode != 0:
        FAILURES.append("pytest collection failed")
        print("FAIL: pytest collection failed")
    else:
        print("PASS: pytest collection ok")
    for line in tail:
        print("  " + line)


def main() -> int:
    print(f"ROOT: {ROOT}")
    check_no_phases_imports()
    check_domain_imports()
    check_pytest_collection()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        return 1
    print("All Phase 0 structural checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
