"""A.S.C.S. - A Smart Coding System.

A local autonomous coding agent backed by Ollama. Intended to integrate with
RISARMS and eventually be usable by ASIS and TIVISS.

Primary components:

    config      - environment/CLI configuration (PLAN / BUILD / AUTO modes)
    workspace   - strict workspace containment for all file operations
    tools       - tool registry + validation + execution (terminal, python,
                  git, file editing, search, plan)
    ollama      - isolated, stdlib-only Ollama HTTP client
    loop        - the autonomous agent execution loop (lifecycle + events)
    events      - structured, JSON-serializable agent events
    state       - explicit lifecycle state machine
    boot        - staged startup with real checks and progress reporting
    web         - local web UI server (stdlib only, SSE, cancellation)
    main        - CLI entry point (``risa``)
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]