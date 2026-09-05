"""A.S.C.S. - A Smart Coding System.

A local autonomous coding agent backed by Ollama. Intended to integrate with
RISARMS and eventually be usable by ASIS and TIVISS.

Runtime layout (Phase 0): flat canonical modules are preserved via shims,
while domain packages provide the architectural address. New code should
import from the domain packages::

    agent.core         - AgentLoop, lifecycle/state, cancellation
    agent.planning     - planner, prompts
    agent.execution    - executor, task graph
    agent.tools        - tool registry + execution
    agent.context      - project index, manifest, toolchain
    agent.experience   - learning memory store
    agent.verification - verification boundary (distributed logic)
    agent.models       - Ollama client, providers, response contract
    agent.terminal     - terminal entry, TUI (primary UI)

Top-level survivors: config, workspace, boot, doctor, web (EventHub/
TaskRunner shared; HTTP serving legacy), main, events. ``phases/`` is never
imported by runtime code.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]