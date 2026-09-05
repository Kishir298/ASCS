"""Compatibility shim: canonical implementation lives at ``agent.core.loop``.

Preserved so existing ``from agent.loop import …`` imports (tests, tooling)
keep working after the Phase 0 move. New code should import from
``agent.core``.
"""

from __future__ import annotations

from agent.core.loop import *  # noqa: F401,F403
from agent.core.loop import AgentLoop, GraphLoopResult, LoopResult, run_agent, run_graph_agent

__all__ = ["AgentLoop", "GraphLoopResult", "LoopResult", "run_agent", "run_graph_agent"]
