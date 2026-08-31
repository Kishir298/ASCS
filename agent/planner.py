"""A.S.C.S. planner: objective decomposition into a dependency-aware task graph.

The planner turns a user objective into a structured :class:`TaskGraph` by
asking the model to decompose it into small, coherent, independently-executable
tasks, then normalising the result and applying automatic size control:

* each task carries title, description, dependencies, files, commands,
  verification and a complexity rating;
* ``auto-chunking`` splits oversized (``large``) tasks into per-file subtasks
  and keeps absurdly-tiny tasks coalesced;
* a real dependency DAG is built (fan-in/fan-out supported), validated for
  cycles, and returned as a :class:`~agent.tasks.TaskGraph`.

The optional ``chat`` hook is model-agnostic: it is any callable
``(messages, format=\"json\") -> str`` (an ``OllamaClient.chat`` works directly);
tests inject fakes so no Ollama server is required.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .project import ProjectStore
from .tasks import TaskGraph, build_graph_from_specs, chunk_graph

COMPLEXITIES = ("small", "medium", "large")
KINDS = ("plan", "inspect", "implement", "verify", "review")

# A huge task is one the model flags as needing broad changes; we re-split it.
MAX_FILES_BEFORE_SPLIT = 3

_PLACEHOLDER_RE = re.compile(
    r"^(no (explicit )?plan|none|n/?a|tbd|placeholder|not provided|todo)$",
    re.IGNORECASE,
)


def _normalise_complexity(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in COMPLEXITIES:
            return lowered
        if lowered in {"s", "tiny", "xs"}:
            return "small"
        if lowered in {"l", "big", "xl"}:
            return "large"
    return "medium"


def _normalise_kind(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in KINDS:
        return value.strip().lower()
    return "implement"


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.splitlines() if item.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif item is not None:
                out.append(str(item).strip())
        return out
    return []


def _as_dependencies(value: Any) -> list[str]:
    deps: list[str] = []
    for dep in _as_str_list(value):
        cleaned = dep.strip()
        if cleaned:
            deps.append(cleaned)
    return deps


def parse_tasks(value: Any) -> list[dict]:
    """Parse a relaxed model response into a list of normalised task specs.

    Accepts ``{"tasks": [...]}``, a bare list, a single dict, or a block of
    text where each line becomes a task. Never raises: unusable entries are
    dropped. Returns normalised dicts consumable by
    :func:`~agent.tasks.build_graph_from_specs`.
    """
    if value is None:
        return []

    if isinstance(value, str):
        lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
        return [{"title": ln} for ln in lines if not _PLACEHOLDER_RE.match(ln)]

    if isinstance(value, dict):
        if "tasks" in value:
            return parse_tasks(value["tasks"])
        # A single task object.
        if "title" in value or "id" in value or "description" in value:
            specs = [value]
        else:
            specs = []
    elif isinstance(value, (list, tuple)):
        specs = list(value)
    else:
        return []

    result: list[dict] = []
    for item in specs:
        if isinstance(item, str):
            title = item.strip()
            if not title or _PLACEHOLDER_RE.match(title):
                continue
            result.append(
                {
                    "title": title,
                    "description": title,
                }
            )
            continue
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or item.get("step") or item.get("task") or "").strip()
        if not title or _PLACEHOLDER_RE.match(title):
            title = str(item.get("description") or "").strip()
            if not title:
                continue

        complexity = _normalise_complexity(item.get("complexity", item.get("size")))
        kind = _normalise_kind(item.get("kind"))

        # Auto-split broad tasks that list many files.
        files = _as_str_list(item.get("files", item.get("touch")))
        if complexity == "large" and len(files) > MAX_FILES_BEFORE_SPLIT:
            files = files[:MAX_FILES_BEFORE_SPLIT]

        dependencies = _as_dependencies(
            item.get("dependencies", item.get("depends_on", item.get("deps")))
        )

        description = str(item.get("description") or "").strip() or title

        result.append(
            {
                "id": str(item["id"]).strip() if item.get("id") else "",
                "title": title,
                "description": description,
                "dependencies": sorted(set(dependencies)),
                "files": sorted(set(files)),
                "commands": sorted(set(_as_str_list(item.get("commands", item.get("command"))))),
                "verification": sorted(
                    set(_as_str_list(item.get("verification", item.get("verify"))))
                ),
                "complexity": complexity,
                "kind": kind,
            }
        )
    return result


def _derive_verification(task: dict, project_manifest_text: str) -> list[str]:
    """Fill a gap in verification with deterministic heuristics.

    Falls back to a toolchain-derived test command when the task clearly
    involves code and the project toolchain is known, otherwise to a generic
    "inspect + report" step. The heuristic never fabricates a passing run: it
    only emits a command that the toolchain detection actually identified.
    """
    if task.get("verification"):
        return list(task["verification"])

    verification: list[str] = []
    title_desc = (task.get("title", "") + " " + task.get("description", "")).lower()
    has_code = bool(task.get("files")) or task.get("kind") in ("implement", "review")

    if has_code:
        # Prefer a real toolchain-derived test command when one was detected.
        test_cmd = _first_test_command(project_manifest_text)
        if test_cmd:
            verification.append(f"run {test_cmd}")
        verification.append("verify the changed files behave as intended")
    else:
        verification.append("report inspection findings and confirm nothing changed")
    return verification


def _first_test_command(project_manifest_text: str) -> str:
    """Pull the first likely test command out of a toolchain summary line."""
    marker = "Toolchain:"
    idx = project_manifest_text.find(marker)
    if idx < 0:
        return ""
    # Locate the "Likely test commands: <semis>" fragment on the same block.
    head = project_manifest_text[idx:]
    for label in ("Likely test commands:", "Test commands:"):
        start = head.find(label)
        if start < 0:
            continue
        rest = head[start + len(label):]
        first = rest.split(";")[0].strip()
        if first and not first.startswith("Likely"):
            return first
    return ""


def planner_prompt(
    objective: str,
    project_intelligence: str,
) -> str:
    """Build the decomposition prompt handed to the model."""
    return (
        "You are the A.S.C.S. planner. Decompose the objective below into a set "
        "of small, coherent, independently-executable tasks.\n"
        "\n"
        "OBJECTIVE:\n"
        f"{objective}\n"
        "\n"
        "PROJECT INTELLIGENCE (trust this; it is scan-derived):\n"
        f"{project_intelligence}\n"
        "\n"
        "Rules:\n"
        "- Each task must be ONE coherent unit that can be implemented and "
        "verified independently (small/medium size). Do not make giant tasks "
        "such as 'implement the entire system'. Do not make absurdly tiny tasks "
        "such as 'open a file'.\n"
        "- Represent REAL dependencies between tasks using their ids.\n"
        "- Assign likely affected files, the commands to run, and an explicit "
        "verification step for each task.\n"
        "- Rate complexity small|medium|large and a kind of "
        "inspect|implement|verify|review.\n"
        "- Prefer a pipeline like: inspect -> design -> implement -> test -> "
        "review, with dependencies such as T2 and T3 both feeding T4.\n"
        "\n"
        "Reply with ONLY a JSON object:\n"
        '{"tasks": [{"id": "T1", "title": "...", "description": "...", '
        '"dependencies": [], "files": [...], "commands": [...], '
        '"verification": [...], "complexity": "medium", "kind": "implement"}]}'
        "\n"
    )


def project_intelligence(store: ProjectStore, objective: str) -> str:
    """Build a compact project-intelligence block to seed the planner.

    Combines the persisted manifest summary with task-specific retrieval so the
    planner understands the project shape before decomposing.
    """
    from .project import project_prompt_text

    lines: list[str] = []
    manifest_text = project_prompt_text(store)
    lines.append(manifest_text)

    if store.index.records:
        try:
            bundle = store.index.retrieve(
                objective,
                level=2,
                max_tokens=1_600,
                max_files=8,
            )
            if bundle.chunks:
                lines.append("\nRelevant files:")
                for chunk in bundle.chunks:
                    lines.append(f"- {chunk.text.strip()}")
        except Exception:  # noqa: BLE001 - retrieval must never break planning
            pass
    return "\n".join(lines)


def plan_objective(
    store: ProjectStore,
    objective: str,
    chat: Callable[[list[dict], dict], str],
    *,
    max_tokens_context: int = 4_000,
) -> TaskGraph:
    """Decompose ``objective`` into a validated, size-controlled task graph.

    ``chat`` is any ``(messages, options) -> str`` callable (an
    ``OllamaClient.chat`` works), enabling full model-agnosticism. The result is
    re-chunked with automatic size control (split oversized tasks) before being
    returned.
    """
    intelligence = project_intelligence(store, objective)
    prompt = planner_prompt(objective, intelligence)

    raw = chat(
        [{"role": "user", "content": prompt}],
        {"format": "json"},
    )

    parsed = _parse_chat_to_value(raw)
    specs = parse_tasks(parsed)
    if not specs:
        # Fall back to a single review-style task so execution still proceeds.
        specs = [
            {
                "title": f"Implement and verify: {objective}",
                "description": objective,
                "complexity": "medium",
                "kind": "implement",
                "verification": ["confirm the objective is satisfied"],
            }
        ]

    graph = build_graph_from_specs(specs)

    # Automatic size control: split oversized tasks.
    if any(t.complexity == "large" for t in graph.tasks.values()):
        graph = chunk_graph(graph)

    # Guarantee verification on every task.
    for task in graph.tasks.values():
        if not task.verification:
            task.verification = _derive_verification(
                {"title": task.title, "description": task.description,
                 "kind": task.kind, "files": task.files},
                intelligence,
            )

    graph.recompute_statuses()
    graph.validate()
    return graph


def _parse_chat_to_value(raw: str) -> Any:
    """Extract a JSON value from a model reply (tolerant).

    Uses :func:`agent.models.parse_model_reply`'s extraction logic without the
    tool/done contract, returning the first top-level JSON object found, or the
    raw text if no JSON is present (so textual lists still work).
    """
    import json

    text = (raw or "").strip()
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return text
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        return obj
    except (json.JSONDecodeError, ValueError):
        return text


def plan_text(graph: TaskGraph) -> str:
    """Render a human-readable plan (used for plan inspection/step log)."""
    lines: list[str] = []
    for index, task in enumerate(graph.ordered(), start=1):
        status = task.status
        lines.append(f"{index}. [{status}] {task.title}")
        if task.files:
            lines.append(f"     files: {', '.join(task.files)}")
        if task.verification:
            lines.append(f"     verify: {'; '.join(task.verification)}")
    return "\n".join(lines)


__all__ = [
    "COMPLEXITIES",
    "KINDS",
    "parse_tasks",
    "plan_objective",
    "plan_text",
    "planner_prompt",
    "project_intelligence",
]
