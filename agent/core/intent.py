"""Intent classification — the boundary between user language and authorized work.

Phase 1 brain contract: the model must not turn every message into a tool
call. This module deterministically classifies each incoming request BEFORE
any orchestration decision is made, and the AgentLoop uses that classification
to decide:

* whether the request can be answered conversationally (no tools at all),
* whether repository context must be loaded at all (demand-driven context),
* which tool categories are actually authorized (no writes without a reason).

Design rules:

* HIGH-CONFIDENCE ONLY. The deterministic layer only asserts what it is sure
  of (greetings, thanks, small talk, general-knowledge questions). Everything
  else is ``ambiguous`` with moderate confidence and the model — operating
  inside mode gating — makes the final call. This keeps terse but legitimate
  work orders ("do the thing", "make a.txt") working.
* This layer NEVER executes anything and NEVER invents objectives. It is a
  pure function of the request text.
* The model provides intelligence; this layer provides the boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Intent categories (documented in docs/architecture/RUNTIME.md and
# phases/phase-01-brain-audit/). The tuple is the public contract; the
# strings are also used in ``intent`` events and experience tags.
CONVERSATION = "conversation"
QUESTION = "question"
PROJECT_INSPECTION = "project_inspection"
CODE_CHANGE = "code_change"
FILE_OPERATION = "file_operation"
COMMAND_REQUEST = "command_request"
VERIFICATION_REQUEST = "verification_request"
AMBIGUOUS = "ambiguous"

INTENT_CATEGORIES: tuple[str, ...] = (
    CONVERSATION,
    QUESTION,
    PROJECT_INSPECTION,
    CODE_CHANGE,
    FILE_OPERATION,
    COMMAND_REQUEST,
    VERIFICATION_REQUEST,
    AMBIGUOUS,
)

# Confidence levels used by the loop to decide how much to trust a decision.
HIGH = "high"
MODERATE = "moderate"

# Intents that must never reach a mutating tool. The AgentLoop rejects write
# and command calls for these intents regardless of what the model requests;
# the model is told to answer directly instead. ``ambiguous`` is deliberately
# NOT excluded: terse but legitimate work orders stay workable, protected by
# mode gating and the prompt contract instead.
WRITE_EXCLUDED_INTENTS: frozenset[str] = frozenset(
    {CONVERSATION, QUESTION, PROJECT_INSPECTION}
)

# Tools that mutate the workspace or execute commands. Kept in sync with
# ``agent.config.MODIFY_TOOLS`` (imported lazily by callers to avoid a
# config -> intent import cycle; this module stays dependency-free).
MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "apply_patch",
        "delete_file",
        "move_file",
        "copy_file",
        "run_command",
    }
)

# ---------------------------------------------------------------------------
# High-confidence conversation patterns: social input that must NEVER become
# a coding task. Match against a normalized (lower-cased, stripped) request,
# anchored so "hello world program" is NOT classified as conversation.
# ---------------------------------------------------------------------------

_CONVERSATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^(hello|hi|hey|yo|greetings|good\s(morning|afternoon|evening)|howdy)[\s!.?]*$",
        r"^(hi|hello|hey)\s+(there|ascs|agent|bot|again)[\s!.?]*$",
        r"^(thanks|thank\syou|thx|ty|cheers|cool|nice|awesome|ok|okay|got\sit|great)[\s!.?]*$",
        r"^(how\sare\syou|how's\sit\sgoing|what's\sup|whats\sup|who\sare\syou|"
        r"what\sare\syou|what\scan\syou\sdo|what\scan\syou\shelp\swith|"
        r"who\smade\syou|are\syou\s(alive|real|an\sai|a\s(bot|robot)))[\s!.?]*$",
        r"^(bye|goodbye|see\syou|later|good\snight)[\s!.?]*$",
        r"^why\sdid\syou\b",
    )
)

# General-knowledge / explanation questions answerable without any tool use.
# Distinguished from project questions by the PROJECT patterns below.
_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^what\s(is|are)\s"
        r"(python|java(script)?|rust|golang|go|c\+\+|c#|html|css|sql|git|docker|"
        r"recursion|recursion\?|ai|machine\slearning|rest|an\sapi|a\sapi|json|yaml|"
        r"pytest|unittest|oop|regex|http|https|tcp|dns|linux|windows|powershell|"
        r"bash|zsh|ascs)\b",
        r"^(explain|what\sdoes|how\sdoes|how\sdo\s(i\s)?|what\sis)\s+"
        r"(recursion|polymorphism|inheritance|encapsulation|asynchronous|async\sio|"
        r"a\slist\scomprehension|list\scomprehensions?|big\so|memoization|"
        r"garbage\s.collection|a\sdecorator|decorators?|dependency\sinjection|"
        r"the\sdifference)\b",
        r"^(explain|teach\sme|tell\sme\sabout)\s+(recursion|oop|git|docker|"
        r"python|javascript|rest|api)s?\b",
        r"^(why|how)\s+(is|does|do|would|can)\b.*\?$",  # explicit question mark guard
        r"^(why|how)\s+(is|does|do|would|can)\b",
        r"^(what|which|when|who|where)\s.*\?$",
    )
)

# Project-directed questions: answerable by READ-ONLY inspection of the repo.
_PROJECT_INSPECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(what|which)\sfiles\b",
        r"\b(what|which)\s(is|are|'s)\s(in|inside)\s(this|the)\s"
        r"(project|repo|codebase|directory|folder|workspace)\b",
        r"\bshow\s(me\s)?(the\s)?(files|structure|tree|layout|directory|project)\b",
        r"\blist\s(the\s)?files\b",
        r"\b(where|which\sfile|which\smodule)\s(is|are|does|do|handles?|"
        r"implements?|defines?|contains?|configures?)\b",
        r"\bhow\sdoes\s(this\sproject|the\sproject|authentication|auth|login|"
        r"the\sconfig|the\sbuild)\b",
        r"\b(project|codebase|repo)s?\s*(structure|overview|layout|summary)\b",
        r"\bexplain\s(this\s(error|exception|traceback|failure)|the\serror)\b",
        r"\bfind\s(where|usages?|references?)\b",
        r"\bhow\sis\s(\w+)\s(configured|implemented|structured|organized)\b",
    )
)

# Explicit coding/file/command work orders. These confirm work intent but the
# model still chooses the minimal tool sequence inside mode gating.
_CODING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(create|write|add|make|implement|build|fix|refactor|rename|move|"
        r"update|change|modify|remove|delete|extend)\b"
        r".*\b(function|class|method|module|file|script|test|feature|bug|"
        r"endpoint|api|component|app|calculator|todo|flag|argument|mode)\b",
        r"\b(delete|remove)\s+(the\s)?(file|directory|folder)\b",
        r"\b(delete|remove)\s+(the\s)?[\w./\\-]+\.\w+\b",
        r"\b(run|execute)\s+(the\s)?(tests?|pytest|unittest|build|app|"
        r"command|script|suite)\b",
        r"\b(re)?name\s+(this|the|that)\b",
        r"\bset\s+up\b|\bscaffold\b",
    )
)


@dataclass(frozen=True)
class Decision:
    """Outcome of classifying one user request.

    ``requires_*`` flags are the orchestration contract:

    * ``requires_workspace`` — repository context (project intelligence /
      experience retrieval) is proportional to the request; conversation and
      world-knowledge questions skip it entirely (demand-driven context).
    * ``requires_write`` / ``requires_command`` — the request as stated can
      only be satisfied by mutating tools; the loop uses this to reject
      unauthorized tool calls for read-only intents.
    * ``requires_planning`` — a task graph may help; planning stays
      conditional (simple requests execute directly).
    """

    intent: str
    confidence: str = MODERATE
    requires_workspace: bool = False
    requires_read: bool = False
    requires_write: bool = False
    requires_command: bool = False
    requires_planning: bool = False
    requires_verification: bool = False
    scope: str = "none"  # none | conversation | project | files | commands
    reason: str = ""
    matched_pattern: str = field(default="", repr=False, compare=False)

    @property
    def is_conversational(self) -> bool:
        """True when the request must be answered without touching the workspace."""
        return (
            self.intent in (CONVERSATION, QUESTION) and self.confidence == HIGH
        )


def classify_request(text: str) -> Decision:
    """Classify ``text`` into an intent with requires_* orchestration flags.

    Pure function: no I/O, no model call, no state. The AgentLoop calls this
    once per run before any orchestration decision.
    """
    raw = (text or "").strip()
    normalized = re.sub(r"\s+", " ", raw.lower()).strip(" !.?")

    for pattern in _CONVERSATION_PATTERNS:
        if pattern.match(normalized):
            return Decision(
                intent=CONVERSATION,
                confidence=HIGH,
                requires_workspace=False,
                requires_read=False,
                requires_write=False,
                requires_command=False,
                requires_planning=False,
                requires_verification=False,
                scope="conversation",
                reason="social input; answer conversationally without tools",
                matched_pattern=pattern.pattern,
            )

    for pattern in _PROJECT_INSPECTION_PATTERNS:
        if pattern.search(raw):
            return Decision(
                intent=PROJECT_INSPECTION,
                confidence=HIGH,
                requires_workspace=True,
                requires_read=True,
                requires_write=False,
                requires_command=False,
                requires_planning=False,
                requires_verification=False,
                scope="project",
                reason="project question; read-only inspection is allowed, "
                "workspace modification is not",
                matched_pattern=pattern.pattern,
            )

    for pattern in _QUESTION_PATTERNS:
        if pattern.match(normalized):
            return Decision(
                intent=QUESTION,
                confidence=HIGH,
                requires_workspace=False,
                requires_read=False,
                requires_write=False,
                requires_command=False,
                requires_planning=False,
                requires_verification=False,
                scope="conversation",
                reason="general-knowledge question; answer without tools",
                matched_pattern=pattern.pattern,
            )

    for pattern in _CODING_PATTERNS:
        if pattern.search(raw):
            return Decision(
                intent=CODE_CHANGE,
                confidence=MODERATE,
                requires_workspace=True,
                requires_read=True,
                requires_write=True,
                requires_command=True,
                requires_planning=True,
                requires_verification=True,
                scope="files",
                reason="explicit coding request; mode gating still applies",
                matched_pattern=pattern.pattern,
            )

    # Everything else: ambiguous. The model decides inside mode gating; the
    # orchestrator must not invent an implementation objective on its behalf.
    return Decision(
        intent=AMBIGUOUS,
        confidence=MODERATE,
        requires_workspace=True,
        requires_read=True,
        requires_write=False,  # the model must justify writes per its contract
        requires_command=False,
        requires_planning=False,
        requires_verification=False,
        scope="project",
        reason="request type unclear; model decides inside mode gating with "
        "no invented objective",
    )


def fallback_spec_for(objective: str) -> dict:
    """Intent-aware single-task fallback for planner failure.

    Planner failure must never silently convert conversation/questions/
    ambiguous input into implementation work. Conversational and
    question-style objectives fall back to a *review* task that reports
    findings instead of implementing; genuine work keeps the honest
    implement-and-verify fallback.
    """
    intent = classify_request(objective).intent
    if intent in (CONVERSATION, QUESTION):
        return {
            "title": f"Review request: {objective}",
            "description": (
                f"The planner could not decompose this request ({objective}). "
                "Report findings to the operator; no changes were requested."
            ),
            "kind": "review",
            "verification": [
                "report the outcome to the operator without modifying the workspace"
            ],
        }
    return {
        "title": f"Implement and verify: {objective}",
        "description": objective,
        "kind": "implement",
        "verification": ["confirm the objective is satisfied"],
    }
