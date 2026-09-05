"""Persistent experience memory for A.S.C.S.

This module provides bounded, local, experience-based learning without
modifying the model weights or the source code.

Experiences record what happened during previous runs so future planning can
retrieve relevant successful and failed approaches.

Runtime dependencies: standard library only.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class Experience:
    """A single completed or evaluated agent experience."""

    task: str
    outcome: str
    success: bool
    plan: str = ""
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    feedback: str = ""
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    experience_id: str = ""

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("experience task cannot be empty")

        self.success = bool(self.success)
        self.score = max(-1.0, min(1.0, float(self.score)))

        if not self.experience_id:
            self.experience_id = f"{int(self.timestamp * 1000)}"

    def to_record(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> "Experience":
        """Construct an experience from persisted JSON data."""
        if not isinstance(record, dict):
            raise ValueError("experience record must be an object")

        return cls(
            task=str(record.get("task", "")),
            outcome=str(record.get("outcome", "")),
            success=bool(record.get("success", False)),
            plan=str(record.get("plan", "")),
            actions=_string_list(record.get("actions")),
            observations=_string_list(record.get("observations")),
            errors=_string_list(record.get("errors")),
            feedback=str(record.get("feedback", "")),
            tags=_string_list(record.get("tags")),
            score=float(record.get("score", 0.0)),
            timestamp=float(record.get("timestamp", time.time())),
            experience_id=str(record.get("experience_id", "")),
        )


def _string_list(value: object) -> list[str]:
    """Normalize persisted/user-provided list values."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 2
    }


def _experience_text(experience: Experience) -> str:
    return " ".join(
        [
            experience.task,
            experience.outcome,
            experience.plan,
            *experience.actions,
            *experience.observations,
            *experience.errors,
            experience.feedback,
            *experience.tags,
        ]
    )


class ExperienceStore:
    """Thread-safe append-only local experience store.

    The store deliberately uses JSON Lines rather than a database so that the
    learning layer remains portable, inspectable, and dependency-free.

    Default storage is outside the repository:

        ~/.risa/ascs/experiences.jsonl

    This prevents runtime experiences from polluting the source tree or Git
    history. A custom path can be supplied for tests or project-specific
    storage.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_records: int = 5000,
    ) -> None:
        if max_records < 1:
            raise ValueError("max_records must be at least 1")

        self.path = (
            Path(path).expanduser()
            if path is not None
            else Path.home() / ".risa" / "ascs" / "experiences.jsonl"
        )
        self.max_records = max_records
        self._lock = threading.RLock()

    def save(self, experience: Experience) -> Experience:
        """Persist one experience and return it unchanged."""
        if not isinstance(experience, Experience):
            raise TypeError("experience must be an Experience instance")

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        experience.to_record(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")

            self._compact_if_needed()

        return experience

    def save_run(
        self,
        *,
        task: str,
        outcome: str,
        success: bool,
        plan: str = "",
        actions: Iterable[str] = (),
        observations: Iterable[str] = (),
        errors: Iterable[str] = (),
        feedback: str = "",
        tags: Iterable[str] = (),
        score: float | None = None,
    ) -> Experience:
        """Create and persist an experience from a completed run.

        When no explicit score is supplied:
            successful run -> +1.0
            failed run     -> -1.0
        """
        if score is None:
            score = 1.0 if success else -1.0

        experience = Experience(
            task=task,
            outcome=outcome,
            success=success,
            plan=plan,
            actions=list(actions),
            observations=list(observations),
            errors=list(errors),
            feedback=feedback,
            tags=list(tags),
            score=score,
        )
        return self.save(experience)

    def load(self, *, limit: int | None = None) -> list[Experience]:
        """Load persisted experiences in chronological order."""
        with self._lock:
            if not self.path.exists():
                return []

            records: list[Experience] = []

            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            records.append(
                                Experience.from_record(json.loads(line))
                            )
                        except (ValueError, TypeError, json.JSONDecodeError):
                            # One corrupt record must not destroy the entire
                            # experience history.
                            continue
            except OSError:
                return []

            if limit is not None:
                if limit < 0:
                    raise ValueError("limit cannot be negative")
                if limit == 0:
                    return []
                records = records[-limit:]

            return records

    def recent(self, limit: int = 10) -> list[Experience]:
        """Return the most recent experiences."""
        return self.load(limit=limit)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        successful_only: bool = False,
    ) -> list[Experience]:
        """Retrieve experiences relevant to a query.

        Relevance combines token overlap with outcome quality. Successful
        experiences receive a small ranking boost, while failed experiences
        remain retrievable so the planner can learn what to avoid.
        """
        if not query.strip():
            return []

        if limit < 1:
            raise ValueError("limit must be at least 1")

        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        candidates = self.load()

        ranked: list[tuple[float, Experience]] = []

        for experience in candidates:
            if successful_only and not experience.success:
                continue

            text_tokens = _tokens(_experience_text(experience))
            overlap = len(query_tokens & text_tokens)

            if overlap == 0:
                continue

            coverage = overlap / len(query_tokens)

            success_bonus = 0.15 if experience.success else 0.0
            score_bonus = max(-0.1, min(0.1, experience.score * 0.1))

            age_days = max(
                0.0,
                (time.time() - experience.timestamp) / 86400.0,
            )
            recency_bonus = 0.05 / (1.0 + age_days / 30.0)

            relevance = (
                coverage
                + success_bonus
                + score_bonus
                + recency_bonus
            )

            ranked.append((relevance, experience))

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].success,
                item[1].timestamp,
            ),
            reverse=True,
        )

        return [experience for _, experience in ranked[:limit]]

    def update_feedback(
        self,
        experience_id: str,
        *,
        feedback: str,
        score: float | None = None,
        success: bool | None = None,
    ) -> Experience | None:
        """Update evaluation data for an existing experience.

        The underlying JSONL file is rewritten atomically through a temporary
        file. The original record order is preserved.
        """
        if not experience_id:
            raise ValueError("experience_id cannot be empty")

        with self._lock:
            experiences = self.load()

            target: Experience | None = None

            for experience in experiences:
                if experience.experience_id == experience_id:
                    target = experience
                    break

            if target is None:
                return None

            target.feedback = feedback

            if score is not None:
                target.score = max(-1.0, min(1.0, float(score)))

            if success is not None:
                target.success = bool(success)

            self._rewrite(experiences)

            return target

    def penalize_contradictions(
        self,
        task: str,
        *,
        exclude_id: str = "",
        delta: float = 0.2,
    ) -> list[str]:
        """Lower the score of successful experiences a newer failed run contradicts.

        This is the contradiction learning signal: when a later run about a
        strongly overlapping task fails, previously stored successful
        experiences for that task become less trusted. Each contradiction
        reduces their score by ``delta`` (floor -1.0), so a repeatedly
        contradicted experience is progressively superseded.

        Returns the ids of the adjusted experiences.
        """
        if not task.strip():
            return []
        if delta <= 0:
            raise ValueError("delta must be positive")

        query_tokens = _tokens(task)
        if not query_tokens:
            return []

        touched: list[str] = []

        with self._lock:
            experiences = self.load()

            changed = False
            required = max(1, len(query_tokens) // 2)

            for experience in experiences:
                if not experience.success:
                    continue
                if experience.experience_id == exclude_id:
                    continue
                overlap = len(query_tokens & _tokens(experience.task))
                if overlap < required:
                    continue

                experience.score = max(-1.0, experience.score - delta)
                touched.append(experience.experience_id)
                changed = True

            if changed:
                self._rewrite(experiences)

        return touched

    def clear(self) -> None:
        """Delete all stored experiences."""
        with self._lock:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def count(self) -> int:
        """Return the number of valid persisted experiences."""
        return len(self.load())

    def _compact_if_needed(self) -> None:
        """Keep the store bounded without changing normal append behavior."""
        experiences = self.load()

        if len(experiences) <= self.max_records:
            return

        self._rewrite(experiences[-self.max_records :])

    def _rewrite(self, experiences: Iterable[Experience]) -> None:
        """Atomically rewrite the experience store."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        temporary = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for experience in experiences:
                    handle.write(
                        json.dumps(
                            experience.to_record(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    handle.write("\n")

            temporary.replace(self.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def format_for_prompt(
    experiences: Iterable[Experience],
    *,
    max_chars: int = 6000,
) -> str:
    """Format retrieved experiences into compact planner context.

    This function intentionally returns plain text. The caller decides where
    the resulting context belongs in the model prompt.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    sections: list[str] = []
    used = 0

    for index, experience in enumerate(experiences, start=1):
        status = "SUCCESS" if experience.success else "FAILURE"

        section = (
            f"Experience {index} [{status}]\n"
            f"Task: {experience.task}\n"
            f"Outcome: {experience.outcome}\n"
        )

        if experience.plan:
            section += f"Plan: {experience.plan}\n"

        if experience.actions:
            section += "Actions: " + "; ".join(experience.actions) + "\n"

        if experience.observations:
            section += (
                "Observations: "
                + "; ".join(experience.observations)
                + "\n"
            )

        if experience.errors:
            section += "Errors: " + "; ".join(experience.errors) + "\n"

        if experience.feedback:
            section += f"Feedback: {experience.feedback}\n"

        if experience.tags:
            section += "Tags: " + ", ".join(experience.tags) + "\n"

        section += f"Score: {experience.score:.2f}\n"

        remaining = max_chars - used

        if remaining <= 0:
            break

        if len(section) > remaining:
            section = section[:remaining].rstrip() + "\n"

        sections.append(section)
        used += len(section)

        if used >= max_chars:
            break

    return "\n".join(sections)


__all__ = [
    "Experience",
    "ExperienceStore",
    "format_for_prompt",
]
