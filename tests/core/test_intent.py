"""Unit tests for the Phase 1 intent/decision layer (``agent.core.intent``).

The classifier must be deterministic, pure, and high-confidence-only: social
input and general-knowledge questions must never be mistaken for coding work,
while terse-but-legitimate work orders must stay workable (ambiguous).
"""

from __future__ import annotations

import pytest

from agent.core.intent import (
    AMBIGUOUS,
    CODE_CHANGE,
    CONVERSATION,
    FILE_OPERATION,
    INTENT_CATEGORIES,
    PROJECT_INSPECTION,
    QUESTION,
    WRITE_EXCLUDED_INTENTS,
    Decision,
    classify_request,
    fallback_spec_for,
)


# ---------------------------------------------------------------------------
# Scenario 1/2: conversational input must never look like work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "hi",
        "hey there",
        "Hello!",
        "thanks",
        "thank you",
        "cool",
        "nice",
        "ok",
        "how are you",
        "what can you do?",
        "what can you do",
        "who are you",
        "bye",
        "good night",
        "why did you do that",
        "why did you create that file?",
    ],
)
def test_conversational_inputs_classify_high_confidence(text):
    decision = classify_request(text)
    assert decision.intent == CONVERSATION
    assert decision.confidence == "high"
    assert decision.is_conversational


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "hi",
        "thanks",
        "what can you do?",
        "how are you",
    ],
)
def test_conversational_inputs_require_nothing(text):
    decision = classify_request(text)
    assert not decision.requires_workspace
    assert not decision.requires_read
    assert not decision.requires_write
    assert not decision.requires_command
    assert not decision.requires_planning


@pytest.mark.parametrize(
    "text",
    [
        "what is Python?",
        "what is recursion?",
        "explain recursion",
        "explain polymorphism",
        "how does memoization work?",
        "what is a decorator?",
    ],
)
def test_world_knowledge_questions_classify_as_question(text):
    decision = classify_request(text)
    assert decision.intent == QUESTION
    assert decision.confidence == "high"
    assert decision.is_conversational
    assert not decision.requires_write


def test_hello_world_program_is_not_conversation():
    """The anchored patterns must not swallow legitimate coding requests."""
    decision = classify_request("write a hello world program in python")
    assert decision.intent != CONVERSATION


# ---------------------------------------------------------------------------
# Scenario 3: project inspection is read-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "show me the files in this project",
        "what files are in this project?",
        "list the files",
        "where is the database configured?",
        "where is authentication implemented?",
        "how does authentication work?",
        "what is in this project?",
        "explain this error",
        "find where the config is loaded",
    ],
)
def test_inspection_requests_are_read_only(text):
    decision = classify_request(text)
    assert decision.intent == PROJECT_INSPECTION
    assert decision.confidence == "high"
    assert decision.requires_read
    assert not decision.requires_write
    assert not decision.requires_command
    assert decision.intent in WRITE_EXCLUDED_INTENTS


# ---------------------------------------------------------------------------
# Coding / file-operation intents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "create a Python calculator",
        "create a file named example.py",
        "fix the login bug",
        "add dark mode",
        "rename this function",
        "run the tests",
        "delete the old file",
    ],
)
def test_coding_requests_authorize_work(text):
    decision = classify_request(text)
    assert decision.intent == CODE_CHANGE
    assert decision.requires_write
    assert decision.requires_command
    assert decision.intent not in WRITE_EXCLUDED_INTENTS


def test_delete_file_with_extension_is_file_operation_intent():
    decision = classify_request("delete example.py")
    assert decision.intent in (CODE_CHANGE, FILE_OPERATION)
    assert decision.requires_write


# ---------------------------------------------------------------------------
# Ambiguity discipline: terse work orders must remain workable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "make a.txt",
        "write hello",
        "do the thing",
        "anything",
        "task",
        "debug",
        "refactor",
        "keep going",
        "plan a change",
    ],
)
def test_terse_work_orders_stay_ambiguous_not_conversation(text):
    decision = classify_request(text)
    assert decision.intent == AMBIGUOUS
    # ambiguous is deliberately NOT write-excluded: the model decides inside
    # mode gating, and legitimate terse work orders must keep working.
    assert decision.intent not in WRITE_EXCLUDED_INTENTS


def test_ambiguous_does_not_preauthorize_writes():
    decision = classify_request("do the thing")
    assert not decision.requires_write
    assert not decision.requires_command


# ---------------------------------------------------------------------------
# Contract surface
# ---------------------------------------------------------------------------


def test_decision_has_required_fields():
    decision = classify_request("hello")
    for field in (
        "intent",
        "confidence",
        "requires_workspace",
        "requires_read",
        "requires_write",
        "requires_command",
        "requires_planning",
        "requires_verification",
        "scope",
        "reason",
    ):
        assert hasattr(decision, field), f"Decision missing {field}"


def test_intent_categories_exist():
    for category in (
        CONVERSATION,
        QUESTION,
        PROJECT_INSPECTION,
        CODE_CHANGE,
        FILE_OPERATION,
        AMBIGUOUS,
    ):
        assert category in INTENT_CATEGORIES


def test_classifier_is_pure_and_deterministic():
    first = classify_request("hello")
    second = classify_request("hello")
    assert first == second


def test_empty_and_whitespace_input_is_ambiguous():
    assert classify_request("").intent == AMBIGUOUS
    assert classify_request("   ").intent == AMBIGUOUS


# ---------------------------------------------------------------------------
# Intent-aware planner fallback
# ---------------------------------------------------------------------------


def test_fallback_for_conversation_is_review_not_implement():
    spec = fallback_spec_for("hello")
    assert spec["kind"] == "review"
    assert "Implement" not in spec["title"]


def test_fallback_for_question_is_review_not_implement():
    spec = fallback_spec_for("what is recursion?")
    assert spec["kind"] == "review"


def test_fallback_for_work_keeps_implement():
    spec = fallback_spec_for("Do the thing")
    assert spec["kind"] == "implement"
    assert spec["title"].startswith("Implement and verify")
