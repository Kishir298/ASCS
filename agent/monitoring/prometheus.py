"""Prometheus metrics exporter for ASCS."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Task execution metrics
ascs_tasks_total = Counter(
    "ascs_tasks_total",
    "Total tasks executed",
    ["status"],  # completed, failed, cancelled, skipped
)
ascs_task_latency_seconds = Histogram(
    "ascs_task_latency_seconds",
    "Task execution latency",
)
ascs_task_iterations_total = Counter(
    "ascs_task_iterations_total",
    "Total task iterations",
)

# Tool execution metrics
ascs_tool_executions_total = Counter(
    "ascs_tool_executions_total",
    "Total tool executions",
    ["tool", "status"],
)
ascs_tool_latency_seconds = Histogram(
    "ascs_tool_latency_seconds",
    "Tool execution latency",
    ["tool"],
)

# Verification metrics
ascs_verification_total = Counter(
    "ascs_verification_total",
    "Total verifications",
    ["status"],  # passed, failed
)
ascs_verification_retries_total = Counter(
    "ascs_verification_retries_total",
    "Total verification retries",
)
ascs_verification_latency_seconds = Histogram(
    "ascs_verification_latency_seconds",
    "Verification latency",
)

# Git metrics
ascs_git_operations_total = Counter(
    "ascs_git_operations_total",
    "Total git operations",
    ["operation", "status"],
)

# File operations
ascs_file_operations_total = Counter(
    "ascs_file_operations_total",
    "Total file operations",
    ["operation", "status"],
)
ascs_files_changed_total = Counter(
    "ascs_files_changed_total",
    "Total files changed",
)

# Loop metrics
ascs_loop_iterations_total = Counter(
    "ascs_loop_iterations_total",
    "Total loop iterations",
)
ascs_loop_duration_seconds = Histogram(
    "ascs_loop_duration_seconds",
    "Loop iteration duration",
)

# Experience/pipeline metrics
ascs_experience_stored_total = Counter(
    "ascs_experience_stored_total",
    "Total experience entries stored",
)
ascs_experience_retrieved_total = Counter(
    "ascs_experience_retrieved_total",
    "Total experience entries retrieved",
)


def metrics_endpoint() -> Response:
    """FastAPI endpoint for Prometheus metrics."""
    from fastapi import Response as FastAPIResponse
    return FastAPIResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )