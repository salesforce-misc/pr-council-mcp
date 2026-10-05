"""Holistic LangChain aggregation and summary agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from localmcp.retry import is_retryable_error
from localmcp.structured_output import SubmitResultError, SubmitResultMiddleware

from pr_council.agents.review.common import (
    BASE_SAFETY,
    ResponseUsageCallback,
    Usage,
    extract_callback_usage,
)
from pr_council.review.models import AggregationCandidate, ReviewError

_AGGREGATION_PROMPT = """Aggregate overlapping retained findings and write a minimal top-level PR review summary.
The only valid source_finding_ids are the IDs listed in requiredSourceFindingIds. Every aggregate must cite one or
more of those IDs, and every required ID must occur exactly once across the aggregates. IDs appearing only in
baselineFindings or priorFindingAssessments are context identifiers, not source finding IDs. Do not create a new
issue. Preserve concrete locations. Keep each finding body concise but actionable and able to stand alone if the
finding cannot be posted inline. Never list or restate finding titles, locations, severity, evidence, explanations, or
remediation in the summary; those belong with the findings. The reviewKind input explicitly identifies an initial or
follow-up review. For an initial review, write exactly one short sentence stating only how many critical or
high-severity items need to be addressed before merge. For a follow-up review, write at most two short sentences
containing only status counts. Example: "2 previously reported issues were resolved; none remain unresolved. 1 new
issue was found." Never describe a new or prior issue, give its severity, make a merge recommendation, or use a colon
to introduce issue details in a follow-up summary. Do not include any other narrative."""


async def aggregate(
    model: BaseChatModel,
    *,
    findings: list[dict[str, object]],
    prior_assessments: list[dict[str, object]],
    baseline_findings: list[dict[str, object]],
    context_changes: list[dict[str, object]],
    context: list[dict[str, object]],
    validate_candidate: Callable[[AggregationCandidate], AggregationCandidate] | None = None,
    usage: Usage | None = None,
    langfuse_callback: Any | None = None,
) -> tuple[AggregationCandidate, Usage]:
    # The same unforced submission tool as the reviewers. This is a cross-model requirement: forced tool_choice
    # and provider-native schemas are not supported by every model and gateway, so the unforced tool and its
    # correction rounds are the lowest common denominator.
    agent = create_agent(
        model,
        [],
        system_prompt=f"{BASE_SAFETY}\n\n{_AGGREGATION_PROMPT}",
        middleware=[SubmitResultMiddleware(AggregationCandidate, max_attempts=3)],
        name="pr-review-aggregator",
    )
    required_source_ids = [str(finding["id"]) for finding in findings]
    messages: list[AnyMessage | dict[str, Any]] = [
        HumanMessage(
            content=json.dumps(
                {
                    "reviewKind": "follow_up" if baseline_findings else "initial",
                    "retainedFindings": findings,
                    "requiredSourceFindingIds": required_source_ids,
                    "priorFindingAssessments": prior_assessments,
                    "baselineFindings": baseline_findings,
                    "reviewContextChangesSinceBaseline": context_changes,
                    "reviewContext": context,
                }
            )
        ),
    ]
    # Accumulate into a caller-supplied Usage when given so token spend from an
    # invocation that later re-raises a transient error (retried by the caller's
    # call_with_rate_limit_retry) is preserved across outer attempts rather than
    # discarded with the raised exception.
    usage = usage if usage is not None else Usage()
    last_error: object = "structured output was missing"
    # Outer correction rounds re-prompt with the error when the submission tool's own rounds are exhausted or a
    # well-formed result fails application validation. Transport faults go to the caller's backoff instead.
    for _attempt in range(3):
        usage_callback = ResponseUsageCallback()
        callbacks: list[Any] = [usage_callback]
        if langfuse_callback is not None:
            callbacks.append(langfuse_callback)
        config: RunnableConfig = {"run_name": "pr-review-aggregator", "callbacks": callbacks}
        candidate: AggregationCandidate | None = None
        try:
            result = await agent.ainvoke({"messages": messages}, config=config)
        except SubmitResultError as exc:
            last_error = exc
        except Exception as exc:
            # Transient transport/API faults (429, 5xx, connection, timeout) are not
            # fixable by the correction retries below: re-raise them so the caller's
            # call_with_rate_limit_retry applies its reset-aware backoff instead of
            # burning all three local attempts with no delay.
            if is_retryable_error(exc):
                raise
            last_error = exc
        else:
            candidate = result["structured_response"]
            # Continue from the submitted result so a validation correction has the prior answer in context.
            messages = list(result["messages"])
        finally:
            _add_usage(usage, extract_callback_usage(usage_callback))
        if candidate is None:
            # Feed the failure back: aggregation runs at temperature 0, so without a
            # corrective message every retry replays the same malformed output
            # against identical input and wastes the attempt.
            messages.append(
                HumanMessage(
                    content=json.dumps(
                        {
                            "correction": "The previous attempt did not submit a valid result. "
                            "Submit a corrected AggregationCandidate.",
                            "schemaError": str(last_error),
                            "requiredSourceFindingIds": required_source_ids,
                        }
                    )
                )
            )
            continue
        if validate_candidate is None:
            return candidate, usage
        try:
            return validate_candidate(candidate), usage
        except (ReviewError, ValueError) as exc:
            last_error = exc
            messages.append(
                HumanMessage(
                    content=json.dumps(
                        {
                            "correction": "The previous aggregation failed validation. Submit a corrected result.",
                            "validationError": str(exc),
                            "requiredSourceFindingIds": required_source_ids,
                        }
                    )
                )
            )
    raise ReviewError(f"aggregation did not return a valid result after 3 attempts: {last_error}")


def _add_usage(total: Usage, current: Usage) -> None:
    total.input_tokens += current.input_tokens
    total.output_tokens += current.output_tokens
    total.total_tokens += current.total_tokens
    total.tool_calls += current.tool_calls
    total.complete = total.complete and current.complete
