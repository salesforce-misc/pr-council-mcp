"""Holistic LangChain aggregation and summary agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pr_council.agents.review.common import BASE_SAFETY, Usage, extract_usage, is_retryable_error
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
    structured = model.with_structured_output(AggregationCandidate, method="json_schema", include_raw=True)
    required_source_ids = [str(finding["id"]) for finding in findings]
    messages = [
        SystemMessage(content=f"{BASE_SAFETY}\n\n{_AGGREGATION_PROMPT}"),
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
    for _attempt in range(3):
        try:
            callbacks = [langfuse_callback] if langfuse_callback is not None else []
            result = await structured.ainvoke(
                messages,
                config={"run_name": "pr-review-aggregator", "callbacks": callbacks},
            )
        except Exception as exc:
            # Transient transport/API faults (429, 5xx, connection, timeout) are not
            # fixable by the schema-correction retries below: re-raise them so the
            # caller's call_with_rate_limit_retry applies its reset-aware backoff
            # instead of burning all three local attempts with no delay.
            if is_retryable_error(exc):
                raise
            last_error = exc
            continue
        result_dict = cast(dict[str, Any], result)
        raw = result_dict.get("raw")
        current_usage = extract_usage([raw] if raw is not None else [])
        usage.input_tokens += current_usage.input_tokens
        usage.output_tokens += current_usage.output_tokens
        usage.total_tokens += current_usage.total_tokens
        usage.tool_calls += current_usage.tool_calls
        usage.complete = usage.complete and current_usage.complete
        parsed = result_dict.get("parsed")
        candidate: AggregationCandidate | None = None
        if isinstance(parsed, AggregationCandidate):
            candidate = parsed
        elif parsed is not None:
            try:
                candidate = AggregationCandidate.model_validate(parsed)
            except ValueError as exc:
                last_error = exc
        else:
            last_error = result_dict.get("parsing_error") or last_error
        if candidate is None:
            # Feed the schema failure back: aggregation runs at temperature 0, so
            # without a corrective message every retry replays the same malformed
            # output against identical input and wastes the attempt.
            messages.append(
                HumanMessage(
                    content=json.dumps(
                        {
                            "correction": "The previous response did not match the required schema. "
                            "Return a corrected AggregationCandidate.",
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
                            "correction": "The previous aggregation failed validation. Return a corrected result.",
                            "validationError": str(exc),
                            "requiredSourceFindingIds": required_source_ids,
                        }
                    )
                )
            )
    raise ReviewError(f"aggregation did not return a valid result after 3 attempts: {last_error}")
