import json
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from pr_council.agents.review.aggregator import aggregate
from pr_council.agents.review.common import Usage
from pr_council.review.models import AggregatedFinding, AggregationCandidate, ReviewError, Severity


class FakeStructuredModel:
    def __init__(self):
        self.calls = 0
        self.configs = []
        self.messages = []
        self.structured_output_kwargs = {}

    def with_structured_output(self, *args, **kwargs):
        self.structured_output_kwargs = kwargs
        return self

    async def ainvoke(self, messages, config=None):
        self.configs.append(config)
        self.messages.append(list(messages))
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary gateway failure")
        if self.calls == 2:
            return {"parsed": None, "parsing_error": ValueError("malformed"), "raw": None}
        return {
            "parsed": AggregationCandidate(summary="Brief outcome.", findings=[]),
            "parsing_error": None,
            "raw": None,
        }


async def test_aggregation_retries_malformed_structured_output():
    model = FakeStructuredModel()
    langfuse_callback = object()

    candidate, usage = await aggregate(
        cast(BaseChatModel, cast(Any, model)),
        findings=[],
        prior_assessments=[],
        baseline_findings=[],
        context_changes=[],
        context=[],
        langfuse_callback=langfuse_callback,
    )

    assert candidate.summary == "Brief outcome."
    assert model.structured_output_kwargs == {"method": "json_schema", "include_raw": True}
    assert model.calls == 3
    assert usage.complete is False
    assert model.configs == [{"run_name": "pr-review-aggregator", "callbacks": [langfuse_callback]}] * 3
    prompt = model.messages[0][0].content
    assert "exactly one short sentence" in prompt
    assert "at most two short sentences" in prompt
    assert "containing only status counts" in prompt
    assert "Never describe a new or prior issue" in prompt
    assert "Do not include any other narrative" in prompt
    payload = json.loads(model.messages[0][1].content)
    assert payload["reviewKind"] == "initial"


async def test_aggregation_omits_callback_when_not_provided():
    # With no langfuse_callback the per-attempt config must carry an empty callbacks
    # list, never a [None] entry. This mirrors the reviewer/deliberator omit-branch tests.
    model = FakeStructuredModel()

    candidate, _usage = await aggregate(
        cast(BaseChatModel, cast(Any, model)),
        findings=[],
        prior_assessments=[],
        baseline_findings=[],
        context_changes=[],
        context=[],
    )

    assert candidate.summary == "Brief outcome."
    assert model.calls == 3
    assert model.configs == [{"run_name": "pr-review-aggregator", "callbacks": []}] * 3


class RateLimitedStructuredModel:
    """Raises a transient gateway 429 on every structured invocation."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        error = RuntimeError("Error code: 429 - rate limit exceeded")
        error.status_code = 429  # type: ignore[attr-defined]
        raise error


async def test_aggregation_reraises_retryable_error_for_outer_backoff():
    # A transient 429 must propagate on the first attempt so the caller's
    # call_with_rate_limit_retry applies reset-aware backoff, rather than being
    # swallowed by the local schema-correction loop and burning all three attempts.
    model = RateLimitedStructuredModel()

    with pytest.raises(RuntimeError):
        await aggregate(
            cast(BaseChatModel, cast(Any, model)),
            findings=[],
            prior_assessments=[],
            baseline_findings=[],
            context_changes=[],
            context=[],
        )

    assert model.calls == 1


class PartialUsageThenRateLimitModel:
    """Extracts usage from a schema-correction turn, then hits a transient 429."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        if self.calls == 1:
            # A billable turn whose output fails schema parsing: usage is extracted
            # and the loop appends a correction before retrying.
            return {
                "parsed": None,
                "parsing_error": ValueError("schema"),
                "raw": AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                ),
            }
        error = RuntimeError("Error code: 429 - rate limit exceeded")
        error.status_code = 429  # type: ignore[attr-defined]
        raise error


async def test_aggregation_preserves_usage_when_transient_error_reraises():
    # Tokens burned before a transient re-raise must survive in the caller-supplied
    # accumulator so the outer retry does not under-report aggregation spend.
    model = PartialUsageThenRateLimitModel()
    usage = Usage()

    with pytest.raises(RuntimeError):
        await aggregate(
            cast(BaseChatModel, cast(Any, model)),
            findings=[],
            prior_assessments=[],
            baseline_findings=[],
            context_changes=[],
            context=[],
            usage=usage,
        )

    assert model.calls == 2
    assert usage.total_tokens == 120
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20


class FollowUpStructuredModel:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def with_structured_output(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        self.messages.append(list(messages))
        source_id = "baseline-id" if self.calls == 1 else "synthetic-retained-id"
        return {
            "parsed": AggregationCandidate(
                summary="A prior finding remains actionable.",
                findings=[
                    AggregatedFinding(
                        id="model-id",
                        title="Still broken",
                        body="The issue remains.",
                        severity=Severity.HIGH,
                        path="app.py",
                        line=1,
                        dispositions=[],
                        reviewer_models=[],
                        source_finding_ids=[source_id],
                    )
                ],
            ),
            "parsing_error": None,
            "raw": None,
        }


def test_aggregated_finding_derives_dispositions_and_reviewer_models():
    # The model may omit these server-derived fields; parsing must still succeed
    # so _validate_aggregation can populate them from the source findings.
    finding = AggregatedFinding(
        id="model-id",
        title="Something",
        body="Detail.",
        severity=Severity.HIGH,
        path="app.py",
        line=1,
        source_finding_ids=["source-1"],
    )
    assert finding.dispositions == []
    assert finding.reviewer_models == []


class SchemaFailureStructuredModel:
    """First response fails schema parsing; the retry returns a valid candidate."""

    def __init__(self):
        self.calls = 0
        self.messages = []

    def with_structured_output(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        self.messages.append(list(messages))
        if self.calls == 1:
            return {
                "parsed": None,
                "parsing_error": ValueError("findings.0.dispositions Field required"),
                "raw": None,
            }
        return {
            "parsed": AggregationCandidate(summary="Recovered.", findings=[]),
            "parsing_error": None,
            "raw": None,
        }


async def test_aggregation_feeds_schema_error_back_on_parse_failure():
    model = SchemaFailureStructuredModel()

    candidate, _usage = await aggregate(
        cast(BaseChatModel, cast(Any, model)),
        findings=[],
        prior_assessments=[],
        baseline_findings=[],
        context_changes=[],
        context=[],
    )

    assert candidate.summary == "Recovered."
    assert model.calls == 2
    correction = json.loads(model.messages[1][-1].content)
    assert "findings.0.dispositions" in correction["schemaError"]


async def test_aggregation_retries_follow_up_source_id_coverage_failure():
    model = FollowUpStructuredModel()

    def validate(candidate):
        used = candidate.findings[0].source_finding_ids
        if used != ["synthetic-retained-id"]:
            raise ReviewError("missing synthetic-retained-id; baseline-id is contextual")
        return candidate

    candidate, _usage = await aggregate(
        cast(BaseChatModel, cast(Any, model)),
        findings=[{"id": "synthetic-retained-id"}],
        prior_assessments=[{"finding_id": "baseline-id", "status": "still_present"}],
        baseline_findings=[{"id": "baseline-id"}],
        context_changes=[],
        context=[],
        validate_candidate=validate,
    )

    initial_payload = json.loads(model.messages[0][1].content)
    assert initial_payload["requiredSourceFindingIds"] == ["synthetic-retained-id"]
    assert candidate.findings[0].source_finding_ids == ["synthetic-retained-id"]
    assert model.calls == 2
    payload = json.loads(model.messages[0][1].content)
    assert payload["reviewKind"] == "follow_up"
    correction = json.loads(model.messages[1][-1].content)
    assert correction["requiredSourceFindingIds"] == ["synthetic-retained-id"]
    assert "baseline-id is contextual" in correction["validationError"]
