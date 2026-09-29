import json
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from pr_council.agents.review.aggregator import aggregate
from pr_council.agents.review.common import Usage
from pr_council.review.models import AggregatedFinding, AggregationCandidate, ReviewError, Severity


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted turns in order; an exception entry is raised from that model call."""

    script: list[Any] = []
    requests: list[list[Any]] = []
    bound: list[dict[str, Any]] = []

    def bind_tools(self, tools, **kwargs):
        self.bound.append({"tools": [getattr(tool, "name", None) for tool in tools], **kwargs})
        return self

    def _generate(self, messages, *args, **kwargs):
        self.requests.append(list(messages))
        turn = self.script.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        self.messages = iter([turn])
        return super()._generate(messages, *args, **kwargs)


def _model(*turns: Any) -> ScriptedModel:
    return ScriptedModel(messages=iter([]), script=list(turns), requests=[], bound=[])


def _submit(candidate: AggregationCandidate | dict[str, Any], call_id: str = "s1") -> AIMessage:
    args = candidate.model_dump(mode="json") if isinstance(candidate, AggregationCandidate) else candidate
    return AIMessage(
        content="", tool_calls=[{"name": "submit_result", "args": args, "id": call_id, "type": "tool_call"}]
    )


def _rate_limited() -> RuntimeError:
    error = RuntimeError("Error code: 429 - rate limit exceeded")
    error.status_code = 429  # type: ignore[attr-defined]
    return error


def _finding(source_id: str) -> AggregatedFinding:
    return AggregatedFinding(
        id="model-id",
        title="Still broken",
        body="The issue remains.",
        severity=Severity.HIGH,
        path="app.py",
        line=1,
        source_finding_ids=[source_id],
    )


async def _aggregate(model: ScriptedModel, **values: Any) -> tuple[AggregationCandidate, Usage]:
    defaults: dict[str, Any] = {
        "findings": [],
        "prior_assessments": [],
        "baseline_findings": [],
        "context_changes": [],
        "context": [],
    }
    return await aggregate(model, **{**defaults, **values})


def _corrections(request: list[Any]) -> list[dict[str, Any]]:
    payloads = [json.loads(m.content) for m in request if m.type == "human" and m.content.startswith("{")]
    return [payload for payload in payloads if "correction" in payload]


class _ChatStarts(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.starts += 1


async def test_aggregation_submits_through_an_unforced_tool():
    model = _model(_submit(AggregationCandidate(summary="Brief outcome.", findings=[])))
    langfuse_callback = _ChatStarts()

    candidate, usage = await _aggregate(model, langfuse_callback=langfuse_callback)

    assert candidate.summary == "Brief outcome."
    assert model.bound[0]["tools"] == ["submit_result"]
    assert model.bound[0].get("tool_choice") is None
    assert langfuse_callback.starts == 1
    assert usage.complete is False
    prompt = model.requests[0][0].content
    assert "exactly one short sentence" in prompt
    assert "at most two short sentences" in prompt
    assert "containing only status counts" in prompt
    assert "Never describe a new or prior issue" in prompt
    assert "Do not include any other narrative" in prompt
    payload = json.loads(model.requests[0][1].content)
    assert payload["reviewKind"] == "initial"


async def test_aggregation_runs_without_a_langfuse_callback():
    model = _model(_submit(AggregationCandidate(summary="Brief outcome.", findings=[])))

    candidate, _usage = await _aggregate(model)

    assert candidate.summary == "Brief outcome."


async def test_aggregation_retries_after_a_non_retryable_failure():
    model = _model(
        RuntimeError("temporary gateway failure"),
        _submit(AggregationCandidate(summary="Recovered.", findings=[])),
    )

    candidate, _usage = await _aggregate(model)

    assert candidate.summary == "Recovered."
    assert len(model.requests) == 2
    assert "temporary gateway failure" in _corrections(model.requests[1])[0]["schemaError"]


async def test_aggregation_returns_schema_errors_to_the_model():
    model = _model(
        _submit({"summary": "Missing findings.", "findings": [{"id": "x"}]}),
        _submit(AggregationCandidate(summary="Recovered.", findings=[]), "s2"),
    )

    candidate, _usage = await _aggregate(model)

    assert candidate.summary == "Recovered."
    error = next(m for m in model.requests[1] if m.type == "tool")
    assert error.status == "error"
    assert "findings.0." in error.content


async def test_aggregation_corrects_after_submission_attempts_are_exhausted():
    model = _model(
        AIMessage(content="no"),
        AIMessage(content="no"),
        AIMessage(content="no"),
        _submit(AggregationCandidate(summary="Recovered.", findings=[])),
    )

    candidate, _usage = await _aggregate(model)

    assert candidate.summary == "Recovered."
    assert len(model.requests) == 4
    assert "did not submit a valid" in _corrections(model.requests[3])[0]["schemaError"]


async def test_aggregation_reraises_retryable_error_for_outer_backoff():
    # A transient 429 must propagate on the first attempt so the caller's
    # call_with_rate_limit_retry applies reset-aware backoff, rather than being
    # swallowed by the local correction loop and burning all three attempts.
    model = _model(_rate_limited())

    with pytest.raises(RuntimeError):
        await _aggregate(model)

    assert len(model.requests) == 1


async def test_aggregation_preserves_usage_when_transient_error_reraises():
    # Tokens burned before a transient re-raise must survive in the caller-supplied
    # accumulator so the outer retry does not under-report aggregation spend.
    model = _model(
        AIMessage(
            content="no",
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            response_metadata={"model_name": "fake"},
        ),
        _rate_limited(),
    )
    usage = Usage()

    with pytest.raises(RuntimeError):
        await _aggregate(model, usage=usage)

    assert len(model.requests) == 2
    assert usage.total_tokens == 120
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20


def _reported(message: AIMessage) -> AIMessage:
    message.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    message.response_metadata = {"model_name": "fake"}
    return message


async def test_rejected_request_does_not_mark_retried_usage_incomplete():
    # The caller's rate-limit retry reuses the accumulator; a request rejected before any response must not
    # leave a later, fully reported attempt marked incomplete.
    usage = Usage()
    with pytest.raises(RuntimeError):
        await _aggregate(_model(_rate_limited()), usage=usage)

    await _aggregate(_model(_reported(_submit(AggregationCandidate(summary="Done.", findings=[])))), usage=usage)

    assert usage.complete is True
    assert usage.total_tokens == 12


async def test_unreported_correction_turn_marks_usage_incomplete():
    model = _model(
        AIMessage(content="no"),
        _reported(_submit(AggregationCandidate(summary="Done.", findings=[]))),
    )

    _candidate, usage = await _aggregate(model)

    assert len(model.requests) == 2
    assert usage.total_tokens == 12
    assert usage.complete is False


async def test_aggregation_fails_after_three_attempts():
    model = _model(*(RuntimeError(f"failure {i}") for i in range(3)))

    with pytest.raises(ReviewError, match="after 3 attempts: failure 2"):
        await _aggregate(model)


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


async def test_aggregation_retries_follow_up_source_id_coverage_failure():
    model = _model(
        _submit(AggregationCandidate(summary="A prior finding remains.", findings=[_finding("baseline-id")])),
        _submit(
            AggregationCandidate(summary="A prior finding remains.", findings=[_finding("synthetic-retained-id")]),
            "s2",
        ),
    )

    def validate(candidate):
        used = candidate.findings[0].source_finding_ids
        if used != ["synthetic-retained-id"]:
            raise ReviewError("missing synthetic-retained-id; baseline-id is contextual")
        return candidate

    candidate, _usage = await _aggregate(
        model,
        findings=[{"id": "synthetic-retained-id"}],
        prior_assessments=[{"finding_id": "baseline-id", "status": "still_present"}],
        baseline_findings=[{"id": "baseline-id"}],
        validate_candidate=validate,
    )

    assert candidate.findings[0].source_finding_ids == ["synthetic-retained-id"]
    assert len(model.requests) == 2
    payload = json.loads(model.requests[0][1].content)
    assert payload["reviewKind"] == "follow_up"
    assert payload["requiredSourceFindingIds"] == ["synthetic-retained-id"]
    retry = model.requests[1]
    assert any(m.type == "ai" and m.tool_calls and m.tool_calls[0]["id"] == "s1" for m in retry)
    correction = _corrections(retry)[0]
    assert correction["requiredSourceFindingIds"] == ["synthetic-retained-id"]
    assert "baseline-id is contextual" in correction["validationError"]
