import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from pr_council.agents.review.aggregator import aggregate
from pr_council.agents.review.common import Usage
from pr_council.review.models import AggregatedFinding, AggregationCandidate, Severity


def _finding(source_id: str, *, path: str = "app.py", line: int = 1, severity: str = "medium") -> dict[str, Any]:
    return {
        "id": source_id,
        "title": f"Title {source_id}",
        "body": f"Body {source_id}.",
        "severity": severity,
        "path": path,
        "line": line,
        "disposition": "quality",
        "model": "model-a",
        "evidence": "",
        "context_refs": [],
    }


def _call(ids: list[str], *, path: str = "app.py", line: int = 1, call_id: str = "m1") -> dict[str, Any]:
    args = {"source_finding_ids": ids, "title": "Merged title", "body": "Merged body.", "path": path, "line": line}
    return {"name": "merge_findings", "args": args, "id": call_id, "type": "tool_call"}


def _merge(ids: list[str], **values: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[_call(ids, **values)])


def _done() -> AIMessage:
    return AIMessage(content="All merges are recorded.")


def _rate_limited() -> RuntimeError:
    error = RuntimeError("Error code: 429 - rate limit exceeded")
    error.status_code = 429  # type: ignore[attr-defined]
    return error


def _reported(message: AIMessage) -> AIMessage:
    message.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    message.response_metadata = {"model_name": "fake"}
    return message


async def _aggregate(model: Any, **values: Any) -> tuple[AggregationCandidate, Usage]:
    defaults: dict[str, Any] = {
        "findings": [_finding("a"), _finding("b", line=2)],
        "prior_assessments": [],
        "baseline_findings": [],
        "context_changes": [],
        "context": [],
    }
    return await aggregate(model, **{**defaults, **values})


def _tool_replies(request: list[Any]) -> dict[str, Any]:
    return {message.tool_call_id: message for message in request if message.type == "tool"}


async def test_unmerged_findings_are_published_exactly_as_written(scripted_model, run_recorder):
    findings = [_finding("a", severity="low"), _finding("b", path="lib.py", line=7, severity="high")]
    model = scripted_model(_done())
    langfuse_callback = run_recorder

    candidate, _usage = await _aggregate(model, findings=findings, langfuse_callback=langfuse_callback)

    assert [
        (finding.title, finding.body, finding.severity, finding.path, finding.line, finding.source_finding_ids)
        for finding in candidate.findings
    ] == [
        ("Title a", "Body a.", Severity.LOW, "app.py", 1, ["a"]),
        ("Title b", "Body b.", Severity.HIGH, "lib.py", 7, ["b"]),
    ]
    assert model.bound[0]["tools"] == ["merge_findings"]
    assert model.bound[0].get("tool_choice") is None
    assert langfuse_callback.starts == 1
    prompt = " ".join(model.requests[0][0].content.split())
    assert "Merge findings only when they describe the same defect" in prompt
    assert "Distinct defects at the same location stay separate" in prompt
    assert "findings in different places that each need their own change, even when they share a cause" in prompt
    assert "Findings you do not merge are published exactly as written" in prompt
    # The top-level summary is rendered from counts after aggregation, so the model is never asked for one.
    assert "summary" not in prompt.lower()
    payload = json.loads(model.requests[0][1].content)
    assert [finding["id"] for finding in payload["retainedFindings"]] == ["a", "b"]
    assert "reviewKind" not in payload


async def test_a_merge_takes_the_position_of_its_first_member(scripted_model):
    findings = [
        _finding("a", severity="low"),
        _finding("b", path="lib.py", line=7),
        _finding("c", line=5, severity="high"),
    ]
    model = scripted_model(_merge(["c", "a"], line=5), _done())

    candidate, _usage = await _aggregate(model, findings=findings)

    merged, unmerged = candidate.findings
    assert (merged.title, merged.body, merged.path, merged.line) == ("Merged title", "Merged body.", "app.py", 5)
    assert merged.source_finding_ids == ["c", "a"]
    assert merged.severity == Severity.HIGH
    assert unmerged.source_finding_ids == ["b"]
    assert _tool_replies(model.requests[1])["m1"].content == (
        "Merged 2 findings. 1 unmerged findings will be published as written."
    )


async def test_several_merges_in_one_turn_are_all_recorded(scripted_model):
    findings = [_finding("a"), _finding("b"), _finding("c", line=3), _finding("d", line=3)]
    turn = AIMessage(
        content="",
        tool_calls=[_call(["a", "b"], call_id="m1"), _call(["c", "d"], line=3, call_id="m2")],
    )
    model = scripted_model(turn, _done())

    candidate, _usage = await _aggregate(model, findings=findings)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a", "b"], ["c", "d"]]
    assert len(model.requests) == 2


async def test_rejected_merges_record_nothing_and_are_returned_to_the_model(scripted_model):
    findings = [_finding("a"), _finding("b"), _finding("c", line=3), _finding("d", line=4)]
    rejected = AIMessage(
        content="",
        tool_calls=[
            _call(["b", "c"], call_id="merged-twice"),
            _call(["c", "baseline-id"], line=3, call_id="unknown"),
            _call(["c", "c"], line=3, call_id="repeated"),
            _call(["c", "d"], line=9, call_id="moved"),
            _call(["c"], line=3, call_id="single"),
        ],
    )
    model = scripted_model(_merge(["a", "b"]), rejected, _done())

    candidate, _usage = await _aggregate(model, findings=findings)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a", "b"], ["c"], ["d"]]
    replies = _tool_replies(model.requests[2])
    assert all(replies[call_id].status == "error" for call_id in ("merged-twice", "unknown", "repeated", "moved"))
    assert replies["merged-twice"].content == "Not recorded: ['b'] were already merged."
    assert replies["unknown"].content == "Not recorded: ['baseline-id'] are not retainedFindings IDs."
    assert replies["repeated"].content == "Not recorded: list each ID once."
    assert replies["moved"].content == "Not recorded: use the path and line of one of the merged findings."
    assert "source_finding_ids" in replies["single"].content


async def test_an_unparsed_tool_call_is_answered_and_the_run_continues(scripted_model):
    truncated = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "merge_findings",
                "args": '{"source_finding_ids": ["a"',
                "id": "truncated",
                "error": None,
                "type": "invalid_tool_call",
            }
        ],
    )
    model = scripted_model(truncated, _merge(["a", "b"]), _done())

    candidate, _usage = await _aggregate(model)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a", "b"]]
    retry = model.requests[1]
    replayed = next(message for message in retry if message.type == "ai")
    assert replayed.invalid_tool_calls == []
    assert [(call["id"], call["args"]) for call in replayed.tool_calls] == [("truncated", {})]
    assert _tool_replies(retry)["truncated"].status == "error"


@pytest.mark.parametrize(
    "metadata",
    [
        {"stop_reason": "max_tokens"},
        {"finish_reason": "length"},
        {"incomplete_details": {"reason": "max_output_tokens"}},
    ],
)
async def test_a_reply_cut_off_before_any_tool_call_is_dropped_and_the_model_reminded(scripted_model, metadata):
    # Reasoning can use the whole output limit; ending there would publish every finding unmerged.
    cut_off = AIMessage(content="First, the findings that overlap", response_metadata=metadata)
    model = scripted_model(cut_off, _merge(["a", "b"]), _done())

    candidate, _usage = await _aggregate(model)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a", "b"]]
    retry = model.requests[1]
    assert [message.type for message in retry[1:]] == ["human", "human"]
    assert "reached the output limit before it made a tool call" in retry[-1].content


async def test_repeated_cut_off_replies_end_the_run_with_the_merges_already_recorded(scripted_model):
    findings = [_finding("a"), _finding("b"), _finding("c", line=3), _finding("d", line=4)]
    cut_off = [AIMessage(content="", response_metadata={"stop_reason": "max_tokens"}) for _ in range(3)]
    model = scripted_model(_merge(["a", "b"]), *cut_off)
    incomplete: list[Exception] = []

    candidate, _usage = await _aggregate(model, findings=findings, on_incomplete=incomplete.append)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a", "b"], ["c"], ["d"]]
    assert [type(exc).__name__ for exc in incomplete] == ["OutputLimitError"]
    assert len(model.requests) == 4


async def test_a_cut_off_reply_ends_the_run_once_no_merge_remains_possible(scripted_model):
    findings = [_finding("a"), _finding("b"), _finding("c", line=3)]
    cut_off = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
    model = scripted_model(_merge(["a", "b"]), cut_off)
    incomplete: list[Exception] = []

    candidate, _usage = await _aggregate(model, findings=findings, on_incomplete=incomplete.append)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a", "b"], ["c"]]
    assert incomplete == []
    assert len(model.requests) == 2


async def test_the_last_call_of_a_cut_off_reply_is_rejected_and_the_rest_recorded(scripted_model):
    # The final call of a truncated response can still parse with its text cut short.
    findings = [_finding("a"), _finding("b"), _finding("c", line=3), _finding("d", line=3)]
    cut_off = AIMessage(
        content="",
        tool_calls=[_call(["a", "b"], call_id="m1"), _call(["c", "d"], line=3, call_id="m2")],
        response_metadata={"stop_reason": "max_tokens"},
    )
    model = scripted_model(cut_off, _merge(["c", "d"], line=3, call_id="m3"), _done())

    candidate, _usage = await _aggregate(model, findings=findings)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a", "b"], ["c", "d"]]
    replies = _tool_replies(model.requests[1])
    assert replies["m1"].status == "success"
    assert (replies["m2"].status, replies["m2"].content) == (
        "error",
        "Not recorded: this call was cut off at the output limit. Record it again.",
    )


async def test_a_deterministic_failure_keeps_the_merges_already_recorded(scripted_model):
    findings = [_finding("a"), _finding("b"), _finding("c", line=3)]
    model = scripted_model(_merge(["a", "b"]), RuntimeError("context too long"))
    incomplete: list[Exception] = []

    candidate, _usage = await _aggregate(model, findings=findings, on_incomplete=incomplete.append)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a", "b"], ["c"]]
    assert [str(exc) for exc in incomplete] == ["context too long"]


async def test_a_run_that_never_finishes_ends_at_the_recursion_limit(scripted_model):
    model = scripted_model(*(_merge(["a", "unknown"], call_id=f"m{index}") for index in range(60)))
    incomplete: list[Exception] = []

    candidate, _usage = await _aggregate(model, on_incomplete=incomplete.append)

    assert [finding.source_finding_ids for finding in candidate.findings] == [["a"], ["b"]]
    assert [type(exc).__name__ for exc in incomplete] == ["GraphRecursionError"]


async def test_fewer_than_two_findings_never_call_the_model(scripted_model):
    model = scripted_model()

    single, _usage = await _aggregate(model, findings=[_finding("a")])
    empty, _usage = await _aggregate(model, findings=[])

    assert [finding.source_finding_ids for finding in single.findings] == [["a"]]
    assert empty.findings == []
    assert model.requests == []


async def test_aggregation_reraises_retryable_error_for_outer_backoff(scripted_model):
    # A transient 429 must propagate so the caller's call_with_rate_limit_retry applies reset-aware backoff.
    model = scripted_model(_rate_limited())

    with pytest.raises(RuntimeError):
        await _aggregate(model)

    assert len(model.requests) == 1


async def test_aggregation_preserves_usage_when_transient_error_reraises(scripted_model):
    # Tokens burned before a transient re-raise must survive in the caller-supplied
    # accumulator so the outer retry does not under-report aggregation spend.
    model = scripted_model(_reported(_merge(["a", "b"], line=2)), _rate_limited())
    usage = Usage()

    with pytest.raises(RuntimeError):
        await _aggregate(model, usage=usage)

    assert len(model.requests) == 2
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (10, 2, 12)


async def test_rejected_request_does_not_mark_retried_usage_incomplete(scripted_model):
    # The caller's rate-limit retry reuses the accumulator; a request rejected before any response must not
    # leave a later, fully reported attempt marked incomplete.
    usage = Usage()
    with pytest.raises(RuntimeError):
        await _aggregate(scripted_model(_rate_limited()), usage=usage)

    await _aggregate(scripted_model(_reported(_done())), usage=usage)

    assert usage.complete is True
    assert usage.total_tokens == 12


async def test_unreported_turn_marks_usage_incomplete(scripted_model):
    model = scripted_model(_merge(["a", "b"], line=2), _reported(_done()))

    _candidate, usage = await _aggregate(model)

    assert len(model.requests) == 2
    assert usage.total_tokens == 12
    assert usage.complete is False


def test_aggregated_finding_derives_dispositions_and_reviewer_models():
    # These server-derived fields default empty so _validate_aggregation can populate them from the source findings.
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
