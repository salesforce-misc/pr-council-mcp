import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from localmcp.retry import ModelCallRetryMiddleware
from localmcp.sandbox import RootAccess, SandboxProfile, SandboxRoot

import pr_council.agents.review.deliberator as deliberator_module
from pr_council.agents.review.common import SEVERITY_RUBRIC, ToolBudget, Usage
from pr_council.agents.review.deliberator import DeliberationCandidate, deliberate


@pytest.fixture(autouse=True)
def _fake_sandbox(monkeypatch):
    # Stands in for the sandboxed Bash tool: it claims the budget like the real one, without running anything.
    def sandbox_tools(profile, *, budget, timeout_seconds):
        async def bash(command: str) -> dict[str, Any]:
            await budget.claim()
            return {"result": {"stdout": "", "exit_code": 0}, "tool_budget": budget.metadata()}

        return [StructuredTool.from_function(coroutine=bash, name="Bash", description="Run a read-only command.")]

    monkeypatch.setattr(deliberator_module, "sandbox_tools", sandbox_tools)


def _finding(finding_id: str) -> dict[str, Any]:
    return {
        "id": finding_id,
        "title": f"Title {finding_id}",
        "body": f"Body {finding_id}.",
        "severity": "high",
        "path": "app.py",
        "line": 1,
        "disposition": "quality",
        "model": "model-a",
        "evidence": "",
        "context_refs": [],
    }


def _call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _decide(finding_id: str, decision: str = "keep", *, call_id: str | None = None, **values: Any) -> dict[str, Any]:
    args = {"finding_id": finding_id, "decision": decision, "rationale": "Checked the source.", **values}
    return _call("decide_finding", call_id or f"decide-{finding_id}", **args)


def _assess(finding_id: str, status: str = "still_present", *, call_id: str | None = None) -> dict[str, Any]:
    args = {"finding_id": finding_id, "status": status, "rationale": "Still reachable."}
    return _call("assess_prior_finding", call_id or f"assess-{finding_id}", **args)


def _turn(*calls: dict[str, Any], **metadata: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls), response_metadata=metadata)


def _done() -> AIMessage:
    return AIMessage(content="Everything is recorded.")


def _rate_limited() -> RuntimeError:
    error = RuntimeError("Error code: 429 - rate limit exceeded")
    error.status_code = 429  # type: ignore[attr-defined]
    return error


def _retry(max_attempts: int = 2) -> ModelCallRetryMiddleware:
    return ModelCallRetryMiddleware(max_attempts=max_attempts, base_delay=0, max_delay=0)


_REVISION = {"title": "Narrower claim", "body": "Only empty input is affected.", "severity": "medium"}


async def _deliberate(model: Any, **values: Any) -> tuple[DeliberationCandidate, Usage]:
    defaults: dict[str, Any] = {
        "disposition": "quality",
        "findings": [_finding("a"), _finding("b")],
        "assessments": [],
        "prior_findings": [],
        "context": [],
        "tool_budget": ToolBudget(5),
        "source_tool_timeout_seconds": 30.0,
    }
    profile = SandboxProfile((SandboxRoot(values.pop("root"), RootAccess.READ_ONLY),))
    return await deliberate(model, profile, "a" * 40, "b" * 40, **{**defaults, **values})


def _tool_replies(request: list[Any]) -> dict[str, Any]:
    return {message.tool_call_id: message for message in request if message.type == "tool"}


async def test_each_decision_and_assessment_is_recorded_as_it_arrives(scripted_model, run_recorder, tmp_path):
    # The calls of one turn run concurrently in no fixed order, so each gets its own turn for its progress to be fixed.
    model = scripted_model(
        _turn(_decide("b", "revise", revision=_REVISION)),
        _turn(_decide("a")),
        _turn(_assess("p")),
        _done(),
    )

    candidate, _usage = await _deliberate(
        model, root=tmp_path, prior_findings=[_finding("p")], langfuse_callback=run_recorder
    )

    # Decisions come back in input order, whatever order they were recorded in.
    assert [(item.finding_id, item.decision) for item in candidate.finding_decisions] == [
        ("a", "keep"),
        ("b", "revise"),
    ]
    assert candidate.finding_decisions[1].revision is not None
    assert candidate.finding_decisions[1].revision.title == "Narrower claim"
    assert [(item.finding_id, item.status) for item in candidate.assessment_decisions] == [("p", "still_present")]
    assert _tool_replies(model.requests[1])["decide-b"].content == (
        "Recorded revise for b. 1 findings still need a decision and 1 prior findings still need an assessment."
    )
    assert (
        _tool_replies(model.requests[2])["decide-a"].content
        == "Recorded keep for a. 1 prior findings still need an assessment."
    )
    assert (
        _tool_replies(model.requests[3])["assess-p"].content == "Recorded still_present for p. Everything is recorded."
    )
    assert model.bound[0]["tools"] == ["Bash", "decide_finding", "assess_prior_finding"]
    assert model.bound[0].get("tool_choice") is None
    assert run_recorder.run_names == ["pr-review-deliberator-quality-agent"]
    assert run_recorder.starts == 4
    prompt = " ".join(model.requests[0][0].content.split())
    assert "Do not remove a finding whose defect is real but overstated; revise it instead." in prompt
    assert "Duplicates are not false positives" in prompt
    assert SEVERITY_RUBRIC.split("\n", 1)[0] in prompt
    # Deliberation shares the reviewers' scope, so a finding in a moved file is not removed as out of scope.
    assert "A renamed or moved file is in scope for the effects of the move" in prompt
    payload = json.loads(model.requests[0][1].content)
    assert [finding["id"] for finding in payload["candidateFindings"]] == ["a", "b"]
    assert payload["requiredPriorAssessmentFindingIds"] == ["p"]


async def test_rejected_calls_record_nothing_and_are_returned_to_the_model(scripted_model, tmp_path):
    rejected = _turn(
        _decide("a", "remove_as_false_positive", call_id="second"),
        _decide("baseline-id", call_id="unknown"),
        _decide("b", "revise", call_id="no-revision"),
        _assess("a", call_id="not-prior"),
    )
    model = scripted_model(_turn(_decide("a")), rejected, _turn(_decide("b")), _done())

    candidate, _usage = await _deliberate(model, root=tmp_path)

    assert [(item.finding_id, item.decision) for item in candidate.finding_decisions] == [("a", "keep"), ("b", "keep")]
    replies = _tool_replies(model.requests[2])
    assert all(reply.status == "error" for reply in replies.values() if reply.tool_call_id != "decide-a")
    assert replies["second"].content == "Not recorded: ['a'] were already decided."
    assert replies["unknown"].content == "Not recorded: ['baseline-id'] are not candidateFindings IDs."
    assert "a revision is required for a revise decision" in replies["no-revision"].content
    assert replies["not-prior"].content == "Not recorded: ['a'] are not requiredPriorAssessmentFindingIds IDs."


async def test_stopping_early_is_answered_with_what_remains(scripted_model, tmp_path):
    model = scripted_model(_turn(_decide("a")), _done(), _turn(_decide("b")), _done())

    candidate, _usage = await _deliberate(model, root=tmp_path, prior_findings=[_finding("p")])

    assert [item.finding_id for item in candidate.finding_decisions] == ["a", "b"]
    reminder = model.requests[2][-1]
    assert (reminder.type, reminder.content) == (
        "human",
        "Record a decision with `decide_finding` for each of these candidate findings: ['b']. Record an assessment "
        "with `assess_prior_finding` for each of these prior findings: ['p']. Record a few per turn.",
    )
    # The early reply stays in the conversation; the reminder follows it.
    assert model.requests[2][-2].content == "Everything is recorded."


async def test_reminders_are_bounded_and_unrecorded_items_left_to_the_caller(scripted_model, tmp_path):
    model = scripted_model(_turn(_decide("a")), _done(), _done(), _done())
    incomplete: list[Exception] = []

    candidate, _usage = await _deliberate(model, root=tmp_path, on_incomplete=incomplete.append)

    assert [item.finding_id for item in candidate.finding_decisions] == ["a"]
    assert len(model.requests) == 4
    assert incomplete == []


async def test_a_cut_off_reply_is_dropped_and_the_model_told_what_remains(scripted_model, tmp_path):
    cut_off = AIMessage(content="Checking finding b", response_metadata={"stop_reason": "max_tokens"})
    model = scripted_model(_turn(_decide("a")), cut_off, _turn(_decide("b")), _done())

    candidate, _usage = await _deliberate(model, root=tmp_path)

    assert [item.finding_id for item in candidate.finding_decisions] == ["a", "b"]
    retry = model.requests[2]
    assert "Checking finding b" not in [message.content for message in retry]
    assert retry[-1].content == (
        "Your last reply reached the output limit before it made a tool call, so nothing was recorded. Record a "
        "decision with `decide_finding` for each of these candidate findings: ['b']. Record a few per turn."
    )


async def test_a_cut_off_reply_ends_the_run_once_everything_is_recorded(scripted_model, tmp_path):
    cut_off = AIMessage(content="All", response_metadata={"finish_reason": "length"})
    model = scripted_model(_turn(_decide("a"), _decide("b")), cut_off)

    candidate, _usage = await _deliberate(model, root=tmp_path)

    assert [item.finding_id for item in candidate.finding_decisions] == ["a", "b"]
    assert len(model.requests) == 2


async def test_a_deterministic_failure_keeps_what_was_recorded(scripted_model, tmp_path):
    model = scripted_model(_turn(_decide("a")), RuntimeError("context too long"))
    incomplete: list[Exception] = []

    candidate, _usage = await _deliberate(model, root=tmp_path, on_incomplete=incomplete.append)

    assert [item.finding_id for item in candidate.finding_decisions] == ["a"]
    assert [str(exc) for exc in incomplete] == ["context too long"]


async def test_a_failure_before_anything_is_recorded_propagates(scripted_model, tmp_path):
    # Source inspection alone records nothing, so there is no deliberation to keep.
    model = scripted_model(_turn(_call("Bash", "bash-1", command="git status")), RuntimeError("model not found"))
    incomplete: list[Exception] = []

    with pytest.raises(RuntimeError, match="model not found"):
        await _deliberate(model, root=tmp_path, on_incomplete=incomplete.append)

    assert incomplete == []


async def test_a_transient_error_retries_the_model_call_without_repeating_source_tool_calls(scripted_model, tmp_path):
    first = _turn(_call("Bash", "bash-1", command="git status"), _decide("a"))
    first.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    first.response_metadata = {"model_name": "fake"}
    model = scripted_model(first, _rate_limited(), _turn(_decide("b")), _done())

    candidate, usage = await _deliberate(model, root=tmp_path, retry=_retry())

    assert [item.finding_id for item in candidate.finding_decisions] == ["a", "b"]
    assert len(model.requests) == 4
    assert model.requests[1] == model.requests[2]
    # Recording a decision is not a source-tool call, and the retry did not repeat the Bash call.
    assert (usage.total_tokens, usage.tool_calls) == (12, 1)


async def test_decisions_survive_a_transient_error_that_outlasts_its_retries(scripted_model, tmp_path):
    model = scripted_model(_turn(_decide("a")), _rate_limited(), _rate_limited())
    incomplete: list[Exception] = []

    candidate, _usage = await _deliberate(model, root=tmp_path, retry=_retry(), on_incomplete=incomplete.append)

    assert [item.finding_id for item in candidate.finding_decisions] == ["a"]
    assert [getattr(exc, "status_code", None) for exc in incomplete] == [429]
