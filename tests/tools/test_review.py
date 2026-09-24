"""Behavioral tests for the durable PR-review lifecycle MCP tools.

Each tool is a thin boundary over ``review_runtime()``; these tests monkeypatch
that module-level factory to a fake and pin the observable contract: the exact
returned dict shape and the arguments forwarded to the runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pr_council.tools.review as review_tool
from pr_council.review.models import OperationStatus, ReviewContextInput, ReviewMode, ReviewSourceMode


async def test_get_review_surfaces_structured_reviewer_failures(monkeypatch):
    failure = {
        "disposition": "quality",
        "model": "model-a",
        "error": "structured output failed",
        "retries": 2,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "tool_calls": 4,
    }
    record = SimpleNamespace(
        id="operation-1",
        label="PR #1",
        status=OperationStatus.READY,
        preview={"reviewer_failures": [failure]},
        error=None,
        result=None,
        created_at="created",
        updated_at="updated",
        commit_ready=True,
    )

    class FakeRuntime:
        async def get_owned(self, operation_id):
            assert operation_id == "operation-1"
            return record

    monkeypatch.setattr(review_tool, "review_runtime", lambda: FakeRuntime())

    result = await review_tool.pr_council_get("operation-1")

    # Pin the FULL mapped dict so every one of the 11 keys is discriminating:
    # a mutation swapping any single source field (e.g. ``operation_id`` <-
    # ``label`` or ``status`` <- ``error``) fails here rather than passing
    # silently. This subsumes the reviewer_failures/preview_ready/commit_ready/
    # status_reason behavioral checks that used to stand alone.
    assert result == {
        "operation_id": "operation-1",
        "label": "PR #1",
        "status": "ready",
        "preview_ready": True,
        "commit_ready": True,
        "status_reason": None,
        "error": None,
        "reviewer_failures": [failure],
        "result": None,
        "created_at": "created",
        "updated_at": "updated",
    }


async def test_get_stale_review_explains_head_change_and_keeps_preview_available(monkeypatch):
    stale_result = {
        "status": "stale",
        "reason": "Pull request head changed after preview; no review was published.",
        "preview_head_sha": "old-head",
        "current_head_sha": "new-head",
        "next_action": "Start a new PR review for the current head.",
    }
    record = SimpleNamespace(
        id="operation-1",
        label="PR #1",
        status=OperationStatus.STALE,
        preview={"reviewer_failures": []},
        error=None,
        result=stale_result,
        created_at="created",
        updated_at="updated",
        commit_ready=False,
    )

    class FakeRuntime:
        async def get_owned(self, operation_id):
            return record

    monkeypatch.setattr(review_tool, "review_runtime", lambda: FakeRuntime())

    result = await review_tool.pr_council_get("operation-1")

    # Full-dict equality on the STALE record too, so status_reason gating is
    # pinned in BOTH states: here status == STALE must surface the result's
    # ``reason`` while the READY test asserts status_reason is None.
    assert result == {
        "operation_id": "operation-1",
        "label": "PR #1",
        "status": "stale",
        "preview_ready": True,
        "commit_ready": False,
        "status_reason": stale_result["reason"],
        "error": None,
        "reviewer_failures": [],
        "result": stale_result,
        "created_at": "created",
        "updated_at": "updated",
    }


async def test_start_forwards_params_to_runtime_and_maps_record(monkeypatch):
    context_items = [ReviewContextInput(label="design", content="finalized snapshot")]
    record = SimpleNamespace(id="operation-7", label="PR #7", status=OperationStatus.QUEUED)
    captured = {}

    class FakeRuntime:
        async def start_operation(self, **kwargs):
            captured.update(kwargs)
            return record

    monkeypatch.setattr(review_tool, "review_runtime", lambda: FakeRuntime())

    result = await review_tool.pr_council_start(
        pr_url="https://github.com/owner/repo/pull/7",
        source_mode=ReviewSourceMode.LOCAL,
        local_source_path=".",
        review_context=context_items,
        mode=ReviewMode.INITIAL,
        baseline_operation_id="baseline-3",
        models={"quality": ["model-a"], "security": ["model-b"]},
        iterations=5,
        deliberation_model="delib-model",
        aggregation_model="agg-model",
    )

    assert result == {"operation_id": "operation-7", "label": "PR #7", "status": "queued"}
    # The ``review_context`` param must arrive as the ``context=`` kwarg; a
    # regression swapping the mapping would surface here.
    assert captured == {
        "pr_url": "https://github.com/owner/repo/pull/7",
        "source_mode": ReviewSourceMode.LOCAL,
        "local_source_path": ".",
        "context": context_items,
        "mode": ReviewMode.INITIAL,
        "baseline_operation_id": "baseline-3",
        "models": {"quality": ["model-a"], "security": ["model-b"]},
        "iterations": 5,
        "deliberation_model": "delib-model",
        "aggregation_model": "agg-model",
    }


async def test_preview_returns_json_model_dump_of_runtime_preview(monkeypatch):
    sentinel = {"revision": 4, "payload_hash": "abc123", "comments": []}
    dump_calls = []

    class FakePreview:
        def model_dump(self, **kwargs):
            dump_calls.append(kwargs)
            return sentinel

    class FakeRuntime:
        async def preview(self, operation_id):
            assert operation_id == "operation-2"
            return FakePreview()

    monkeypatch.setattr(review_tool, "review_runtime", lambda: FakeRuntime())

    result = await review_tool.pr_council_preview("operation-2")

    assert result is sentinel
    assert dump_calls == [{"mode": "json"}]


async def test_commit_forwards_positional_args_and_maps_record(monkeypatch):
    record = SimpleNamespace(id="operation-3", label="PR #3", status=OperationStatus.COMMITTING)
    captured = {}

    class FakeRuntime:
        async def commit(self, operation_id, revision, payload_hash):
            captured["args"] = (operation_id, revision, payload_hash)
            return record

    monkeypatch.setattr(review_tool, "review_runtime", lambda: FakeRuntime())

    result = await review_tool.pr_council_commit("operation-3", 4, "hash-9")

    assert captured["args"] == ("operation-3", 4, "hash-9")
    assert result == {"operation_id": "operation-3", "label": "PR #3", "status": "committing"}


async def test_cancel_forwards_operation_id_and_maps_record(monkeypatch):
    record = SimpleNamespace(id="operation-4", label="PR #4", status=OperationStatus.CANCELLED)
    captured = {}

    class FakeRuntime:
        async def cancel(self, operation_id):
            captured["operation_id"] = operation_id
            return record

    monkeypatch.setattr(review_tool, "review_runtime", lambda: FakeRuntime())

    result = await review_tool.pr_council_cancel("operation-4")

    assert captured["operation_id"] == "operation-4"
    assert result == {"operation_id": "operation-4", "label": "PR #4", "status": "cancelled"}


def test_tools_aggregate_registers_only_review_lifecycle_tools():
    from pr_council.tools import TOOLS

    expected = [
        review_tool.pr_council_start,
        review_tool.pr_council_get,
        review_tool.pr_council_preview,
        review_tool.pr_council_commit,
        review_tool.pr_council_cancel,
    ]

    # Identity AND ORDER are part of the advertised contract: a client discovers
    # tools in list order, so a reorder of TOOLS must fail here, not pass.
    assert TOOLS == expected
    assert [tool.__name__ for tool in TOOLS] == [
        "pr_council_start",
        "pr_council_get",
        "pr_council_preview",
        "pr_council_commit",
        "pr_council_cancel",
    ]
    assert all(callable(tool) for tool in TOOLS)


def _registered_annotations(fn):
    """Return the ToolAnnotations FastMCP advertises for ``fn`` to clients.

    fastmcp 4.0.4's ``@tool`` returns the plain function unchanged, so the
    annotations are not introspectable on the bare callable. Registering the
    tool on a throwaway ``FastMCP`` yields the ``FunctionTool`` whose
    ``.annotations`` is the true client-facing contract.
    """
    from fastmcp import FastMCP

    return FastMCP("annotation-probe").add_tool(fn).annotations


# Expected client-facing annotation flags, read from the tool sources and
# pinned here. Order: (title, read_only, destructive, idempotent, open_world).
_EXPECTED_ANNOTATIONS = {
    "pr_council_start": ("Start a PR review", False, False, False, True),
    "pr_council_get": ("Get a PR review", True, False, True, True),
    "pr_council_preview": ("Preview a PR review", True, False, True, True),
    "pr_council_commit": ("Commit a PR review", False, False, True, True),
    "pr_council_cancel": ("Cancel a PR review", False, False, True, True),
}


@pytest.mark.parametrize(
    ("name", "title", "read_only", "destructive", "idempotent", "open_world"),
    [(name, *flags) for name, flags in _EXPECTED_ANNOTATIONS.items()],
)
def test_tool_advertises_expected_annotations(name, title, read_only, destructive, idempotent, open_world):
    fns = {
        "pr_council_start": review_tool.pr_council_start,
        "pr_council_get": review_tool.pr_council_get,
        "pr_council_preview": review_tool.pr_council_preview,
        "pr_council_commit": review_tool.pr_council_commit,
        "pr_council_cancel": review_tool.pr_council_cancel,
    }

    annotations = _registered_annotations(fns[name])

    # Pin each behavioral hint individually so flipping any single flag
    # (e.g. commit.destructive_hint or start.idempotent_hint) fails the suite.
    assert annotations.title == title
    assert annotations.read_only_hint is read_only
    assert annotations.destructive_hint is destructive
    assert annotations.idempotent_hint is idempotent
    assert annotations.open_world_hint is open_world
