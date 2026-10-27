from typing import Any, cast

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from localmcp.retry import ModelCallRetryMiddleware
from localmcp.sandbox import RootAccess, SandboxProfile, SandboxRoot

import pr_council.agents.review.reviewer as reviewer_module
from pr_council.agents.review.common import ToolBudget
from pr_council.agents.review.reviewer import RawIterationCandidate, ReviewerAgent


def _build_agent(monkeypatch, tmp_path, captured):
    class FakeAgent:
        async def ainvoke(self, payload, config=None):
            captured["config"] = config
            return {"structured_response": RawIterationCandidate(), "messages": []}

    monkeypatch.setattr(reviewer_module, "create_agent", lambda *a, **k: FakeAgent())
    monkeypatch.setattr(reviewer_module, "sandbox_tools", lambda *a, **k: [])
    return ReviewerAgent(
        cast(BaseChatModel, object()),
        SandboxProfile((SandboxRoot(tmp_path, RootAccess.READ_ONLY),)),
        "a" * 40,
        "b" * 40,
        "quality",
        "model",
        ToolBudget(1),
        source_tool_timeout_seconds=30.0,
    )


async def test_reviewer_threads_langfuse_callback_and_run_name(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    agent = _build_agent(monkeypatch, tmp_path, captured)
    usage_callback = UsageMetadataCallbackHandler()
    langfuse_callback = object()

    await agent.run_iteration(
        iteration=1,
        total_iterations=1,
        context=[],
        prior_findings=[],
        previous=None,
        usage_callback=usage_callback,
        langfuse_callback=langfuse_callback,
    )

    config = captured["config"]
    assert config["run_name"] == "pr-review-quality-agent"
    assert config["callbacks"] == [usage_callback, langfuse_callback]
    assert "recursion_limit" in config


async def test_reviewer_omits_langfuse_callback_when_absent(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    agent = _build_agent(monkeypatch, tmp_path, captured)
    usage_callback = UsageMetadataCallbackHandler()

    await agent.run_iteration(
        iteration=1,
        total_iterations=1,
        context=[],
        prior_findings=[],
        previous=None,
        usage_callback=usage_callback,
    )

    config = captured["config"]
    assert config["run_name"] == "pr-review-quality-agent"
    assert config["callbacks"] == [usage_callback]
    assert "recursion_limit" in config


async def test_a_transient_error_retries_the_model_call_without_repeating_source_tool_calls(
    monkeypatch, tmp_path, scripted_model
):
    def sandbox_tools(profile, *, budget, timeout_seconds):
        async def bash(command: str) -> dict[str, Any]:
            await budget.claim()
            return {"result": {"stdout": "", "exit_code": 0}, "tool_budget": budget.metadata()}

        return [StructuredTool.from_function(coroutine=bash, name="Bash", description="Run a read-only command.")]

    rate_limited = RuntimeError("Error code: 429 - rate limit exceeded")
    rate_limited.status_code = 429  # type: ignore[attr-defined]
    inspect = {"name": "Bash", "args": {"command": "git status"}, "id": "bash-1", "type": "tool_call"}
    submit = {"name": "submit_result", "args": {"findings": []}, "id": "submit-1", "type": "tool_call"}
    model = scripted_model(
        AIMessage(content="", tool_calls=[inspect]), rate_limited, AIMessage(content="", tool_calls=[submit])
    )
    monkeypatch.setattr(reviewer_module, "sandbox_tools", sandbox_tools)
    budget = ToolBudget(5)
    agent = ReviewerAgent(
        model,
        SandboxProfile((SandboxRoot(tmp_path, RootAccess.READ_ONLY),)),
        "a" * 40,
        "b" * 40,
        "quality",
        "model",
        budget,
        source_tool_timeout_seconds=30.0,
        retry=ModelCallRetryMiddleware(max_attempts=2, base_delay=0, max_delay=0),
    )

    candidate, _usage = await agent.run_iteration(
        iteration=1,
        total_iterations=1,
        context=[],
        prior_findings=[],
        previous=None,
        usage_callback=UsageMetadataCallbackHandler(),
    )

    assert candidate.findings == []
    assert len(model.requests) == 3
    assert model.requests[1] == model.requests[2]
    assert budget.used == 1
