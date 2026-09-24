from typing import Any, cast

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
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
