from typing import Any, cast

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from localmcp.sandbox import RootAccess, SandboxProfile, SandboxRoot

import pr_council.agents.review.deliberator as deliberator_module
from pr_council.agents.review.common import ToolBudget
from pr_council.agents.review.deliberator import DeliberationCandidate, deliberate


def _install_fake_agent(monkeypatch, captured):
    class FakeAgent:
        async def ainvoke(self, payload, config=None):
            captured["config"] = config
            return {"structured_response": DeliberationCandidate(), "messages": []}

    monkeypatch.setattr(deliberator_module, "create_agent", lambda *a, **k: FakeAgent())
    monkeypatch.setattr(deliberator_module, "sandbox_tools", lambda *a, **k: [])


async def test_deliberator_threads_langfuse_callback_and_run_name(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    _install_fake_agent(monkeypatch, captured)
    usage_callback = UsageMetadataCallbackHandler()
    langfuse_callback = object()

    await deliberate(
        cast(BaseChatModel, object()),
        SandboxProfile((SandboxRoot(tmp_path, RootAccess.READ_ONLY),)),
        "a" * 40,
        "b" * 40,
        disposition="quality",
        findings=[],
        assessments=[],
        prior_findings=[],
        context=[],
        tool_budget=ToolBudget(1),
        source_tool_timeout_seconds=30.0,
        usage_callback=usage_callback,
        langfuse_callback=langfuse_callback,
    )

    config = captured["config"]
    assert config["run_name"] == "pr-review-deliberator-quality-agent"
    assert config["callbacks"] == [usage_callback, langfuse_callback]
    assert "recursion_limit" in config


async def test_deliberator_omits_langfuse_callback_when_absent(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    _install_fake_agent(monkeypatch, captured)
    usage_callback = UsageMetadataCallbackHandler()

    await deliberate(
        cast(BaseChatModel, object()),
        SandboxProfile((SandboxRoot(tmp_path, RootAccess.READ_ONLY),)),
        "a" * 40,
        "b" * 40,
        disposition="quality",
        findings=[],
        assessments=[],
        prior_findings=[],
        context=[],
        tool_budget=ToolBudget(1),
        source_tool_timeout_seconds=30.0,
        usage_callback=usage_callback,
    )

    config = captured["config"]
    assert config["run_name"] == "pr-review-deliberator-quality-agent"
    assert config["callbacks"] == [usage_callback]
    assert "recursion_limit" in config
