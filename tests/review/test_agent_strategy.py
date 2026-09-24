from typing import Any, cast

from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from localmcp.sandbox import RootAccess, SandboxProfile, SandboxRoot

import pr_council.agents.review.deliberator as deliberator_module
import pr_council.agents.review.reviewer as reviewer_module
from pr_council.agents.review.common import ToolBudget
from pr_council.agents.review.deliberator import DeliberationCandidate
from pr_council.agents.review.reviewer import ReviewerAgent


def test_reviewer_forces_tool_strategy(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    def fake_create_agent(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(reviewer_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(reviewer_module, "sandbox_tools", lambda *a, **k: [])
    ReviewerAgent(
        cast(BaseChatModel, object()),
        SandboxProfile((SandboxRoot(tmp_path, RootAccess.READ_ONLY),)),
        "a" * 40,
        "b" * 40,
        "quality",
        "model",
        ToolBudget(1),
        source_tool_timeout_seconds=30.0,
    )

    assert isinstance(captured["response_format"], ToolStrategy)


async def test_deliberator_forces_tool_strategy(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    class FakeAgent:
        async def ainvoke(self, *args, **kwargs):
            return {"structured_response": DeliberationCandidate(), "messages": []}

    def fake_create_agent(*args, **kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(deliberator_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(deliberator_module, "sandbox_tools", lambda *a, **k: [])
    await deliberator_module.deliberate(
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
        usage_callback=UsageMetadataCallbackHandler(),
    )

    assert isinstance(captured["response_format"], ToolStrategy)
