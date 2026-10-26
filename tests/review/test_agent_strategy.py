from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from localmcp.sandbox import RootAccess, SandboxProfile, SandboxRoot
from localmcp.structured_output import SubmitResultMiddleware

import pr_council.agents.review.reviewer as reviewer_module
from pr_council.agents.review.common import ToolBudget
from pr_council.agents.review.reviewer import ReviewerAgent


def test_reviewer_uses_unforced_submit_tool(monkeypatch, tmp_path):
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

    assert captured.get("response_format") is None
    assert any(isinstance(m, SubmitResultMiddleware) for m in captured["middleware"])
