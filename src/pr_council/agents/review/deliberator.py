"""Closed-set LangChain deliberation agent."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from localmcp.sandbox import SandboxProfile
from localmcp.sandbox.seatbelt import sandbox_tools
from localmcp.structured_output import SubmitResultMiddleware
from pydantic import BaseModel, Field

from pr_council.agents.review.common import (
    ToolBudget,
    Usage,
    agent_recursion_limit,
    extract_callback_usage,
    review_safety_prompt,
)
from pr_council.review.models import AssessmentDecision, FindingDecision

_DELIBERATION_PROMPT = """Deliberate the closed set of candidate findings. Decide keep or remove_as_false_positive for
every supplied finding ID, and reconcile every supplied prior-finding assessment. You may inspect the source tree.
You must not add findings or IDs. Prefer removal when the claimed behavior is disproven by code or supplied context,
but retain actionable issues with adequate evidence."""


class DeliberationCandidate(BaseModel):
    finding_decisions: list[FindingDecision] = Field(default_factory=list)
    assessment_decisions: list[AssessmentDecision] = Field(default_factory=list)


async def deliberate(
    model: BaseChatModel,
    sandbox_profile: SandboxProfile,
    base_sha: str,
    head_sha: str,
    *,
    disposition: str,
    findings: list[dict[str, object]],
    assessments: list[dict[str, object]],
    prior_findings: list[dict[str, object]],
    context: list[dict[str, object]],
    tool_budget: ToolBudget,
    source_tool_timeout_seconds: float,
    usage_callback: UsageMetadataCallbackHandler,
    langfuse_callback: Any | None = None,
) -> tuple[DeliberationCandidate, Usage]:
    agent = create_agent(
        model,
        sandbox_tools(
            sandbox_profile,
            budget=tool_budget,
            timeout_seconds=source_tool_timeout_seconds,
        ),
        system_prompt=(
            f"{review_safety_prompt(base_sha, head_sha)}\n\n{_DELIBERATION_PROMPT}\n\n"
            f"You have at most {tool_budget.max_calls} source-tool calls in this pass. "
            "Every tool result reports the remaining budget. Submit your result before it is exhausted."
        ),
        # Cross-model requirement: ToolStrategy forces tool_choice, which some models reject, and gateways may
        # drop ProviderStrategy's schema or let a model answer without inspecting the source. An unforced
        # submission tool is the lowest common denominator; its in-run correction rounds are deliberate.
        middleware=[SubmitResultMiddleware(DeliberationCandidate)],
        name=f"pr-review-deliberate-{disposition}",
    )
    callbacks: list[Any] = [usage_callback]
    if langfuse_callback is not None:
        callbacks.append(langfuse_callback)
    invoke_config: RunnableConfig = {
        "run_name": f"pr-review-deliberator-{disposition}-agent",
        "recursion_limit": agent_recursion_limit(tool_budget.max_calls),
        "callbacks": callbacks,
    }
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "disposition": disposition,
                            "candidateFindings": findings,
                            "priorAssessments": assessments,
                            "priorReviewFindings": prior_findings,
                            "requiredPriorAssessmentFindingIds": [finding["id"] for finding in prior_findings],
                            "priorAssessmentRule": (
                                "Return exactly one assessment_decision for each required ID and no other IDs."
                                if prior_findings
                                else "There are no baseline findings. assessment_decisions must be empty."
                            ),
                            "reviewContext": context,
                        }
                    ),
                }
            ]
        },
        config=invoke_config,
    )
    raw = result.get("structured_response")
    candidate = raw if isinstance(raw, DeliberationCandidate) else DeliberationCandidate.model_validate(raw)
    usage = extract_callback_usage(usage_callback, tool_calls=tool_budget.used)
    return candidate, usage
