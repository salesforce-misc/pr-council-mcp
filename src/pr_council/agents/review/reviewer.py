"""Iterative LangChain reviewer agent."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from localmcp.sandbox import SandboxProfile
from localmcp.sandbox.seatbelt import sandbox_tools
from pydantic import BaseModel, Field

from pr_council.agents.review.common import (
    ToolBudget,
    Usage,
    agent_recursion_limit,
    extract_callback_usage,
    review_safety_prompt,
)
from pr_council.review.models import Finding, IterationCandidate, PriorFindingAssessment, Severity

_DISPOSITION_PROMPTS = {
    "quality": """Review for concrete correctness, reliability, maintainability, compatibility, test coverage, and
alignment with the supplied context. Report only actionable defects introduced or exposed by this PR, not style
preferences. A finding needs a precise changed-file location and evidence.""",
    "security": """Review for concrete security vulnerabilities, unsafe trust boundaries, injection, authorization,
authentication, sensitive-data exposure, cryptographic misuse, and denial-of-service risks. Trace relevant data
flows before reporting. Avoid speculative findings without an exploitable or realistic failure path.""",
}


class RawFinding(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    body: str = Field(min_length=1, max_length=8_000)
    severity: Severity
    path: str
    line: int = Field(gt=0)
    evidence: str = Field(default="", max_length=4_000)
    context_refs: list[str] = Field(default_factory=list)


class RawIterationCandidate(BaseModel):
    findings: list[RawFinding] = Field(default_factory=list)
    prior_assessments: list[PriorFindingAssessment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReviewerAgent:
    def __init__(
        self,
        model: BaseChatModel,
        sandbox_profile: SandboxProfile,
        base_sha: str,
        head_sha: str,
        disposition: str,
        model_id: str,
        tool_budget: ToolBudget,
        *,
        source_tool_timeout_seconds: float,
    ):
        self.disposition = disposition
        self.model_id = model_id
        self.tool_budget = tool_budget
        self._agent = create_agent(
            model,
            sandbox_tools(
                sandbox_profile,
                budget=tool_budget,
                timeout_seconds=source_tool_timeout_seconds,
            ),
            system_prompt=(
                f"{review_safety_prompt(base_sha, head_sha)}\n\n{_DISPOSITION_PROMPTS[disposition]}\n\n"
                f"You have at most {tool_budget.max_calls} source-tool calls in this pass. "
                "Every tool result reports the remaining budget. Return your structured response before "
                "it is exhausted."
            ),
            response_format=ToolStrategy(RawIterationCandidate),
            name=f"pr-review-{disposition}",
        )

    async def run_iteration(
        self,
        *,
        iteration: int,
        total_iterations: int,
        context: list[dict[str, object]],
        prior_findings: list[dict[str, object]],
        previous: dict[str, object] | None,
        context_changes: list[dict[str, object]] | None = None,
        usage_callback: UsageMetadataCallbackHandler,
        langfuse_callback: Any | None = None,
    ) -> tuple[IterationCandidate, Usage]:
        prompt = {
            "task": "Review the checked-out PR and produce the structured result.",
            "iteration": iteration,
            "totalIterations": total_iterations,
            "reviewContext": context,
            "reviewContextChangesSinceBaseline": context_changes or [],
            "priorReviewFindings": prior_findings,
            "priorAssessmentRule": (
                "Return exactly one prior_assessment for each priorReviewFindings ID and no other IDs."
                if prior_findings
                else "There are no baseline findings. prior_assessments must be empty."
            ),
            "previousIterationCandidate": previous,
            "iterationGuidance": (
                "Explore broadly and draft findings."
                if iteration == 1
                else (
                    "Challenge every prior candidate, gather more evidence, "
                    "remove disproven items, and look for misses."
                )
            ),
        }
        callbacks: list[Any] = [usage_callback]
        if langfuse_callback is not None:
            callbacks.append(langfuse_callback)
        invoke_config: RunnableConfig = {
            "run_name": f"pr-review-{self.disposition}-agent",
            "recursion_limit": agent_recursion_limit(self.tool_budget.max_calls),
            "callbacks": callbacks,
        }
        result = await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": json.dumps(prompt)}]},
            config=invoke_config,
        )
        raw = result.get("structured_response")
        candidate = raw if isinstance(raw, RawIterationCandidate) else RawIterationCandidate.model_validate(raw)
        findings: list[Finding] = []
        for item in candidate.findings:
            material = f"{self.disposition}\0{self.model_id}\0{item.path}\0{item.line}\0{item.title}\0{item.body}"
            finding_id = hashlib.sha256(material.encode()).hexdigest()[:20]
            findings.append(
                Finding(
                    id=finding_id,
                    disposition=self.disposition,
                    model=self.model_id,
                    **item.model_dump(),
                )
            )
        usage = extract_callback_usage(usage_callback, tool_calls=self.tool_budget.used)
        return (
            IterationCandidate(
                findings=findings,
                prior_assessments=candidate.prior_assessments,
                notes=candidate.notes,
            ),
            usage,
        )
