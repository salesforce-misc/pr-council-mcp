"""Closed-set LangChain deliberation agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from localmcp.sandbox import SandboxProfile
from localmcp.sandbox.seatbelt import sandbox_tools
from pydantic import BaseModel, Field

from pr_council.agents.review.common import SEVERITY_RUBRIC, ToolBudget, Usage, review_safety_prompt
from pr_council.agents.review.recording import ClosedSet, Recorder, recording_tool, run_recording_agent
from pr_council.review.models import AssessmentDecision, FindingDecision

_DECIDE_TOOL = "decide_finding"
_ASSESS_TOOL = "assess_prior_finding"

_DELIBERATION_PROMPT = f"""Deliberate the closed set of candidate findings. Record one decision for every
candidateFindings ID with `{_DECIDE_TOOL}`, and one assessment for every requiredPriorAssessmentFindingIds ID with
`{_ASSESS_TOOL}`. You may record several in one turn. Each ID can be recorded only once, and a rejected call records
nothing. You may inspect the source tree. You must not add findings or IDs.
- keep: the finding is accurate as written.
- revise: the underlying defect is real, but the finding misstates its impact, trigger, severity, or fix. Supply a
  revision whose title, body, severity, and evidence describe what the code supports. A revision must describe the
  same defect at the same location; it cannot replace the finding with a different issue. Severity may move up or
  down.
- remove_as_false_positive: the claimed defect is disproven by code or supplied context. Do not remove a finding
  whose defect is real but overstated; revise it instead.
Findings from different reviewers often describe the same defect. Duplicates are not false positives: decide each one
on its own merits, and give every duplicate of a defect the same correction. A later step merges duplicates and
credits every reviewer that found them. When every decision and assessment is recorded, reply with a short
confirmation and no tool calls."""


class DeliberationCandidate(BaseModel):
    finding_decisions: list[FindingDecision] = Field(default_factory=list)
    assessment_decisions: list[AssessmentDecision] = Field(default_factory=list)


class _DeliberationRecorder(Recorder):
    """Record each decision and assessment through its own call, so no single response carries them all."""

    required = True

    def __init__(self, findings: list[dict[str, object]], prior_findings: list[dict[str, object]]):
        self._decisions = ClosedSet[FindingDecision](
            (str(finding["id"]) for finding in findings), source="candidateFindings", recorded="decided"
        )
        self._assessments = ClosedSet[AssessmentDecision](
            (str(finding["id"]) for finding in prior_findings),
            source="requiredPriorAssessmentFindingIds",
            recorded="assessed",
        )
        self.items = len(self._decisions.ids) + len(self._assessments.ids)
        self.tools = [
            recording_tool(
                self.decide,
                name=_DECIDE_TOOL,
                description="Record the keep, revise, or remove_as_false_positive decision for one candidate finding.",
                schema=FindingDecision,
            ),
            recording_tool(
                self.assess,
                name=_ASSESS_TOOL,
                description="Record whether one prior review finding is resolved at the current head.",
                schema=AssessmentDecision,
            ),
        ]

    async def decide(self, **values: Any) -> str:
        decision = FindingDecision.model_validate(values)
        self._decisions.check([decision.finding_id])
        self._decisions.record([decision.finding_id], decision)
        return f"Recorded {decision.decision} for {decision.finding_id}. {self._progress()}"

    async def assess(self, **values: Any) -> str:
        assessment = AssessmentDecision.model_validate(values)
        self._assessments.check([assessment.finding_id])
        self._assessments.record([assessment.finding_id], assessment)
        return f"Recorded {assessment.status} for {assessment.finding_id}. {self._progress()}"

    def _progress(self) -> str:
        needed = []
        if count := len(self._decisions.open()):
            needed.append(f"{count} findings still need a decision")
        if count := len(self._assessments.open()):
            needed.append(f"{count} prior findings still need an assessment")
        return f"{' and '.join(needed)}." if needed else "Everything is recorded."

    def remaining(self) -> str | None:
        unrecorded = []
        if ids := self._decisions.open():
            unrecorded.append(f"Record a decision with `{_DECIDE_TOOL}` for each of these candidate findings: {ids}.")
        if ids := self._assessments.open():
            unrecorded.append(f"Record an assessment with `{_ASSESS_TOOL}` for each of these prior findings: {ids}.")
        return f"{' '.join(unrecorded)} Record a few per turn." if unrecorded else None

    def has_records(self) -> bool:
        return bool(self._decisions.records or self._assessments.records)

    def candidate(self) -> DeliberationCandidate:
        """Return what was recorded, in input order; the graph closes both sets over the supplied IDs."""
        return DeliberationCandidate(
            finding_decisions=self._decisions.recorded(), assessment_decisions=self._assessments.recorded()
        )


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
    usage: Usage | None = None,
    langfuse_callback: Any | None = None,
    on_incomplete: Callable[[Exception], None] | None = None,
) -> tuple[DeliberationCandidate, Usage]:
    # As for aggregation, a caller-supplied accumulator keeps spend from an attempt that re-raises a transient error.
    usage = usage if usage is not None else Usage()
    recorder = _DeliberationRecorder(findings, prior_findings)
    await run_recording_agent(
        model,
        recorder,
        name=f"pr-review-deliberate-{disposition}",
        run_name=f"pr-review-deliberator-{disposition}-agent",
        system_prompt=(
            f"{review_safety_prompt(base_sha, head_sha)}\n\n{_DELIBERATION_PROMPT}\n\n{SEVERITY_RUBRIC}\n\n"
            f"You have at most {tool_budget.max_calls} source-tool calls in this pass. "
            "Every source-tool result reports the remaining budget. Recording does not use it."
        ),
        payload={
            "disposition": disposition,
            "candidateFindings": findings,
            "priorAssessments": assessments,
            "priorReviewFindings": prior_findings,
            "requiredPriorAssessmentFindingIds": [finding["id"] for finding in prior_findings],
            "priorAssessmentRule": (
                f"Record exactly one assessment with `{_ASSESS_TOOL}` for each required ID and no other IDs."
                if prior_findings
                else "There are no baseline findings, so record no assessments."
            ),
            "reviewContext": context,
        },
        usage=usage,
        tools=sandbox_tools(sandbox_profile, budget=tool_budget, timeout_seconds=source_tool_timeout_seconds),
        tool_budget=tool_budget,
        langfuse_callback=langfuse_callback,
        # Undecided findings are kept and missing assessments become cannot_verify, so a deterministic failure keeps
        # what was already recorded instead of failing the review.
        on_incomplete=on_incomplete,
    )
    return recorder.candidate(), usage
