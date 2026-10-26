"""Holistic LangChain aggregation agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from pr_council.agents.review.common import BASE_SAFETY, Usage
from pr_council.agents.review.recording import ClosedSet, Recorder, recording_tool, run_recording_agent
from pr_council.review.models import AggregatedFinding, AggregationCandidate

_MERGE_TOOL = "merge_findings"

_AGGREGATION_PROMPT = f"""Merge retained findings that describe the same defect. For each such group, call
`{_MERGE_TOOL}` with two or more IDs from retainedFindings, a merged title and body, and the path and line of one of
the merged findings. You may record several merges in one turn. Each ID can be merged only once, and a rejected call
records nothing. Findings you do not merge are published exactly as written, so never merge a finding only to rephrase
it. Merge findings only when they describe the same defect, which one change at one location fixes. Distinct defects at
the same location stay separate, and so do findings in different places that each need their own change, even when
they share a cause. IDs
appearing only in baselineFindings or priorFindingAssessments are context identifiers, not retained finding IDs. Do
not create a new issue. Keep each merged body concise but actionable and able to stand alone if the finding cannot be
posted inline. When every merge is recorded, or if no findings overlap, reply with a short confirmation and no tool
calls."""


class FindingMerge(BaseModel):
    source_finding_ids: list[str] = Field(min_length=2)
    title: str = Field(min_length=1, max_length=240)
    body: str = Field(min_length=1, max_length=8_000)
    path: str
    line: int = Field(gt=0)


class _MergeRecorder(Recorder):
    """Record each merge through its own call, so no single response has to carry the whole result.

    Unmerged findings never pass through the model at all.
    """

    def __init__(self, findings: list[dict[str, Any]]):
        self._findings = {str(finding["id"]): finding for finding in findings}
        self._merged = ClosedSet[int](self._findings, source="retainedFindings", recorded="merged")
        self.merges: list[FindingMerge] = []
        self.items = len(findings)
        self.tools = [
            recording_tool(
                self.merge,
                name=_MERGE_TOOL,
                description="Merge two or more retained findings that describe the same defect into one finding.",
                schema=FindingMerge,
            )
        ]

    async def merge(self, **values: Any) -> str:
        merge = FindingMerge.model_validate(values)
        ids = merge.source_finding_ids
        self._merged.check(ids)
        if (merge.path, merge.line) not in {(self._findings[i]["path"], self._findings[i]["line"]) for i in ids}:
            raise ToolException("Not recorded: use the path and line of one of the merged findings.")
        self._merged.record(ids, len(self.merges))
        self.merges.append(merge)
        return f"Merged {len(ids)} findings. {len(self._merged.open())} unmerged findings will be published as written."

    def remaining(self) -> str | None:
        if len(self._merged.open()) < 2:
            return None
        return f"Record any remaining merges with `{_MERGE_TOOL}`, a few per turn, or reply briefly if none remain."

    def has_records(self) -> bool:
        return bool(self.merges)

    def candidate(self) -> AggregationCandidate:
        """Return every finding once, in input order, with each merge at the position of its first member.

        A merge carries its first member's severity as a placeholder: IDs, severity, and attribution are derived from
        the source findings during graph validation.
        """
        findings: list[AggregatedFinding] = []
        emitted: set[int] = set()
        for source_id, source in self._findings.items():
            index = self._merged.records.get(source_id)
            if index is None:
                findings.append(
                    AggregatedFinding(
                        id=source_id,
                        title=source["title"],
                        body=source["body"],
                        severity=source["severity"],
                        path=source["path"],
                        line=source["line"],
                        source_finding_ids=[source_id],
                    )
                )
            elif index not in emitted:
                emitted.add(index)
                merge = self.merges[index]
                findings.append(
                    AggregatedFinding(
                        id=source_id,
                        title=merge.title,
                        body=merge.body,
                        severity=source["severity"],
                        path=merge.path,
                        line=merge.line,
                        source_finding_ids=merge.source_finding_ids,
                    )
                )
        return AggregationCandidate(findings=findings)


async def aggregate(
    model: BaseChatModel,
    *,
    findings: list[dict[str, Any]],
    prior_assessments: list[dict[str, object]],
    baseline_findings: list[dict[str, object]],
    context_changes: list[dict[str, object]],
    context: list[dict[str, object]],
    usage: Usage | None = None,
    langfuse_callback: Any | None = None,
    on_incomplete: Callable[[Exception], None] | None = None,
) -> tuple[AggregationCandidate, Usage]:
    # Accumulate into a caller-supplied Usage when given so token spend from an
    # invocation that later re-raises a transient error (retried by the caller's
    # call_with_rate_limit_retry) is preserved across outer attempts rather than
    # discarded with the raised exception.
    usage = usage if usage is not None else Usage()
    recorder = _MergeRecorder(findings)
    if len(findings) < 2:
        return recorder.candidate(), usage
    await run_recording_agent(
        model,
        recorder,
        name="pr-review-aggregator",
        run_name="pr-review-aggregator",
        system_prompt=f"{BASE_SAFETY}\n\n{_AGGREGATION_PROMPT}",
        payload={
            "retainedFindings": findings,
            "priorFindingAssessments": prior_assessments,
            "baselineFindings": baseline_findings,
            "reviewContextChangesSinceBaseline": context_changes,
            "reviewContext": context,
        },
        usage=usage,
        langfuse_callback=langfuse_callback,
        # Every finding is publishable without merging, so a deterministic failure keeps the merges already
        # recorded instead of failing the review.
        on_incomplete=on_incomplete,
    )
    return recorder.candidate(), usage
