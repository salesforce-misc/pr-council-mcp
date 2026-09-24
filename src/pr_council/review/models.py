"""Typed contracts shared by PR-review agents, workflows, and MCP tools."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from pr_council.errors import PrCouncilError


class ReviewError(PrCouncilError):
    """Raised when a PR review cannot proceed safely."""


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewMode(StrEnum):
    AUTO = "auto"
    INITIAL = "initial"
    FOLLOW_UP = "follow_up"


class ReviewSourceMode(StrEnum):
    MANAGED = "managed"
    LOCAL = "local"


class OperationStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    REVIEWING = "reviewing"
    DELIBERATING = "deliberating"
    AGGREGATING = "aggregating"
    READY = "ready"
    COMMIT_QUEUED = "commit_queued"
    COMMITTING = "committing"
    COMPLETED = "completed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


TERMINAL_STATUSES = {
    OperationStatus.COMPLETED,
    OperationStatus.CANCELLED,
    OperationStatus.STALE,
    OperationStatus.FAILED,
}


class ReviewContextInput(BaseModel):
    label: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=32_000)
    source: str | None = Field(default=None, max_length=200)

    @field_validator("label", "content", "source")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ReviewContextItem(ReviewContextInput):
    id: str
    sha256: str


class PrRef(BaseModel):
    host: str
    owner: str
    repo: str
    number: int = Field(gt=0)

    @property
    def repo_key(self) -> str:
        return f"{self.host}/{self.owner}/{self.repo}"

    @property
    def url(self) -> str:
        return f"https://{self.host}/{self.owner}/{self.repo}/pull/{self.number}"


class Finding(BaseModel):
    id: str
    title: str = Field(min_length=1, max_length=240)
    body: str = Field(min_length=1, max_length=8_000)
    severity: Severity
    path: str
    line: int = Field(gt=0)
    disposition: str
    model: str
    contributing_models: list[str] = Field(default_factory=list)
    contributing_dispositions: list[str] = Field(default_factory=list)
    evidence: str = Field(default="", max_length=4_000)
    context_refs: list[str] = Field(default_factory=list)


class PriorFindingAssessment(BaseModel):
    finding_id: str
    status: Literal["resolved", "still_present", "partially_resolved", "cannot_verify"]
    explanation: str = Field(min_length=1, max_length=4_000)
    path: str | None = None
    line: int | None = Field(default=None, gt=0)


class IterationCandidate(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    prior_assessments: list[PriorFindingAssessment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReviewerResult(BaseModel):
    disposition: str
    model: str
    findings: list[Finding] = Field(default_factory=list)
    prior_assessments: list[PriorFindingAssessment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    iterations_completed: int = 0
    tool_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_accounting_complete: bool = True
    failed: bool = False
    error: str | None = None


class FindingDecision(BaseModel):
    finding_id: str
    decision: Literal["keep", "remove_as_false_positive"]
    rationale: str = Field(min_length=1, max_length=4_000)


class AssessmentDecision(BaseModel):
    finding_id: str
    status: Literal["resolved", "still_present", "partially_resolved", "cannot_verify"]
    rationale: str = Field(min_length=1, max_length=4_000)
    path: str | None = None
    line: int | None = Field(default=None, gt=0)


class DeliberationResult(BaseModel):
    disposition: str
    model: str
    finding_decisions: list[FindingDecision] = Field(default_factory=list)
    assessment_decisions: list[AssessmentDecision] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_accounting_complete: bool = True
    tool_calls: int = 0


class AggregatedFinding(BaseModel):
    id: str
    title: str = Field(min_length=1, max_length=240)
    body: str = Field(min_length=1, max_length=8_000)
    severity: Severity
    path: str
    line: int = Field(gt=0)
    # Server-derived from the source findings during aggregation validation
    # (see _validate_aggregation); the model's values are always discarded, so
    # these default empty rather than being required and aborting the parse.
    dispositions: list[str] = Field(default_factory=list)
    reviewer_models: list[str] = Field(default_factory=list)
    source_finding_ids: list[str] = Field(min_length=1)
    context_refs: list[str] = Field(default_factory=list)


class AggregationCandidate(BaseModel):
    summary: str = Field(max_length=4_000)
    findings: list[AggregatedFinding] = Field(default_factory=list)


class ReviewMetrics(BaseModel):
    schema_version: int = 2
    models: dict[str, list[str]]
    findings: dict[str, dict[str, int]]
    overlapping_findings: int
    duration_ms: int
    total_tokens: int
    tokens: dict[str, object]
    timing: dict[str, int]
    reviewers: dict[str, int]
    results: dict[str, int]
    follow_up: dict[str, int] | None = None
    context: dict[str, int]


class PreviewComment(BaseModel):
    finding_id: str
    path: str
    line: int
    severity: Severity
    body: str
    reviewer_models: list[str]
    dispositions: list[str]
    source_finding_ids: list[str]


class ReviewerFailure(BaseModel):
    disposition: str
    model: str
    error: str = Field(max_length=8_000)
    retries: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls: int


class ReviewPreview(BaseModel):
    revision: int
    payload_hash: str
    base_sha: str
    head_sha: str
    summary: str
    summary_body: str
    comments: list[PreviewComment]
    context_manifest: list[dict[str, object]]
    metrics: ReviewMetrics
    reviewer_failures: list[ReviewerFailure] = Field(default_factory=list)
    mode: ReviewMode
    baseline_operation_id: str | None = None
