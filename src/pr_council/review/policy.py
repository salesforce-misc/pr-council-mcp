"""Pure rendering and validation policy for PR-review publication."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping

from pr_council.review.models import (
    AggregatedFinding,
    AssessmentDecision,
    FindingDecision,
    PriorFindingAssessment,
    ReviewMetrics,
    Severity,
)

_SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
    Severity.INFO: "⚪",
}
_MENTION = re.compile(r"(^|[^A-Za-z0-9_])@([A-Za-z0-9][A-Za-z0-9-]*(?:/[A-Za-z0-9][A-Za-z0-9-]*)?)")
_CLOSES = re.compile(r"\b((?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+)#(\d+)", re.IGNORECASE)


def neutralize_github_markup(value: str) -> str:
    value = _MENTION.sub(lambda match: f"{match.group(1)}@\u200b{match.group(2)}", value)
    return _CLOSES.sub(lambda match: f"{match.group(1)}#\u200b{match.group(2)}", value)


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def normalize_prior_assessments(
    assessments: Iterable[PriorFindingAssessment], allowed_finding_ids: Iterable[str]
) -> list[PriorFindingAssessment]:
    """Keep at most one reviewer assessment for each allowed baseline finding."""
    allowed = set(allowed_finding_ids)
    normalized: list[PriorFindingAssessment] = []
    seen: set[str] = set()
    for assessment in assessments:
        if assessment.finding_id not in allowed or assessment.finding_id in seen:
            continue
        seen.add(assessment.finding_id)
        normalized.append(assessment)
    return normalized


def _close_decisions[D: (FindingDecision, AssessmentDecision)](
    decisions: Iterable[D], findings: Iterable[dict[str, object]], default: Callable[[str], D]
) -> list[D]:
    """Return one decision per finding, in finding order, discarding unknown and duplicate IDs."""
    ordered_ids = unique(str(finding["id"]) for finding in findings)
    allowed = set(ordered_ids)
    by_id: dict[str, D] = {}
    for decision in decisions:
        if decision.finding_id in allowed and decision.finding_id not in by_id:
            by_id[decision.finding_id] = decision
    return [by_id[finding_id] if finding_id in by_id else default(finding_id) for finding_id in ordered_ids]


def reconcile_finding_decisions(
    decisions: Iterable[FindingDecision], findings: Iterable[dict[str, object]]
) -> list[FindingDecision]:
    """Close deliberation output over the ordered candidate finding set.

    An undecided finding is kept as written: removal needs a deliberated disproof, and a revision its corrected claim.
    """
    return _close_decisions(
        decisions,
        findings,
        lambda finding_id: FindingDecision(
            finding_id=finding_id,
            decision="keep",
            rationale="The deliberator did not decide this finding, so it is kept as written.",
        ),
    )


def reconcile_assessment_decisions(
    decisions: Iterable[AssessmentDecision], prior_findings: Iterable[dict[str, object]]
) -> list[AssessmentDecision]:
    """Close deliberation output over the ordered baseline finding set.

    A model omission is represented as ``cannot_verify`` so follow-up publication never silently treats it as
    resolved.
    """
    return _close_decisions(
        decisions,
        prior_findings,
        lambda finding_id: AssessmentDecision(
            finding_id=finding_id,
            status="cannot_verify",
            rationale="The deliberator did not return an assessment for this baseline finding.",
        ),
    )


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def render_comment_footer(
    *,
    reviewer_models: list[str],
    deliberation_model: str,
    aggregation_model: str,
    severity: Severity,
) -> str:
    reviewers = " | ".join(f"`{model}`" for model in unique(reviewer_models)) or "(none)"
    return (
        "\n\n> 🤖 **pr-council-mcp**"
        f"\n> Reviewers: {reviewers}"
        f"\n> Deliberator: `{deliberation_model}` · Aggregator: `{aggregation_model}`"
        f"\n> {_SEVERITY_EMOJI[severity]} severity: {severity.value}"
    )


def render_summary_footer(models: dict[str, list[str]], *, deliberation_model: str, aggregation_model: str) -> str:
    lines = ["\n\n> 🤖 **pr-council-mcp**"]
    for disposition, model_ids in models.items():
        rendered = " | ".join(f"`{model}`" for model in unique(model_ids)) or "(none)"
        lines.append(f"> {disposition.title()}: {rendered}")
    lines.append(f"> Deliberation: `{deliberation_model}` · Aggregation: `{aggregation_model}`")
    return "\n".join(lines)


def _count(count: int, singular: str, plural: str, *, zero: str) -> str:
    if count == 0:
        return zero
    return f"1 {singular}" if count == 1 else f"{count} {plural}"


def render_review_summary(
    findings: Iterable[AggregatedFinding],
    *,
    follow_up: Mapping[str, int] | None = None,
) -> str:
    """Render the top-level status from the published findings and follow-up counts alone.

    The text is built from counts so it can never disagree with the comments it summarizes.
    """
    if follow_up is None:
        blocking = sum(finding.severity in {Severity.CRITICAL, Severity.HIGH} for finding in findings)
        if blocking == 0:
            return "No critical or high-severity issues need to be addressed before merge."
        noun = "issue needs" if blocking == 1 else "issues need"
        return f"{blocking} critical or high-severity {noun} to be addressed before merge."
    resolved = follow_up.get("resolved", 0)
    unresolved = follow_up.get("still_present", 0) + follow_up.get("partially_resolved", 0)
    unverified = follow_up.get("cannot_verify", 0)
    if resolved + unresolved + unverified == 0:
        prior = "No issues were previously reported."
    else:
        clauses = [
            _count(
                resolved,
                "previously reported issue was resolved",
                "previously reported issues were resolved",
                zero="No previously reported issues were resolved",
            ),
            _count(unresolved, "remains unresolved", "remain unresolved", zero="none remain unresolved"),
        ]
        if unverified:
            clauses.append(f"{unverified} could not be verified")
        prior = "; ".join(clauses) + "."
    new = _count(
        follow_up.get("new", 0),
        "new issue was found.",
        "new issues were found.",
        zero="No new issues were found.",
    )
    return f"{prior} {new}"


def render_additional_findings(findings: Iterable[AggregatedFinding]) -> str:
    """Render non-inline findings with their only actionable explanation."""
    items = list(findings)
    if not items:
        return ""
    return "\n\n## Additional findings (not on changed lines)\n" + "\n".join(
        f"- **{finding.path}:{finding.line}** [{finding.severity.value}] — {finding.title}: {finding.body}"
        for finding in items
    )


def render_html_metadata(marker: str, value: object) -> str:
    # Metadata is built only from validated identifiers/counts, never model prose.
    rendered = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    if "-->" in rendered:
        raise ValueError("unsafe HTML-comment metadata")
    return f"\n\n<!-- {marker}\n{rendered}\n-->"


def render_stats_comment(metrics: ReviewMetrics) -> str:
    data = metrics.model_dump(by_alias=True)
    # Preserve the TypeScript harness's established camelCase metadata contract.
    compatibility = {
        "schemaVersion": data.pop("schema_version"),
        "models": data.pop("models"),
        "findings": data.pop("findings"),
        "overlappingFindings": data.pop("overlapping_findings"),
        "durationMs": data.pop("duration_ms"),
        "totalTokens": data.pop("total_tokens"),
        "followUp": data.pop("follow_up"),
        **data,
    }
    return render_html_metadata("pr-council-mcp-stats", compatibility)


def render_comment_metadata(
    finding: AggregatedFinding,
    *,
    operation_id: str,
    revision: int,
    deliberation_model: str,
    aggregation_model: str,
) -> str:
    return render_html_metadata(
        "pr-council-mcp-comment",
        {
            "schemaVersion": 1,
            "operationId": operation_id,
            "revision": revision,
            "findingId": finding.id,
            "sourceFindingIds": finding.source_finding_ids,
            "dispositions": finding.dispositions,
            "reviewerModels": finding.reviewer_models,
            "deliberationModel": deliberation_model,
            "aggregationModel": aggregation_model,
            "severity": finding.severity.value,
        },
    )
