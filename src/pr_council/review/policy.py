"""Pure rendering and validation policy for PR-review publication."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from pr_council.review.models import (
    AggregatedFinding,
    AssessmentDecision,
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


def reconcile_assessment_decisions(
    decisions: Iterable[AssessmentDecision], prior_findings: Iterable[dict[str, object]]
) -> list[AssessmentDecision]:
    """Close deliberation output over the ordered baseline finding set.

    Unknown and duplicate IDs are discarded. A model omission is represented as
    ``cannot_verify`` so follow-up publication never silently treats it as resolved.
    """
    ordered_ids = unique(str(finding["id"]) for finding in prior_findings)
    allowed = set(ordered_ids)
    by_id: dict[str, AssessmentDecision] = {}
    for decision in decisions:
        if decision.finding_id in allowed and decision.finding_id not in by_id:
            by_id[decision.finding_id] = decision
    return [
        by_id.get(
            finding_id,
            AssessmentDecision(
                finding_id=finding_id,
                status="cannot_verify",
                rationale="The deliberator did not return an assessment for this baseline finding.",
            ),
        )
        for finding_id in ordered_ids
    ]


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
    lines.append(f"> Deliberation: `{deliberation_model}` · Summary: `{aggregation_model}`")
    return "\n".join(lines)


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
