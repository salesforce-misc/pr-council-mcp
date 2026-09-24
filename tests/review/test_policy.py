import json
import re

import pytest

from pr_council.review.models import (
    AggregatedFinding,
    AssessmentDecision,
    PriorFindingAssessment,
    ReviewMetrics,
    Severity,
)
from pr_council.review.policy import (
    canonical_json,
    neutralize_github_markup,
    normalize_prior_assessments,
    payload_hash,
    reconcile_assessment_decisions,
    render_additional_findings,
    render_comment_footer,
    render_comment_metadata,
    render_html_metadata,
    render_stats_comment,
    render_summary_footer,
)


def test_prior_assessments_are_limited_to_unique_baseline_ids():
    assessments = [
        PriorFindingAssessment(finding_id="unknown", status="resolved", explanation="invented"),
        PriorFindingAssessment(finding_id="known", status="still_present", explanation="first"),
        PriorFindingAssessment(finding_id="known", status="resolved", explanation="duplicate"),
    ]

    normalized = normalize_prior_assessments(assessments, ["known"])

    assert [(item.finding_id, item.status) for item in normalized] == [("known", "still_present")]


def test_assessment_decisions_are_closed_over_baseline_and_fill_omissions():
    decisions = [
        AssessmentDecision(finding_id="unknown", status="resolved", rationale="invented"),
        AssessmentDecision(finding_id="first", status="resolved", rationale="confirmed"),
        AssessmentDecision(finding_id="first", status="still_present", rationale="duplicate"),
    ]

    normalized = reconcile_assessment_decisions(decisions, [{"id": "first"}, {"id": "second"}])

    assert [(item.finding_id, item.status) for item in normalized] == [
        ("first", "resolved"),
        ("second", "cannot_verify"),
    ]


def test_comment_footer_attributes_models_by_role_and_severity():
    footer = render_comment_footer(
        reviewer_models=["model-a", "model-b", "model-a"],
        deliberation_model="model-d",
        aggregation_model="model-s",
        severity=Severity.CRITICAL,
    )
    assert "Reviewers: `model-a` | `model-b`" in footer
    assert "Deliberator: `model-d` · Aggregator: `model-s`" in footer
    assert "🔴 severity: critical" in footer


def test_comment_footer_carries_the_pr_council_brand_line():
    footer = render_comment_footer(
        reviewer_models=["model-a"],
        deliberation_model="model-d",
        aggregation_model="model-s",
        severity=Severity.CRITICAL,
    )
    assert "> 🤖 **pr-council-mcp**" in footer


def test_stats_comment_preserves_typescript_core_fields_and_round_trips():
    metrics = ReviewMetrics(
        models={"quality": ["model-a"]},
        findings={"quality": {"model-a": 2}},
        overlapping_findings=1,
        duration_ms=123,
        total_tokens=456,
        tokens={"input": 300, "output": 156},
        timing={"reviewMs": 123},
        reviewers={"requested": 1},
        results={"rawFindings": 2},
        context={"items": 1, "characters": 20},
    )
    rendered = render_stats_comment(metrics)
    match = re.search(r"<!-- pr-council-mcp-stats\n(.*?)\n-->", rendered, re.DOTALL)
    assert match
    value = json.loads(match.group(1))
    assert value["overlappingFindings"] == 1
    assert value["durationMs"] == 123
    assert value["totalTokens"] == 456


def test_neutralize_github_markup_defuses_mentions_and_issue_closers():
    # A zero-width space (​) is inserted after @ and # so GitHub no longer
    # renders a live mention or auto-closes the referenced issue.
    assert neutralize_github_markup("@octocat please, fixes #42") == "@​octocat please, fixes #​42"
    assert neutralize_github_markup("hey @acme/team resolved #7") == "hey @​acme/team resolved #​7"


def test_neutralize_github_markup_leaves_ordinary_text_untouched():
    assert neutralize_github_markup("email user@host and issue number 5") == "email user@host and issue number 5"


def test_render_summary_footer_labels_dispositions_and_dedupes_models():
    footer = render_summary_footer(
        {"quality": ["m-a", "m-a", "m-b"], "security": []},
        deliberation_model="d",
        aggregation_model="s",
    )
    assert footer == (
        "\n\n> 🤖 **pr-council-mcp**\n> Quality: `m-a` | `m-b`\n> Security: (none)\n> Deliberation: `d` · Summary: `s`"
    )


def test_additional_findings_include_actionable_body_when_no_inline_comment_exists():
    finding = AggregatedFinding(
        id="f-1",
        title="Backend compatibility is not checked",
        body="Concise evidence, impact, and remediation.",
        severity=Severity.HIGH,
        path="runtime.py",
        line=12,
        source_finding_ids=["s-1"],
    )

    rendered = render_additional_findings([finding])

    assert rendered == (
        "\n\n## Additional findings (not on changed lines)\n"
        "- **runtime.py:12** [high] — Backend compatibility is not checked: "
        "Concise evidence, impact, and remediation."
    )
    assert finding.body in rendered


def test_render_html_metadata_renders_sorted_indented_comment_block():
    rendered = render_html_metadata("marker", {"b": 1, "a": 2})
    assert rendered == '\n\n<!-- marker\n{\n  "a": 2,\n  "b": 1\n}\n-->'


def test_render_html_metadata_rejects_comment_terminator_injection():
    with pytest.raises(ValueError, match="unsafe HTML-comment metadata"):
        render_html_metadata("marker", {"payload": "abc-->def"})


def test_render_comment_metadata_emits_camelcase_contract_for_a_finding():
    finding = AggregatedFinding(
        id="f-1",
        title="T",
        body="B",
        severity=Severity.HIGH,
        path="a.py",
        line=3,
        dispositions=["quality"],
        reviewer_models=["m-a"],
        source_finding_ids=["s-1"],
    )
    rendered = render_comment_metadata(
        finding,
        operation_id="op-1",
        revision=2,
        deliberation_model="d",
        aggregation_model="s",
    )
    match = re.search(r"<!-- pr-council-mcp-comment\n(.*?)\n-->", rendered, re.DOTALL)
    assert match
    assert json.loads(match.group(1)) == {
        "schemaVersion": 1,
        "operationId": "op-1",
        "revision": 2,
        "findingId": "f-1",
        "sourceFindingIds": ["s-1"],
        "dispositions": ["quality"],
        "reviewerModels": ["m-a"],
        "deliberationModel": "d",
        "aggregationModel": "s",
        "severity": "high",
    }


def test_canonical_json_sorts_keys_and_omits_whitespace():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_payload_hash_is_deterministic_sha256_of_canonical_json():
    assert payload_hash({"a": 1}) == "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
