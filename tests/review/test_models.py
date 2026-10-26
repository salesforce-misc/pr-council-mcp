import pytest
from pydantic import ValidationError

from pr_council.review.models import FindingDecision, FindingRevision, PrRef, ReviewContextInput, Severity


def test_review_context_input_strips_surrounding_whitespace():
    item = ReviewContextInput(label="  Design note  ", content="  body text  ", source="  origin  ")
    assert item.label == "Design note"
    assert item.content == "body text"
    assert item.source == "origin"


def test_review_context_input_allows_absent_optional_source():
    item = ReviewContextInput(label="label", content="content")
    assert item.source is None


def test_review_context_input_passes_explicit_none_source_through_validator():
    item = ReviewContextInput(label="label", content="content", source=None)
    assert item.source is None


def test_review_context_input_rejects_blank_after_stripping():
    with pytest.raises(ValidationError, match="must not be blank"):
        ReviewContextInput(label="   ", content="content")


def test_pr_ref_url_composes_canonical_pull_request_url():
    ref = PrRef(host="github.com", owner="acme", repo="widget", number=42)
    assert ref.url == "https://github.com/acme/widget/pull/42"


def test_finding_decision_requires_a_revision_exactly_for_revise():
    revision = FindingRevision(title="Narrower claim", body="Only empty input fails.", severity=Severity.LOW)

    revised = FindingDecision(finding_id="f-1", decision="revise", rationale="Overstated.", revision=revision)

    assert revised.revision == revision
    assert revised.revision.evidence is None
    with pytest.raises(ValidationError, match="a revision is required"):
        FindingDecision(finding_id="f-1", decision="revise", rationale="Overstated.")
    for decision in ("keep", "remove_as_false_positive"):
        with pytest.raises(ValidationError, match="not allowed for any other decision"):
            FindingDecision(finding_id="f-1", decision=decision, rationale="Reason.", revision=revision)
