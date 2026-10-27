import os
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

import pr_council.workflows.review.graph as graph_module
from pr_council.agents.review.common import Usage
from pr_council.agents.review.deliberator import DeliberationCandidate
from pr_council.config import Config, PRReviewConfig, PRReviewLimitsConfig
from pr_council.review.git import LocalSourceSnapshot
from pr_council.review.models import (
    AggregatedFinding,
    AggregationCandidate,
    AssessmentDecision,
    Finding,
    FindingDecision,
    FindingRevision,
    IterationCandidate,
    OperationStatus,
    PriorFindingAssessment,
    ReviewError,
    Severity,
)
from pr_council.workflows.review.graph import ReviewGraphDeps, build_review_graph


class FakeModelOwner:
    def __init__(self) -> None:
        self.model = object()

    async def __aenter__(self):
        return self.model

    async def __aexit__(self, *exc):
        return None


class FakeModelFactory:
    def __init__(self, calls=None) -> None:
        self.calls = calls

    def create(self, model_id, *, max_output_tokens=None, reasoning_effort=None):
        if self.calls is not None:
            self.calls.append((model_id, max_output_tokens, reasoning_effort))
        return FakeModelOwner()


class FakeStore:
    def __init__(self, baseline=None):
        self.statuses = []
        self.preview = None
        self.result = None
        self.baseline = baseline

    async def is_cancel_requested(self, operation_id):
        return False

    async def update(self, operation_id, **values):
        if values.get("status"):
            self.statuses.append(values["status"])
        if values.get("preview"):
            self.preview = values["preview"]
        if values.get("result"):
            self.result = values["result"]

    async def latest_completed(self, *args):
        return self.baseline

    async def require_owned(self, *args):
        return self.baseline


class FakeGit:
    def __init__(self):
        self.posted = []
        self.removed = []
        self.released = []
        self.clone_calls = 0
        self.worktree_calls = 0
        self.local_validation_calls = []
        self.local_source_valid = True
        self.excluded_paths = ()
        self.base_sha = "base-sha"
        self.head_sha = "head-sha"
        self.login = "reviewer"
        self.bindings = []

    def bind(self, host, account):
        self.bindings.append((host, account))
        return self

    async def clone_or_fetch(self, ref, base_dir):
        self.clone_calls += 1
        path = base_dir / ref.repo_key
        path.mkdir(parents=True)
        return path

    async def fetch_head(self, repo_path, pr_number):
        return self.head_sha

    async def pr_shas(self, ref):
        return self.base_sha, self.head_sha

    async def materialize_comparison(self, repo_path, **kwargs):
        return None

    async def create_worktree(self, repo_path, worktree_path, head_sha):
        self.worktree_calls += 1
        worktree_path.mkdir(parents=True)
        (worktree_path / "app.py").write_text("value = 1\n")

    def linked_worktree_metadata_root(self, repo_path, worktree_path, repositories_root):
        return repo_path

    async def validate_local_source(self, source_path, allowed_root, **kwargs):
        self.local_validation_calls.append((source_path, allowed_root, kwargs))
        if not self.local_source_valid:
            raise ReviewError("local source HEAD does not match the pull request head")
        return LocalSourceSnapshot(source_path / ".git", self.excluded_paths)

    async def diff(self, repo_path, ref):
        return "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+value = 1\n"

    async def authenticated_user(self, host):
        return self.login

    async def remove_worktree(self, repo_path, worktree_path):
        self.removed.append(worktree_path)
        return None

    async def release_comparison(self, repo_path, operation_id):
        self.released.append(operation_id)

    async def review_already_posted(self, *args):
        return False

    async def post_comment_review(self, ref, **values):
        self.posted.append(values)


class FakeLock:
    @asynccontextmanager
    async def acquire(self, key):
        yield


class FakeReviewerAgent:
    profiles = []

    def __init__(
        self,
        model,
        sandbox_profile,
        base_sha,
        head_sha,
        disposition,
        model_id,
        tool_budget,
        **kwargs,
    ):
        type(self).profiles.append(sandbox_profile)
        self.disposition = disposition
        self.model_id = model_id
        self.tool_budget = tool_budget

    async def run_iteration(self, **kwargs):
        finding = Finding(
            id=f"{self.disposition}-{self.model_id}",
            title="Incorrect value",
            body="The new value is incorrect.",
            severity=Severity.HIGH,
            path="app.py",
            line=1,
            disposition=self.disposition,
            model=self.model_id,
        )
        return IterationCandidate(
            findings=[finding],
            prior_assessments=[
                PriorFindingAssessment(
                    finding_id="hallucinated-baseline-id",
                    status="resolved",
                    explanation="No such baseline exists.",
                )
            ],
        ), Usage(total_tokens=10)


class FlakyReviewerAgent(FakeReviewerAgent):
    """Sends one model call through the retry policy the graph supplies; model-a's call is rate limited first."""

    failures = 0
    calls: Counter[str] = Counter()

    def __init__(self, *args, retry, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry = retry

    async def run_iteration(self, **kwargs):
        type(self).calls[self.model_id] += 1
        attempts = 0

        async def model_call(request):
            nonlocal attempts
            attempts += 1
            if self.model_id == "model-a" and attempts <= self.failures:
                error = Exception("Error code: 429 - rate limit exceeded")
                error.status_code = 429  # type: ignore[attr-defined]
                raise error

        await self.retry.awrap_model_call(None, model_call)
        return await super().run_iteration(**kwargs)


class NonRetryableReviewerAgent(FakeReviewerAgent):
    """Always fails with a deterministic (non-transient) error."""

    calls = 0

    async def run_iteration(self, **kwargs):
        type(self).calls += 1
        raise ValueError("invalid model configuration")


async def fake_deliberate(model, sandbox_profile, base_sha, head_sha, **values):
    return (
        DeliberationCandidate(
            finding_decisions=[
                FindingDecision(finding_id=finding["id"], decision="keep", rationale="confirmed")
                for finding in values["findings"]
            ],
            assessment_decisions=[
                AssessmentDecision(
                    finding_id=finding["id"],
                    status="still_present",
                    rationale="The prior issue remains actionable.",
                )
                for finding in values["prior_findings"]
            ],
        ),
        Usage(total_tokens=5),
    )


async def fake_aggregate(model, **values):
    findings = values["findings"]
    first = findings[0]
    candidate = AggregationCandidate(
        findings=[
            AggregatedFinding(
                id="model-value-is-replaced",
                title="Incorrect value",
                body="The new value is incorrect.",
                severity=Severity.INFO,
                path=first["path"],
                line=first["line"],
                dispositions=[],
                reviewer_models=[],
                source_finding_ids=[finding["id"] for finding in findings],
            )
        ],
    )
    return candidate, Usage(total_tokens=5)


def _initial_graph_state(operation_id: str) -> dict[str, object]:
    return {
        "operation_id": operation_id,
        "request": {
            "owner_uid": 1,
            "mode": "initial",
            "baseline_operation_id": None,
            "context": [],
            "models": {"quality": ["model-a"], "security": ["model-b"]},
            "iterations": 1,
            "deliberation_model": "model-d",
            "aggregation_model": "model-s",
        },
        "ref": {"host": "github.com", "owner": "org", "repo": "repo", "number": 7},
        "reviewer_results": [],
        "deliberation_results": [],
    }


async def test_prepare_preserves_failure_and_attempts_both_post_pin_cleanups(tmp_path):
    class FailingGit(FakeGit):
        async def diff(self, repo_path, ref):
            raise RuntimeError("authoritative diff failed")

        async def remove_worktree(self, repo_path, worktree_path):
            self.removed.append(worktree_path)
            raise RuntimeError("worktree cleanup failed")

        async def release_comparison(self, repo_path, operation_id):
            self.released.append(operation_id)
            raise RuntimeError("ref cleanup failed")

    git = FailingGit()
    deps = ReviewGraphDeps(
        config=Config(),
        model_factory=FakeModelFactory(),
        store=FakeStore(),
        git=git,
        lock=FakeLock(),
        repos_dir=tmp_path / "repos",
        worktrees_dir=tmp_path / "worktrees",
    )
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "failure.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        with pytest.raises(RuntimeError, match="authoritative diff failed"):
            await graph.ainvoke(
                _initial_graph_state("operation-failure"),
                config={"configurable": {"thread_id": "operation-failure"}},
            )

    assert git.removed == [tmp_path / "worktrees" / "operation-failure"]
    assert git.released == ["operation-failure"]


async def test_prepare_rejects_endpoint_movement_during_authoritative_diff(tmp_path):
    class MovingGit(FakeGit):
        async def diff(self, repo_path, ref):
            result = await super().diff(repo_path, ref)
            self.base_sha = "moved-base-sha"
            return result

    git = MovingGit()
    deps = ReviewGraphDeps(
        config=Config(),
        model_factory=FakeModelFactory(),
        store=FakeStore(),
        git=git,
        lock=FakeLock(),
        repos_dir=tmp_path / "repos",
        worktrees_dir=tmp_path / "worktrees",
    )
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "movement.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        with pytest.raises(ReviewError, match="base or head changed while"):
            await graph.ainvoke(
                _initial_graph_state("operation-movement"),
                config={"configurable": {"thread_id": "operation-movement"}},
            )

    assert git.removed == [tmp_path / "worktrees" / "operation-movement"]
    assert git.released == ["operation-movement"]


@pytest.mark.parametrize(
    ("source_mode", "changed_endpoint"),
    [
        ("managed", None),
        ("managed", "head"),
        ("managed", "base"),
        ("local", None),
        ("local", "head"),
        ("local", "base"),
        ("local", "source"),
    ],
)
async def test_graph_checkpoints_preview_then_resumes_comment_only_commit(
    monkeypatch, tmp_path, changed_endpoint, source_mode
):
    monkeypatch.setattr(graph_module, "ReviewerAgent", FakeReviewerAgent)
    monkeypatch.setattr(graph_module, "deliberate", fake_deliberate)
    monkeypatch.setattr(graph_module, "aggregate", fake_aggregate)
    model_calls = []

    model_factory = FakeModelFactory(model_calls)

    config = Config()
    store = FakeStore()
    git = FakeGit()
    FakeReviewerAgent.profiles = []
    local_repository = tmp_path / "local-repository"
    local_repository.mkdir()
    local_secret = local_repository / ".mcp.json"
    if source_mode == "local":
        local_secret.write_text("secret\n")
        git.excluded_paths = (local_secret,)
    deps = ReviewGraphDeps(
        config=config,
        model_factory=model_factory,
        store=store,
        git=git,
        lock=FakeLock(),
        repos_dir=tmp_path / "repos",
        worktrees_dir=tmp_path / "worktrees",
        local_source_root=tmp_path,
    )
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    async with AsyncSqliteSaver.from_conn_string(os.fspath(checkpoint_path)) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
        initial = {
            "operation_id": "operation-1",
            "request": {
                "owner_uid": 1,
                "source_mode": source_mode,
                "local_source_path": str(local_repository) if source_mode == "local" else None,
                "mode": "initial",
                "baseline_operation_id": None,
                "context": [],
                "models": {"quality": ["model-a"], "security": ["model-b"]},
                "iterations": 2,
                "deliberation_model": "model-d",
                "aggregation_model": "model-s",
                "review_efforts": {"model-a": "xhigh", "model-b": None},
                "deliberation_effort": "medium",
                "aggregation_effort": "low",
            },
            "ref": {"host": "github.com", "owner": "org", "repo": "repo", "number": 7},
            "reviewer_results": [],
            "deliberation_results": [],
        }

        paused = await graph.ainvoke(initial, config=graph_config)
        assert paused["__interrupt__"]
        assert store.statuses[-1] == OperationStatus.READY
        assert store.preview["comments"][0]["body"].endswith("-->")
        assert set(store.preview["comments"][0]["reviewer_models"]) == {"model-a", "model-b"}
        assert "`model-a`" in store.preview["comments"][0]["body"]
        assert "`model-b`" in store.preview["comments"][0]["body"]
        assert "<!-- pr-council-mcp-stats" in store.preview["summary_body"]
        assert "<!-- pr-council-mcp-operation" in store.preview["summary_body"]
        assert store.preview["metrics"]["reviewers"]["iterationsCompleted"] == 4
        assert store.preview["metrics"]["total_tokens"] == 55
        # Each role runs at its persisted effort, and no call caps the model's output.
        assert Counter(model_calls) == {
            ("model-a", None, "xhigh"): 2,
            ("model-b", None, None): 2,
            ("model-d", None, "medium"): 2,
            ("model-s", None, "low"): 1,
        }

    async with AsyncSqliteSaver.from_conn_string(os.fspath(checkpoint_path)) as restarted_checkpointer:
        await restarted_checkpointer.setup()
        restarted_graph = build_review_graph(deps, restarted_checkpointer)
        if changed_endpoint == "head":
            git.head_sha = "new-head-sha"
        elif changed_endpoint == "base":
            git.base_sha = "new-base-sha"
        elif changed_endpoint == "source":
            git.local_source_valid = False
        await restarted_graph.ainvoke(
            Command(
                resume={
                    "action": "commit",
                    "revision": store.preview["revision"],
                    "payload_hash": store.preview["payload_hash"],
                }
            ),
            config=graph_config,
        )

    if changed_endpoint:
        assert store.statuses[-1] == OperationStatus.STALE
        assert git.posted == []
        if changed_endpoint == "source":
            assert store.result == {
                "operation_id": "operation-1",
                "status": "stale",
                "reason": "Local source changed or no longer matches the reviewed revision; no review was published.",
                "preview_base_sha": "base-sha",
                "current_base_sha": "base-sha",
                "preview_head_sha": "head-sha",
                "current_head_sha": "head-sha",
                "next_action": "Restore a clean local checkout at the pull request head and start a new review.",
            }
        else:
            assert store.result == {
                "operation_id": "operation-1",
                "status": "stale",
                "reason": "Pull request base or head changed after preview; no review was published.",
                "preview_base_sha": "base-sha",
                "current_base_sha": "new-base-sha" if changed_endpoint == "base" else "base-sha",
                "preview_head_sha": "head-sha",
                "current_head_sha": "new-head-sha" if changed_endpoint == "head" else "head-sha",
                "next_action": "Start a new PR review for the current base and head.",
            }
    else:
        assert store.statuses[-1] == OperationStatus.COMPLETED
        assert len(git.posted) == 1
        assert git.posted[0]["comments"][0].severity == Severity.HIGH
        assert store.result["status"] == "completed"
    if source_mode == "local":
        assert git.clone_calls == 0
        assert git.worktree_calls == 0
        assert len(git.local_validation_calls) == 4
        assert git.removed == []
        assert git.released == []
        assert FakeReviewerAgent.profiles
        assert all(profile.denied_paths == (local_secret,) for profile in FakeReviewerAgent.profiles)
    else:
        assert git.clone_calls == 1
        assert git.worktree_calls == 1
        assert git.local_validation_calls == []
        assert git.released == ["operation-1"]
        assert all(profile.denied_paths == () for profile in FakeReviewerAgent.profiles)
    assert all("git" in profile.tools for profile in FakeReviewerAgent.profiles)


async def test_follow_up_aggregation_uses_synthetic_retained_id(monkeypatch, tmp_path):
    monkeypatch.setattr(graph_module, "ReviewerAgent", FakeReviewerAgent)
    monkeypatch.setattr(graph_module, "deliberate", fake_deliberate)
    monkeypatch.setattr(graph_module, "aggregate", fake_aggregate)

    baseline_finding = AggregatedFinding(
        id="baseline-id",
        title="Existing defect",
        body="The original defect.",
        severity=Severity.HIGH,
        path="app.py",
        line=1,
        dispositions=["quality"],
        reviewer_models=["model-a"],
        source_finding_ids=["original-reviewer-id"],
    )
    baseline = SimpleNamespace(
        id="baseline-operation",
        result={"context": [], "findings": [baseline_finding.model_dump(mode="json")]},
    )
    config = Config()
    store = FakeStore(baseline)
    deps = ReviewGraphDeps(
        config=config,
        model_factory=FakeModelFactory(),
        store=store,
        git=FakeGit(),
        lock=FakeLock(),
        repos_dir=tmp_path / "repos",
        worktrees_dir=tmp_path / "worktrees",
    )

    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "follow-up.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        paused = await graph.ainvoke(
            {
                "operation_id": "follow-up-operation",
                "request": {
                    "owner_uid": 1,
                    "mode": "follow_up",
                    "baseline_operation_id": baseline.id,
                    "context": [],
                    "models": {"quality": ["model-a"], "security": ["model-b"]},
                    "iterations": 1,
                    "deliberation_model": "model-d",
                    "aggregation_model": "model-s",
                },
                "ref": {"host": "github.com", "owner": "org", "repo": "repo", "number": 7},
                "reviewer_results": [],
                "deliberation_results": [],
            },
            config={"configurable": {"thread_id": "follow-up-operation"}, "recursion_limit": 100},
        )

    retained_ids = {finding["id"] for finding in paused["retained_findings"]}
    used_ids = {
        source_id for finding in paused["aggregation"]["findings"] for source_id in finding["source_finding_ids"]
    }
    assert paused["__interrupt__"]
    assert "baseline-id" not in retained_ids
    assert len(retained_ids) == 3
    assert retained_ids == used_ids
    assert any(assessment["status"] == "still_present" for assessment in paused["prior_assessments"])


async def _run_flaky_reviewers(monkeypatch, tmp_path, *, failures):
    FlakyReviewerAgent.failures = failures
    FlakyReviewerAgent.calls = Counter()
    _patch_graph_fakes(monkeypatch, reviewer_agent=FlakyReviewerAgent)
    limits = PRReviewLimitsConfig(model_max_attempts=2, rate_limit_base_delay_seconds=0)
    store = FakeStore()
    deps = _make_deps(tmp_path, store, FakeGit())
    deps.config = Config(pr_review=PRReviewConfig(limits=limits))
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "retry.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        paused = await graph.ainvoke(
            _initial_state("retry-operation", {"quality": ["model-a", "model-c"], "security": ["model-b"]}),
            config={"configurable": {"thread_id": "retry-operation"}, "recursion_limit": 100},
        )
    assert paused["__interrupt__"]
    return store


async def test_a_reviewer_model_call_retries_with_the_configured_policy(monkeypatch, tmp_path):
    store = await _run_flaky_reviewers(monkeypatch, tmp_path, failures=1)

    # The rate-limited model call was retried inside the reviewer pass, which ran once and recovered.
    assert store.statuses[-1] == OperationStatus.READY
    assert FlakyReviewerAgent.calls == Counter({"model-a": 1, "model-b": 1, "model-c": 1})
    metrics = store.preview["metrics"]["reviewers"]
    assert (metrics["failed"], metrics["retries"]) == (0, 1)
    assert store.preview["reviewer_failures"] == []
    assert len(store.preview["comments"]) == 1


async def test_a_reviewer_fails_without_rerunning_its_pass_when_retries_run_out(monkeypatch, tmp_path):
    store = await _run_flaky_reviewers(monkeypatch, tmp_path, failures=2)

    # model_max_attempts=2 bounds the model call; the graph does not start the pass again.
    assert FlakyReviewerAgent.calls == Counter({"model-a": 1, "model-b": 1, "model-c": 1})
    assert [(item["model"], item["retries"]) for item in store.preview["reviewer_failures"]] == [("model-a", 1)]
    assert store.preview["metrics"]["reviewers"]["retries"] == 1


async def test_reviewer_does_not_retry_non_retryable_error(monkeypatch, tmp_path):
    NonRetryableReviewerAgent.calls = 0
    _patch_graph_fakes(monkeypatch, reviewer_agent=NonRetryableReviewerAgent)
    deps = _make_deps(tmp_path, FakeStore(), FakeGit())

    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "nonretry.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        with pytest.raises(ReviewError):
            await graph.ainvoke(
                _initial_state("nonretry-operation", {"quality": ["model-a"], "security": ["model-b"]}),
                config={"configurable": {"thread_id": "nonretry-operation"}, "recursion_limit": 100},
            )

    # Each of the two reviewers runs exactly once before failing out.
    assert NonRetryableReviewerAgent.calls == 2


def _make_deps(tmp_path, store, git):
    return ReviewGraphDeps(
        config=Config(),
        model_factory=FakeModelFactory(),
        store=store,
        git=git,
        lock=FakeLock(),
        repos_dir=tmp_path / "repos",
        worktrees_dir=tmp_path / "worktrees",
    )


def _initial_state(operation_id, models):
    return {
        "operation_id": operation_id,
        "request": {
            "owner_uid": 1,
            "mode": "initial",
            "baseline_operation_id": None,
            "context": [],
            "models": models,
            "iterations": 1,
            "deliberation_model": "model-d",
            "aggregation_model": "model-s",
        },
        "ref": {"host": "github.com", "owner": "org", "repo": "repo", "number": 7},
        "reviewer_results": [],
        "deliberation_results": [],
    }


def _patch_graph_fakes(monkeypatch, reviewer_agent=FakeReviewerAgent):
    monkeypatch.setattr(graph_module, "ReviewerAgent", reviewer_agent)
    monkeypatch.setattr(graph_module, "deliberate", fake_deliberate)
    monkeypatch.setattr(graph_module, "aggregate", fake_aggregate)


async def test_commit_rejects_mismatched_payload_and_never_posts(monkeypatch, tmp_path):
    # The commit gate is a trust boundary: a resume payload whose revision/payload_hash
    # does not match the reviewed preview must be rejected before anything reaches GitHub.
    _patch_graph_fakes(monkeypatch)
    store = FakeStore()
    git = FakeGit()
    deps = _make_deps(tmp_path, store, git)
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "mismatch.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        paused = await graph.ainvoke(
            _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]}),
            config=graph_config,
        )
        assert paused["__interrupt__"]

        with pytest.raises(ReviewError):
            await graph.ainvoke(
                Command(
                    resume={
                        "action": "commit",
                        # Tampered revision that does not match the previewed payload.
                        "revision": store.preview["revision"] + 1,
                        "payload_hash": store.preview["payload_hash"],
                    }
                ),
                config=graph_config,
            )

    # Removing the revision/payload_hash mismatch guard would let commit_node proceed and
    # publish; the empty posted list proves nothing was written to GitHub.
    assert git.posted == []


async def test_commit_rejects_when_authenticated_user_changed(monkeypatch, tmp_path):
    # If the authenticated GitHub identity changes between preview and commit, the review
    # must not be published under a different account.
    class UserChangesGit(FakeGit):
        def __init__(self):
            super().__init__()
            self.auth_calls = 0

        async def authenticated_user(self, host):
            self.auth_calls += 1
            # "reviewer" while preparing (sets owner_login); a different login at commit.
            return "reviewer" if self.auth_calls == 1 else "attacker"

    _patch_graph_fakes(monkeypatch)
    store = FakeStore()
    git = UserChangesGit()
    deps = _make_deps(tmp_path, store, git)
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "userchange.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        paused = await graph.ainvoke(
            _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]}),
            config=graph_config,
        )
        assert paused["__interrupt__"]

        with pytest.raises(ReviewError):
            await graph.ainvoke(
                Command(
                    resume={
                        "action": "commit",
                        "revision": store.preview["revision"],
                        "payload_hash": store.preview["payload_hash"],
                    }
                ),
                config=graph_config,
            )

    # The authenticated-user guard must fire after the payload match check; dropping it
    # would post under the changed identity.
    assert git.auth_calls == 2
    assert git.posted == []


async def test_prepare_and_commit_use_the_account_stored_with_the_operation(monkeypatch, tmp_path):
    _patch_graph_fakes(monkeypatch)
    store = FakeStore()
    git = FakeGit()
    # GitHub logins are case-insensitive, so its canonical spelling still matches the mapped account.
    git.login = "Mapped-User"
    deps = _make_deps(tmp_path, store, git)
    state = _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]})
    state["request"]["github_account"] = "mapped-user"
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "account.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        assert (await graph.ainvoke(state, config=graph_config))["__interrupt__"]
        await graph.ainvoke(
            Command(
                resume={
                    "action": "commit",
                    "revision": store.preview["revision"],
                    "payload_hash": store.preview["payload_hash"],
                }
            ),
            config=graph_config,
        )

    assert git.bindings == [("github.com", "mapped-user"), ("github.com", "mapped-user")]
    assert len(git.posted) == 1


async def test_prepare_rejects_a_token_for_a_different_account(monkeypatch, tmp_path):
    _patch_graph_fakes(monkeypatch)
    git = FakeGit()
    git.login = "someone-else"
    deps = _make_deps(tmp_path, FakeStore(), git)
    state = _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]})
    state["request"]["github_account"] = "mapped-user"
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "wrong.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        with pytest.raises(ReviewError, match='"mapped-user" authenticated as "someone-else"'):
            await graph.ainvoke(state, config={"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100})


async def test_cancel_decision_cleans_up_without_posting(monkeypatch, tmp_path):
    # The decline/cancel branch of the interrupt gate must tear down the worktree, publish
    # nothing, and mark the operation CANCELLED.
    _patch_graph_fakes(monkeypatch)
    store = FakeStore()
    git = FakeGit()
    deps = _make_deps(tmp_path, store, git)
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "cancel.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        paused = await graph.ainvoke(
            _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]}),
            config=graph_config,
        )
        assert paused["__interrupt__"]

        await graph.ainvoke(
            Command(resume={"action": "cancel"}),
            config=graph_config,
        )

    # route_decision sends a non-commit action to cleanup, which removes the worktree,
    # never posts, and records CANCELLED.
    assert git.posted == []
    assert len(git.removed) == 1
    assert store.statuses[-1] == OperationStatus.CANCELLED


async def test_cancellation_request_aborts_with_review_cancelled(monkeypatch, tmp_path):
    # A cancel request observed at a checkpoint must abort the run via ReviewCancelled
    # rather than continuing to a preview/commit.
    class CancelStore(FakeStore):
        async def is_cancel_requested(self, operation_id):
            return True

    _patch_graph_fakes(monkeypatch)
    store = CancelStore()
    git = FakeGit()
    deps = _make_deps(tmp_path, store, git)
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "cancelled.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        with pytest.raises(graph_module.ReviewCancelled):
            await graph.ainvoke(
                _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]}),
                config=graph_config,
            )

    # If check_cancelled stopped raising, the run would reach the interrupt/preview and
    # nothing would be posted-but no ReviewCancelled would surface.
    assert git.posted == []


class DispositionPartialFailureReviewerAgent(FakeReviewerAgent):
    """Fails one specific model with a non-retryable error; all others succeed."""

    FAIL_MODEL = "model-fail"

    async def run_iteration(self, **kwargs):
        if self.model_id == self.FAIL_MODEL:
            raise ValueError("model unavailable")
        return await super().run_iteration(**kwargs)


class GraphSpyTelemetry:
    """Telemetry double that records span kwargs and hands back a fixed callback.

    Each ``span(...)`` records the span ``name``, its ``observation_type``, and
    whether a ``model`` was passed, then yields a no-op observation exposing the
    ``update`` method the graph nodes call. ``langchain_callback()`` returns a
    fixed sentinel so a test can prove the graph threads it into every agent call.
    """

    def __init__(self, callback):
        self.span_calls = []
        self._callback = callback

    @contextmanager
    def span(self, name, *, observation_type="span", metadata=None, trace_seed=None, model=None):
        self.span_calls.append({"name": name, "observation_type": observation_type, "model": model})
        yield SimpleNamespace(update=lambda **kwargs: None)

    def langchain_callback(self):
        return self._callback


async def test_graph_threads_langfuse_callback_and_marks_agent_spans_as_span(monkeypatch, tmp_path):
    # The three agent-enclosing spans must be plain "span" observations (no model=),
    # and the native LangChain callback returned by langchain_callback() must be
    # threaded into every reviewer, deliberator, and aggregator invocation.
    sentinel = object()
    received: dict[str, object] = {}

    class RecordingReviewerAgent(FakeReviewerAgent):
        async def run_iteration(self, **kwargs):
            received["reviewer"] = kwargs.get("langfuse_callback")
            return await super().run_iteration(**kwargs)

    async def recording_deliberate(model, sandbox_profile, base_sha, head_sha, **values):
        received["deliberator"] = values.get("langfuse_callback")
        return await fake_deliberate(model, sandbox_profile, base_sha, head_sha, **values)

    async def recording_aggregate(model, **values):
        received["aggregator"] = values.get("langfuse_callback")
        return await fake_aggregate(model, **values)

    monkeypatch.setattr(graph_module, "ReviewerAgent", RecordingReviewerAgent)
    monkeypatch.setattr(graph_module, "deliberate", recording_deliberate)
    monkeypatch.setattr(graph_module, "aggregate", recording_aggregate)

    telemetry = GraphSpyTelemetry(sentinel)
    store = FakeStore()
    git = FakeGit()
    deps = ReviewGraphDeps(
        config=Config(),
        model_factory=FakeModelFactory(),
        store=store,
        git=git,
        lock=FakeLock(),
        repos_dir=tmp_path / "repos",
        worktrees_dir=tmp_path / "worktrees",
        telemetry=telemetry,
    )
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "spans.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        paused = await graph.ainvoke(
            _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]}),
            config=graph_config,
        )
        assert paused["__interrupt__"]

    agent_spans = {
        call["name"]: call
        for call in telemetry.span_calls
        if call["name"] in {"pr_review.reviewer", "pr_review.deliberator", "pr_review.aggregator"}
    }
    # All three agent spans were opened as plain spans with no model= kwarg. Reverting
    # any of them to observation_type="generation"/model=... trips these assertions.
    assert set(agent_spans) == {"pr_review.reviewer", "pr_review.deliberator", "pr_review.aggregator"}
    for call in agent_spans.values():
        assert call["observation_type"] == "span"
        assert call["model"] is None

    # The native callback was threaded into every agent call. Dropping any
    # langfuse_callback=deps.telemetry.langchain_callback() argument leaves a None here.
    assert received["reviewer"] is sentinel
    assert received["deliberator"] is sentinel
    assert received["aggregator"] is sentinel


async def test_preview_warns_on_degraded_coverage_when_one_reviewer_fails(monkeypatch, tmp_path):
    # With two models in a disposition, one failing and one succeeding, deliberation still
    # proceeds and the preview must surface a degraded-coverage warning naming the failed job.
    _patch_graph_fakes(monkeypatch, reviewer_agent=DispositionPartialFailureReviewerAgent)
    store = FakeStore()
    git = FakeGit()
    deps = _make_deps(tmp_path, store, git)
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "degraded.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(deps, checkpointer)
        paused = await graph.ainvoke(
            _initial_state("operation-1", {"quality": ["model-a", "model-fail"], "security": ["model-c"]}),
            config=graph_config,
        )
        assert paused["__interrupt__"]

    # The failed reviewer is recorded and the preview's summary carries the warning block.
    failures = store.preview["reviewer_failures"]
    assert [(item["disposition"], item["model"]) for item in failures] == [("quality", "model-fail")]
    assert "Review coverage degraded" in store.preview["summary_body"]
    assert "quality/model-fail" in store.preview["summary_body"]


async def test_deliberation_revision_corrects_the_claim_without_changing_identity(monkeypatch, tmp_path):
    _patch_graph_fakes(monkeypatch)

    async def revise_quality_remove_security(model, sandbox_profile, base_sha, head_sha, **values):
        decisions = []
        for finding in values["findings"]:
            if finding["disposition"] == "quality":
                revision = FindingRevision(
                    title="Narrower claim",
                    body="Only empty input is affected.",
                    severity=Severity.MEDIUM,
                    evidence="app.py:1 handles every non-empty value.",
                )
                decisions.append(
                    FindingDecision(
                        finding_id=finding["id"], decision="revise", rationale="Overstated.", revision=revision
                    )
                )
            else:
                decisions.append(
                    FindingDecision(
                        finding_id=finding["id"], decision="remove_as_false_positive", rationale="Disproven."
                    )
                )
        return DeliberationCandidate(finding_decisions=decisions), Usage(total_tokens=5)

    monkeypatch.setattr(graph_module, "deliberate", revise_quality_remove_security)
    store = FakeStore()
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "revise.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(_make_deps(tmp_path, store, FakeGit()), checkpointer)
        paused = await graph.ainvoke(
            _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]}),
            config={"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100},
        )

    [retained] = paused["retained_findings"]
    assert retained["id"] == "quality-model-a"
    assert (retained["path"], retained["line"], retained["model"]) == ("app.py", 1, "model-a")
    assert (retained["title"], retained["body"], retained["severity"]) == (
        "Narrower claim",
        "Only empty input is affected.",
        "medium",
    )
    assert retained["evidence"] == "app.py:1 handles every non-empty value."
    # The reviewer's original claim is kept for comparison.
    original = next(result for result in paused["reviewer_results"] if result["disposition"] == "quality")
    assert (original["findings"][0]["title"], original["findings"][0]["severity"]) == ("Incorrect value", "high")
    # Merged severity follows the corrected claim rather than the reviewer's original rating.
    assert [comment["severity"] for comment in store.preview["comments"]] == ["medium"]
    assert store.preview["metrics"]["results"]["revisedFindings"] == 1
    assert store.preview["metrics"]["results"]["falsePositivesRemoved"] == 1
    assert store.preview["summary"] == "No critical or high-severity issues need to be addressed before merge."


class _RecordedLog:
    def __init__(self):
        self.events = []

    def __getattr__(self, level):
        return lambda event, **values: self.events.append((level, event, values))


async def test_an_incomplete_deliberation_keeps_its_undecided_findings(monkeypatch, tmp_path):
    _patch_graph_fakes(monkeypatch)
    recorded_log = _RecordedLog()
    monkeypatch.setattr(graph_module, "log", recorded_log)

    async def decide_nothing(model, sandbox_profile, base_sha, head_sha, **values):
        values["on_incomplete"](RuntimeError("context too long"))
        return DeliberationCandidate(), Usage(total_tokens=5, tool_calls=2)

    monkeypatch.setattr(graph_module, "deliberate", decide_nothing)
    store = FakeStore()
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "incomplete.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(_make_deps(tmp_path, store, FakeGit()), checkpointer)
        paused = await graph.ainvoke(
            _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]}),
            config={"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100},
        )

    assert [finding["id"] for finding in paused["retained_findings"]] == ["quality-model-a", "security-model-b"]
    assert [
        (decision["decision"], decision["rationale"])
        for result in paused["deliberation_results"]
        for decision in result["finding_decisions"]
    ] == [("keep", "The deliberator did not decide this finding, so it is kept as written.")] * 2
    assert store.preview["metrics"]["reviewers"]["deliberatorToolCalls"] == 4
    # The human approving the preview sees that these findings were never deliberated, not only the log.
    assert [result["undecided_findings"] for result in paused["deliberation_results"]] == [1, 1]
    assert store.preview["summary"].endswith(
        "> ⚠️ **Deliberation incomplete:** 2 findings were not deliberated and kept as the reviewers wrote them."
    )
    events = [(level, event, values.get("disposition")) for level, event, values in recorded_log.events]
    for disposition in ("quality", "security"):
        assert ("warning", "review.deliberation_incomplete", disposition) in events
        assert ("warning", "review.finding_decisions_defaulted", disposition) in events


@pytest.mark.parametrize(
    ("source", "description"),
    [("config", "the configured `min_severity`"), ("override", "this review's `min_severity` override")],
)
async def test_findings_below_min_severity_are_previewed_but_never_published(
    monkeypatch, tmp_path, source, description
):
    _patch_graph_fakes(monkeypatch)
    store = FakeStore()
    git = FakeGit()
    state = _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]})
    state["request"] |= {"min_severity": "critical", "min_severity_source": source}
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "threshold.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(_make_deps(tmp_path, store, git), checkpointer)
        await graph.ainvoke(state, config=graph_config)
        preview = store.preview
        await graph.ainvoke(
            Command(
                resume={"action": "commit", "revision": preview["revision"], "payload_hash": preview["payload_hash"]}
            ),
            config=graph_config,
        )

    assert preview["min_severity"] == "critical"
    assert preview["comments"] == []
    assert [finding["severity"] for finding in preview["suppressed_findings"]] == ["high"]
    assert preview["suppression_note"] == (
        f"1 finding below critical severity was suppressed by {description} and will not be published."
    )
    assert preview["summary"] == "No critical or high-severity issues need to be addressed before merge."
    assert "Incorrect value" not in preview["summary_body"]
    results = preview["metrics"]["results"]
    assert (results["aggregatedFindings"], results["suppressedFindings"], results["inlineComments"]) == (1, 1, 0)
    assert git.posted[0]["comments"] == []
    # Only published findings become a later follow-up's baseline.
    assert store.result["findings"] == []
    assert store.result["summary"] == preview["summary"]


async def test_incomplete_aggregation_is_logged_and_shown_in_the_preview(monkeypatch, tmp_path):
    _patch_graph_fakes(monkeypatch)
    warnings: list[tuple[str, dict[str, object]]] = []

    class RecordingLog:
        def warning(self, event, **values):
            warnings.append((event, values))

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    async def aggregate_incompletely(model, **values):
        values["on_incomplete"](RuntimeError("context too long"))
        return await fake_aggregate(model, **values)

    monkeypatch.setattr(graph_module, "log", RecordingLog())
    monkeypatch.setattr(graph_module, "aggregate", aggregate_incompletely)
    store = FakeStore()
    state = _initial_state("operation-1", {"quality": ["model-a"], "security": ["model-b"]})
    graph_config = {"configurable": {"thread_id": "operation-1"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "incomplete.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(_make_deps(tmp_path, store, FakeGit()), checkpointer)
        await graph.ainvoke(state, config=graph_config)

    assert (
        "review.aggregation_incomplete",
        {
            "operation_id": "operation-1",
            "model": "model-s",
            "error_type": "RuntimeError",
        },
    ) in warnings
    assert len(store.preview["comments"]) == 1
    assert "**Aggregation incomplete:**" in store.preview["summary"]
    assert "Deliberation incomplete" not in store.preview["summary"]


async def aggregate_separately(model, **values):
    candidate = AggregationCandidate(
        findings=[
            AggregatedFinding(
                id="replaced",
                title=finding["title"],
                body=finding["body"],
                severity=finding["severity"],
                path=finding["path"],
                line=finding["line"],
                source_finding_ids=[finding["id"]],
            )
            for finding in values["findings"]
        ]
    )
    return candidate, Usage(total_tokens=5)


def _baseline_operation(severity=Severity.HIGH):
    baseline_finding = AggregatedFinding(
        id="baseline-id",
        title="Existing defect",
        body="The original defect.",
        severity=severity,
        path="app.py",
        line=1,
        dispositions=["quality"],
        reviewer_models=["model-a"],
        source_finding_ids=["original-reviewer-id"],
    )
    return SimpleNamespace(
        id="baseline-operation",
        result={"context": [], "findings": [baseline_finding.model_dump(mode="json")]},
    )


async def test_follow_up_summary_counts_only_findings_not_carried_from_the_baseline(monkeypatch, tmp_path):
    _patch_graph_fakes(monkeypatch)
    monkeypatch.setattr(graph_module, "aggregate", aggregate_separately)
    baseline = _baseline_operation()
    store = FakeStore(baseline)
    state = _initial_state("follow-up-operation", {"quality": ["model-a"], "security": ["model-b"]})
    state["request"] |= {"mode": "follow_up", "baseline_operation_id": baseline.id}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "follow-up.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(_make_deps(tmp_path, store, FakeGit()), checkpointer)
        await graph.ainvoke(
            state, config={"configurable": {"thread_id": "follow-up-operation"}, "recursion_limit": 100}
        )

    # Two reviewer findings are new; the third published finding carries the still-present baseline finding.
    assert len(store.preview["comments"]) == 3
    assert store.preview["metrics"]["follow_up"]["new"] == 2
    assert store.preview["summary"] == (
        "No previously reported issues were resolved; 1 remains unresolved. 2 new issues were found."
    )


async def test_a_suppressed_unresolved_prior_finding_is_disclosed_and_stays_tracked(monkeypatch, tmp_path):
    _patch_graph_fakes(monkeypatch)
    monkeypatch.setattr(graph_module, "aggregate", aggregate_separately)
    baseline = _baseline_operation(Severity.LOW)
    store = FakeStore(baseline)
    git = FakeGit()
    state = _initial_state("follow-up-operation", {"quality": ["model-a"], "security": ["model-b"]})
    state["request"] |= {
        "mode": "follow_up",
        "baseline_operation_id": baseline.id,
        "min_severity": "medium",
        "min_severity_source": "override",
    }
    graph_config = {"configurable": {"thread_id": "follow-up-operation"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(os.fspath(tmp_path / "follow-up.sqlite3")) as checkpointer:
        await checkpointer.setup()
        graph = build_review_graph(_make_deps(tmp_path, store, git), checkpointer)
        await graph.ainvoke(state, config=graph_config)
        preview = store.preview
        await graph.ainvoke(
            Command(
                resume={"action": "commit", "revision": preview["revision"], "payload_hash": preview["payload_hash"]}
            ),
            config=graph_config,
        )

    # The low-severity prior finding is still present but below the threshold, so it has no comment of its own.
    assert [finding["title"] for finding in preview["suppressed_findings"]] == ["Existing defect"]
    assert len(git.posted[0]["comments"]) == 2
    assert preview["metrics"]["follow_up"]["unresolved_suppressed"] == 1
    assert preview["summary"] == (
        "No previously reported issues were resolved; 1 remains unresolved. 1 of them is below this review's "
        "severity threshold and not repeated. 2 new issues were found."
    )
    # An earlier review published it, so the next follow-up still assesses it.
    assert [finding["title"] for finding in store.result["findings"]] == [
        "Incorrect value",
        "Incorrect value",
        "Existing defect",
    ]
