"""Checkpointed LangGraph for initial and follow-up PR reviews."""

from __future__ import annotations

import asyncio
import hashlib
import operator
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from localmcp.llm import ModelFactory
from localmcp.observability.logging import get_logger
from localmcp.observability.telemetry import LangfuseTelemetry
from localmcp.sandbox import RootAccess, SandboxProfile, SandboxRoot

from pr_council.agents.review.aggregator import aggregate
from pr_council.agents.review.common import (
    ResponseUsageCallback,
    ToolBudget,
    Usage,
    call_with_rate_limit_retry,
    extract_callback_usage,
    is_rate_limit_error,
    is_retryable_error,
    rate_limit_delay,
)
from pr_council.agents.review.deliberator import deliberate
from pr_council.agents.review.reviewer import ReviewerAgent
from pr_council.config import Config
from pr_council.review.diff import commentable_lines
from pr_council.review.git import GitHubCli, LocalSourceSnapshot
from pr_council.review.locks import RepoLock
from pr_council.review.models import (
    AggregatedFinding,
    AggregationCandidate,
    DeliberationResult,
    Finding,
    OperationStatus,
    PreviewComment,
    PriorFindingAssessment,
    PrRef,
    ReviewerFailure,
    ReviewerResult,
    ReviewError,
    ReviewMetrics,
    ReviewMode,
    ReviewPreview,
    ReviewSourceMode,
    Severity,
)
from pr_council.review.policy import (
    neutralize_github_markup,
    normalize_prior_assessments,
    payload_hash,
    reconcile_assessment_decisions,
    render_additional_findings,
    render_comment_footer,
    render_comment_metadata,
    render_stats_comment,
    render_summary_footer,
    unique,
)
from pr_council.review.store import OperationStore

log = get_logger(__name__)

_SEVERITY_RANK = {severity: rank for rank, severity in enumerate(Severity)}


class GraphState(TypedDict, total=False):
    operation_id: str
    request: dict[str, Any]
    ref: dict[str, Any]
    mode: str
    baseline_operation_id: str | None
    baseline_findings: list[dict[str, Any]]
    baseline_context: list[dict[str, Any]]
    context_changes: list[dict[str, Any]]
    context: list[dict[str, Any]]
    source_mode: str
    repo_path: str
    worktree_path: str
    git_metadata_root: str
    source_excluded_paths: list[str] | None
    base_sha: str
    head_sha: str
    diff: str
    owner_login: str
    started_ms: int
    timing: dict[str, int]
    reviewer_results: Annotated[list[dict[str, Any]], operator.add]
    deliberation_results: Annotated[list[dict[str, Any]], operator.add]
    deliberation_target: str
    retained_findings: list[dict[str, Any]]
    prior_assessments: list[dict[str, Any]]
    aggregation: dict[str, Any]
    aggregation_usage: dict[str, Any]
    preview: dict[str, Any]
    decision: dict[str, Any]
    final_status: str


class ReviewerState(TypedDict, total=False):
    operation_id: str
    job: dict[str, str]
    worktree_path: str
    git_metadata_root: str
    source_excluded_paths: list[str] | None
    base_sha: str
    head_sha: str
    context: list[dict[str, Any]]
    context_changes: list[dict[str, Any]]
    prior_findings: list[dict[str, Any]]
    iterations: int
    operation_tool_limit: int
    model_output_tokens_per_call: int
    source_tool_timeout_seconds: float
    iteration: int
    candidate: dict[str, Any]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_accounting_complete: bool
    tool_calls: int
    retries: int
    reviewer_results: list[dict[str, Any]]


class ReviewerOutput(TypedDict):
    reviewer_results: list[dict[str, Any]]


class ReviewCancelled(Exception):
    pass


def _review_sandbox_profile(
    worktree_path: str,
    git_metadata_root: str,
    source_excluded_paths: list[str] | None = None,
) -> SandboxProfile:
    """Build the generic read-only profile used by review agents."""
    return SandboxProfile(
        (
            SandboxRoot(Path(worktree_path), RootAccess.READ_ONLY),
            SandboxRoot(Path(git_metadata_root), RootAccess.READ_ONLY),
        ),
        denied_paths=tuple(Path(path) for path in source_excluded_paths or ()),
    )


def _require_same_local_snapshot(state: GraphState, snapshot: LocalSourceSnapshot) -> None:
    """Reject local authority changes that ordinary Git cleanliness cannot see."""
    expected_metadata = Path(state["git_metadata_root"])
    raw_paths = state.get("source_excluded_paths")
    if raw_paths is None:
        raise ReviewError("local source operation is missing its ignored-path sandbox snapshot")
    expected_paths = tuple(Path(path) for path in raw_paths)
    if snapshot.git_metadata_root != expected_metadata or snapshot.excluded_paths != expected_paths:
        raise ReviewError("local source metadata or ignored-path sandbox snapshot changed")


@dataclass
class ReviewGraphDeps:
    config: Config
    model_factory: ModelFactory
    store: OperationStore
    git: GitHubCli
    lock: RepoLock
    repos_dir: Path
    worktrees_dir: Path
    local_source_root: Path = field(default_factory=Path.cwd)
    telemetry: LangfuseTelemetry = field(default_factory=lambda: LangfuseTelemetry({}))

    async def check_cancelled(self, operation_id: str) -> None:
        if await self.store.is_cancel_requested(operation_id):
            raise ReviewCancelled

    def tool_budget(self, operation_id: str, local_limit: int, operation_limit: int) -> ToolBudget:
        async def claim_operation_call() -> int | None:
            return await self.store.claim_tool_call(operation_id, operation_limit)

        return ToolBudget(local_limit, claim_operation_call)


def build_review_graph(deps: ReviewGraphDeps, checkpointer: Any) -> Any:
    reviewer_builder = StateGraph(ReviewerState, output_schema=ReviewerOutput)

    async def review_iteration(state: ReviewerState) -> dict[str, Any]:
        operation_id = state["operation_id"]
        await deps.check_cancelled(operation_id)
        job = state["job"]
        iteration = state.get("iteration", 0) + 1
        last_error: Exception | None = None
        usage_callback = ResponseUsageCallback()
        limits = deps.config.pr_review.limits
        max_attempts = limits.model_max_attempts
        # Intentionally span retries: failed attempts still consume billable tokens,
        # and operation metrics report total usage rather than only the successful call.
        attempts_made = 0
        failed_tool_calls = 0
        for attempt in range(max_attempts):
            attempts_made += 1
            # Fresh local budget per attempt: ToolBudget.claim() only increments, so
            # reusing one instance would leave a retry pre-charged and prematurely
            # exhausted. The operation-wide claim is durable (store-backed), so the
            # cross-attempt operation limit is still enforced.
            tool_budget = deps.tool_budget(
                operation_id,
                deps.config.pr_review.limits.reviewer_tool_calls_per_iteration,
                state["operation_tool_limit"],
            )
            try:
                with deps.telemetry.span(
                    "pr_review.reviewer",
                    observation_type="span",
                    metadata={
                        "role": "reviewer",
                        "disposition": job["disposition"],
                        "iteration": iteration,
                        "attempt": attempt + 1,
                    },
                ) as observation:
                    model_owner = deps.model_factory.create(
                        model_id=job["model"],
                        max_output_tokens=state["model_output_tokens_per_call"],
                    )
                    async with model_owner as model:
                        agent = ReviewerAgent(
                            model,
                            _review_sandbox_profile(
                                state["worktree_path"],
                                state["git_metadata_root"],
                                state.get("source_excluded_paths"),
                            ),
                            state["base_sha"],
                            state["head_sha"],
                            job["disposition"],
                            job["model"],
                            tool_budget,
                            source_tool_timeout_seconds=state["source_tool_timeout_seconds"],
                        )
                        candidate, usage = await agent.run_iteration(
                            iteration=iteration,
                            total_iterations=state["iterations"],
                            context=state["context"],
                            prior_findings=state.get("prior_findings", []),
                            previous=state.get("candidate"),
                            context_changes=state.get("context_changes", []),
                            usage_callback=usage_callback,
                            langfuse_callback=deps.telemetry.langchain_callback(),
                        )
                    # Fold in tool calls charged by earlier transiently-failed attempts:
                    # the operation-wide store already billed them, so the successful
                    # result must report them too or node telemetry underreports usage.
                    total_tool_calls = usage.tool_calls + failed_tool_calls
                    observation.update(
                        usage_details={
                            "input": usage.input_tokens,
                            "output": usage.output_tokens,
                            "total": usage.total_tokens,
                        },
                        metadata={
                            "tool_calls": total_tool_calls,
                            "token_accounting_complete": usage.complete,
                            "finding_count": len(candidate.findings),
                        },
                    )
                allowed_prior_ids = {str(finding["id"]) for finding in state.get("prior_findings", [])}
                normalized_assessments = normalize_prior_assessments(candidate.prior_assessments, allowed_prior_ids)
                if len(normalized_assessments) != len(candidate.prior_assessments):
                    log.info(
                        "review.prior_assessments_normalized",
                        operation_id=operation_id,
                        disposition=job["disposition"],
                        model=job["model"],
                        supplied=len(candidate.prior_assessments),
                        retained=len(normalized_assessments),
                    )
                candidate = candidate.model_copy(update={"prior_assessments": normalized_assessments})
                return {
                    "iteration": iteration,
                    "candidate": candidate.model_dump(mode="json"),
                    "input_tokens": state.get("input_tokens", 0) + usage.input_tokens,
                    "output_tokens": state.get("output_tokens", 0) + usage.output_tokens,
                    "total_tokens": state.get("total_tokens", 0) + usage.total_tokens,
                    "token_accounting_complete": state.get("token_accounting_complete", True) and usage.complete,
                    "tool_calls": state.get("tool_calls", 0) + total_tool_calls,
                    "retries": state.get("retries", 0) + attempt,
                }
            except Exception as exc:
                last_error = exc
                failed_tool_calls += tool_budget.used
                # Deterministic failures (validation, unknown-model config, exhausted
                # tool budget) are not transient: retrying only re-runs the reviewer's
                # multi-turn loop and burns tokens. Match call_with_rate_limit_retry and
                # stop immediately, falling through to the failed-result return below.
                if not is_retryable_error(exc):
                    break
                if attempt + 1 < max_attempts:
                    delay = rate_limit_delay(
                        exc,
                        attempt + 1,
                        base_delay=limits.rate_limit_base_delay_seconds,
                        max_delay=limits.rate_limit_max_delay_seconds,
                    )
                    log.warning(
                        "review.reviewer_retry",
                        operation_id=operation_id,
                        disposition=job["disposition"],
                        model=job["model"],
                        iteration=iteration,
                        attempt=attempt + 1,
                        delay_seconds=round(delay, 2),
                        rate_limited=is_rate_limit_error(exc),
                        error_type=type(exc).__name__,
                        status_code=getattr(exc, "status_code", None),
                    )
                    await asyncio.sleep(delay)
        assert last_error is not None
        failed_usage = extract_callback_usage(usage_callback, tool_calls=failed_tool_calls)
        retries = max(0, attempts_made - 1)
        return {
            "iteration": state["iterations"],
            "candidate": {"findings": [], "prior_assessments": [], "notes": []},
            "retries": state.get("retries", 0) + retries,
            "input_tokens": state.get("input_tokens", 0) + failed_usage.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + failed_usage.output_tokens,
            "total_tokens": state.get("total_tokens", 0) + failed_usage.total_tokens,
            "tool_calls": state.get("tool_calls", 0) + failed_usage.tool_calls,
            "token_accounting_complete": False,
            "reviewer_results": [
                ReviewerResult(
                    disposition=job["disposition"],
                    model=job["model"],
                    failed=True,
                    error=str(last_error),
                    retries=state.get("retries", 0) + retries,
                    input_tokens=state.get("input_tokens", 0) + failed_usage.input_tokens,
                    output_tokens=state.get("output_tokens", 0) + failed_usage.output_tokens,
                    total_tokens=state.get("total_tokens", 0) + failed_usage.total_tokens,
                    tool_calls=state.get("tool_calls", 0) + failed_usage.tool_calls,
                    token_accounting_complete=False,
                ).model_dump(mode="json")
            ],
        }

    def continue_iterations(state: ReviewerState) -> str:
        if state.get("reviewer_results"):
            return "done"
        return "again" if state.get("iteration", 0) < state["iterations"] else "done"

    async def finalize_reviewer(state: ReviewerState) -> dict[str, Any]:
        if state.get("reviewer_results"):
            return {}
        job = state["job"]
        candidate = state.get("candidate", {})
        return {
            "reviewer_results": [
                ReviewerResult(
                    disposition=job["disposition"],
                    model=job["model"],
                    findings=candidate.get("findings", []),
                    prior_assessments=candidate.get("prior_assessments", []),
                    notes=candidate.get("notes", []),
                    iterations_completed=state.get("iteration", 0),
                    tool_calls=state.get("tool_calls", 0),
                    retries=state.get("retries", 0),
                    input_tokens=state.get("input_tokens", 0),
                    output_tokens=state.get("output_tokens", 0),
                    total_tokens=state.get("total_tokens", 0),
                    token_accounting_complete=state.get("token_accounting_complete", True),
                ).model_dump(mode="json")
            ]
        }

    reviewer_builder.add_node("iteration", review_iteration)
    reviewer_builder.add_node("finalize", finalize_reviewer)
    reviewer_builder.add_edge(START, "iteration")
    reviewer_builder.add_conditional_edges("iteration", continue_iterations, {"again": "iteration", "done": "finalize"})
    reviewer_builder.add_edge("finalize", END)
    reviewer_graph = reviewer_builder.compile()

    builder = StateGraph(GraphState)

    async def initialize(state: GraphState) -> dict[str, Any]:
        started = int(time.time() * 1000)
        request = state["request"]
        ref = PrRef.model_validate(state["ref"])
        baseline_id = request.get("baseline_operation_id")
        mode = ReviewMode(request.get("mode", ReviewMode.AUTO))
        baseline = None
        if baseline_id:
            baseline = await deps.store.require_owned(baseline_id, int(request["owner_uid"]))
        elif mode != ReviewMode.INITIAL:
            baseline = await deps.store.latest_completed(ref.repo_key, ref.number, int(request["owner_uid"]))
        if mode == ReviewMode.FOLLOW_UP and baseline is None:
            raise ReviewError("follow-up review requires a completed baseline operation")
        effective = ReviewMode.FOLLOW_UP if baseline is not None and mode != ReviewMode.INITIAL else ReviewMode.INITIAL
        context = request.get("context")
        baseline_context = (baseline.result or {}).get("context", []) if baseline else []
        if context is None and baseline and baseline.result:
            context = baseline_context
        context = context or []
        await deps.store.update(state["operation_id"], status=OperationStatus.PREPARING)
        return {
            "started_ms": started,
            "mode": effective.value,
            "baseline_operation_id": baseline.id if baseline else None,
            "baseline_findings": (baseline.result or {}).get("findings", []) if baseline else [],
            "baseline_context": baseline_context,
            "context_changes": _context_changes(baseline_context, context),
            "context": context,
            "timing": {},
        }

    async def prepare(state: GraphState) -> dict[str, Any]:
        began = int(time.time() * 1000)
        operation_id = state["operation_id"]
        await deps.check_cancelled(operation_id)
        ref = PrRef.model_validate(state["ref"])
        source_mode = ReviewSourceMode(state["request"].get("source_mode", ReviewSourceMode.MANAGED.value))
        async with deps.lock.acquire(ref.repo_key):
            base_sha, advertised_head = await deps.git.pr_shas(ref)
            head_sha = advertised_head
            if source_mode == ReviewSourceMode.LOCAL:
                raw_local_path = state["request"].get("local_source_path")
                if not isinstance(raw_local_path, str):
                    raise ReviewError("local source operation is missing its repository path")
                repo_path = Path(raw_local_path)
                worktree_path = repo_path
                local_snapshot = await deps.git.validate_local_source(
                    repo_path,
                    deps.local_source_root,
                    base_sha=base_sha,
                    head_sha=head_sha,
                )
                git_metadata_root = local_snapshot.git_metadata_root
                source_excluded_paths = [os.fspath(path) for path in local_snapshot.excluded_paths]
                # This is GitHub's canonical patch solely for later inline-comment
                # eligibility. Reviewers inspect the validated local base/head
                # objects directly through their source sandbox.
                diff = await deps.git.diff(repo_path, ref)
                confirmed_base, confirmed_head = await deps.git.pr_shas(ref)
                confirmed_snapshot = await deps.git.validate_local_source(
                    repo_path,
                    deps.local_source_root,
                    base_sha=base_sha,
                    head_sha=head_sha,
                )
                # Revalidate the original revision and tracked-path authority to
                # close local mutations during the canonical-diff request window;
                # the tuple comparison below independently detects PR endpoint drift.
                if confirmed_snapshot != local_snapshot:
                    raise ReviewError("local source changed while the review was being prepared")
                if (confirmed_base, confirmed_head) != (base_sha, head_sha):
                    raise ReviewError("pull request base or head changed while the review was being prepared")
            else:
                source_excluded_paths = None
                repo_path = await deps.git.clone_or_fetch(ref, deps.repos_dir)
                fetched_head = await deps.git.fetch_head(repo_path, ref.number)
                if fetched_head != advertised_head:
                    raise ReviewError("pull request head changed while the review was being prepared")
                await deps.git.materialize_comparison(
                    repo_path,
                    base_sha=base_sha,
                    head_sha=head_sha,
                    operation_id=operation_id,
                )
                worktree_path = deps.worktrees_dir / operation_id
                try:
                    # Persist cleanup coordinates as soon as refs are pinned so
                    # crash recovery can release them after an interrupted setup.
                    await deps.store.update(
                        operation_id,
                        state={
                            "source_mode": source_mode.value,
                            "repo_path": os.fspath(repo_path),
                            "worktree_path": os.fspath(worktree_path),
                        },
                    )
                    await deps.git.create_worktree(repo_path, worktree_path, head_sha)
                    git_metadata_root = deps.git.linked_worktree_metadata_root(
                        repo_path,
                        worktree_path,
                        deps.repos_dir,
                    )
                    diff = await deps.git.diff(repo_path, ref)
                    confirmed_base, confirmed_head = await deps.git.pr_shas(ref)
                    confirmed_fetched_head = await deps.git.fetch_head(repo_path, ref.number)
                    if (confirmed_base, confirmed_head, confirmed_fetched_head) != (base_sha, head_sha, head_sha):
                        raise ReviewError("pull request base or head changed while the review was being prepared")
                except BaseException:
                    cleanup_failures: list[str] = []
                    try:
                        await deps.git.remove_worktree(repo_path, worktree_path)
                    except Exception as exc:
                        cleanup_failures.append(type(exc).__name__)
                    try:
                        await deps.git.release_comparison(repo_path, operation_id)
                    except Exception as exc:
                        cleanup_failures.append(type(exc).__name__)
                    if cleanup_failures:
                        log.warning(
                            "review.prepare_cleanup_failed",
                            operation_id=operation_id,
                            error_types=cleanup_failures,
                        )
                    raise
        login = await deps.git.authenticated_user(ref.host)
        timing = {**state.get("timing", {}), "setupMs": int(time.time() * 1000) - began}
        catalog_state = {
            "source_mode": source_mode.value,
            "repo_path": os.fspath(repo_path),
            "worktree_path": os.fspath(worktree_path),
            "git_metadata_root": os.fspath(git_metadata_root),
            "source_excluded_paths": source_excluded_paths,
            "base_sha": base_sha,
            "head_sha": head_sha,
        }
        await deps.store.update(
            operation_id,
            status=OperationStatus.REVIEWING,
            owner_login=login,
            state=catalog_state,
        )
        return {
            **catalog_state,
            "diff": diff,
            "owner_login": login,
            "timing": timing,
            "reviewer_results": [],
            "deliberation_results": [],
        }

    def dispatch_reviewers(state: GraphState) -> list[Send]:
        request = state["request"]
        models = request["models"]
        jobs = [
            {"disposition": disposition, "model": model}
            for disposition, model_ids in models.items()
            for model in model_ids
        ]
        return [
            Send(
                "reviewer",
                {
                    "operation_id": state["operation_id"],
                    "job": job,
                    "worktree_path": state["worktree_path"],
                    "git_metadata_root": state["git_metadata_root"],
                    "source_excluded_paths": state.get("source_excluded_paths"),
                    "base_sha": state["base_sha"],
                    "head_sha": state["head_sha"],
                    "context": state["context"],
                    "prior_findings": [
                        finding
                        for finding in state.get("baseline_findings", [])
                        if job["disposition"] in finding.get("dispositions", [finding.get("disposition")])
                    ],
                    "context_changes": state.get("context_changes", []),
                    "iterations": request["iterations"],
                    "operation_tool_limit": request["operation_tool_limit"],
                    "model_output_tokens_per_call": request["model_output_tokens_per_call"],
                    "source_tool_timeout_seconds": request.get(
                        "source_tool_timeout_seconds",
                        deps.config.pr_review.limits.source_tool_timeout_seconds,
                    ),
                    "iteration": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "token_accounting_complete": True,
                    "tool_calls": 0,
                    "retries": 0,
                    "reviewer_results": [],
                },
            )
            for job in jobs
        ]

    async def begin_deliberation(state: GraphState) -> dict[str, Any]:
        await deps.check_cancelled(state["operation_id"])
        for disposition in state["request"]["models"]:
            successes = [
                result
                for result in state["reviewer_results"]
                if result["disposition"] == disposition and not result.get("failed")
            ]
            if not successes:
                raise ReviewError(f"all {disposition} reviewers failed")
        await deps.store.update(state["operation_id"], status=OperationStatus.DELIBERATING)
        return {
            "timing": {
                **state["timing"],
                "reviewMs": int(time.time() * 1000) - state["started_ms"] - state["timing"]["setupMs"],
            }
        }

    def dispatch_deliberators(state: GraphState) -> list[Send]:
        return [
            Send(
                "deliberator",
                {
                    **state,
                    "deliberation_target": disposition,
                },
            )
            for disposition in state["request"]["models"]
        ]

    async def deliberator_node(state: GraphState) -> dict[str, Any]:
        disposition = state["deliberation_target"]
        prior_findings = [
            finding for finding in state.get("baseline_findings", []) if disposition in finding.get("dispositions", [])
        ]
        prior_ids = {str(finding["id"]) for finding in prior_findings}
        findings = [
            finding
            for result in state["reviewer_results"]
            if result["disposition"] == disposition and not result.get("failed")
            for finding in result["findings"]
        ]
        assessments = [
            PriorFindingAssessment.model_validate(assessment).model_dump(mode="json")
            for result in state["reviewer_results"]
            if result["disposition"] == disposition and not result.get("failed")
            for assessment in result["prior_assessments"]
            if assessment.get("finding_id") in prior_ids
        ]
        model_id = state["request"]["deliberation_model"]
        usage_callback = ResponseUsageCallback()
        limits = deps.config.pr_review.limits
        failed_tool_calls = 0

        async def run_deliberation() -> Any:
            # Allocate a fresh local budget per attempt: ToolBudget.claim() only ever
            # increments, so reusing one instance across call_with_rate_limit_retry
            # attempts would leave a retry pre-charged and prematurely exhausted. The
            # operation-wide claim is durable (store-backed by operation_id), so the
            # cross-attempt operation limit is still enforced.
            nonlocal failed_tool_calls
            tool_budget = deps.tool_budget(
                state["operation_id"],
                deps.config.pr_review.limits.deliberator_tool_calls_per_disposition,
                state["request"]["operation_tool_limit"],
            )
            try:
                model_owner = deps.model_factory.create(
                    model_id=model_id,
                    max_output_tokens=state["request"]["model_output_tokens_per_call"],
                )
                async with model_owner as model:
                    return await deliberate(
                        model,
                        _review_sandbox_profile(
                            state["worktree_path"],
                            state["git_metadata_root"],
                            state.get("source_excluded_paths"),
                        ),
                        state["base_sha"],
                        state["head_sha"],
                        disposition=disposition,
                        findings=findings,
                        assessments=assessments,
                        prior_findings=prior_findings,
                        context=state["context"],
                        tool_budget=tool_budget,
                        source_tool_timeout_seconds=state["request"].get(
                            "source_tool_timeout_seconds",
                            deps.config.pr_review.limits.source_tool_timeout_seconds,
                        ),
                        usage_callback=usage_callback,
                        langfuse_callback=deps.telemetry.langchain_callback(),
                    )
            except Exception:
                # A retried attempt's tool calls are still charged against the durable
                # operation budget, so fold them into the successful attempt's usage
                # rather than under-reporting the deliberator's true tool spend.
                failed_tool_calls += tool_budget.used
                raise

        with deps.telemetry.span(
            "pr_review.deliberator",
            observation_type="span",
            metadata={"role": "deliberator", "disposition": disposition},
        ) as observation:
            candidate, usage = await call_with_rate_limit_retry(
                run_deliberation,
                max_attempts=limits.model_max_attempts,
                base_delay=limits.rate_limit_base_delay_seconds,
                max_delay=limits.rate_limit_max_delay_seconds,
                on_retry=lambda attempt, delay, exc: log.warning(
                    "review.deliberator_retry",
                    operation_id=state["operation_id"],
                    disposition=disposition,
                    model=model_id,
                    attempt=attempt,
                    delay_seconds=round(delay, 2),
                    rate_limited=is_rate_limit_error(exc),
                    error_type=type(exc).__name__,
                    status_code=getattr(exc, "status_code", None),
                ),
            )
            # Fold in tool calls charged by earlier transiently-failed attempts so the
            # deliberator's reported spend matches what the durable budget was billed.
            total_tool_calls = usage.tool_calls + failed_tool_calls
            observation.update(
                usage_details={
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                    "total": usage.total_tokens,
                },
                metadata={
                    "tool_calls": total_tool_calls,
                    "token_accounting_complete": usage.complete,
                    "decision_count": len(candidate.finding_decisions),
                },
            )
        finding_ids = {finding["id"] for finding in findings}
        decision_ids = [decision.finding_id for decision in candidate.finding_decisions]
        if set(decision_ids) != finding_ids or len(decision_ids) != len(set(decision_ids)):
            raise ReviewError(f"{disposition} deliberation did not decide the exact finding set")
        supplied_assessment_decisions = candidate.assessment_decisions
        candidate = candidate.model_copy(
            update={
                "assessment_decisions": reconcile_assessment_decisions(supplied_assessment_decisions, prior_findings)
            }
        )
        if (
            len(candidate.assessment_decisions) != len(supplied_assessment_decisions)
            or {decision.finding_id for decision in supplied_assessment_decisions} != prior_ids
        ):
            log.info(
                "review.assessment_decisions_reconciled",
                operation_id=state["operation_id"],
                disposition=disposition,
                supplied=len(supplied_assessment_decisions),
                required=len(prior_ids),
            )
        assessment_ids = [decision.finding_id for decision in candidate.assessment_decisions]
        if set(assessment_ids) != prior_ids or len(assessment_ids) != len(set(assessment_ids)):
            raise ReviewError(f"{disposition} deliberation did not assess the exact prior finding set")
        result = DeliberationResult(
            disposition=disposition,
            model=model_id,
            finding_decisions=candidate.finding_decisions,
            assessment_decisions=candidate.assessment_decisions,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            token_accounting_complete=usage.complete,
            tool_calls=total_tool_calls,
        )
        for decision in result.finding_decisions:
            if decision.decision == "remove_as_false_positive":
                log.info(
                    "review.finding_removed",
                    operation_id=state["operation_id"],
                    finding_id=decision.finding_id,
                    disposition=disposition,
                )
        return {"deliberation_results": [result.model_dump(mode="json")]}

    async def aggregate_node(state: GraphState) -> dict[str, Any]:
        began = int(time.time() * 1000)
        await deps.check_cancelled(state["operation_id"])
        await deps.store.update(state["operation_id"], status=OperationStatus.AGGREGATING)
        all_findings = {
            finding["id"]: finding
            for result in state["reviewer_results"]
            if not result.get("failed")
            for finding in result["findings"]
        }
        retained_ids = {
            decision["finding_id"]
            for result in state["deliberation_results"]
            for decision in result["finding_decisions"]
            if decision["decision"] == "keep"
        }
        retained = [all_findings[finding_id] for finding_id in retained_ids]
        raw_assessments = [
            decision for result in state["deliberation_results"] for decision in result["assessment_decisions"]
        ]
        assessments = _merge_assessments(raw_assessments)
        baseline = {finding["id"]: finding for finding in state.get("baseline_findings", [])}
        remaining_prior: list[dict[str, Any]] = []
        for assessment in assessments:
            if assessment["status"] not in {"still_present", "partially_resolved"}:
                continue
            old = baseline.get(assessment["finding_id"])
            if old is None:
                log.warning(
                    "review.unknown_baseline_assessment_skipped",
                    operation_id=state["operation_id"],
                    finding_id=assessment["finding_id"],
                )
                continue
            material = f"follow-up\0{assessment['finding_id']}\0{assessment['status']}"
            remaining_prior.append(
                Finding(
                    id=hashlib.sha256(material.encode()).hexdigest()[:20],
                    title=old["title"],
                    body=assessment["rationale"],
                    severity=old["severity"],
                    path=assessment.get("path") or old["path"],
                    line=assessment.get("line") or old["line"],
                    disposition=(old.get("dispositions") or ["quality"])[0],
                    model=(old.get("reviewer_models") or ["prior-review"])[0],
                    contributing_models=old.get("reviewer_models", []),
                    contributing_dispositions=old.get("dispositions", []),
                    evidence="Follow-up deliberation confirmed this prior finding remains actionable.",
                    context_refs=old.get("context_refs", []),
                ).model_dump(mode="json")
            )
        retained.extend(remaining_prior)
        model_id = state["request"]["aggregation_model"]
        limits = deps.config.pr_review.limits
        # Shared across outer retries so token spend from an attempt that re-raises
        # a transient error is preserved rather than discarded with the exception.
        aggregation_usage = Usage()

        async def run_aggregation() -> Any:
            model_owner = deps.model_factory.create(
                model_id=model_id,
                max_output_tokens=state["request"]["model_output_tokens_per_call"],
            )
            async with model_owner as model:
                return await aggregate(
                    model,
                    findings=retained,
                    prior_assessments=assessments,
                    baseline_findings=state.get("baseline_findings", []),
                    context_changes=state.get("context_changes", []),
                    context=state["context"],
                    validate_candidate=lambda value: _validate_aggregation(value, retained),
                    usage=aggregation_usage,
                    langfuse_callback=deps.telemetry.langchain_callback(),
                )

        with deps.telemetry.span(
            "pr_review.aggregator",
            observation_type="span",
            metadata={"role": "aggregator", "retained_finding_count": len(retained)},
        ) as observation:
            candidate, usage = await call_with_rate_limit_retry(
                run_aggregation,
                max_attempts=limits.model_max_attempts,
                base_delay=limits.rate_limit_base_delay_seconds,
                max_delay=limits.rate_limit_max_delay_seconds,
                on_retry=lambda attempt, delay, exc: log.warning(
                    "review.aggregator_retry",
                    operation_id=state["operation_id"],
                    model=model_id,
                    attempt=attempt,
                    delay_seconds=round(delay, 2),
                    rate_limited=is_rate_limit_error(exc),
                    error_type=type(exc).__name__,
                    status_code=getattr(exc, "status_code", None),
                ),
            )
            observation.update(
                usage_details={
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                    "total": usage.total_tokens,
                },
                metadata={"finding_count": len(candidate.findings), "token_accounting_complete": usage.complete},
            )
        normalized = _validate_aggregation(candidate, retained)
        timing = {
            **state["timing"],
            "deliberationMs": max(0, began - state["started_ms"] - sum(state["timing"].values())),
            "aggregationMs": int(time.time() * 1000) - began,
        }
        return {
            "retained_findings": retained,
            "prior_assessments": assessments,
            "aggregation": normalized.model_dump(mode="json"),
            "aggregation_usage": usage.model_dump(),
            "timing": timing,
        }

    async def preview_node(state: GraphState) -> dict[str, Any]:
        if state.get("source_mode") == ReviewSourceMode.LOCAL.value:
            ref = PrRef.model_validate(state["ref"])
            async with deps.lock.acquire(ref.repo_key):
                snapshot = await deps.git.validate_local_source(
                    Path(state["worktree_path"]),
                    deps.local_source_root,
                    base_sha=state["base_sha"],
                    head_sha=state["head_sha"],
                )
                _require_same_local_snapshot(state, snapshot)
        preview = _build_preview(state)
        await deps.store.update(
            state["operation_id"],
            status=OperationStatus.READY,
            preview=preview.model_dump(mode="json"),
        )
        return {"preview": preview.model_dump(mode="json")}

    async def await_commit(state: GraphState) -> dict[str, Any]:
        decision = interrupt(
            {
                "operationId": state["operation_id"],
                "revision": state["preview"]["revision"],
                "payloadHash": state["preview"]["payload_hash"],
            }
        )
        return {"decision": decision}

    def route_decision(state: GraphState) -> str:
        return "commit" if state["decision"].get("action") == "commit" else "cleanup"

    async def commit_node(state: GraphState) -> dict[str, Any]:
        decision = state["decision"]
        preview = ReviewPreview.model_validate(state["preview"])
        if decision.get("revision") != preview.revision or decision.get("payload_hash") != preview.payload_hash:
            raise ReviewError("commit revision or payload hash does not match the preview")
        await deps.store.update(state["operation_id"], status=OperationStatus.COMMITTING)
        ref = PrRef.model_validate(state["ref"])
        current_user = await deps.git.authenticated_user(ref.host)
        if current_user != state["owner_login"]:
            raise ReviewError("authenticated GitHub user changed after the review was prepared")
        source_mode = ReviewSourceMode(state.get("source_mode", ReviewSourceMode.MANAGED.value))
        local_source_error: ReviewError | None = None
        async with deps.lock.acquire(ref.repo_key):
            current_base, current_head = await deps.git.pr_shas(ref)
            if source_mode == ReviewSourceMode.LOCAL:
                try:
                    snapshot = await deps.git.validate_local_source(
                        Path(state["worktree_path"]),
                        deps.local_source_root,
                        base_sha=preview.base_sha,
                        head_sha=preview.head_sha,
                    )
                    _require_same_local_snapshot(state, snapshot)
                except ReviewError as exc:
                    local_source_error = exc
                fetched_head = current_head
            else:
                fetched_head = await deps.git.fetch_head(Path(state["repo_path"]), ref.number)
        if (
            local_source_error is not None
            or fetched_head != current_head
            or current_head != preview.head_sha
            or current_base != preview.base_sha
        ):
            reason = "Pull request base or head changed after preview; no review was published."
            next_action = "Start a new PR review for the current base and head."
            if local_source_error is not None and (current_base, current_head) == (preview.base_sha, preview.head_sha):
                reason = "Local source changed or no longer matches the reviewed revision; no review was published."
                next_action = "Restore a clean local checkout at the pull request head and start a new review."
            stale_result = {
                "operation_id": state["operation_id"],
                "status": OperationStatus.STALE.value,
                "reason": reason,
                "preview_head_sha": preview.head_sha,
                "current_head_sha": current_head,
                "preview_base_sha": preview.base_sha,
                "current_base_sha": current_base,
                "next_action": next_action,
            }
            await deps.store.update(
                state["operation_id"],
                status=OperationStatus.STALE,
                result=stale_result,
            )
            log.info(
                "review.commit_stale",
                operation_id=state["operation_id"],
                preview_head_sha=preview.head_sha,
                current_head_sha=current_head,
                preview_base_sha=preview.base_sha,
                current_base_sha=current_base,
            )
            return {"final_status": OperationStatus.STALE.value}
        marker = f'"operationId": "{state["operation_id"]}"'
        if not await deps.git.review_already_posted(ref, marker, state["owner_login"]):
            await deps.git.post_comment_review(
                ref,
                head_sha=preview.head_sha,
                summary_body=preview.summary_body,
                comments=preview.comments,
            )
        result = {
            "operation_id": state["operation_id"],
            "status": OperationStatus.COMPLETED.value,
            "repo": ref.repo_key,
            "pr_number": ref.number,
            "base_sha": preview.base_sha,
            "head_sha": preview.head_sha,
            "mode": state["mode"],
            "baseline_operation_id": state.get("baseline_operation_id"),
            "context": state["context"],
            "findings": state["aggregation"]["findings"],
            "prior_assessments": state.get("prior_assessments", []),
            "summary": state["aggregation"]["summary"],
            "metrics": preview.metrics.model_dump(mode="json"),
        }
        await deps.store.update(state["operation_id"], status=OperationStatus.COMPLETED, result=result)
        return {"final_status": OperationStatus.COMPLETED.value}

    async def cleanup(state: GraphState) -> dict[str, Any]:
        if (
            state.get("source_mode", ReviewSourceMode.MANAGED.value) == ReviewSourceMode.MANAGED.value
            and state.get("repo_path")
            and state.get("worktree_path")
        ):
            ref = PrRef.model_validate(state["ref"])
            async with deps.lock.acquire(ref.repo_key):
                try:
                    await deps.git.remove_worktree(Path(state["repo_path"]), Path(state["worktree_path"]))
                finally:
                    await deps.git.release_comparison(Path(state["repo_path"]), state["operation_id"])
        if state.get("decision", {}).get("action") == "cancel":
            await deps.store.update(state["operation_id"], status=OperationStatus.CANCELLED)
            return {"final_status": OperationStatus.CANCELLED.value}
        return {}

    builder.add_node("initialize", initialize)
    builder.add_node("prepare", prepare)
    builder.add_node("dispatch_reviewers", lambda state: {})
    builder.add_node("reviewer", reviewer_graph)
    builder.add_node("begin_deliberation", begin_deliberation)
    builder.add_node("dispatch_deliberators", lambda state: {})
    builder.add_node("deliberator", deliberator_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("preview", preview_node)
    builder.add_node("await_commit", await_commit)
    builder.add_node("commit", commit_node)
    builder.add_node("cleanup", cleanup)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "prepare")
    builder.add_edge("prepare", "dispatch_reviewers")
    builder.add_conditional_edges("dispatch_reviewers", dispatch_reviewers, ["reviewer"])
    builder.add_edge("reviewer", "begin_deliberation")
    builder.add_edge("begin_deliberation", "dispatch_deliberators")
    builder.add_conditional_edges("dispatch_deliberators", dispatch_deliberators, ["deliberator"])
    builder.add_edge("deliberator", "aggregate")
    builder.add_edge("aggregate", "preview")
    builder.add_edge("preview", "await_commit")
    builder.add_conditional_edges("await_commit", route_decision, {"commit": "commit", "cleanup": "cleanup"})
    builder.add_edge("commit", "cleanup")
    builder.add_edge("cleanup", END)
    return builder.compile(checkpointer=checkpointer)


def _validate_aggregation(candidate: AggregationCandidate, retained: list[dict[str, Any]]) -> AggregationCandidate:
    source = {finding["id"]: Finding.model_validate(finding) for finding in retained}
    used = [source_id for finding in candidate.findings for source_id in finding.source_finding_ids]
    if set(used) != set(source) or len(used) != len(set(used)):
        missing = sorted(set(source) - set(used))
        unknown = sorted(set(used) - set(source))
        duplicated = sorted(source_id for source_id in set(used) if used.count(source_id) > 1)
        raise ReviewError(
            "aggregation source finding coverage mismatch "
            f"(missing={missing}, unknown={unknown}, duplicated={duplicated})"
        )
    normalized: list[AggregatedFinding] = []
    for finding in candidate.findings:
        members = [source[source_id] for source_id in finding.source_finding_ids]
        if (finding.path, finding.line) not in {(member.path, member.line) for member in members}:
            raise ReviewError("aggregated finding location must come from a source finding")
        finding.id = hashlib.sha256("\0".join(sorted(finding.source_finding_ids)).encode()).hexdigest()[:20]
        finding.severity = max((member.severity for member in members), key=_SEVERITY_RANK.__getitem__)
        finding.dispositions = unique(
            disposition
            for member in members
            for disposition in (member.contributing_dispositions or [member.disposition])
        )
        finding.reviewer_models = unique(
            model for member in members for model in (member.contributing_models or [member.model])
        )
        finding.context_refs = unique(ref for member in members for ref in member.context_refs)
        normalized.append(finding)
    return AggregationCandidate(summary=candidate.summary, findings=normalized)


def _merge_assessments(assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {"resolved": 0, "cannot_verify": 1, "partially_resolved": 2, "still_present": 3}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for assessment in assessments:
        grouped.setdefault(assessment["finding_id"], []).append(assessment)
    merged: list[dict[str, Any]] = []
    for finding_id, values in grouped.items():
        selected = max(values, key=lambda value: rank[value["status"]])
        merged.append(
            {
                **selected,
                "finding_id": finding_id,
                "rationale": "\n\n".join(unique(value["rationale"] for value in values)),
            }
        )
    return merged


def _context_changes(baseline: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[object, object]:
        return item.get("label"), item.get("source")

    old = {key(item): item for item in baseline}
    new = {key(item): item for item in current}
    changes: list[dict[str, Any]] = []
    for item_key in sorted(set(old) | set(new), key=str):
        if item_key not in old:
            changes.append({"label": item_key[0], "source": item_key[1], "change": "added"})
        elif item_key not in new:
            changes.append({"label": item_key[0], "source": item_key[1], "change": "removed"})
        elif old[item_key].get("sha256") != new[item_key].get("sha256"):
            changes.append({"label": item_key[0], "source": item_key[1], "change": "changed"})
    return changes


def _build_preview(state: GraphState) -> ReviewPreview:
    aggregation = AggregationCandidate.model_validate(state["aggregation"])
    request = state["request"]
    reviewers = [ReviewerResult.model_validate(result) for result in state["reviewer_results"]]
    deliberations = [DeliberationResult.model_validate(result) for result in state["deliberation_results"]]
    findings_by_model: dict[str, dict[str, int]] = {}
    for reviewer in reviewers:
        findings_by_model.setdefault(reviewer.disposition, {})[reviewer.model] = len(reviewer.findings)
    total_input = sum(result.input_tokens for result in reviewers) + sum(
        result.input_tokens for result in deliberations
    )
    total_output = sum(result.output_tokens for result in reviewers) + sum(
        result.output_tokens for result in deliberations
    )
    total_tokens = sum(result.total_tokens for result in reviewers) + sum(
        result.total_tokens for result in deliberations
    )
    aggregation_usage = state["aggregation_usage"]
    total_input += aggregation_usage["input_tokens"]
    total_output += aggregation_usage["output_tokens"]
    total_tokens += aggregation_usage["total_tokens"]
    by_model: dict[str, int] = {}
    by_disposition: dict[str, int] = {}
    for reviewer in reviewers:
        by_model[reviewer.model] = by_model.get(reviewer.model, 0) + reviewer.total_tokens
        by_disposition[reviewer.disposition] = by_disposition.get(reviewer.disposition, 0) + reviewer.total_tokens
    for deliberation in deliberations:
        by_model[deliberation.model] = by_model.get(deliberation.model, 0) + deliberation.total_tokens
    aggregation_model = request["aggregation_model"]
    by_model[aggregation_model] = by_model.get(aggregation_model, 0) + aggregation_usage["total_tokens"]
    removed = sum(
        decision["decision"] == "remove_as_false_positive"
        for result in state["deliberation_results"]
        for decision in result["finding_decisions"]
    )
    commentable = commentable_lines(state["diff"])
    inline = [finding for finding in aggregation.findings if finding.line in commentable.get(finding.path, set())]
    summary_only = [finding for finding in aggregation.findings if finding not in inline]
    follow_counts = None
    if state["mode"] == ReviewMode.FOLLOW_UP.value:
        follow_counts = {
            status: sum(item["status"] == status for item in state.get("prior_assessments", []))
            for status in ("resolved", "still_present", "partially_resolved", "cannot_verify")
        }
        follow_counts["new"] = len(aggregation.findings)
    metrics = ReviewMetrics(
        models=request["models"],
        findings=findings_by_model,
        overlapping_findings=sum(len(finding.reviewer_models) > 1 for finding in aggregation.findings),
        duration_ms=sum(state["timing"].values()),
        total_tokens=total_tokens,
        tokens={
            "input": total_input,
            "output": total_output,
            "accountingComplete": all(result.token_accounting_complete for result in reviewers)
            and all(result.token_accounting_complete for result in deliberations)
            and aggregation_usage["complete"],
            "byPhase": {
                "review": sum(result.total_tokens for result in reviewers),
                "deliberation": sum(result.total_tokens for result in deliberations),
                "aggregation": aggregation_usage["total_tokens"],
            },
            "byModel": by_model,
            "byDisposition": by_disposition,
        },
        timing=state["timing"],
        reviewers={
            "requested": len(reviewers),
            "succeeded": sum(not result.failed for result in reviewers),
            "failed": sum(result.failed for result in reviewers),
            "iterationsConfigured": request["iterations"],
            "iterationsCompleted": sum(result.iterations_completed for result in reviewers),
            "toolCalls": sum(result.tool_calls for result in reviewers)
            + sum(result.tool_calls for result in deliberations),
            "reviewerToolCalls": sum(result.tool_calls for result in reviewers),
            "deliberatorToolCalls": sum(result.tool_calls for result in deliberations),
            "retries": sum(result.retries for result in reviewers),
        },
        results={
            "rawFindings": sum(len(result.findings) for result in reviewers),
            "falsePositivesRemoved": removed,
            "retainedFindings": len(state["retained_findings"]),
            "aggregatedFindings": len(aggregation.findings),
            "inlineComments": len(inline),
            "summaryOnlyFindings": len(summary_only),
        },
        follow_up=follow_counts,
        context={"items": len(state["context"]), "characters": sum(len(item["content"]) for item in state["context"])},
    )
    revision = 1
    comments: list[PreviewComment] = []
    for finding in inline:
        body = neutralize_github_markup(finding.body)
        body += render_comment_footer(
            reviewer_models=finding.reviewer_models,
            deliberation_model=request["deliberation_model"],
            aggregation_model=request["aggregation_model"],
            severity=finding.severity,
        )
        body += render_comment_metadata(
            finding,
            operation_id=state["operation_id"],
            revision=revision,
            deliberation_model=request["deliberation_model"],
            aggregation_model=request["aggregation_model"],
        )
        comments.append(
            PreviewComment(
                finding_id=finding.id,
                path=finding.path,
                line=finding.line,
                severity=finding.severity,
                body=body,
                reviewer_models=finding.reviewer_models,
                dispositions=finding.dispositions,
                source_finding_ids=finding.source_finding_ids,
            )
        )
    extra = render_additional_findings(summary_only)
    reviewer_failures = [
        ReviewerFailure(
            disposition=result.disposition,
            model=result.model,
            error=result.error or "unknown reviewer failure",
            retries=result.retries,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            tool_calls=result.tool_calls,
        )
        for result in reviewers
        if result.failed
    ]
    coverage_warning = ""
    if reviewer_failures:
        failed_jobs = ", ".join(f"{item.disposition}/{item.model}" for item in reviewer_failures)
        coverage_warning = (
            "\n\n> ⚠️ **Review coverage degraded:** the following reviewer jobs failed after retries: "
            f"{failed_jobs}. Findings do not include their independent analysis."
        )
    summary = neutralize_github_markup(aggregation.summary + extra + coverage_warning)
    summary_body = summary + render_stats_comment(metrics)
    summary_body += (
        f'\n\n<!-- pr-council-mcp-operation\n{{"operationId": "{state["operation_id"]}", "revision": {revision}}}\n-->'
    )
    summary_body += render_summary_footer(
        request["models"],
        deliberation_model=request["deliberation_model"],
        aggregation_model=request["aggregation_model"],
    )
    hash_input = {
        "base_sha": state["base_sha"],
        "head_sha": state["head_sha"],
        "summary_body": summary_body,
        "comments": [comment.model_dump(mode="json") for comment in comments],
        "context": state["context"],
    }
    return ReviewPreview(
        revision=revision,
        payload_hash=payload_hash(hash_input),
        base_sha=state["base_sha"],
        head_sha=state["head_sha"],
        summary=summary,
        summary_body=summary_body,
        comments=comments,
        context_manifest=[
            {
                "id": item["id"],
                "label": item["label"],
                "source": item.get("source"),
                "sha256": item["sha256"],
                "characters": len(item["content"]),
            }
            for item in state["context"]
        ],
        metrics=metrics,
        reviewer_failures=reviewer_failures,
        mode=ReviewMode(state["mode"]),
        baseline_operation_id=state.get("baseline_operation_id"),
    )
