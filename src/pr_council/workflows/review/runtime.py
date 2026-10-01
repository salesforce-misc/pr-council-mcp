"""Lifecycle service that runs and resumes checkpointed PR-review graphs."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command
from localmcp.llm import ModelConfigurationError, ModelFactory
from localmcp.observability.logging import get_logger
from localmcp.observability.telemetry import LangfuseTelemetry
from localmcp.server import RuntimeUnavailableError, current_runtime
from localmcp.workflows.runtime import WorkflowRuntime

from pr_council.config import Config
from pr_council.review.git import GitHubCli, parse_pr_url
from pr_council.review.locks import RepoLock
from pr_council.review.models import (
    TERMINAL_STATUSES,
    OperationStatus,
    ReviewContextInput,
    ReviewContextItem,
    ReviewError,
    ReviewMode,
    ReviewPreview,
    ReviewSourceMode,
)
from pr_council.review.store import OperationRecord, OperationStore
from pr_council.workflows.review.graph import ReviewCancelled, ReviewGraphDeps, build_review_graph

log = get_logger(__name__)
_MODEL_ID = re.compile(r"^[A-Za-z0-9._:/-]+$")
_DISPOSITIONS = ("quality", "security")


class ReviewRuntime(WorkflowRuntime[OperationRecord, OperationStore]):
    runtime_name = "review"
    state_subdirectory = "review"

    def __init__(
        self,
        config: Config,
        state_root: Path,
        model_factory: ModelFactory,
        *,
        telemetry: LangfuseTelemetry | None = None,
        local_source_root: Path | None = None,
    ):
        telemetry_service = telemetry or LangfuseTelemetry({})
        super().__init__(state_root, telemetry=telemetry_service, trace_namespace="pr-council-mcp")
        self.review_telemetry = telemetry_service
        self.config = config
        self.model_factory = model_factory
        try:
            self.local_source_root = (local_source_root or Path.cwd()).resolve(strict=True)
        except OSError as exc:
            raise ReviewError("could not resolve the local-source sandbox root") from exc
        if not self.local_source_root.is_dir():
            raise ReviewError("local-source sandbox root is not a directory")
        self.git = GitHubCli(
            max_repository_bytes=config.pr_review.limits.repository_size_mib * 1024 * 1024,
            allowed_hosts=config.pr_review.allowed_hosts,
            accounts=config.pr_review.github_accounts,
        )
        self.lock = RepoLock(self.root / "locks")

    async def _open_store(self) -> OperationStore:
        return await OperationStore.open(self.root / self.operations_database_name)

    def _build_graph(self, checkpointer: Any) -> Any:
        return build_review_graph(
            ReviewGraphDeps(
                config=self.config,
                model_factory=self.model_factory,
                store=self._require_store(),
                git=self.git,
                lock=self.lock,
                repos_dir=self.root / "repos",
                worktrees_dir=self.root / "worktrees",
                local_source_root=self.local_source_root,
                telemetry=self.review_telemetry,
            ),
            checkpointer,
        )

    async def _recover(self) -> None:
        assert self.store is not None
        for record in await self.store.recoverable():
            if record.status == OperationStatus.CANCEL_REQUESTED:
                await self._cleanup(record)
                await self.store.update(record.id, status=OperationStatus.CANCELLED)
                continue
            ref_data = record.request.get("ref")
            host = ref_data.get("host") if isinstance(ref_data, dict) else None
            if host not in self.git.allowed_hosts:
                await self._cleanup(record)
                await self.store.update(
                    record.id,
                    status=OperationStatus.FAILED,
                    state={key: value for key, value in record.state.items() if key != "pending_commit"},
                    error="PR host is no longer allowed by server configuration",
                )
                continue
            if record.request.get("source_mode") == ReviewSourceMode.LOCAL.value:
                try:
                    self._resolve_local_source_path(record.request.get("local_source_path"))
                except ReviewError as exc:
                    await self.store.update(record.id, status=OperationStatus.FAILED, error=str(exc))
                    continue
            if record.status == OperationStatus.COMMIT_QUEUED and record.state.get("pending_commit"):
                self._spawn(record.id, command=Command(resume=record.state["pending_commit"]))
            elif record.status != OperationStatus.READY:
                self._spawn(record.id, resume=True)

    def _spawn(self, operation_id: str, *, resume: bool = False, command: Command[Any] | None = None) -> None:
        self._track_task(operation_id, lambda: self._run(operation_id, resume=resume, command=command))

    async def _run(self, operation_id: str, *, resume: bool, command: Command[Any] | None) -> None:
        async def execute(record: OperationRecord) -> None:
            store = self._require_store()
            config = {
                "configurable": {"thread_id": operation_id},
                "recursion_limit": 200,
                "max_concurrency": int(
                    record.request.get("concurrent_model_calls", self.config.pr_review.limits.concurrent_model_calls)
                ),
            }
            try:
                if command is not None:
                    await self.graph.ainvoke(command, config=config)
                elif resume:
                    snapshot = await self.graph.aget_state(config)
                    if snapshot.values:
                        await self.graph.ainvoke(None, config=config)
                    else:
                        await self.graph.ainvoke(self._initial_state(record), config=config)
                else:
                    await self.graph.ainvoke(self._initial_state(record), config=config)
            except ReviewCancelled:
                await self._cleanup(record)
                await store.update(operation_id, status=OperationStatus.CANCELLED)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("review.operation_failed", operation_id=operation_id, error_type=type(exc).__name__)
                await self._cleanup(await store.get(operation_id) or record)
                await store.update(operation_id, status=OperationStatus.FAILED, error=str(exc))

        await self._run_leased(operation_id, execute)

    @staticmethod
    def _initial_state(record: OperationRecord) -> dict[str, Any]:
        return {
            "operation_id": record.id,
            "request": record.request,
            "ref": record.request["ref"],
            "reviewer_results": [],
            "deliberation_results": [],
        }

    async def _cleanup(self, record: OperationRecord) -> None:
        if record.request.get("source_mode", ReviewSourceMode.MANAGED.value) != ReviewSourceMode.MANAGED.value:
            return
        repo_path = record.state.get("repo_path")
        worktree_path = record.state.get("worktree_path")
        if not repo_path or not worktree_path:
            return
        try:
            async with self.lock.acquire(record.repo_key):
                try:
                    await self.git.remove_worktree(Path(repo_path), Path(worktree_path))
                finally:
                    await self.git.release_comparison(Path(repo_path), record.id)
        except Exception:
            log.warning("review.cleanup_failed", operation_id=record.id)

    async def start_operation(
        self,
        *,
        pr_url: str,
        source_mode: ReviewSourceMode = ReviewSourceMode.MANAGED,
        local_source_path: str | None = None,
        context: list[ReviewContextInput] | None,
        mode: ReviewMode,
        baseline_operation_id: str | None,
        iterations: int,
        models: dict[str, list[str]] | None = None,
        deliberation_model: str | None = None,
        aggregation_model: str | None = None,
    ) -> OperationRecord:
        store = self._require_store()
        ref = parse_pr_url(pr_url, allowed_hosts=self.config.pr_review.allowed_hosts)
        try:
            source_mode = ReviewSourceMode(source_mode)
        except ValueError as exc:
            raise ReviewError("source_mode must be managed or local") from exc
        if source_mode == ReviewSourceMode.LOCAL:
            resolved_local_source = self._resolve_local_source_path(local_source_path)
        else:
            if local_source_path is not None:
                raise ReviewError("local_source_path is valid only when source_mode is local")
            resolved_local_source = None
        owner_uid = os.getuid()
        if iterations < 1 or iterations > 10:
            raise ReviewError("review iterations must be between 1 and 10")
        configured_models = self.config.pr_review.models
        resolved_models = (
            {
                "quality": list(configured_models.quality),
                "security": list(configured_models.security),
            }
            if models is None
            else {disposition: list(values) for disposition, values in models.items()}
        )
        if set(resolved_models) != set(_DISPOSITIONS) or any(not values for values in resolved_models.values()):
            raise ReviewError("models must provide non-empty quality and security lists")
        resolved_models = {disposition: list(dict.fromkeys(values)) for disposition, values in resolved_models.items()}
        if any(
            len(values) > self.config.pr_review.limits.models_per_disposition for values in resolved_models.values()
        ):
            raise ReviewError("reviewer model count exceeds the configured per-disposition safety limit")
        resolved_deliberation_model = (
            configured_models.deliberation if deliberation_model is None else deliberation_model
        )
        resolved_aggregation_model = configured_models.aggregation if aggregation_model is None else aggregation_model
        configured_model_ids = [model for values in resolved_models.values() for model in values]
        configured_model_ids.extend((resolved_deliberation_model, resolved_aggregation_model))
        if any(not _MODEL_ID.fullmatch(model) for model in configured_model_ids):
            raise ReviewError("model IDs may contain only letters, numbers, dot, underscore, colon, slash, and dash")
        for model_id in configured_model_ids:
            try:
                self.model_factory.validate(model_id)
            except ModelConfigurationError as exc:
                raise ReviewError(str(exc)) from exc
        github_account = self.git.account_for(ref)
        if github_account is not None:
            # Fail before any review work when the mapped account cannot act on this host.
            login = await self.git.bind(ref.host, github_account).authenticated_user(ref.host)
            if login.lower() != github_account.lower():
                raise ReviewError(
                    f'the token for GitHub account "{github_account}" authenticated as "{login}" on {ref.host}'
                )
        baseline = None
        if baseline_operation_id:
            if mode == ReviewMode.INITIAL:
                raise ReviewError("an initial review cannot specify a baseline operation")
            baseline = await store.require_owned(baseline_operation_id, owner_uid)
            if baseline.repo_key != ref.repo_key or baseline.pr_number != ref.number:
                raise ReviewError("baseline operation belongs to a different pull request")
            if baseline.status != OperationStatus.COMPLETED or baseline.result is None:
                raise ReviewError("baseline operation must be completed")
        elif mode != ReviewMode.INITIAL:
            baseline = await store.latest_completed(ref.repo_key, ref.number, owner_uid)
        if mode == ReviewMode.FOLLOW_UP and baseline is None:
            raise ReviewError("follow-up review requires a completed baseline operation")
        normalized_context = None if context is None else _normalize_context(context)
        inherited_context = (baseline.result or {}).get("context", []) if baseline and baseline.result else []
        operation_id = str(uuid.uuid4())
        request = {
            "owner_uid": owner_uid,
            "ref": ref.model_dump(mode="json"),
            "github_account": github_account,
            "source_mode": source_mode.value,
            "local_source_path": os.fspath(resolved_local_source) if resolved_local_source is not None else None,
            "mode": mode.value,
            "baseline_operation_id": baseline.id if baseline else None,
            "context": (
                [item.model_dump(mode="json") for item in normalized_context]
                if normalized_context is not None
                else inherited_context
            ),
            "models": resolved_models,
            "iterations": iterations,
            "operation_tool_limit": self.config.pr_review.limits.operation_tool_calls,
            "model_output_tokens_per_call": self.config.pr_review.limits.model_output_tokens_per_call,
            "concurrent_model_calls": self.config.pr_review.limits.concurrent_model_calls,
            "source_tool_timeout_seconds": self.config.pr_review.limits.source_tool_timeout_seconds,
            "deliberation_model": resolved_deliberation_model,
            "aggregation_model": resolved_aggregation_model,
        }
        record = await store.create(
            operation_id=operation_id,
            owner_uid=owner_uid,
            repo_key=ref.repo_key,
            pr_number=ref.number,
            label=f"PR #{ref.number} in {ref.owner}/{ref.repo}",
            request=request,
            parent_operation_id=baseline.id if baseline else None,
        )
        self._spawn(operation_id)
        return record

    def _resolve_local_source_path(self, value: object) -> Path:
        if value is not None and not isinstance(value, str):
            raise ReviewError("local_source_path must be a string path")
        candidate = Path(value or ".")
        if not candidate.is_absolute():
            candidate = self.local_source_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ReviewError("local source path does not exist or cannot be resolved") from exc
        if not resolved.is_dir():
            raise ReviewError("local source path is not a directory")
        if not resolved.is_relative_to(self.local_source_root):
            raise ReviewError("local source path is outside the server working-directory sandbox")
        return resolved

    async def get_owned(self, operation_id: str) -> OperationRecord:
        return await self._require_store().require_owned(operation_id, os.getuid())

    async def preview(self, operation_id: str) -> ReviewPreview:
        record = await self.get_owned(operation_id)
        if record.preview is None:
            raise ReviewError(f"review preview is unavailable; current status is {record.status.value}")
        return ReviewPreview.model_validate(record.preview)

    async def commit(self, operation_id: str, revision: int, payload_hash: str) -> OperationRecord:
        store = self._require_store()
        record = await self.get_owned(operation_id)
        if record.status == OperationStatus.COMPLETED:
            return record
        if not record.commit_ready:
            raise ReviewError(f"review cannot be committed from status {record.status.value}")
        assert record.preview is not None
        preview = ReviewPreview.model_validate(record.preview)
        if preview.revision != revision or preview.payload_hash != payload_hash:
            raise ReviewError("revision or payload hash does not match the current preview")
        pending = {"action": "commit", "revision": revision, "payload_hash": payload_hash}
        await store.update(
            operation_id,
            status=OperationStatus.COMMIT_QUEUED,
            state={**record.state, "pending_commit": pending},
            clear_error=True,
        )
        self._spawn(operation_id, command=Command(resume=pending))
        updated = await store.get(operation_id)
        assert updated is not None
        return updated

    async def cancel(self, operation_id: str) -> OperationRecord:
        store = self._require_store()
        record = await self.get_owned(operation_id)
        if record.status in TERMINAL_STATUSES:
            return record
        if record.status in {OperationStatus.COMMIT_QUEUED, OperationStatus.COMMITTING}:
            raise ReviewError("publication has begun; cancellation is too late")
        await store.request_cancel(operation_id)
        if record.status == OperationStatus.READY:
            self._spawn(operation_id, command=Command(resume={"action": "cancel"}))
        updated = await store.get(operation_id)
        assert updated is not None
        return updated


def review_runtime() -> ReviewRuntime:
    """Return the lifespan-owned review runtime for model-facing tools."""
    try:
        return current_runtime(ReviewRuntime)
    except RuntimeUnavailableError as exc:
        raise ReviewError("review runtime is not initialized") from exc


def _normalize_context(items: list[ReviewContextInput]) -> list[ReviewContextItem]:
    if len(items) > 20:
        raise ReviewError("review context supports at most 20 items")
    total = sum(len(item.content) for item in items)
    if total > 64_000:
        raise ReviewError("review context content exceeds the 64,000-character limit")
    result: list[ReviewContextItem] = []
    for item in items:
        digest = hashlib.sha256(f"{item.label}\0{item.source or ''}\0{item.content}".encode()).hexdigest()
        result.append(ReviewContextItem(id=f"context-{digest[:12]}", sha256=digest, **item.model_dump()))
    return result
