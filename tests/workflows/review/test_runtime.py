"""Behavioral tests for :class:`ReviewRuntime`, migrated from the upstream suite.

Faithful port of ``vaas-harness``'s ``tests/workflows/review/test_runtime.py``
modulo the sanctioned adaptations:

* ``vaas_harness.*`` -> ``pr_council.*`` imports; ``HarnessConfig`` ->
  :class:`~pr_council.config.Config` plus an inert model factory.
* Our ``Config`` (and every sub-config) is ``frozen=True``, so the roster/limit
  variations upstream expressed by mutating the config after construction are
  built immutably up-front via the ``_config`` factory. The dedup-then-cap case
  therefore constructs two runtimes -- one per frozen config -- to preserve both
  behavioral assertions.
* ``ReviewRuntime`` now takes an explicit ``state_root`` and ``secrets`` instead
  of carrying ``paths.state_dir`` on the config; ``_spawn`` is monkeypatched to a
  no-op (as upstream) so ``start_operation`` never launches a real graph.

Every runtime-lifecycle / operation await is wrapped in
``asyncio.wait_for(timeout=5.0)`` so a lease or lifecycle bug fails fast instead
of hanging the suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from localmcp.llm import DEFAULT_MODEL_REGISTRY, ModelConfigurationError

from pr_council.config import Config, PRReviewConfig, PRReviewLimitsConfig, PRReviewModelsConfig
from pr_council.review.models import (
    OperationStatus,
    ReviewContextInput,
    ReviewError,
    ReviewMode,
    ReviewSourceMode,
)
from pr_council.workflows.review.runtime import ReviewRuntime

_TIMEOUT = 5.0


class FakeModelFactory:
    def validate(self, model_id, *, reasoning_effort=None):
        del reasoning_effort
        return DEFAULT_MODEL_REGISTRY.resolve(model_id)

    def create(self, model_id, *, max_output_tokens=None, reasoning_effort=None):
        raise AssertionError("model construction is not expected in runtime coordination tests")


def _config(
    *,
    quality: list[str] | None = None,
    security: list[str] | None = None,
    deliberation: str | None = None,
    aggregation: str | None = None,
    models_per_disposition: int | None = None,
    concurrent_model_calls: int | None = None,
) -> Config:
    """Build a frozen :class:`Config`, overriding only the supplied fields.

    Our config is immutable, so tests that need a distinct reviewer roster or
    limit express it here at construction time rather than mutating after the
    fact (which would raise on a frozen model).
    """
    models_kwargs: dict[str, object] = {}
    if quality is not None:
        models_kwargs["quality"] = quality
    if security is not None:
        models_kwargs["security"] = security
    if deliberation is not None:
        models_kwargs["deliberation"] = deliberation
    if aggregation is not None:
        models_kwargs["aggregation"] = aggregation
    limits_kwargs: dict[str, object] = {}
    if models_per_disposition is not None:
        limits_kwargs["models_per_disposition"] = models_per_disposition
    if concurrent_model_calls is not None:
        limits_kwargs["concurrent_model_calls"] = concurrent_model_calls
    return Config(
        pr_review=PRReviewConfig(
            models=PRReviewModelsConfig(**models_kwargs),
            limits=PRReviewLimitsConfig(**limits_kwargs),
        ),
    )


def _runtime(
    config: Config,
    state_root: Path,
    *,
    local_source_root: Path | None = None,
    model_factory: FakeModelFactory | None = None,
) -> ReviewRuntime:
    return ReviewRuntime(
        config,
        state_root=state_root,
        model_factory=model_factory or FakeModelFactory(),
        local_source_root=local_source_root,
    )


def _preview(head_sha: str = "head") -> dict[str, object]:
    return {
        "revision": 1,
        "payload_hash": "payload",
        "base_sha": "base",
        "head_sha": head_sha,
        "summary": "summary",
        "summary_body": "summary",
        "comments": [],
        "context_manifest": [],
        "metrics": {
            "models": {},
            "findings": {},
            "overlapping_findings": 0,
            "duration_ms": 0,
            "total_tokens": 0,
            "tokens": {},
            "timing": {},
            "reviewers": {},
            "results": {},
            "context": {},
        },
        "mode": "initial",
    }


async def test_start_snapshots_arbitrary_context_and_auto_followup_inherits_it(monkeypatch, tmp_path):
    runtime = _runtime(_config(), tmp_path / "state")
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    try:
        first = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[ReviewContextInput(label="GUS context", source="user", content="Acceptance criterion")],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=3,
            ),
            timeout=_TIMEOUT,
        )
        assert first.request["context"][0]["id"].startswith("context-")
        assert first.request["context"][0]["sha256"]
        assert first.request["operation_tool_limit"] == 300
        assert first.request["model_output_tokens_per_call"] == 8_000
        assert first.request["concurrent_model_calls"] == 4
        assert first.request["source_tool_timeout_seconds"] == 30
        assert runtime.store is not None
        await asyncio.wait_for(
            runtime.store.update(
                first.id,
                status=OperationStatus.COMPLETED,
                result={"context": first.request["context"], "findings": []},
            ),
            timeout=_TIMEOUT,
        )

        follow_up = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=None,
                mode=ReviewMode.AUTO,
                baseline_operation_id=None,
                iterations=3,
            ),
            timeout=_TIMEOUT,
        )
        assert follow_up.parent_operation_id == first.id
        assert follow_up.request["context"] == first.request["context"]
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_start_snapshots_local_source_beneath_working_directory(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    repository = workspace / "repo"
    repository.mkdir(parents=True)
    runtime = _runtime(_config(), tmp_path / "state", local_source_root=workspace)
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    try:
        record = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                source_mode=ReviewSourceMode.LOCAL,
                local_source_path="repo",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            ),
            timeout=_TIMEOUT,
        )

        assert record.request["source_mode"] == "local"
        assert record.request["local_source_path"] == str(repository.resolve())

        cwd_record = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/8",
                source_mode=ReviewSourceMode.LOCAL,
                local_source_path=None,
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            ),
            timeout=_TIMEOUT,
        )
        assert cwd_record.request["local_source_path"] == str(workspace.resolve())
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_start_rejects_local_source_outside_working_directory_or_on_managed_mode(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime = _runtime(_config(), tmp_path / "state", local_source_root=workspace)
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    common = {
        "pr_url": "https://github.com/acme/repo/pull/7",
        "context": [],
        "mode": ReviewMode.INITIAL,
        "baseline_operation_id": None,
        "iterations": 1,
    }
    try:
        with pytest.raises(ReviewError, match="outside the server working-directory sandbox"):
            await runtime.start_operation(
                **common,
                source_mode=ReviewSourceMode.LOCAL,
                local_source_path=str(outside),
            )
        with pytest.raises(ReviewError, match="valid only when source_mode is local"):
            await runtime.start_operation(
                **common,
                source_mode=ReviewSourceMode.MANAGED,
                local_source_path=".",
            )
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_start_deduplicates_and_caps_configured_reviewer_models(monkeypatch, tmp_path):
    # Dedup: a roster with a repeated model id collapses to a single entry.
    dedup_runtime = _runtime(
        _config(quality=["claude-opus-4-8", "claude-opus-4-8"], security=["gpt-5.6"]),
        tmp_path / "dedup",
    )
    await asyncio.wait_for(dedup_runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(dedup_runtime, "_spawn", lambda *args, **kwargs: None)
    try:
        record = await asyncio.wait_for(
            dedup_runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            ),
            timeout=_TIMEOUT,
        )
        assert record.request["models"]["quality"] == ["claude-opus-4-8"]
    finally:
        await asyncio.wait_for(dedup_runtime.close(), timeout=_TIMEOUT)

    # Cap: a two-model roster under a per-disposition limit of 1 is rejected.
    capped_runtime = _runtime(
        _config(
            quality=["claude-opus-4-8", "gpt-5.6"],
            security=["gpt-5.6"],
            models_per_disposition=1,
        ),
        tmp_path / "cap",
    )
    await asyncio.wait_for(capped_runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(capped_runtime, "_spawn", lambda *args, **kwargs: None)
    try:
        with pytest.raises(ReviewError, match="model count"):
            await asyncio.wait_for(
                capped_runtime.start_operation(
                    pr_url="https://github.com/acme/repo/pull/7",
                    context=[],
                    mode=ReviewMode.INITIAL,
                    baseline_operation_id=None,
                    iterations=1,
                ),
                timeout=_TIMEOUT,
            )
    finally:
        await asyncio.wait_for(capped_runtime.close(), timeout=_TIMEOUT)


async def test_start_uses_configured_models(monkeypatch, tmp_path):
    runtime = _runtime(
        _config(
            quality=["claude-opus-4-8", "claude-opus-4-8"],
            security=["gpt-5.6"],
            deliberation="claude-opus-5",
            aggregation="claude-haiku-4-5-20251001",
        ),
        tmp_path / "state",
    )
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    try:
        record = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            ),
            timeout=_TIMEOUT,
        )
        assert record.request["models"] == {
            "quality": ["claude-opus-4-8"],
            "security": ["gpt-5.6"],
        }
        assert record.request["deliberation_model"] == "claude-opus-5"
        assert record.request["aggregation_model"] == "claude-haiku-4-5-20251001"
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_start_snapshots_per_operation_model_overrides(monkeypatch, tmp_path):
    runtime = _runtime(_config(), tmp_path / "state")
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    try:
        record = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
                models={
                    "quality": ["claude-sonnet-5", "claude-sonnet-5"],
                    "security": ["gpt-5.6"],
                },
                deliberation_model="claude-opus-5",
                aggregation_model="claude-haiku-4-5-20251001",
            ),
            timeout=_TIMEOUT,
        )
        assert record.request["models"] == {
            "quality": ["claude-sonnet-5"],
            "security": ["gpt-5.6"],
        }
        assert record.request["deliberation_model"] == "claude-opus-5"
        assert record.request["aggregation_model"] == "claude-haiku-4-5-20251001"
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_start_rejects_unknown_model_before_persisting(monkeypatch, tmp_path):
    runtime = _runtime(_config(), tmp_path / "state")
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    assert runtime.store is not None
    create = AsyncMock(wraps=runtime.store.create)
    monkeypatch.setattr(runtime.store, "create", create)
    try:
        with pytest.raises(ReviewError, match='unknown model id "unknown-model"'):
            await runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
                models={"quality": ["unknown-model"], "security": ["gpt-5.6"]},
            )
        create.assert_not_awaited()
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_start_rejects_backend_incompatible_model_before_persisting(monkeypatch, tmp_path):
    class NativeFactory(FakeModelFactory):
        def validate(self, model_id, *, reasoning_effort=None):
            spec = super().validate(model_id, reasoning_effort=reasoning_effort)
            if spec.native_provider is None:
                raise ModelConfigurationError(f'model "{model_id}" is available only through a gateway')
            return spec

    runtime = _runtime(
        _config(quality=["gemini-3.1-pro-preview"], security=["gpt-5.6"]),
        tmp_path / "state",
        model_factory=NativeFactory(),
    )
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    assert runtime.store is not None
    create = AsyncMock(wraps=runtime.store.create)
    monkeypatch.setattr(runtime.store, "create", create)
    try:
        with pytest.raises(ReviewError, match='"gemini-3.1-pro-preview" is available only through a gateway'):
            await runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            )
        create.assert_not_awaited()
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_failed_publication_retry_resumes_with_the_persisted_commit(monkeypatch, tmp_path):
    runtime = _runtime(_config(), tmp_path / "state")
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    spawned = []
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: spawned.append((args, kwargs)))
    try:
        record = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            ),
            timeout=_TIMEOUT,
        )
        spawned.clear()
        assert runtime.store is not None
        await asyncio.wait_for(
            runtime.store.update(
                record.id,
                status=OperationStatus.FAILED,
                preview=_preview(),
                state={
                    **record.state,
                    "pending_commit": {"action": "commit", "revision": 1, "payload_hash": "payload"},
                },
            ),
            timeout=_TIMEOUT,
        )

        await asyncio.wait_for(runtime.commit(record.id, 1, "payload"), timeout=_TIMEOUT)

        assert len(spawned) == 1
        command = spawned[0][1]["command"]
        assert command.resume == {"action": "commit", "revision": 1, "payload_hash": "payload"}
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_failed_review_without_persisted_commit_intent_cannot_be_committed(monkeypatch, tmp_path):
    runtime = _runtime(_config(), tmp_path / "state")
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    try:
        record = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            ),
            timeout=_TIMEOUT,
        )
        assert runtime.store is not None
        await asyncio.wait_for(
            runtime.store.update(record.id, status=OperationStatus.FAILED, preview=_preview()),
            timeout=_TIMEOUT,
        )

        with pytest.raises(ReviewError, match="cannot be committed from status failed"):
            await asyncio.wait_for(runtime.commit(record.id, 1, "payload"), timeout=_TIMEOUT)
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_stale_preview_remains_readable_but_cannot_be_recommitted(monkeypatch, tmp_path):
    runtime = _runtime(_config(), tmp_path / "state")
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    try:
        record = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            ),
            timeout=_TIMEOUT,
        )
        assert runtime.store is not None
        await asyncio.wait_for(
            runtime.store.update(record.id, status=OperationStatus.STALE, preview=_preview("old-head")),
            timeout=_TIMEOUT,
        )

        assert (await asyncio.wait_for(runtime.preview(record.id), timeout=_TIMEOUT)).head_sha == "old-head"
        with pytest.raises(ReviewError, match="cannot be committed from status stale"):
            await asyncio.wait_for(runtime.commit(record.id, 1, "payload"), timeout=_TIMEOUT)
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)


async def test_run_applies_the_snapshotted_model_concurrency_limit(monkeypatch, tmp_path):
    runtime = _runtime(_config(concurrent_model_calls=3), tmp_path / "state")
    await asyncio.wait_for(runtime.start(), timeout=_TIMEOUT)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: None)
    captured = {}

    class FakeGraph:
        async def ainvoke(self, value, *, config):
            captured.update(config)

    runtime.graph = FakeGraph()
    try:
        record = await asyncio.wait_for(
            runtime.start_operation(
                pr_url="https://github.com/acme/repo/pull/7",
                context=[],
                mode=ReviewMode.INITIAL,
                baseline_operation_id=None,
                iterations=1,
            ),
            timeout=_TIMEOUT,
        )

        await asyncio.wait_for(runtime._run(record.id, resume=False, command=None), timeout=_TIMEOUT)

        assert captured["max_concurrency"] == 3
    finally:
        await asyncio.wait_for(runtime.close(), timeout=_TIMEOUT)
