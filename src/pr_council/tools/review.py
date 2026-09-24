"""Lifecycle MCP tools for durable pull-request reviews."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastmcp.tools import tool
from localmcp.observability.logging import get_logger
from mcp.types import ToolAnnotations
from pydantic import Field

from pr_council.review.models import OperationStatus, ReviewContextInput, ReviewMode, ReviewSourceMode
from pr_council.workflows.review.runtime import review_runtime

log = get_logger(__name__)


@tool(
    annotations=ToolAnnotations(
        title="Start a PR review",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def pr_council_start(
    pr_url: Annotated[str, Field(description="Full HTTPS pull-request URL on an allowed GitHub host.")],
    source_mode: Annotated[
        ReviewSourceMode,
        Field(description="Use a server-managed checkout or an existing clean local checkout."),
    ] = ReviewSourceMode.MANAGED,
    local_source_path: Annotated[
        str | None,
        Field(
            max_length=4096,
            description=(
                "Local repository root for source_mode=local. Relative paths resolve from the server working "
                "directory, must remain beneath it, and default to '.'. The caller must keep this live checkout "
                "stable until the review is ready. Invalid for managed mode."
            ),
        ),
    ] = None,
    review_context: Annotated[
        list[ReviewContextInput] | None,
        Field(description="Optional finalized context items supplied by the calling agent."),
    ] = None,
    mode: Annotated[
        ReviewMode,
        Field(description="Auto-detect an earlier published review, force initial review, or require follow-up."),
    ] = ReviewMode.AUTO,
    baseline_operation_id: Annotated[
        str | None,
        Field(description="Optional prior completed operation to use as the follow-up baseline."),
    ] = None,
    models: Annotated[
        dict[str, list[str]] | None,
        Field(
            description=(
                "Optional per-operation override containing non-empty quality and security reviewer-model lists; "
                "omit to use [pr_review.models]."
            )
        ),
    ] = None,
    iterations: Annotated[int, Field(ge=1, le=10, description="Refinement iterations per reviewer model.")] = 3,
    deliberation_model: Annotated[
        str | None,
        Field(description="Optional per-operation deliberation-model override; omit to use [pr_review.models]."),
    ] = None,
    aggregation_model: Annotated[
        str | None,
        Field(description="Optional per-operation aggregation-model override; omit to use [pr_review.models]."),
    ] = None,
) -> dict[str, object]:
    """Queue a durable, comment-only PR review and return its operation handle.

    The calling agent gathers and edits arbitrary context in normal conversation,
    then supplies the finalized snapshot here. Poll with ``pr_council_get``.
    This tool never posts to GitHub.
    """
    log.debug("tool.invoked", tool="pr_council_start")
    record = await review_runtime().start_operation(
        pr_url=pr_url,
        source_mode=source_mode,
        local_source_path=local_source_path,
        context=review_context,
        mode=mode,
        baseline_operation_id=baseline_operation_id,
        models=models,
        iterations=iterations,
        deliberation_model=deliberation_model,
        aggregation_model=aggregation_model,
    )
    return {"operation_id": record.id, "label": record.label, "status": record.status.value}


@tool(
    annotations=ToolAnnotations(
        title="Get a PR review",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def pr_council_get(
    operation_id: Annotated[str, Field(description="Opaque operation ID returned by pr_council_start.")],
) -> dict[str, object]:
    """Return durable status, progress, errors, and completion data for one owned review."""
    record = await review_runtime().get_owned(operation_id)
    preview_available = record.preview is not None
    status_reason = (record.result or {}).get("reason") if record.status == OperationStatus.STALE else None
    return {
        "operation_id": record.id,
        "label": record.label,
        "status": record.status.value,
        "preview_ready": preview_available,
        "commit_ready": record.commit_ready,
        "status_reason": status_reason,
        "error": record.error,
        "reviewer_failures": (record.preview or {}).get("reviewer_failures", []),
        "result": record.result,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


@tool(
    annotations=ToolAnnotations(
        title="Preview a PR review",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def pr_council_preview(
    operation_id: Annotated[str, Field(description="Opaque operation ID returned by pr_council_start.")],
) -> dict[str, object]:
    """Return the exact revision-bound summary and inline comments without posting them."""
    return (await review_runtime().preview(operation_id)).model_dump(mode="json")


@tool(
    annotations=ToolAnnotations(
        title="Commit a PR review",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def pr_council_commit(
    operation_id: Annotated[str, Field(description="Opaque operation ID for the reviewed preview.")],
    revision: Annotated[int, Field(gt=0, description="Revision returned by pr_council_preview.")],
    payload_hash: Annotated[str, Field(description="Payload hash returned by pr_council_preview.")],
) -> dict[str, object]:
    """Commit exactly one reviewed preview as inline comments and a COMMENT-only summary.

    Call only after presenting the preview and receiving approval in normal
    conversation or through the MCP client's tool-call authorization policy.
    """
    record = await review_runtime().commit(operation_id, revision, payload_hash)
    return {"operation_id": record.id, "label": record.label, "status": record.status.value}


@tool(
    annotations=ToolAnnotations(
        title="Cancel a PR review",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def pr_council_cancel(
    operation_id: Annotated[str, Field(description="Opaque operation ID returned by pr_council_start.")],
) -> dict[str, object]:
    """Cooperatively cancel one owned review unless publication has already begun."""
    record = await review_runtime().cancel(operation_id)
    return {"operation_id": record.id, "label": record.label, "status": record.status.value}


TOOLS: list[Callable[..., Any]] = [
    pr_council_start,
    pr_council_get,
    pr_council_preview,
    pr_council_commit,
    pr_council_cancel,
]
