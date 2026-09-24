import asyncio

import aiosqlite
import pytest

from pr_council.review.models import OperationStatus, ReviewError
from pr_council.review.store import OperationStore


async def _seed(store, *, operation_id="operation-1", owner_uid=123, repo_key="github.com/acme/repo", pr_number=7):
    await store.create(
        operation_id=operation_id,
        owner_uid=owner_uid,
        repo_key=repo_key,
        pr_number=pr_number,
        label=f"PR #{pr_number}",
        request={},
        parent_operation_id=None,
    )


async def test_operation_store_persists_status_preview_and_followup_lookup(tmp_path):
    path = tmp_path / "operations.sqlite3"
    store = await OperationStore.open(path)
    await store.create(
        operation_id="operation-1",
        owner_uid=123,
        repo_key="github.com/acme/repo",
        pr_number=7,
        label="PR #7",
        request={"value": 1},
        parent_operation_id=None,
    )
    await store.update(
        "operation-1",
        status=OperationStatus.COMPLETED,
        preview={"revision": 1},
        result={"findings": []},
    )
    await store.close()

    reopened = await OperationStore.open(path)
    record = await reopened.get("operation-1")
    assert record is not None
    assert record.status == OperationStatus.COMPLETED
    assert record.preview == {"revision": 1}
    assert await reopened.latest_completed("github.com/acme/repo", 7, 123) == record
    await reopened.close()


async def test_open_pins_a_bounded_busy_timeout_for_writer_contention(tmp_path):
    # Regression guard: writers must serialize on a bounded wait rather than an
    # implicit driver default that a future connect() change could silently drop.
    # We force aiosqlite's connect() default OFF (timeout=0 -> busy_timeout=0) so
    # the baseline is neutralized; the only way busy_timeout can read 5000 is the
    # explicit PRAGMA in OperationStore.setup(). This makes the guard detect the
    # removal of that explicit PRAGMA, which a plain OperationStore.open() (which
    # inherits the driver's timeout=5.0 -> 5000 default) could not.
    path = tmp_path / "operations.sqlite3"
    conn = await aiosqlite.connect(path, timeout=0)
    baseline = await conn.execute("PRAGMA busy_timeout")
    assert (await baseline.fetchone())[0] == 0

    store = OperationStore(conn)
    await store.setup()

    cursor = await store.connection.execute("PRAGMA busy_timeout")
    assert (await cursor.fetchone())[0] == 5000
    await store.close()


async def test_operation_lease_prevents_two_workers_from_claiming_the_same_graph(tmp_path):
    path = tmp_path / "operations.sqlite3"
    first = await OperationStore.open(path)
    second = await OperationStore.open(path)
    await first.create(
        operation_id="operation-1",
        owner_uid=123,
        repo_key="github.com/acme/repo",
        pr_number=7,
        label="PR #7",
        request={},
        parent_operation_id=None,
    )
    assert await first.claim("operation-1", "worker-a") is True
    assert await second.claim("operation-1", "worker-b") is False
    await first.release("operation-1", "worker-a")
    assert await second.claim("operation-1", "worker-b") is True
    await first.close()
    await second.close()


async def test_operation_tool_budget_is_atomic_and_persisted(tmp_path):
    path = tmp_path / "operations.sqlite3"
    first = await OperationStore.open(path)
    second = await OperationStore.open(path)
    await first.create(
        operation_id="operation-1",
        owner_uid=123,
        repo_key="github.com/acme/repo",
        pr_number=7,
        label="PR #7",
        request={},
        parent_operation_id=None,
    )
    claims = await asyncio.gather(
        *(first.claim_tool_call("operation-1", 5) for _ in range(5)),
        *(second.claim_tool_call("operation-1", 5) for _ in range(5)),
    )
    assert sorted(result for result in claims if result is not None) == [0, 1, 2, 3, 4]
    assert sum(result is not None for result in claims) == 5
    await first.close()
    await second.close()

    reopened = await OperationStore.open(path)
    record = await reopened.get("operation-1")
    assert record is not None
    assert record.tool_calls_used == 5
    assert await reopened.claim_tool_call("operation-1", 5) is None
    await reopened.close()


async def test_require_owned_returns_record_for_owner_and_rejects_others(tmp_path):
    store = await OperationStore.open(tmp_path / "operations.sqlite3")
    await _seed(store, owner_uid=123)

    owned = await store.require_owned("operation-1", 123)
    assert owned.owner_uid == 123

    with pytest.raises(ReviewError, match="was not found for the authenticated local caller"):
        await store.require_owned("operation-1", 999)
    with pytest.raises(ReviewError, match="was not found for the authenticated local caller"):
        await store.require_owned("missing", 123)
    await store.close()


async def test_request_cancel_sets_status_and_cancel_flag(tmp_path):
    store = await OperationStore.open(tmp_path / "operations.sqlite3")
    await _seed(store)

    assert await store.is_cancel_requested("operation-1") is False

    await store.request_cancel("operation-1")

    assert await store.is_cancel_requested("operation-1") is True
    record = await store.get("operation-1")
    assert record is not None
    assert record.status == OperationStatus.CANCEL_REQUESTED
    assert record.cancel_requested is True
    await store.close()


async def test_renew_extends_lease_for_owner_and_refuses_non_owner(tmp_path):
    store = await OperationStore.open(tmp_path / "operations.sqlite3")
    await _seed(store)

    assert await store.claim("operation-1", "worker-a", lease_seconds=1.0) is True
    before = (await store.get("operation-1")).lease_expires

    assert await store.renew("operation-1", "worker-a", lease_seconds=500.0) is True
    after = (await store.get("operation-1")).lease_expires
    assert after > before

    assert await store.renew("operation-1", "worker-b", lease_seconds=500.0) is False
    assert (await store.get("operation-1")).lease_expires == after
    await store.close()


async def test_release_clears_lease_ownership(tmp_path):
    store = await OperationStore.open(tmp_path / "operations.sqlite3")
    await _seed(store)
    await store.claim("operation-1", "worker-a")

    await store.release("operation-1", "worker-a")

    record = await store.get("operation-1")
    assert record is not None
    assert record.lease_owner is None
    assert record.lease_expires is None
    await store.close()


async def test_recoverable_returns_only_non_terminal_operations(tmp_path):
    store = await OperationStore.open(tmp_path / "operations.sqlite3")
    await _seed(store, operation_id="active", pr_number=1)
    await _seed(store, operation_id="done", pr_number=2)
    await _seed(store, operation_id="failed", pr_number=3)
    await store.update("active", status=OperationStatus.REVIEWING)
    await store.update("done", status=OperationStatus.COMPLETED)
    await store.update("failed", status=OperationStatus.FAILED)

    recoverable = await store.recoverable()

    assert {record.id for record in recoverable} == {"active"}
    await store.close()


async def test_latest_completed_returns_newest_completed_for_target(tmp_path):
    store = await OperationStore.open(tmp_path / "operations.sqlite3")
    await _seed(store, operation_id="older", pr_number=7)
    await _seed(store, operation_id="newer", pr_number=7)
    # created_at drives ordering; force a later timestamp on the newer record.
    await store.connection.execute(
        "UPDATE review_operations SET created_at = ? WHERE id = ?",
        ("2999-01-01T00:00:00+00:00", "newer"),
    )
    await store.connection.commit()
    await store.update("older", status=OperationStatus.COMPLETED)
    await store.update("newer", status=OperationStatus.COMPLETED)

    latest = await store.latest_completed("github.com/acme/repo", 7, 123)

    assert latest is not None
    assert latest.id == "newer"
    # A different owner sees no completed operation for the same target.
    assert await store.latest_completed("github.com/acme/repo", 7, 999) is None
    await store.close()


async def test_update_sets_and_clears_error(tmp_path):
    store = await OperationStore.open(tmp_path / "operations.sqlite3")
    await _seed(store)

    await store.update("operation-1", status=OperationStatus.FAILED, error="boom")
    assert (await store.get("operation-1")).error == "boom"

    await store.update("operation-1", clear_error=True)
    assert (await store.get("operation-1")).error is None
    await store.close()
