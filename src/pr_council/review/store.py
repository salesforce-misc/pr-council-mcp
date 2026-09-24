"""Durable operation catalog beside LangGraph's checkpoint database."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from pr_council.review.models import TERMINAL_STATUSES, OperationStatus, ReviewError


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class OperationRecord:
    id: str
    owner_uid: int
    owner_login: str | None
    repo_key: str
    pr_number: int
    label: str
    status: OperationStatus
    request: dict[str, Any]
    state: dict[str, Any]
    preview: dict[str, Any] | None
    result: dict[str, Any] | None
    error: str | None
    cancel_requested: bool
    parent_operation_id: str | None
    created_at: str
    updated_at: str
    lease_owner: str | None
    lease_expires: float | None
    tool_calls_used: int

    @property
    def commit_ready(self) -> bool:
        if self.preview is None:
            return False
        if self.status == OperationStatus.READY:
            return True
        if self.status != OperationStatus.FAILED:
            return False
        pending = self.state.get("pending_commit")
        return bool(
            isinstance(pending, dict)
            and pending.get("action") == "commit"
            and pending.get("revision") == self.preview.get("revision")
            and pending.get("payload_hash") == self.preview.get("payload_hash")
        )


class OperationStore:
    def __init__(self, connection: aiosqlite.Connection):
        self.connection = connection
        self._tool_call_lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: Path) -> OperationStore:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = await aiosqlite.connect(path)
        connection.row_factory = aiosqlite.Row
        store = cls(connection)
        await store.setup()
        return store

    async def setup(self) -> None:
        # Pin a bounded busy wait independent of aiosqlite's connect(timeout=...)
        # default, so writer contention blocks-then-fails rather than erroring
        # instantly. Must precede journal_mode=WAL, whose lock acquisition this
        # wait also needs to cover.
        await self.connection.execute("PRAGMA busy_timeout=5000")
        await self.connection.execute("PRAGMA journal_mode=WAL")
        await self.connection.execute("PRAGMA foreign_keys=ON")
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS review_operations (
              id TEXT PRIMARY KEY,
              owner_uid INTEGER NOT NULL,
              owner_login TEXT,
              repo_key TEXT NOT NULL,
              pr_number INTEGER NOT NULL,
              label TEXT NOT NULL,
              status TEXT NOT NULL,
              request_json TEXT NOT NULL,
              state_json TEXT NOT NULL DEFAULT '{}',
              preview_json TEXT,
              result_json TEXT,
              error TEXT,
              cancel_requested INTEGER NOT NULL DEFAULT 0,
              parent_operation_id TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
              ,lease_owner TEXT
              ,lease_expires REAL
              ,tool_calls_used INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        columns = {
            row[1] for row in await (await self.connection.execute("PRAGMA table_info(review_operations)")).fetchall()
        }
        if "lease_owner" not in columns:
            await self.connection.execute("ALTER TABLE review_operations ADD COLUMN lease_owner TEXT")
        if "lease_expires" not in columns:
            await self.connection.execute("ALTER TABLE review_operations ADD COLUMN lease_expires REAL")
        if "tool_calls_used" not in columns:
            await self.connection.execute(
                "ALTER TABLE review_operations ADD COLUMN tool_calls_used INTEGER NOT NULL DEFAULT 0"
            )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS review_operations_target ON review_operations(repo_key, pr_number, created_at)"
        )
        await self.connection.commit()

    async def close(self) -> None:
        await self.connection.close()

    async def create(
        self,
        *,
        operation_id: str,
        owner_uid: int,
        repo_key: str,
        pr_number: int,
        label: str,
        request: dict[str, Any],
        parent_operation_id: str | None,
    ) -> OperationRecord:
        now = _now()
        await self.connection.execute(
            """
            INSERT INTO review_operations
              (id, owner_uid, repo_key, pr_number, label, status, request_json,
               parent_operation_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                operation_id,
                owner_uid,
                repo_key,
                pr_number,
                label,
                OperationStatus.QUEUED.value,
                json.dumps(request),
                parent_operation_id,
                now,
                now,
            ),
        )
        await self.connection.commit()
        record = await self.get(operation_id)
        assert record is not None
        return record

    async def get(self, operation_id: str) -> OperationRecord | None:
        cursor = await self.connection.execute("SELECT * FROM review_operations WHERE id = ?", (operation_id,))
        row = await cursor.fetchone()
        return _record(row) if row is not None else None

    async def require_owned(self, operation_id: str, owner_uid: int) -> OperationRecord:
        record = await self.get(operation_id)
        if record is None or record.owner_uid != owner_uid:
            raise ReviewError("review operation was not found for the authenticated local caller")
        return record

    async def update(
        self,
        operation_id: str,
        *,
        status: OperationStatus | None = None,
        owner_login: str | None = None,
        state: dict[str, Any] | None = None,
        preview: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        clear_error: bool = False,
    ) -> None:
        assignments = ["updated_at = ?"]
        values: list[object] = [_now()]
        for column, value in (
            ("status", status.value if status else None),
            ("owner_login", owner_login),
            ("state_json", json.dumps(state) if state is not None else None),
            ("preview_json", json.dumps(preview) if preview is not None else None),
            ("result_json", json.dumps(result) if result is not None else None),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if error is not None or clear_error:
            assignments.append("error = ?")
            values.append(error)
        values.append(operation_id)
        await self.connection.execute(
            f"UPDATE review_operations SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
            values,
        )
        await self.connection.commit()

    async def request_cancel(self, operation_id: str) -> None:
        await self.connection.execute(
            "UPDATE review_operations SET cancel_requested = 1, status = ?, updated_at = ? WHERE id = ?",
            (OperationStatus.CANCEL_REQUESTED.value, _now(), operation_id),
        )
        await self.connection.commit()

    async def is_cancel_requested(self, operation_id: str) -> bool:
        cursor = await self.connection.execute(
            "SELECT cancel_requested FROM review_operations WHERE id = ?", (operation_id,)
        )
        row = await cursor.fetchone()
        return bool(row and row[0])

    async def claim_tool_call(self, operation_id: str, limit: int) -> int | None:
        """Atomically reserve a call and return operation capacity remaining."""
        async with self._tool_call_lock:
            async with self.connection.execute(
                """
                UPDATE review_operations
                SET tool_calls_used = tool_calls_used + 1, updated_at = ?
                WHERE id = ? AND tool_calls_used < ?
                """,
                (_now(), operation_id, limit),
            ) as cursor:
                claimed = cursor.rowcount == 1
            if not claimed:
                await self.connection.commit()
                return None
            async with self.connection.execute(
                "SELECT tool_calls_used FROM review_operations WHERE id = ?", (operation_id,)
            ) as cursor:
                row = await cursor.fetchone()
            await self.connection.commit()
            assert row is not None
            return limit - int(row[0])

    async def claim(self, operation_id: str, worker_id: str, *, lease_seconds: float = 90.0) -> bool:
        now = time.time()
        cursor = await self.connection.execute(
            """
            UPDATE review_operations SET lease_owner = ?, lease_expires = ?, updated_at = ?
            WHERE id = ? AND (lease_owner IS NULL OR lease_expires < ? OR lease_owner = ?)
            """,
            (worker_id, now + lease_seconds, _now(), operation_id, now, worker_id),
        )
        await self.connection.commit()
        return cursor.rowcount == 1

    async def renew(self, operation_id: str, worker_id: str, *, lease_seconds: float = 90.0) -> bool:
        cursor = await self.connection.execute(
            "UPDATE review_operations SET lease_expires = ? WHERE id = ? AND lease_owner = ?",
            (time.time() + lease_seconds, operation_id, worker_id),
        )
        await self.connection.commit()
        return cursor.rowcount == 1

    async def release(self, operation_id: str, worker_id: str) -> None:
        await self.connection.execute(
            "UPDATE review_operations SET lease_owner = NULL, lease_expires = NULL WHERE id = ? AND lease_owner = ?",
            (operation_id, worker_id),
        )
        await self.connection.commit()

    async def recoverable(self) -> list[OperationRecord]:
        terminal = tuple(status.value for status in TERMINAL_STATUSES)
        placeholders = ",".join("?" for _ in terminal)
        cursor = await self.connection.execute(
            f"SELECT * FROM review_operations WHERE status NOT IN ({placeholders})",
            terminal,  # noqa: S608
        )
        return [_record(row) for row in await cursor.fetchall()]

    async def latest_completed(self, repo_key: str, pr_number: int, owner_uid: int) -> OperationRecord | None:
        cursor = await self.connection.execute(
            """
            SELECT * FROM review_operations
            WHERE repo_key = ? AND pr_number = ? AND owner_uid = ? AND status = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (repo_key, pr_number, owner_uid, OperationStatus.COMPLETED.value),
        )
        row = await cursor.fetchone()
        return _record(row) if row is not None else None


def _loads(value: str | None) -> dict[str, Any] | None:
    return json.loads(value) if value is not None else None


def _record(row: aiosqlite.Row) -> OperationRecord:
    return OperationRecord(
        id=row["id"],
        owner_uid=row["owner_uid"],
        owner_login=row["owner_login"],
        repo_key=row["repo_key"],
        pr_number=row["pr_number"],
        label=row["label"],
        status=OperationStatus(row["status"]),
        request=_loads(row["request_json"]) or {},
        state=_loads(row["state_json"]) or {},
        preview=_loads(row["preview_json"]),
        result=_loads(row["result_json"]),
        error=row["error"],
        cancel_requested=bool(row["cancel_requested"]),
        parent_operation_id=row["parent_operation_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        lease_owner=row["lease_owner"],
        lease_expires=row["lease_expires"],
        tool_calls_used=row["tool_calls_used"],
    )
