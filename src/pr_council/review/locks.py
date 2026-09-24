"""Cross-process advisory locks for shared Git repositories."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO


class RepoLock:
    def __init__(self, lock_dir: Path):
        self._lock_dir = lock_dir

    @asynccontextmanager
    async def acquire(self, repo_key: str) -> AsyncIterator[None]:
        self._lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        name = hashlib.sha256(repo_key.encode()).hexdigest()
        handle = (self._lock_dir / f"{name}.lock").open("a+b")
        try:
            await asyncio.to_thread(_flock, handle, fcntl.LOCK_EX)
            yield
        finally:
            await asyncio.to_thread(_flock, handle, fcntl.LOCK_UN)
            handle.close()


def _flock(handle: BinaryIO, operation: int) -> None:
    fcntl.flock(handle.fileno(), operation)
