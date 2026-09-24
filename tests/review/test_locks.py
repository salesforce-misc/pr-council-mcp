import asyncio
import hashlib
import stat

import pytest

from pr_council.review.locks import RepoLock

# Bounded guard so a serialization/deadlock bug fails fast instead of hanging.
_TIMEOUT = 5.0


async def test_same_repo_key_serializes_acquisitions(tmp_path):
    """Two concurrent acquisitions of the SAME repo_key must not overlap.

    fcntl.flock is per open-file-description, so even within one process the two
    separate acquire() calls (each opening its own handle) contend on the same
    lock file. We track a live active-count and assert peak concurrency == 1.
    A no-op `yield` implementation would let both bodies run at once (peak == 2).
    """
    lock = RepoLock(tmp_path / "locks")
    repo_key = "github.com/acme/repo"

    active = 0
    peak = 0

    async def worker() -> None:
        nonlocal active, peak
        async with lock.acquire(repo_key):
            active += 1
            peak = max(peak, active)
            # Yield to the loop so a broken (non-serializing) lock would let the
            # second worker enter and drive `active` to 2 during this window.
            await asyncio.sleep(0.05)
            active -= 1

    await asyncio.wait_for(asyncio.gather(worker(), worker()), timeout=_TIMEOUT)

    assert peak == 1
    assert active == 0


async def test_different_repo_keys_do_not_serialize(tmp_path):
    """Acquisitions of DIFFERENT repo_keys may be active simultaneously.

    Each worker signals its own entry event, then waits for the other's. Both can
    only complete if both are inside their locks at the same time. If distinct
    keys wrongly shared a lock, the second could not enter while the first waits,
    producing a deadlock that the bounded timeout turns into a failure.
    """
    lock = RepoLock(tmp_path / "locks")
    entered_a = asyncio.Event()
    entered_b = asyncio.Event()

    async def worker(repo_key: str, mine: asyncio.Event, other: asyncio.Event) -> None:
        async with lock.acquire(repo_key):
            mine.set()
            await asyncio.wait_for(other.wait(), timeout=_TIMEOUT)

    await asyncio.wait_for(
        asyncio.gather(
            worker("github.com/acme/repo-one", entered_a, entered_b),
            worker("github.com/acme/repo-two", entered_b, entered_a),
        ),
        timeout=_TIMEOUT,
    )

    assert entered_a.is_set()
    assert entered_b.is_set()


async def test_acquire_creates_lock_dir_with_owner_only_mode(tmp_path):
    """Acquiring creates the lock directory (parents=True) owner-only.

    The security-relevant, umask-robust invariant is that group and other have
    no access while the owner does. Asserting an exact ``== 0o700`` is brittle
    because ``mkdir(mode=0o700)`` is masked by the process umask, which can clear
    owner bits under a pathological umask; we assert the invariant instead.
    """
    lock_dir = tmp_path / "nested" / "locks"
    lock = RepoLock(lock_dir)
    assert not lock_dir.exists()

    repo_key = "github.com/acme/repo"
    async with lock.acquire(repo_key):
        assert lock_dir.is_dir()
        mode = stat.S_IMODE(lock_dir.stat().st_mode)
        # Never group/other accessible regardless of umask nuances.
        assert not mode & (stat.S_IRWXG | stat.S_IRWXO)
        # Owner retains access to the lock dir it must read/write/traverse.
        assert mode & stat.S_IRWXU


async def test_lock_filename_is_sha256_of_repo_key(tmp_path):
    """The on-disk lock file is named ``sha256(repo_key).hexdigest() + '.lock'``.

    This pins the acceptance-required derivation scheme: swapping the digest
    (e.g. sha256 -> sha1) or using the raw key would leave this file absent.
    """
    lock_dir = tmp_path / "locks"
    lock = RepoLock(lock_dir)
    repo_key = "github.com/acme/repo"
    expected = hashlib.sha256(repo_key.encode()).hexdigest()

    async with lock.acquire(repo_key):
        assert (lock_dir / f"{expected}.lock").exists()


async def test_lock_released_after_body_raises(tmp_path):
    """The `finally` (LOCK_UN + close) must run even when the body raises,
    so a subsequent acquisition of the same key succeeds instead of blocking."""
    lock = RepoLock(tmp_path / "locks")
    repo_key = "github.com/acme/repo"

    with pytest.raises(RuntimeError, match="boom"):
        async with lock.acquire(repo_key):
            raise RuntimeError("boom")

    # If release failed, this second acquisition would block until the timeout.
    async def reacquire() -> bool:
        async with lock.acquire(repo_key):
            return True

    assert await asyncio.wait_for(reacquire(), timeout=_TIMEOUT) is True
