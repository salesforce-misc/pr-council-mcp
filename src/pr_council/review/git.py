"""Safe async Git and GitHub CLI adapters for pull-request review."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pr_council.review.models import PreviewComment, PrRef, ReviewError

_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_OBJECT_ID = re.compile(r"^[0-9a-fA-F]{40}$")
_ALLOWED_HOSTS = {"github.com"}
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_COMMAND_TIMEOUT_SECONDS = 300.0
_MAX_LOCAL_SANDBOX_EXCLUSIONS = 4_096
_MAX_LOCAL_SUBMODULES = 1_024
_SAFE_LOCAL_CONFIG_KEY = re.compile(
    r"^(?:"
    r"core\.(?:repositoryformatversion|filemode|bare|logallrefupdates|ignorecase|precomposeunicode|symlinks|worktree)|"
    r"extensions\.(?:objectformat|worktreeconfig|partialclone)|"
    r"remote\..+\.(?:url|fetch|mirror|promisor|partialclonefilter|tagopt)|"
    r"branch\..+\.(?:remote|merge|rebase)|"
    r"submodule\..+\.(?:url|active)"
    r")$"
)
_REMOTE_URL_CONFIG_KEY = re.compile(r"^(?:remote|submodule)\..+\.url$")
_SENSITIVE_GIT_METADATA_NAMES = (
    "AUTO_MERGE",
    "BISECT_LOG",
    "COMMIT_EDITMSG",
    "FETCH_HEAD",
    "MERGE_MSG",
    "ORIG_HEAD",
    "SQUASH_MSG",
    "hooks",
    "logs",
    "rebase-apply",
    "rebase-merge",
    "sequencer",
)


class _OutputLimitExceeded(Exception):
    pass


@dataclass(frozen=True)
class LocalSourceSnapshot:
    """Validated local Git metadata and caller-local sandbox exclusions."""

    git_metadata_root: Path
    excluded_paths: tuple[Path, ...]


def parse_pr_url(value: str, *, allowed_hosts: set[str] | None = None) -> PrRef:
    parsed = urlparse(value.strip())
    hosts = allowed_hosts or _ALLOWED_HOSTS
    if (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ReviewError("PR URL must be an HTTPS URL on an allowed GitHub host")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 4 or parts[2] != "pull" or not parts[3].isdigit():
        raise ReviewError("PR URL must have the form https://host/owner/repo/pull/number")
    owner, repo = parts[0], parts[1]
    if any(part in {".", ".."} or not _SEGMENT.fullmatch(part) for part in (parsed.hostname, owner, repo)):
        raise ReviewError("PR URL contains an invalid repository segment")
    return PrRef(host=parsed.hostname, owner=owner, repo=repo, number=int(parts[3]))


class GitHubCli:
    def __init__(
        self,
        *,
        max_output_bytes: int = _MAX_OUTPUT_BYTES,
        command_timeout_seconds: float = _COMMAND_TIMEOUT_SECONDS,
        max_repository_bytes: int = 5 * 1024 * 1024 * 1024,
    ):
        self.max_output_bytes = max_output_bytes
        self.command_timeout_seconds = command_timeout_seconds
        self.max_repository_bytes = max_repository_bytes

    async def _communicate(self, process: asyncio.subprocess.Process, command: str) -> tuple[bytes, bytes]:
        total_bytes = 0

        async def read(stream: asyncio.StreamReader | None) -> bytes:
            nonlocal total_bytes
            if stream is None:
                return b""
            chunks: list[bytes] = []
            while chunk := await stream.read(64 * 1024):
                total_bytes += len(chunk)
                if total_bytes > self.max_output_bytes:
                    raise _OutputLimitExceeded
                chunks.append(chunk)
            return b"".join(chunks)

        readers = [asyncio.create_task(read(process.stdout)), asyncio.create_task(read(process.stderr))]

        async def stop() -> None:
            for reader in readers:
                reader.cancel()
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await asyncio.gather(*readers, return_exceptions=True)
            await process.wait()

        try:
            stdout, stderr = await asyncio.wait_for(asyncio.gather(*readers), timeout=self.command_timeout_seconds)
            await process.wait()
            return stdout, stderr
        except (TimeoutError, _OutputLimitExceeded) as exc:
            await stop()
            if isinstance(exc, TimeoutError):
                raise ReviewError(
                    f"{command} exceeded the {self.command_timeout_seconds:g}-second safety timeout"
                ) from exc
            raise ReviewError(f"{command} output exceeded the bounded safety limit") from exc
        except BaseException:
            await stop()
            raise

    async def _run(
        self, command: str, args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                command,
                *args,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ReviewError(f"could not start {command}: {exc}") from exc
        stdout, _stderr = await self._communicate(process, command)
        out = stdout.decode(errors="replace")
        if process.returncode != 0:
            raise ReviewError(f"{command} command failed with exit code {process.returncode}")
        return out

    @staticmethod
    def _env(host: str) -> dict[str, str]:
        return {**os.environ, "GH_HOST": host, "LC_ALL": "C"}

    @staticmethod
    def _local_git_env() -> dict[str, str]:
        env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
        return {
            **env,
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_KEY_1": "core.hooksPath",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_VALUE_0": "false",
            "GIT_CONFIG_VALUE_1": os.devnull,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }

    async def authenticated_user(self, host: str) -> str:
        return (await self._run("gh", ["api", "user", "--jq", ".login"], env=self._env(host))).strip()

    async def clone_or_fetch(self, ref: PrRef, base_dir: Path) -> Path:
        metadata = await self._api_json(ref, f"repos/{ref.owner}/{ref.repo}")
        return await self._clone_or_fetch_repository(ref.host, ref.owner, ref.repo, base_dir, metadata)

    async def clone_or_fetch_repository(self, host: str, owner: str, repo: str, base_dir: Path) -> Path:
        if host not in _ALLOWED_HOSTS or any(
            part in {".", ".."} or not _SEGMENT.fullmatch(part) for part in (host, owner, repo)
        ):
            raise ReviewError("repository contains an invalid or unsupported segment")
        metadata = await self._api_json_for_host(host, f"repos/{owner}/{repo}")
        return await self._clone_or_fetch_repository(host, owner, repo, base_dir, metadata)

    async def _clone_or_fetch_repository(self, host: str, owner: str, repo: str, base_dir: Path, metadata: Any) -> Path:
        repo_path = base_dir / host / owner / repo
        repository_kib = metadata.get("size") if isinstance(metadata, dict) else None
        if not isinstance(repository_kib, int) or repository_kib < 0:
            raise ReviewError("GitHub did not return a valid repository size")
        if repository_kib * 1024 > self.max_repository_bytes:
            raise ReviewError("repository exceeds the configured size safety limit")
        if repo_path.exists():
            if await asyncio.to_thread(self._directory_exceeds_limit, repo_path):
                raise ReviewError("local repository exceeds the configured size safety limit")
            await self._run("git", ["fetch", "origin", "--tags", "--prune"], cwd=repo_path)
        else:
            repo_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            await self._run(
                "gh",
                ["repo", "clone", f"{owner}/{repo}", os.fspath(repo_path), "--", "--filter=blob:none"],
                env=self._env(host),
            )
        if await asyncio.to_thread(self._directory_exceeds_limit, repo_path):
            raise ReviewError("local repository exceeds the configured size safety limit")
        return repo_path

    def _directory_exceeds_limit(self, root: Path) -> bool:
        total = 0
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                entries = os.scandir(directory)
            except OSError as exc:
                raise ReviewError(f"could not inspect local repository size: {exc}") from exc
            try:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
                    if total > self.max_repository_bytes:
                        return True
            except OSError as exc:
                raise ReviewError(f"could not inspect local repository size: {exc}") from exc
            finally:
                close = getattr(entries, "close", None)
                if close is not None:
                    close()
        return False

    async def fetch_head(self, repo_path: Path, pr_number: int) -> str:
        ref = f"refs/pull/{pr_number}/head"
        await self._run("git", ["fetch", "origin", f"{ref}:{ref}", "--force"], cwd=repo_path)
        return (await self._run("git", ["rev-parse", ref], cwd=repo_path)).strip()

    async def pr_shas(self, ref: PrRef) -> tuple[str, str]:
        metadata = await self._api_json(ref, f"repos/{ref.owner}/{ref.repo}/pulls/{ref.number}")
        if not isinstance(metadata, dict):
            raise ReviewError("GitHub did not return pull request metadata")
        base = metadata.get("base")
        head = metadata.get("head")
        base_sha = base.get("sha") if isinstance(base, dict) else None
        head_sha = head.get("sha") if isinstance(head, dict) else None
        if not isinstance(base_sha, str) or not _OBJECT_ID.fullmatch(base_sha):
            raise ReviewError("GitHub did not return a valid pull request base SHA")
        if not isinstance(head_sha, str) or not _OBJECT_ID.fullmatch(head_sha):
            raise ReviewError("GitHub did not return a valid pull request head SHA")
        return base_sha.lower(), head_sha.lower()

    async def materialize_comparison(self, repo_path: Path, *, base_sha: str, head_sha: str, operation_id: str) -> None:
        """Hydrate a blobless clone and pin both immutable review endpoints."""
        if (
            not _OBJECT_ID.fullmatch(base_sha)
            or not _OBJECT_ID.fullmatch(head_sha)
            or not _SEGMENT.fullmatch(operation_id)
        ):
            raise ReviewError("comparison contains an invalid object ID or operation ID")
        try:
            await self._run("git", ["cat-file", "-e", f"{base_sha}^{{commit}}"], cwd=repo_path)
        except ReviewError:
            await self._run("git", ["fetch", "origin", base_sha], cwd=repo_path)
        await self._run("git", ["cat-file", "-e", f"{head_sha}^{{commit}}"], cwd=repo_path)
        namespace = f"refs/pr-council-mcp/{operation_id}"
        try:
            await self._run("git", ["update-ref", f"{namespace}/base", base_sha], cwd=repo_path)
            await self._run("git", ["update-ref", f"{namespace}/head", head_sha], cwd=repo_path)
            # Producing the exact canonical comparison on the trusted host
            # forces promised blobs into the clone before no-network model use.
            await self._run("git", ["diff", f"{base_sha}...{head_sha}"], cwd=repo_path)
        except BaseException:
            for name in ("base", "head"):
                try:
                    await self._run("git", ["update-ref", "-d", f"{namespace}/{name}"], cwd=repo_path)
                except Exception:
                    pass
            raise

    async def release_comparison(self, repo_path: Path, operation_id: str) -> None:
        if not _SEGMENT.fullmatch(operation_id):
            raise ReviewError("operation ID contains unsupported characters")
        namespace = f"refs/pr-council-mcp/{operation_id}"
        failure: ReviewError | None = None
        for name in ("base", "head"):
            try:
                await self._run("git", ["update-ref", "-d", f"{namespace}/{name}"], cwd=repo_path)
            except ReviewError as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    async def create_worktree(self, repo_path: Path, worktree_path: Path, head_sha: str) -> None:
        worktree_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if worktree_path.exists():
            await self.remove_worktree(repo_path, worktree_path)
        await self._run("git", ["worktree", "add", "--detach", os.fspath(worktree_path), head_sha], cwd=repo_path)

    @staticmethod
    def linked_worktree_metadata_root(repo_path: Path, worktree_path: Path, repositories_root: Path) -> Path:
        """Validate a managed linked worktree and return its Git metadata root.

        This trust-boundary check intentionally lives in the review adapter,
        not the generic process sandbox. Granting the returned common Git
        directory read access also covers the linked worktree's own gitdir.
        """
        try:
            allowed_root = repositories_root.resolve(strict=True)
            repository = repo_path.resolve(strict=True)
            worktree = worktree_path.resolve(strict=True)
            if not allowed_root.is_dir() or not repository.is_relative_to(allowed_root):
                raise ReviewError("repository is outside the server-controlled repository root")
            common_dir = (repository / ".git").resolve(strict=True)
            if not common_dir.is_dir() or not common_dir.is_relative_to(repository):
                raise ReviewError("repository Git metadata is invalid")

            marker = worktree / ".git"
            prefix, raw_gitdir = marker.read_text(encoding="utf-8").strip().split(":", 1)
            if prefix != "gitdir":
                raise ReviewError("worktree Git marker is invalid")
            gitdir_path = Path(raw_gitdir.strip())
            if not gitdir_path.is_absolute():
                gitdir_path = marker.parent / gitdir_path
            gitdir = gitdir_path.resolve(strict=True)
            expected_worktrees = (common_dir / "worktrees").resolve(strict=True)
            if not gitdir.is_dir() or not gitdir.is_relative_to(expected_worktrees):
                raise ReviewError("worktree Git directory is outside the expected metadata root")
            commondir = (gitdir / "commondir").read_text(encoding="utf-8").strip()
            if (gitdir / commondir).resolve(strict=True) != common_dir:
                raise ReviewError("worktree common Git directory does not match the repository")
        except ReviewError:
            raise
        except (OSError, ValueError) as exc:
            raise ReviewError("could not validate linked-worktree Git metadata") from exc
        return common_dir

    async def validate_local_source(
        self,
        source_path: Path,
        allowed_root: Path,
        *,
        base_sha: str,
        head_sha: str,
    ) -> LocalSourceSnapshot:
        """Validate a caller checkout without modifying its refs or worktree.

        The returned snapshot grants the source and common Git metadata
        recursively while explicitly denying collapsed ignored paths. Ordinary
        untracked content is rejected by the cleanliness check.
        """
        if not _OBJECT_ID.fullmatch(base_sha) or not _OBJECT_ID.fullmatch(head_sha):
            raise ReviewError("local comparison contains an invalid object ID")
        try:
            sandbox_root = allowed_root.resolve(strict=True)
            repository = source_path.resolve(strict=True)
        except OSError as exc:
            raise ReviewError("could not resolve the local source repository") from exc
        if not sandbox_root.is_dir() or not repository.is_dir() or not repository.is_relative_to(sandbox_root):
            raise ReviewError("local source repository is outside the server working-directory sandbox")

        git_env = self._local_git_env()
        top_level_raw = (await self._run("git", ["rev-parse", "--show-toplevel"], cwd=repository, env=git_env)).strip()
        try:
            top_level = Path(top_level_raw).resolve(strict=True)
        except OSError as exc:
            raise ReviewError("local source path is not a valid Git repository root") from exc
        if top_level != repository:
            raise ReviewError("local source path must be the Git repository root")

        common_dir_raw = (
            await self._run("git", ["rev-parse", "--git-common-dir"], cwd=repository, env=git_env)
        ).strip()
        common_dir_path = Path(common_dir_raw)
        if not common_dir_path.is_absolute():
            common_dir_path = repository / common_dir_path
        try:
            common_dir = common_dir_path.resolve(strict=True)
        except OSError as exc:
            raise ReviewError("local source Git metadata cannot be resolved") from exc
        if not common_dir.is_dir() or not common_dir.is_relative_to(sandbox_root):
            raise ReviewError("local source Git metadata is outside the server working-directory sandbox")

        metadata_roots = {common_dir}

        async def validate_local_config(worktree: Path) -> None:
            raw_config = await self._run(
                "git",
                ["config", "--local", "--null", "--list"],
                cwd=worktree,
                env=git_env,
            )
            for entry in (value for value in raw_config.split("\0") if value):
                try:
                    key, value = entry.split("\n", 1)
                except ValueError as exc:
                    raise ReviewError("local source contains malformed Git configuration") from exc
                normalized_key = key.lower()
                if not _SAFE_LOCAL_CONFIG_KEY.fullmatch(normalized_key):
                    raise ReviewError(f'local source Git configuration key "{key}" is not safe for model access')
                if _REMOTE_URL_CONFIG_KEY.fullmatch(normalized_key):
                    parsed_url = urlparse(value)
                    if "::" in value or parsed_url.scheme not in {"", "file", "git", "http", "https", "ssh"}:
                        raise ReviewError("local source Git remote URL uses an executable or unsupported transport")
                    if parsed_url.password is not None or (
                        parsed_url.scheme in {"http", "https"}
                        and (parsed_url.username is not None or parsed_url.query or parsed_url.fragment)
                    ):
                        raise ReviewError("local source Git remote URL contains credentials or secret-bearing fields")

        await validate_local_config(repository)

        objects_dir_raw = (
            await self._run("git", ["rev-parse", "--git-path", "objects"], cwd=repository, env=git_env)
        ).strip()
        objects_dir_path = Path(objects_dir_raw)
        if not objects_dir_path.is_absolute():
            objects_dir_path = repository / objects_dir_path
        try:
            objects_dir = objects_dir_path.resolve(strict=True)
        except OSError as exc:
            raise ReviewError("local source Git object database cannot be resolved") from exc
        if not objects_dir.is_dir() or not objects_dir.is_relative_to(common_dir):
            raise ReviewError("local source Git object database is outside its common metadata directory")
        alternates_file = objects_dir / "info" / "alternates"
        if alternates_file.exists():
            try:
                resolved_alternates_file = alternates_file.resolve(strict=True)
                if not resolved_alternates_file.is_file() or not resolved_alternates_file.is_relative_to(objects_dir):
                    raise ReviewError("local source Git alternates file is outside its object database")
                alternate_paths = [
                    line for line in resolved_alternates_file.read_text(encoding="utf-8").splitlines() if line
                ]
                alternates = [
                    (Path(value) if Path(value).is_absolute() else objects_dir / value).resolve(strict=True)
                    for value in alternate_paths
                ]
            except ReviewError:
                raise
            except (OSError, UnicodeError) as exc:
                raise ReviewError("local source Git alternates cannot be validated") from exc
            if any(
                not alternate.is_dir()
                or not (alternate.is_relative_to(repository) or alternate.is_relative_to(common_dir))
                for alternate in alternates
            ):
                raise ReviewError(
                    "external Git alternates are unsupported by the local-source sandbox; "
                    "alternates must stay beneath the repository or common metadata directory"
                )

        resolved_head = (
            (await self._run("git", ["rev-parse", "--verify", "HEAD"], cwd=repository, env=git_env)).strip().lower()
        )
        if resolved_head != head_sha.lower():
            raise ReviewError("local source HEAD does not match the pull request head")
        status = await self._run(
            "git",
            ["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
            cwd=repository,
            env=git_env,
        )
        if status:
            raise ReviewError("local source worktree must be clean, including untracked files and submodules")

        await self._run("git", ["cat-file", "-e", f"{base_sha}^{{commit}}"], cwd=repository, env=git_env)
        await self._run("git", ["cat-file", "-e", f"{head_sha}^{{commit}}"], cwd=repository, env=git_env)
        await self._run("git", ["merge-base", base_sha, head_sha], cwd=repository, env=git_env)

        tracked_output = await self._run(
            "git",
            ["ls-files", "--stage", "--recurse-submodules", "-z"],
            cwd=repository,
            env=git_env,
        )
        tracked_entries: dict[str, str] = {}
        for record in tracked_output.split("\0"):
            if not record:
                continue
            try:
                metadata, name = record.split("\t", 1)
                mode = metadata.split(" ", 1)[0]
            except ValueError as exc:
                raise ReviewError("local source returned malformed tracked-path metadata") from exc
            tracked_entries[name] = mode
        for name, mode in tracked_entries.items():
            if mode != "120000":
                continue
            relative = Path(name)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ReviewError("local source contains an unsafe tracked path")
            candidate = repository.joinpath(*relative.parts)
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise ReviewError("local source contains an unreadable tracked symlink") from exc
            if not resolved.is_relative_to(repository):
                raise ReviewError("local source contains a tracked symlink outside the repository")

        excluded_paths: set[Path] = set()
        visited_worktrees: set[Path] = set()

        async def collect_ignored_paths(worktree: Path) -> None:
            if worktree in visited_worktrees:
                raise ReviewError("local source contains a recursive submodule worktree")
            visited_worktrees.add(worktree)
            if len(visited_worktrees) > _MAX_LOCAL_SUBMODULES:
                raise ReviewError("local source has too many initialized submodules to validate safely")

            ignored_output = await self._run(
                "git",
                [
                    "ls-files",
                    "--others",
                    "--ignored",
                    "--exclude-standard",
                    "--directory",
                    "--no-empty-directory",
                    "-z",
                ],
                cwd=worktree,
                env=git_env,
            )
            for name in (value for value in ignored_output.split("\0") if value):
                relative = Path(name.rstrip("/"))
                if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                    raise ReviewError("local source contains an unsafe ignored path")
                candidate = worktree.joinpath(*relative.parts)
                try:
                    candidate.lstat()
                except OSError as exc:
                    raise ReviewError("local source contains an unreadable ignored path") from exc
                excluded_paths.add(candidate)
                if len(excluded_paths) > _MAX_LOCAL_SANDBOX_EXCLUSIONS:
                    raise ReviewError("local source has too many ignored paths to sandbox safely")

            direct_entries = await self._run("git", ["ls-files", "--stage", "-z"], cwd=worktree, env=git_env)
            for record in (value for value in direct_entries.split("\0") if value):
                try:
                    metadata, name = record.split("\t", 1)
                    mode = metadata.split(" ", 1)[0]
                except ValueError as exc:
                    raise ReviewError("local source returned malformed tracked-path metadata") from exc
                if mode != "160000":
                    continue
                relative = Path(name)
                if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                    raise ReviewError("local source contains an unsafe submodule path")
                candidate = worktree.joinpath(*relative.parts)
                git_marker = candidate / ".git"
                if not git_marker.exists():
                    continue
                try:
                    submodule = candidate.resolve(strict=True)
                except OSError as exc:
                    raise ReviewError("local source submodule cannot be resolved") from exc
                if not submodule.is_dir() or not submodule.is_relative_to(repository):
                    raise ReviewError("local source submodule is outside the repository sandbox")
                submodule_top = Path(
                    (await self._run("git", ["rev-parse", "--show-toplevel"], cwd=submodule, env=git_env)).strip()
                ).resolve(strict=True)
                if submodule_top != submodule:
                    raise ReviewError("local source submodule path is not its Git repository root")
                submodule_common_raw = (
                    await self._run("git", ["rev-parse", "--git-common-dir"], cwd=submodule, env=git_env)
                ).strip()
                submodule_common_path = Path(submodule_common_raw)
                if not submodule_common_path.is_absolute():
                    submodule_common_path = submodule / submodule_common_path
                try:
                    submodule_common = submodule_common_path.resolve(strict=True)
                except OSError as exc:
                    raise ReviewError("local source submodule Git metadata cannot be resolved") from exc
                if not submodule_common.is_dir() or not (
                    submodule_common.is_relative_to(repository) or submodule_common.is_relative_to(common_dir)
                ):
                    raise ReviewError("local source submodule Git metadata is outside model-readable sandbox roots")
                metadata_roots.add(submodule_common)
                await validate_local_config(submodule)
                await collect_ignored_paths(submodule)

        await collect_ignored_paths(repository)

        for metadata_root in metadata_roots:
            for name in _SENSITIVE_GIT_METADATA_NAMES:
                candidate = metadata_root / name
                excluded_paths.add(candidate)
                if len(excluded_paths) > _MAX_LOCAL_SANDBOX_EXCLUSIONS:
                    raise ReviewError("local source has too many excluded paths to sandbox safely")

        return LocalSourceSnapshot(
            git_metadata_root=common_dir,
            excluded_paths=tuple(sorted(excluded_paths, key=os.fspath)),
        )

    async def resolve_tag(self, repo_path: Path, tag: str) -> str:
        if not _SEGMENT.fullmatch(tag):
            raise ReviewError("tag contains unsupported characters")
        return (await self._run("git", ["rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], cwd=repo_path)).strip()

    async def resolve_head(self, worktree_path: Path) -> str:
        return (await self._run("git", ["rev-parse", "--verify", "HEAD"], cwd=worktree_path)).strip()

    async def show_file(self, repo_path: Path, ref: str, path: str) -> str:
        if not _SEGMENT.fullmatch(ref) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ReviewError("Git object reference contains unsupported characters")
        return await self._run("git", ["show", f"{ref}:{path}"], cwd=repo_path)

    async def remove_worktree(self, repo_path: Path, worktree_path: Path) -> None:
        if not worktree_path.exists():
            return
        try:
            await self._run("git", ["worktree", "remove", "--force", os.fspath(worktree_path)], cwd=repo_path)
        except ReviewError:
            # A broken worktree registration should not strand the operation forever.
            await self._run("git", ["worktree", "prune"], cwd=repo_path)
            await self._remove_orphaned_worktree(repo_path, worktree_path)

    @staticmethod
    async def _remove_orphaned_worktree(repo_path: Path, worktree_path: Path) -> None:
        marker = worktree_path / ".git"
        if not marker.is_file() or worktree_path.is_symlink():
            return
        try:
            marker_text = await asyncio.to_thread(marker.read_text)
            prefix, raw_git_dir = marker_text.strip().split(":", 1)
            git_dir = Path(raw_git_dir.strip()).resolve(strict=False)
            expected_root = (repo_path.resolve() / ".git" / "worktrees").resolve(strict=False)
        except (OSError, ValueError):
            return
        if prefix != "gitdir" or not git_dir.is_relative_to(expected_root):
            return
        await asyncio.to_thread(shutil.rmtree, worktree_path)

    async def diff(self, repo_path: Path, ref: PrRef) -> str:
        return await self._run(
            "gh",
            ["pr", "diff", str(ref.number), "--repo", ref.repo_key],
            cwd=repo_path,
            env=self._env(ref.host),
        )

    async def _api_json(self, ref: PrRef, endpoint: str) -> Any:
        return await self._api_json_for_host(ref.host, endpoint)

    async def _api_json_for_host(self, host: str, endpoint: str) -> Any:
        raw = await self._run("gh", ["api", endpoint, "--paginate"], env=self._env(host))
        documents = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(documents) == 1:
            return documents[0]
        flattened: list[object] = []
        for document in documents:
            flattened.extend(document if isinstance(document, list) else [document])
        return flattened

    async def review_already_posted(self, ref: PrRef, marker: str, authenticated_user: str) -> bool:
        reviews = await self._api_json(ref, f"repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/reviews")
        if not isinstance(reviews, list):
            return False
        return any(
            isinstance(review, dict)
            and isinstance(review.get("user"), dict)
            and review["user"].get("login") == authenticated_user
            and marker in str(review.get("body", ""))
            for review in reviews
        )

    async def post_comment_review(
        self,
        ref: PrRef,
        *,
        head_sha: str,
        summary_body: str,
        comments: list[PreviewComment],
    ) -> None:
        body = {
            "commit_id": head_sha,
            "event": "COMMENT",
            "body": summary_body,
            "comments": [
                {"path": comment.path, "line": comment.line, "side": "RIGHT", "body": comment.body}
                for comment in comments
            ],
        }
        with tempfile.TemporaryDirectory(prefix="pr-council-mcp-gh-") as directory:
            body_path = Path(directory) / "review.json"
            body_path.write_text(json.dumps(body))
            await self._run(
                "gh",
                [
                    "api",
                    "--method",
                    "POST",
                    f"repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/reviews",
                    "--input",
                    os.fspath(body_path),
                ],
                env=self._env(ref.host),
            )
