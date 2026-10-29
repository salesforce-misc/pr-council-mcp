"""The review sandbox profile against the real platform sandbox, with no mocking of OS boundaries."""

import os
import subprocess

from localmcp.sandbox import create_sandbox

from pr_council.review.git import GitHubCli
from pr_council.workflows.review.graph import _review_sandbox_profile


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test User",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test User",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "commit.gpgsign",
            "GIT_CONFIG_VALUE_0": "false",
        },
    ).stdout.strip()


def _commit_base_and_head(repository, *, gitignore=""):
    """Create a repository whose head commit changes app.py, returning the base and head SHAs."""
    repository.mkdir(parents=True)
    _git(repository, "init")
    (repository / ".gitignore").write_text(gitignore)
    (repository / "app.py").write_text("print('base')\n")
    _git(repository, "add", ".gitignore", "app.py")
    _git(repository, "commit", "-m", "base")
    base_sha = _git(repository, "rev-parse", "HEAD")
    (repository / "app.py").write_text("print('head')\n")
    _git(repository, "commit", "-am", "head")
    return base_sha, _git(repository, "rev-parse", "HEAD")


async def _assert_inspects_diff(sandbox, base_sha, head_sha):
    inspected = await sandbox.run(f"git diff --stat {base_sha}...{head_sha} && git show {head_sha}:app.py")
    assert inspected.exit_code == 0, inspected
    assert "app.py | 2 +-" in inspected.stdout
    assert "print('head')" in inspected.stdout


def _assert_denied(result, secret):
    # Exit code 126 is the shell check's rejection; the denial must come from the sandbox itself.
    assert result.exit_code not in (0, 126), result
    assert secret not in result.stdout + result.stderr


async def test_local_source_profile_allows_inspection_and_hides_excluded_paths(tmp_path):
    repository = tmp_path / "repo"
    base_sha, head_sha = _commit_base_and_head(repository, gitignore=".env\nbuild/\n")
    (repository / ".env").write_text("ignored-secret\n")
    (repository / "build").mkdir()
    (repository / "build" / "output").write_text("ignored-secret\n")

    snapshot = await GitHubCli().validate_local_source(repository, tmp_path, base_sha=base_sha, head_sha=head_sha)
    # Most sensitive Git metadata names do not exist in a fresh repository; both platforms must accept them anyway.
    assert any(not path.exists() for path in snapshot.excluded_paths)
    sandbox = create_sandbox(
        _review_sandbox_profile(
            str(repository), str(snapshot.git_metadata_root), [str(path) for path in snapshot.excluded_paths]
        )
    )

    await _assert_inspects_diff(sandbox, base_sha, head_sha)
    for command in ("cat .env", "cat build/output", "ls .git/hooks", "printf changed > app.py"):
        _assert_denied(await sandbox.run(command), "ignored-secret")
    assert (repository / "app.py").read_text() == "print('head')\n"


async def test_managed_worktree_profile_reads_shared_git_metadata_read_only(tmp_path):
    origin = tmp_path / "origin"
    base_sha, head_sha = _commit_base_and_head(origin)
    repos_dir = tmp_path / "repos"
    repo_path = repos_dir / "github.com" / "owner" / "repo"
    _git(tmp_path, "clone", "--quiet", str(origin), str(repo_path))
    worktree_path = tmp_path / "worktrees" / "operation"
    (tmp_path / "outside").write_text("outside-secret\n")

    git = GitHubCli()
    await git.create_worktree(repo_path, worktree_path, head_sha)
    git_metadata_root = git.linked_worktree_metadata_root(repo_path, worktree_path, repos_dir)
    # The linked worktree's history lives only in the clone's Git directory, outside the worktree root.
    assert not git_metadata_root.is_relative_to(worktree_path.resolve())
    sandbox = create_sandbox(_review_sandbox_profile(str(worktree_path), str(git_metadata_root)))

    await _assert_inspects_diff(sandbox, base_sha, head_sha)
    for command in (
        f"cat {tmp_path / 'outside'}",
        "printf changed > app.py",
        f"git update-ref refs/heads/escaped {head_sha}",
    ):
        _assert_denied(await sandbox.run(command), "outside-secret")
    assert (worktree_path / "app.py").read_text() == "print('head')\n"
    assert "refs/heads/escaped" not in _git(repo_path, "show-ref")
