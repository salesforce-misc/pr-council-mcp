"""The review sandbox profile against the real platform sandbox, with no mocking of OS boundaries."""

import os
import subprocess

from localmcp.sandbox import create_sandbox

from pr_council.review.git import GitHubCli
from pr_council.workflows.review.graph import _review_sandbox_profile


async def test_local_source_profile_allows_inspection_and_hides_excluded_paths(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repository,
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

    git("init")
    (repository / ".gitignore").write_text(".env\nbuild/\n")
    (repository / "app.py").write_text("print('base')\n")
    git("add", ".gitignore", "app.py")
    git("commit", "-m", "base")
    base_sha = git("rev-parse", "HEAD")
    (repository / "app.py").write_text("print('head')\n")
    git("commit", "-am", "head")
    head_sha = git("rev-parse", "HEAD")
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

    inspected = await sandbox.run(f"git diff --stat {base_sha}...{head_sha} && git show {head_sha}:app.py")
    assert inspected.exit_code == 0, inspected
    assert "app.py | 2 +-" in inspected.stdout
    assert "print('head')" in inspected.stdout
    for command in ("cat .env", "cat build/output", "ls .git/hooks", "printf changed > app.py"):
        result = await sandbox.run(command)
        assert result.exit_code != 0, result
        assert "ignored-secret" not in result.stdout + result.stderr
    assert (repository / "app.py").read_text() == "print('head')\n"
