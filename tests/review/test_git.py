import subprocess
import sys
from pathlib import Path

import pytest

import pr_council.review.git as git_module
from pr_council.review.git import GitHubCli, parse_pr_url
from pr_council.review.models import PrRef, ReviewError


def test_linked_worktree_metadata_root_validates_and_returns_common_dir(tmp_path: Path):
    repositories = tmp_path / "repos"
    repository = repositories / "github.com" / "acme" / "repo"
    common = repository / ".git"
    gitdir = common / "worktrees" / "operation"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n")
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n")

    assert GitHubCli.linked_worktree_metadata_root(repository, worktree, repositories) == common.resolve()


def test_linked_worktree_metadata_root_rejects_repository_outside_controlled_root(tmp_path: Path):
    repositories = tmp_path / "repos"
    repositories.mkdir()
    repository = tmp_path / "outside"
    (repository / ".git").mkdir(parents=True)
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    with pytest.raises(ReviewError, match="outside the server-controlled"):
        GitHubCli.linked_worktree_metadata_root(repository, worktree, repositories)


def test_parse_pr_url_returns_canonical_ref():
    ref = parse_pr_url("https://github.com/acme/widget/pull/42")
    assert ref.repo_key == "github.com/acme/widget"
    assert ref.number == 42


def test_parse_pr_url_honors_caller_supplied_allowed_hosts():
    ref = parse_pr_url("https://example.test/o/r/pull/3", allowed_hosts={"example.test"})
    assert ref == PrRef(host="example.test", owner="o", repo="r", number=3)


@pytest.mark.parametrize(
    "value",
    [
        "http://github.com/acme/widget/pull/42",
        "https://evil.example/acme/widget/pull/42",
        "https://github.example/acme/widget/pull/42",
        "https://github.com:8443/acme/widget/pull/42",
        "https://github.com/acme/widget/pull/42?foo=bar",
        "https://github.com/acme/widget/pull/42#frag",
    ],
)
def test_parse_pr_url_rejects_non_https_wrong_host_or_extra_url_parts(value):
    with pytest.raises(ReviewError, match="HTTPS URL on an allowed GitHub host"):
        parse_pr_url(value)


@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/acme/widget/issues/42",
        "https://github.com/acme/widget/pull/notanumber",
        "https://github.com/acme/widget/pull",
    ],
)
def test_parse_pr_url_rejects_malformed_pull_path(value):
    with pytest.raises(ReviewError, match="form https://host/owner/repo/pull/number"):
        parse_pr_url(value)


@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/acme/wid$get/pull/42",
        "https://github.com/acme/../pull/42",
        "https://github.com/acme/./pull/42",
    ],
)
def test_parse_pr_url_rejects_invalid_repository_segment(value):
    with pytest.raises(ReviewError, match="invalid repository segment"):
        parse_pr_url(value)


async def test_cli_streams_output_into_a_hard_memory_cap():
    cli = GitHubCli(max_output_bytes=1_024)

    with pytest.raises(ReviewError, match="output exceeded"):
        await cli._run(sys.executable, ["-c", "import sys; sys.stdout.write('x' * 100000)"])


async def test_cli_kills_commands_that_exceed_the_timeout():
    cli = GitHubCli(command_timeout_seconds=0.01)

    with pytest.raises(ReviewError, match="safety timeout"):
        await cli._run(sys.executable, ["-c", "import time; time.sleep(10)"])


async def test_cli_failure_does_not_surface_untrusted_process_output():
    secret = "unrecognized-secret-format-12345"
    cli = GitHubCli()

    with pytest.raises(ReviewError, match="exit code 7") as raised:
        await cli._run(
            sys.executable,
            ["-c", f"import sys; sys.stderr.write({secret!r}); raise SystemExit(7)"],
        )

    assert secret not in str(raised.value)


async def test_clone_rejects_repository_metadata_over_the_size_limit(monkeypatch, tmp_path):
    cli = GitHubCli(max_repository_bytes=1_024)
    commands = []

    async def metadata(*args, **kwargs):
        return {"size": 2}

    async def run(*args, **kwargs):
        commands.append((args, kwargs))
        return ""

    monkeypatch.setattr(cli, "_api_json", metadata)
    monkeypatch.setattr(cli, "_run", run)

    with pytest.raises(ReviewError, match="repository exceeds"):
        await cli.clone_or_fetch(PrRef(host="github.com", owner="acme", repo="large", number=1), tmp_path)
    assert commands == []


def test_repository_size_scan_fails_closed_when_a_directory_is_unreadable(monkeypatch, tmp_path):
    cli = GitHubCli()

    def fail(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(git_module.os, "scandir", fail)

    with pytest.raises(ReviewError, match="could not inspect"):
        cli._directory_exceeds_limit(tmp_path)


def test_repository_size_scan_fails_closed_when_an_entry_cannot_be_inspected(monkeypatch, tmp_path):
    cli = GitHubCli()

    class UnreadableEntry:
        def is_dir(self, *, follow_symlinks):
            raise OSError("permission denied")

    monkeypatch.setattr(git_module.os, "scandir", lambda _path: [UnreadableEntry()])

    with pytest.raises(ReviewError, match="could not inspect"):
        cli._directory_exceeds_limit(tmp_path)


@pytest.mark.parametrize(
    "ref, path",
    [
        ("main", "/etc/passwd"),
        ("main", "../secret"),
        ("bad ref!", "file.py"),
    ],
)
async def test_show_file_rejects_traversal_and_bad_ref_before_subprocess(monkeypatch, ref, path):
    # The line 208 guard runs before _run() is ever reached; make _run explode
    # so a test that slipped past the guard would fail loudly instead of
    # silently spawning git.
    cli = GitHubCli()

    async def forbidden_run(*args, **kwargs):
        raise AssertionError("guard must reject before spawning a subprocess")

    monkeypatch.setattr(cli, "_run", forbidden_run)

    with pytest.raises(ReviewError, match="Git object reference contains unsupported characters"):
        await cli.show_file(Path("/nonexistent/repo"), ref, path)


async def test_resolve_tag_rejects_bad_segment_before_subprocess(monkeypatch):
    cli = GitHubCli()

    async def forbidden_run(*args, **kwargs):
        raise AssertionError("guard must reject before spawning a subprocess")

    monkeypatch.setattr(cli, "_run", forbidden_run)

    with pytest.raises(ReviewError, match="tag contains unsupported characters"):
        await cli.resolve_tag(Path("/nonexistent/repo"), "v1.0;rm -rf")


@pytest.mark.parametrize(
    "host, owner, repo",
    [
        ("evil.example", "acme", "widget"),
        ("github.com", "bad owner!", "widget"),
        ("github.com", "acme", ".."),
    ],
)
async def test_clone_or_fetch_repository_rejects_bad_host_or_segment_before_subprocess(
    monkeypatch, tmp_path, host, owner, repo
):
    # The line 134-137 guard runs before _api_json_for_host (and thus _run);
    # trip both to AssertionError so a leaked case fails loudly.
    cli = GitHubCli()

    async def forbidden(*args, **kwargs):
        raise AssertionError("guard must reject before spawning a subprocess")

    monkeypatch.setattr(cli, "_api_json_for_host", forbidden)
    monkeypatch.setattr(cli, "_run", forbidden)

    with pytest.raises(ReviewError, match="repository contains an invalid or unsupported segment"):
        await cli.clone_or_fetch_repository(host, owner, repo, tmp_path)


async def test_post_comment_review_uses_the_pr_council_tempdir_prefix(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    real_temporary_directory = git_module.tempfile.TemporaryDirectory

    def capture(*args, **kwargs):
        captured["prefix"] = kwargs.get("prefix")
        return real_temporary_directory(dir=tmp_path)

    monkeypatch.setattr(git_module.tempfile, "TemporaryDirectory", capture)

    cli = GitHubCli()

    async def noop_run(*args, **kwargs):
        return ""

    monkeypatch.setattr(cli, "_run", noop_run)

    await cli.post_comment_review(
        PrRef(host="github.com", owner="acme", repo="widget", number=42),
        head_sha="deadbeef",
        summary_body="summary",
        comments=[],
    )

    assert captured["prefix"] == "pr-council-mcp-gh-"


async def test_remove_worktree_deletes_a_validated_orphan_after_git_cleanup_fails(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    git_dir = repo / ".git" / "worktrees" / "orphan"
    git_dir.mkdir(parents=True)
    worktree = tmp_path / "worktrees" / "operation-id"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n")
    (worktree / "large.bin").write_bytes(b"content")
    cli = GitHubCli()
    calls = 0

    async def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReviewError("not a registered worktree")
        return ""

    monkeypatch.setattr(cli, "_run", run)

    await cli.remove_worktree(repo, worktree)

    assert not worktree.exists()
    assert calls == 2


async def test_pr_shas_validates_full_base_and_head_object_ids(monkeypatch):
    cli = GitHubCli()

    async def metadata(*args):
        return {"base": {"sha": "A" * 40}, "head": {"sha": "b" * 40}}

    monkeypatch.setattr(cli, "_api_json", metadata)

    assert await cli.pr_shas(PrRef(host="github.com", owner="acme", repo="widget", number=42)) == (
        "a" * 40,
        "b" * 40,
    )


async def test_materialize_comparison_pins_refs_and_hydrates_exact_diff(monkeypatch, tmp_path):
    cli = GitHubCli()
    calls = []

    async def run(command, args, **kwargs):
        calls.append((command, args))
        return ""

    monkeypatch.setattr(cli, "_run", run)
    await cli.materialize_comparison(
        tmp_path,
        base_sha="a" * 40,
        head_sha="b" * 40,
        operation_id="operation-1",
    )

    assert ("git", ["diff", f"{'a' * 40}...{'b' * 40}"]) in calls
    assert ("git", ["update-ref", "refs/pr-council-mcp/operation-1/base", "a" * 40]) in calls
    assert ("git", ["update-ref", "refs/pr-council-mcp/operation-1/head", "b" * 40]) in calls


async def test_release_comparison_deletes_both_pinned_refs(monkeypatch, tmp_path):
    cli = GitHubCli()
    calls = []

    async def run(command, args, **kwargs):
        calls.append((command, args))
        return ""

    monkeypatch.setattr(cli, "_run", run)
    await cli.release_comparison(tmp_path, "operation-1")

    assert calls == [
        ("git", ["update-ref", "-d", "refs/pr-council-mcp/operation-1/base"]),
        ("git", ["update-ref", "-d", "refs/pr-council-mcp/operation-1/head"]),
    ]


async def test_validate_local_source_accepts_clean_matching_checkout_without_mutating_it(monkeypatch, tmp_path):
    sandbox = tmp_path / "workspace"
    repository = sandbox / "repo"
    metadata = repository / ".git"
    (metadata / "objects").mkdir(parents=True)
    base_sha = "a" * 40
    head_sha = "b" * 40
    calls = []
    environments = []
    cli = GitHubCli()

    async def run(command, args, **kwargs):
        calls.append((command, args, kwargs["cwd"]))
        environments.append(kwargs["env"])
        if args == ["rev-parse", "--show-toplevel"]:
            return f"{repository}\n"
        if args == ["rev-parse", "--git-common-dir"]:
            return ".git\n"
        if args == ["rev-parse", "--git-path", "objects"]:
            return ".git/objects\n"
        if args == ["rev-parse", "--verify", "HEAD"]:
            return f"{head_sha}\n"
        return ""

    monkeypatch.setattr(cli, "_run", run)

    snapshot = await cli.validate_local_source(
        repository,
        sandbox,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    assert snapshot.git_metadata_root == metadata.resolve()
    assert metadata / "hooks" in snapshot.excluded_paths
    assert metadata / "logs" in snapshot.excluded_paths
    assert not any(args[0] in {"fetch", "update-ref", "worktree"} for _, args, _ in calls)
    assert ("git", ["cat-file", "-e", f"{base_sha}^{{commit}}"], repository.resolve()) in calls
    assert ("git", ["merge-base", base_sha, head_sha], repository.resolve()) in calls
    assert all(env["GIT_OPTIONAL_LOCKS"] == "0" for env in environments)
    assert all(env["GIT_NO_LAZY_FETCH"] == "1" for env in environments)


async def test_validate_local_source_against_real_git_repository(tmp_path):
    repository = tmp_path / "repo"

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            env={
                **git_module.os.environ,
                "GIT_AUTHOR_NAME": "Test User",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "Test User",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "commit.gpgsign",
                "GIT_CONFIG_VALUE_0": "false",
            },
        ).stdout.strip()

    repository.mkdir()
    git("init")
    (repository / "example.txt").write_text("base\n")
    git("add", "example.txt")
    git("commit", "-m", "base")
    base_sha = git("rev-parse", "HEAD")
    (repository / "example.txt").write_text("head\n")
    git("commit", "-am", "head")
    head_sha = git("rev-parse", "HEAD")

    snapshot = await GitHubCli().validate_local_source(
        repository,
        tmp_path,
        base_sha=base_sha,
        head_sha=head_sha,
    )

    assert snapshot.git_metadata_root == (repository / ".git").resolve()
    assert repository / ".git" / "hooks" in snapshot.excluded_paths
    assert repository / ".git" / "logs" in snapshot.excluded_paths
    assert git("status", "--porcelain=v1") == ""


async def test_validate_local_source_rejects_unsafe_repository_config(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    git_env = {
        **git_module.os.environ,
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test User",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "commit.gpgsign",
        "GIT_CONFIG_VALUE_0": "false",
    }

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repository, check=True, capture_output=True, text=True, env=git_env
        ).stdout.strip()

    git("init")
    (repository / "example.txt").write_text("head\n")
    git("add", "example.txt")
    git("commit", "-m", "head")
    head_sha = git("rev-parse", "HEAD")
    git("config", "http.https://github.com/.extraHeader", "Authorization: bearer secret")

    with pytest.raises(ReviewError, match="configuration key.*is not safe"):
        await GitHubCli().validate_local_source(
            repository,
            tmp_path,
            base_sha=head_sha,
            head_sha=head_sha,
        )


async def test_validate_local_source_excludes_collapsed_ignored_paths(tmp_path):
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
                **git_module.os.environ,
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
    (repository / ".gitignore").write_text(".mcp.json\nignored/\n")
    (repository / "tracked.txt").write_text("tracked\n")
    git("add", ".gitignore", "tracked.txt")
    git("commit", "-m", "head")
    head_sha = git("rev-parse", "HEAD")
    (repository / ".mcp.json").write_text('{"secret": true}\n')
    ignored = repository / "ignored"
    ignored.mkdir()
    (ignored / "many-files.txt").write_text("secret\n")

    snapshot = await GitHubCli().validate_local_source(
        repository,
        tmp_path,
        base_sha=head_sha,
        head_sha=head_sha,
    )

    assert repository / ".mcp.json" in snapshot.excluded_paths
    assert ignored in snapshot.excluded_paths


async def test_validate_local_source_excludes_ignored_paths_in_initialized_submodules(tmp_path):
    submodule_source = tmp_path / "submodule-source"
    repository = tmp_path / "repo"
    submodule_source.mkdir()
    repository.mkdir()
    git_env = {
        **git_module.os.environ,
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test User",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "commit.gpgsign",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_ALLOW_PROTOCOL": "file",
    }

    def git(cwd, *args):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=git_env
        ).stdout.strip()

    git(submodule_source, "init")
    (submodule_source / ".gitignore").write_text("secret.env\n")
    (submodule_source / "tracked.txt").write_text("tracked\n")
    git(submodule_source, "add", ".gitignore", "tracked.txt")
    git(submodule_source, "commit", "-m", "submodule head")

    git(repository, "init")
    git(repository, "submodule", "add", str(submodule_source), "modules/child")
    git(repository, "commit", "-m", "parent head")
    head_sha = git(repository, "rev-parse", "HEAD")
    ignored_secret = repository / "modules" / "child" / "secret.env"
    ignored_secret.write_text("submodule secret\n")
    assert git(repository, "status", "--porcelain=v1", "--ignore-submodules=none") == ""

    snapshot = await GitHubCli().validate_local_source(
        repository,
        tmp_path,
        base_sha=head_sha,
        head_sha=head_sha,
    )

    assert ignored_secret in snapshot.excluded_paths


async def test_validate_local_source_rejects_tracked_symlink_outside_repository(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n")

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            env={
                **git_module.os.environ,
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
    (repository / "escape").symlink_to(outside)
    git("add", "escape")
    git("commit", "-m", "head")
    head_sha = git("rev-parse", "HEAD")

    with pytest.raises(ReviewError, match="tracked symlink outside"):
        await GitHubCli().validate_local_source(
            repository,
            tmp_path,
            base_sha=head_sha,
            head_sha=head_sha,
        )


async def test_validate_local_source_accepts_tracked_symlink_to_internal_directory(tmp_path):
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
                **git_module.os.environ,
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
    docs = repository / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("guide\n")
    (repository / "guide").symlink_to(docs, target_is_directory=True)
    git("add", "docs/guide.md", "guide")
    git("commit", "-m", "head")
    head_sha = git("rev-parse", "HEAD")

    snapshot = await GitHubCli().validate_local_source(
        repository,
        tmp_path,
        base_sha=head_sha,
        head_sha=head_sha,
    )

    assert snapshot.git_metadata_root == (repository / ".git").resolve()


async def test_validate_local_source_rejects_external_git_alternate(monkeypatch, tmp_path):
    repository = tmp_path / "repo"
    objects = repository / ".git" / "objects"
    alternates_file = objects / "info" / "alternates"
    alternates_file.parent.mkdir(parents=True)
    external_objects = tmp_path / "external-objects"
    external_objects.mkdir()
    alternates_file.write_text(f"{external_objects}\n")
    head_sha = "b" * 40
    cli = GitHubCli()

    async def run(command, args, **kwargs):
        if args == ["rev-parse", "--show-toplevel"]:
            return f"{repository}\n"
        if args == ["rev-parse", "--git-common-dir"]:
            return ".git\n"
        if args == ["rev-parse", "--git-path", "objects"]:
            return ".git/objects\n"
        if args == ["rev-parse", "--verify", "HEAD"]:
            return f"{head_sha}\n"
        return ""

    monkeypatch.setattr(cli, "_run", run)

    with pytest.raises(ReviewError, match="external Git alternates are unsupported"):
        await cli.validate_local_source(
            repository,
            tmp_path,
            base_sha="a" * 40,
            head_sha=head_sha,
        )


@pytest.mark.parametrize(
    ("head", "status", "message"),
    [
        ("c" * 40, "", "HEAD does not match"),
        ("b" * 40, "?? generated.txt\n", "worktree must be clean"),
    ],
)
async def test_validate_local_source_rejects_wrong_revision_or_dirty_tree(monkeypatch, tmp_path, head, status, message):
    sandbox = tmp_path / "workspace"
    repository = sandbox / "repo"
    (repository / ".git" / "objects").mkdir(parents=True)
    cli = GitHubCli()

    async def run(command, args, **kwargs):
        if args == ["rev-parse", "--show-toplevel"]:
            return f"{repository}\n"
        if args == ["rev-parse", "--git-common-dir"]:
            return ".git\n"
        if args == ["rev-parse", "--git-path", "objects"]:
            return ".git/objects\n"
        if args == ["rev-parse", "--verify", "HEAD"]:
            return f"{head}\n"
        if args[0] == "status":
            return status
        return ""

    monkeypatch.setattr(cli, "_run", run)

    with pytest.raises(ReviewError, match=message):
        await cli.validate_local_source(
            repository,
            sandbox,
            base_sha="a" * 40,
            head_sha="b" * 40,
        )


async def test_validate_local_source_rejects_repository_outside_sandbox_before_git(monkeypatch, tmp_path):
    sandbox = tmp_path / "workspace"
    sandbox.mkdir()
    repository = tmp_path / "outside"
    repository.mkdir()
    cli = GitHubCli()

    async def forbidden_run(*args, **kwargs):
        raise AssertionError("sandbox boundary must reject before spawning git")

    monkeypatch.setattr(cli, "_run", forbidden_run)

    with pytest.raises(ReviewError, match="outside the server working-directory sandbox"):
        await cli.validate_local_source(
            repository,
            sandbox,
            base_sha="a" * 40,
            head_sha="b" * 40,
        )


async def test_diff_targets_the_pr_repository_explicitly(monkeypatch, tmp_path):
    cli = GitHubCli()
    captured = {}

    async def run(command, args, **kwargs):
        captured.update(command=command, args=args, kwargs=kwargs)
        return "diff"

    monkeypatch.setattr(cli, "_run", run)
    ref = PrRef(host="github.com", owner="acme", repo="widget", number=42)

    assert await cli.diff(tmp_path, ref) == "diff"
    assert captured["args"] == ["pr", "diff", "42", "--repo", "github.com/acme/widget"]
