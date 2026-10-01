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

    with pytest.raises(ReviewError, match="allowed GitHub host"):
        parse_pr_url("https://github.com/o/r/pull/3", allowed_hosts=set())


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


async def test_clone_accepts_configured_enterprise_host(monkeypatch, tmp_path):
    cli = GitHubCli(allowed_hosts=["github.com", "github.enterprise.example"])
    ref = PrRef(host="github.enterprise.example", owner="acme", repo="repo", number=1)
    calls = []

    async def metadata(*args):
        calls.append(args)
        return {"size": 0}

    async def clone(*args):
        calls.append(args)
        return tmp_path / "repo"

    monkeypatch.setattr(cli, "_api_json", metadata)
    monkeypatch.setattr(cli, "_clone_or_fetch_repository", clone)

    assert await cli.clone_or_fetch(ref, tmp_path) == tmp_path / "repo"
    assert calls[0][0] == ref

    with pytest.raises(ReviewError, match="invalid or unsupported segment"):
        await cli.clone_or_fetch(PrRef(host="other.example", owner="acme", repo="repo", number=1), tmp_path)
    assert len(calls) == 2

    async def metadata_for_host(*args):
        calls.append(args)
        return {"size": 0}

    monkeypatch.setattr(cli, "_api_json_for_host", metadata_for_host)
    assert await cli.clone_or_fetch_repository(ref.host, ref.owner, ref.repo, tmp_path) == tmp_path / "repo"
    assert calls[2][0] == ref.host


async def test_cli_rejects_revoked_host_before_github_requests(monkeypatch):
    cli = GitHubCli(allowed_hosts=["github.com"])
    ref = PrRef(host="github.enterprise.example", owner="acme", repo="repo", number=1)

    async def forbidden_run(*args, **kwargs):
        raise AssertionError("GitHub CLI must not be called for a revoked host")

    monkeypatch.setattr(cli, "_run", forbidden_run)

    with pytest.raises(ReviewError, match="host is not allowed"):
        await cli.authenticated_user(ref.host)
    with pytest.raises(ReviewError, match="host is not allowed"):
        await cli.pr_shas(ref)
    with pytest.raises(ReviewError, match="host is not allowed"):
        await cli.post_comment_review(ref, head_sha="a" * 40, summary_body="summary", comments=[])


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


def _ref(host="github.com", owner="acme", repo="repo"):
    return PrRef(host=host, owner=owner, repo=repo, number=1)


def test_account_for_prefers_the_most_specific_case_insensitive_match():
    cli = GitHubCli(
        accounts={"github.com": "host-user", "github.com/Acme": "owner-user", "github.com/acme/special": "repo-user"}
    )

    assert cli.account_for(_ref(owner="ACME", repo="Special")) == "repo-user"
    assert cli.account_for(_ref(owner="acme", repo="other")) == "owner-user"
    assert cli.account_for(_ref(owner="acme-labs")) == "host-user"
    assert GitHubCli().account_for(_ref()) is None


def test_bind_rejects_a_host_that_is_not_allowed():
    with pytest.raises(ReviewError, match="host is not allowed"):
        GitHubCli().bind("other.example", "user")


def _token_stub(monkeypatch, cli, calls, token="gho_mapped"):
    real_run = cli._run

    async def run(command, args, **kwargs):
        if command == "gh" and args[:2] == ["auth", "token"]:
            calls.append((args, kwargs["env"]))
            return f"{token}\n"
        if command == "gh":
            calls.append((args, kwargs["env"]))
            return "mapped-user\n"
        return await real_run(command, args, **kwargs)

    monkeypatch.setattr(git_module.GitHubCli, "_run", lambda self, *a, **k: run(*a, **k))


async def test_unbound_calls_keep_the_inherited_environment(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "inherited")
    cli = GitHubCli()
    calls = []
    _token_stub(monkeypatch, cli, calls)

    await cli.authenticated_user("github.com")

    assert [args for args, _env in calls] == [["api", "user", "--jq", ".login"]]
    assert calls[0][1]["GH_TOKEN"] == "inherited"
    assert await cli.bind("github.com", None)._git_network_env() is None


async def test_bound_calls_get_only_the_mapped_account_token(monkeypatch):
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        monkeypatch.setenv(name, "inherited")
    cli = GitHubCli().bind("github.com", "mapped-user")
    calls = []
    _token_stub(monkeypatch, cli, calls)

    assert await cli.authenticated_user("github.com") == "mapped-user"

    (token_args, token_env), (api_args, api_env) = calls
    assert token_args == ["auth", "token", "--hostname", "github.com", "--user", "mapped-user"]
    # An inherited token would make gh ignore --user, so it must not reach the lookup.
    assert not {"GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"} & set(token_env)
    assert api_args == ["api", "user", "--jq", ".login"]
    assert api_env["GH_TOKEN"] == api_env["GH_ENTERPRISE_TOKEN"] == "gho_mapped"
    assert "GITHUB_TOKEN" not in api_env and "GITHUB_ENTERPRISE_TOKEN" not in api_env


async def test_bound_calls_reject_another_host(monkeypatch):
    cli = GitHubCli(allowed_hosts=["github.com", "ghe.example"]).bind("github.com", "mapped-user")

    with pytest.raises(ReviewError, match="does not match the account-bound adapter"):
        await cli.authenticated_user("ghe.example")


async def test_missing_account_token_names_the_account(monkeypatch):
    cli = GitHubCli().bind("github.com", "mapped-user")

    async def fail(self, command, args, **kwargs):
        raise ReviewError("gh command failed with exit code 1")

    monkeypatch.setattr(git_module.GitHubCli, "_run", fail)

    with pytest.raises(ReviewError, match='no token for GitHub account "mapped-user" on github.com'):
        await cli.authenticated_user("github.com")


async def test_bound_git_fetches_rewrite_ssh_remotes_to_https_with_gh_credentials(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "git@github.com:acme/repo.git"], check=True)
    cli = GitHubCli().bind("github.com", "mapped-user")
    _token_stub(monkeypatch, cli, [])

    env = await cli._git_network_env()
    assert env is not None
    url = subprocess.run(
        ["git", "ls-remote", "--get-url", "origin"], cwd=repo, env=env, capture_output=True, text=True, check=True
    ).stdout.strip()
    helpers = subprocess.run(
        ["git", "config", "--get-all", "credential.helper"], cwd=repo, env=env, capture_output=True, text=True
    ).stdout.splitlines()

    assert url == "https://github.com/acme/repo.git"
    # The empty entry discards inherited helpers so only gh supplies credentials.
    assert helpers[-2:] == ["", "!gh auth git-credential"]
    assert env["GH_TOKEN"] == "gho_mapped"


async def test_bound_fetch_head_uses_the_account_network_environment(monkeypatch, tmp_path):
    cli = GitHubCli().bind("github.com", "mapped-user")
    calls = []

    async def run(self, command, args, **kwargs):
        if args[:2] == ["auth", "token"]:
            return "gho_mapped\n"
        calls.append((command, args, kwargs.get("env")))
        return "a" * 40 + "\n"

    monkeypatch.setattr(git_module.GitHubCli, "_run", run)

    await cli.fetch_head(tmp_path, 7)

    fetch = next(env for command, args, env in calls if args[0] == "fetch")
    assert fetch is not None and fetch["GH_TOKEN"] == "gho_mapped"
    assert "!gh auth git-credential" in fetch.values()
