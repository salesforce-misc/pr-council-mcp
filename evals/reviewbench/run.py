"""Run pr-council-mcp over ReviewBench pull requests and write judge-ready findings.

The harness drives the real review runtime to its preview interrupt and then cancels, so it never publishes.
Every GitHub interaction is replaced: pull-request endpoints come from the benchmark manifest, the canonical diff
is computed locally, and each pull request gets its own repository holding only the history reachable from its
pinned base and head, fetched from the benchmark's public mirror. Reviewers therefore cannot see later fix commits
or other benchmark pull requests from the same repository, and no GitHub API, token, or account is used.

    uv run python evals/reviewbench/run.py --reviewbench ../ReviewBench --pr 0

Outputs, beneath the run directory:

- ``findings/``: final aggregated findings, the judge input for the system score;
- ``stages/reviewers/`` and ``stages/deliberated/``: per-stage findings in the same format;
- ``artifacts/``: reviewer, deliberation, and aggregation state with candidate-to-finding ID maps;
- ``run.json``: run provenance and per-PR status.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localmcp.llm import ModelFactory
from localmcp.observability.logging import configure_logging
from localmcp.observability.telemetry import LangfuseTelemetry

from pr_council.config import SERVER_NAME, Config, parse_config
from pr_council.review.git import GitHubCli
from pr_council.review.models import (
    TERMINAL_STATUSES,
    OperationStatus,
    PrRef,
    ReviewContextInput,
    ReviewError,
    ReviewMode,
    ReviewSourceMode,
)
from pr_council.server_setup import PrCouncilServer
from pr_council.workflows.review.runtime import ReviewRuntime

AGENT = "pr-council-mcp"
_EVAL_LOGIN = "reviewbench-eval"
_GITHUB = "https://github.com/"
_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
_SETS = {"test": Path("corpus/test/test.json"), "full": Path("corpus/manifest.json")}
_TOKEN_ENV_VARS = ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN")
_POLL_SECONDS = 5.0


@dataclass(frozen=True)
class BenchmarkPr:
    index: int
    repo: str
    pr_number: int
    base: str
    head: str
    title: str
    body: str

    @property
    def nwo(self) -> str:
        return self.repo.removeprefix(_GITHUB)

    @property
    def key(self) -> str:
        return f"{self.nwo.replace('/', '_')}_{self.pr_number}-{self.head[:8]}"

    @property
    def url(self) -> str:
        return f"{self.repo}/pull/{self.pr_number}"


def load_prs(reviewbench: Path, set_name: str, indices: Sequence[int]) -> list[BenchmarkPr]:
    entries = json.loads((reviewbench / _SETS[set_name]).read_text(encoding="utf-8"))
    prs: list[BenchmarkPr] = []
    for index, entry in enumerate(entries):
        if indices and index not in indices:
            continue
        repo, base, head = str(entry["repo"]), str(entry["base"]).lower(), str(entry["head"]).lower()
        if not repo.startswith(_GITHUB) or len(repo.removeprefix(_GITHUB).split("/")) != 2:
            raise SystemExit(f"{set_name}[{index}]: unsupported repository {repo!r}")
        if not _OBJECT_ID.fullmatch(base) or not _OBJECT_ID.fullmatch(head):
            raise SystemExit(f"{set_name}[{index}]: base and head must be 40-character SHAs")
        prs.append(
            BenchmarkPr(
                index=index,
                repo=repo,
                pr_number=int(entry["pr_number"]),
                base=base,
                head=head,
                title=str(entry.get("title") or ""),
                body=str(entry.get("body") or ""),
            )
        )
    missing = sorted(set(indices) - {pr.index for pr in prs})
    if missing:
        raise SystemExit(f"{set_name} set has no pull requests at indices {missing}")
    return prs


def repository_path(repos_dir: Path, pr: BenchmarkPr) -> Path:
    """Return the PR's own repository, which holds only the history reachable from its base and head."""
    owner, repo = pr.nwo.split("/")
    return repos_dir / "github.com" / owner / f"{repo}-{pr.pr_number}-{pr.head[:12]}"


class BenchmarkGit(GitHubCli):
    """Git adapter that serves pinned benchmark endpoints and never runs ``gh``."""

    def __init__(self, prs: Sequence[BenchmarkPr], *, mirror_org: str, max_repository_bytes: int):
        super().__init__(max_repository_bytes=max_repository_bytes, allowed_hosts=("github.com",))
        self._prs = {(pr.nwo.lower(), pr.pr_number): pr for pr in prs}
        # Repositories record their upstream URL; process-only configuration redirects fetches to the mirror, so
        # nothing the reviewers can read names the benchmark.
        self._rewrites = sorted(
            {
                (f"url.{_GITHUB}{mirror_org}/{pr.nwo.replace('/', '_')}.git.insteadOf", f"{_GITHUB}{pr.nwo}.git")
                for pr in prs
            }
        )
        self._mirror_org = mirror_org

    def _pr(self, ref: PrRef) -> BenchmarkPr:
        pr = self._prs.get((f"{ref.owner}/{ref.repo}".lower(), ref.number))
        if pr is None:
            raise ReviewError(f"{ref.url} is not a selected benchmark pull request")
        return pr

    def _benchmark_env(self) -> dict[str, str]:
        env = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("GIT_") and name not in _TOKEN_ENV_VARS
        }
        env.update(
            {
                # Global config may install LFS smudging, credential helpers, or URL rewrites. Benchmark
                # repositories are public, and LFS objects stay pointers as they do in the official runner.
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "LC_ALL": "C",
                "GIT_CONFIG_COUNT": str(len(self._rewrites)),
            }
        )
        for index, (key, value) in enumerate(self._rewrites):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        return env

    async def _run(
        self, command: str, args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> str:
        # The single choke point for GitHub API calls, PR diffs, clones, and publication.
        if command != "git":
            raise ReviewError(f"the ReviewBench harness does not run {command}")
        return await super()._run(command, args, cwd=cwd, env=env or self._benchmark_env())

    async def _git_network_env(self) -> dict[str, str] | None:
        return self._benchmark_env()

    async def authenticated_user(self, host: str) -> str:
        return _EVAL_LOGIN

    async def pr_shas(self, ref: PrRef) -> tuple[str, str]:
        pr = self._pr(ref)
        return pr.base, pr.head

    async def clone_or_fetch(self, ref: PrRef, base_dir: Path) -> Path:
        self._validate_repository(ref.host, ref.owner, ref.repo)
        pr = self._pr(ref)
        repo_path = repository_path(base_dir, pr)
        if not (repo_path / ".git").is_dir():
            shutil.rmtree(repo_path, ignore_errors=True)
            repo_path.mkdir(parents=True, mode=0o700)
            await self._run("git", ["init", "--quiet", os.fspath(repo_path)])
            await self._run("git", ["remote", "add", "origin", f"{_GITHUB}{pr.nwo}.git"], cwd=repo_path)
        try:
            await self._run(
                "git",
                ["fetch", "--quiet", "--no-tags", "--filter=blob:none", "origin", pr.base, pr.head],
                cwd=repo_path,
            )
        except ReviewError as exc:
            mirror = f"{self._mirror_org}/{pr.nwo.replace('/', '_')}"
            raise ReviewError(f"could not fetch base {pr.base} and head {pr.head} from {mirror}") from exc
        # FETCH_HEAD records the mirror URL.
        (repo_path / ".git" / "FETCH_HEAD").unlink(missing_ok=True)
        await self._run("git", ["update-ref", f"refs/pull/{ref.number}/head", pr.head], cwd=repo_path)
        if await asyncio.to_thread(self._directory_exceeds_limit, repo_path):
            raise ReviewError("local repository exceeds the configured size safety limit")
        return repo_path

    async def fetch_head(self, repo_path: Path, pr_number: int) -> str:
        return (await self._run("git", ["rev-parse", f"refs/pull/{pr_number}/head"], cwd=repo_path)).strip()

    async def diff(self, repo_path: Path, ref: PrRef) -> str:
        pr = self._pr(ref)
        return await self._run("git", ["diff", "--no-color", "--no-ext-diff", f"{pr.base}...{pr.head}"], cwd=repo_path)


class BenchmarkReviewRuntime(ReviewRuntime, register=False):
    def __init__(
        self,
        config: Config,
        state_root: Path,
        model_factory: ModelFactory,
        *,
        git: GitHubCli,
        telemetry: LangfuseTelemetry | None = None,
    ):
        super().__init__(config, state_root, model_factory, telemetry=telemetry)
        self.git = git


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def findings(self) -> Path:
        return self.root / "findings"

    @property
    def reviewers(self) -> Path:
        return self.root / "stages" / "reviewers"

    @property
    def deliberated(self) -> Path:
        return self.root / "stages" / "deliberated"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def manifest(self) -> Path:
        return self.root / "run.json"


def _candidate(finding: dict[str, Any], producer: str) -> dict[str, Any]:
    line = int(finding["line"])
    return {
        "producer": producer,
        "file": str(finding["path"]),
        "start_line": line,
        "end_line": line,
        "message": f"{finding['title']}\n\n{finding['body']}",
    }


def _judge_input(pr: BenchmarkPr, findings: list[dict[str, Any]], usage: dict[str, int] | None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "pr": {"repo": pr.repo, "pr_number": pr.pr_number, "base": pr.base, "head": pr.head},
        "agent": AGENT,
        "findings": findings,
    }
    if usage is not None:
        document["usage"] = usage
    return document


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_outputs(paths: RunPaths, pr: BenchmarkPr, operation_id: str, values: dict[str, Any]) -> dict[str, int]:
    """Write judge inputs for each stage plus the state needed to attribute judgments back to findings."""
    reviewers = [result for result in values["reviewer_results"] if not result.get("failed")]
    reviewer_findings = [
        (finding, f"{AGENT}/{result['disposition']}/{result['model']}")
        for result in reviewers
        for finding in result["findings"]
    ]
    retained = list(values["retained_findings"])
    aggregated = list(values["aggregation"]["findings"])
    metrics = values["preview"]["metrics"]
    usage = {
        "prompt_tokens": int(metrics["tokens"]["input"]),
        "completion_tokens": int(metrics["tokens"]["output"]),
        "total_tokens": int(metrics["total_tokens"]),
        "time_in_ms": int(metrics["duration_ms"]),
    }
    _write_json(
        paths.findings / f"{pr.key}.json",
        _judge_input(pr, [_candidate(finding, AGENT) for finding in aggregated], usage),
    )
    _write_json(
        paths.reviewers / f"{pr.key}.json",
        _judge_input(pr, [_candidate(finding, producer) for finding, producer in reviewer_findings], None),
    )
    _write_json(
        paths.deliberated / f"{pr.key}.json",
        _judge_input(pr, [_candidate(finding, f"{AGENT}/{finding['disposition']}") for finding in retained], None),
    )
    _write_json(
        paths.artifacts / f"{pr.key}.json",
        {
            "operation_id": operation_id,
            "pr": {"index": pr.index, "repo": pr.repo, "pr_number": pr.pr_number, "base": pr.base, "head": pr.head},
            # Candidate order per stage file, so judge decisions map back to finding IDs.
            "candidate_ids": {
                "findings": [finding["id"] for finding in aggregated],
                "reviewers": [finding["id"] for finding, _producer in reviewer_findings],
                "deliberated": [finding["id"] for finding in retained],
            },
            "reviewer_results": values["reviewer_results"],
            "deliberation_results": values["deliberation_results"],
            "retained_findings": retained,
            "aggregation": values["aggregation"],
            "metrics": metrics,
            "reviewer_failures": values["preview"].get("reviewer_failures", []),
        },
    )
    return {"reviewers": len(reviewer_findings), "deliberated": len(retained), "findings": len(aggregated)}


async def _wait(
    runtime: ReviewRuntime, operation_id: str, statuses: set[OperationStatus], timeout_seconds: float
) -> OperationStatus:
    deadline = time.monotonic() + timeout_seconds
    while True:
        record = await runtime.get_owned(operation_id)
        if record.status in statuses:
            return record.status
        if time.monotonic() >= deadline:
            raise TimeoutError(f"operation still {record.status.value} after {timeout_seconds:.0f}s")
        await asyncio.sleep(_POLL_SECONDS)


async def _has_pi_resources(git: GitHubCli, repo_path: Path, head: str) -> bool:
    """Report whether the head tree has a root ``.pi`` entry.

    The judge's classifier runs a pi agent session inside a checkout of the head, and pi loads project extensions,
    which are executable code, from ``.pi/`` in its working directory.
    """
    return bool((await git._run("git", ["ls-tree", "--name-only", head, "--", ".pi"], cwd=repo_path)).strip())


def _log(message: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


async def review_pr(
    runtime: ReviewRuntime,
    paths: RunPaths,
    pr: BenchmarkPr,
    *,
    models: dict[str, list[str]] | None,
    deliberation_model: str | None,
    aggregation_model: str | None,
    iterations: int,
    timeout_seconds: float,
    keep_repos: bool,
) -> dict[str, Any]:
    began = time.monotonic()
    outcome: dict[str, Any] = {"index": pr.index, "key": pr.key, "url": pr.url}
    _log(f"{pr.key}: starting")
    context = [
        ReviewContextInput(
            label="Pull request description",
            source=pr.url,
            content=f"Title: {pr.title}\n\n{pr.body}".strip()[:32_000],
        )
    ]
    record = await runtime.start_operation(
        pr_url=pr.url,
        source_mode=ReviewSourceMode.MANAGED,
        context=context,
        mode=ReviewMode.INITIAL,
        baseline_operation_id=None,
        iterations=iterations,
        models=models,
        deliberation_model=deliberation_model,
        aggregation_model=aggregation_model,
    )
    outcome["operation_id"] = record.id
    try:
        status = await _wait(runtime, record.id, {OperationStatus.READY, *TERMINAL_STATUSES}, timeout_seconds)
        if status == OperationStatus.READY:
            snapshot = await runtime.graph.aget_state({"configurable": {"thread_id": record.id}})
            values = dict(snapshot.values)
            outcome["counts"] = write_outputs(paths, pr, record.id, values)
            failures = values["preview"].get("reviewer_failures", [])
            if failures:
                outcome["reviewer_failures"] = [f"{failure['disposition']}/{failure['model']}" for failure in failures]
            if await _has_pi_resources(runtime.git, Path(values["repo_path"]), pr.head):
                outcome["judge_hazard"] = ".pi/ at head"
                _log(f"{pr.key}: head contains .pi/, which the judge's classifier would load; do not judge it")
        else:
            outcome["error"] = (await runtime.get_owned(record.id)).error or f"operation ended as {status.value}"
    except Exception as exc:  # Recorded per PR so one failure does not abort the run.
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Cancelling at the preview interrupt runs the graph's own worktree and ref cleanup.
        try:
            await runtime.cancel(record.id)
            await _wait(runtime, record.id, TERMINAL_STATUSES, 300)
        except (ReviewError, TimeoutError) as exc:
            _log(f"{pr.key}: cleanup did not finish ({exc})")
        if not keep_repos:
            shutil.rmtree(repository_path(runtime.root / "repos", pr), ignore_errors=True)
    # A degraded review ran without some of its configured reviewers, so it does not measure the configuration.
    outcome["status"] = "failed" if "counts" not in outcome else "degraded" if "reviewer_failures" in outcome else "ok"
    outcome["elapsed_s"] = round(time.monotonic() - began, 1)
    detail = outcome.get("counts") or outcome.get("error")
    if "reviewer_failures" in outcome:
        detail = f"{detail}; failed reviewers: {', '.join(outcome['reviewer_failures'])}"
    _log(f"{pr.key}: {outcome['status']} in {outcome['elapsed_s']}s ({detail})")
    return outcome


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", os.fspath(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


async def run(args: argparse.Namespace) -> int:
    reviewbench = args.reviewbench.resolve()
    prs = load_prs(reviewbench, args.set, args.pr)
    paths = RunPaths(args.run_dir.resolve())
    if paths.root.exists() and any(paths.root.iterdir()):
        raise SystemExit(f"{paths.root} is not empty; give every run its own directory")
    paths.root.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=paths.root / "pr-council.log", level=logging.INFO)

    runtimes: list[BenchmarkReviewRuntime] = []

    def runtime_factory(
        config: Config, state_root: Path, model_factory: ModelFactory, *, telemetry: LangfuseTelemetry
    ) -> BenchmarkReviewRuntime:
        git = BenchmarkGit(
            prs,
            mirror_org=args.mirror_org,
            max_repository_bytes=config.pr_review.limits.repository_size_mib * 1024 * 1024,
        )
        runtime = BenchmarkReviewRuntime(config, paths.state, model_factory, git=git, telemetry=telemetry)
        runtimes.append(runtime)
        return runtime

    server = PrCouncilServer(name=SERVER_NAME, tools=(), config_parser=parse_config, runtime_factory=runtime_factory)
    config = await server.start_runtime(env=os.environ, home=Path.home())
    runtime = runtimes[-1]
    try:
        configured = config.pr_review.models
        models = None
        if args.quality or args.security:
            models = {
                "quality": args.quality or list(configured.quality),
                "security": args.security or list(configured.security),
            }
        provenance: dict[str, Any] = {
            "agent": AGENT,
            "started_at": datetime.now(UTC).isoformat(),
            "reviewbench": {"path": os.fspath(reviewbench), "revision": _git_revision(reviewbench), "set": args.set},
            "pr_council_revision": _git_revision(Path(__file__).resolve().parent),
            "models": models or {"quality": list(configured.quality), "security": list(configured.security)},
            "deliberation_model": args.deliberation or configured.deliberation,
            "aggregation_model": args.aggregation or configured.aggregation,
            "iterations": args.iterations,
            "prs": [],
        }
        _write_json(paths.manifest, provenance)
        semaphore = asyncio.Semaphore(args.jobs)

        async def bounded(pr: BenchmarkPr) -> dict[str, Any]:
            async with semaphore:
                try:
                    return await review_pr(
                        runtime,
                        paths,
                        pr,
                        models=models,
                        deliberation_model=args.deliberation,
                        aggregation_model=args.aggregation,
                        iterations=args.iterations,
                        timeout_seconds=args.timeout_minutes * 60,
                        keep_repos=args.keep_repos,
                    )
                except Exception as exc:
                    _log(f"{pr.key}: failed to start ({exc})")
                    return {"index": pr.index, "key": pr.key, "url": pr.url, "status": "failed", "error": str(exc)}

        outcomes = await asyncio.gather(*(bounded(pr) for pr in prs))
    finally:
        await server.stop_runtime()
    failed = [outcome for outcome in outcomes if outcome["status"] == "failed"]
    degraded = [outcome for outcome in outcomes if outcome["status"] == "degraded"]
    provenance["prs"] = sorted(outcomes, key=lambda outcome: int(outcome["index"]))
    provenance["finished_at"] = datetime.now(UTC).isoformat()
    _write_json(paths.manifest, provenance)
    ok = len(outcomes) - len(failed) - len(degraded)
    _log(f"done: {ok} ok, {len(degraded)} degraded, {len(failed)} failed; findings in {paths.findings}")
    hazards = [outcome["key"] for outcome in outcomes if "judge_hazard" in outcome]
    if hazards:
        _log(f"remove these from the judge input first: {', '.join(hazards)}")
    return 1 if failed or degraded else 0


def _parser() -> argparse.ArgumentParser:
    default_run = Path(__file__).resolve().parent / "runs" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0] if __doc__ else None)
    parser.add_argument("--reviewbench", type=Path, required=True, help="ReviewBench checkout")
    parser.add_argument("--set", choices=sorted(_SETS), default="test", help="test (25 PRs) or full (219 PRs)")
    parser.add_argument("--pr", type=int, action="append", default=[], help="set index to run; repeatable")
    parser.add_argument("--run-dir", type=Path, default=default_run, help="fresh output directory")
    parser.add_argument("--iterations", type=int, default=3, help="refinement iterations per reviewer model")
    parser.add_argument("--quality", action="append", help="quality reviewer model; repeatable")
    parser.add_argument("--security", action="append", help="security reviewer model; repeatable")
    parser.add_argument("--deliberation", help="deliberation model")
    parser.add_argument("--aggregation", help="aggregation model")
    parser.add_argument("--jobs", type=int, default=1, help="pull requests reviewed concurrently")
    parser.add_argument("--timeout-minutes", type=float, default=60, help="per-PR review timeout")
    parser.add_argument("--mirror-org", default="review-bench", help="GitHub organization hosting the mirrors")
    parser.add_argument("--keep-repos", action="store_true", help="keep per-PR repositories after review")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
