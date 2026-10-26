"""Score a run's findings with ReviewBench's judge through the configured localmcp gateway.

    uv run python evals/reviewbench/judge.py --reviewbench ../ReviewBench evals/reviewbench/runs/<run>

The judge is ReviewBench's own, unmodified code. It reaches the ``openai_compatible`` endpoint from the shared
localmcp config over OpenAI chat completions, authenticated with the ``llm_api_key`` secret. Both are resolved at
run time: the endpoint goes into a pi ``models.json`` in a private temporary directory that is removed afterwards,
and the key reaches the judge only through its environment.

Each stage is scored separately into ``scoring/<stage>/`` beneath the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from localmcp.config import ConfigDocument
from localmcp.secrets import SecretCatalog, SecretResolver

from pr_council.config import SERVER_NAME

STAGES = {
    "findings": Path("findings"),
    "reviewers": Path("stages/reviewers"),
    "deliberated": Path("stages/deliberated"),
}
_PROVIDER = "localmcp-gateway"
_KEY_ENV = "REVIEWBENCH_JUDGE_API_KEY"
_METRICS = ("grounded_precision", "grounded_recall", "augmented_precision", "augmented_recall")


def gateway(env: dict[str, str]) -> tuple[str, str]:
    """Return the configured gateway base URL and API key."""
    document = ConfigDocument.load(env=env, home=Path.home(), missing_ok=False)
    llm = document.for_server(SERVER_NAME).values.get("llm")
    if not isinstance(llm, Mapping) or llm.get("backend") != "openai_compatible" or not llm.get("base_url"):
        raise SystemExit("the judge wrapper requires [llm].backend = 'openai_compatible' with a base_url")
    if llm.get("default_headers") or llm.get("verify") not in (None, True):
        raise SystemExit("the judge wrapper does not support [llm].default_headers or a custom [llm].verify")
    reference = SecretCatalog.from_config(document, server=SERVER_NAME).require("llm_api_key")
    secret = SecretResolver(env, tolerate_keyring_errors=True).require(reference)
    return str(llm["base_url"]), secret.reveal()


def pi_models(base_url: str, model: str) -> dict[str, Any]:
    return {
        "providers": {
            _PROVIDER: {
                "baseUrl": base_url,
                "api": "openai-completions",
                # The name of the environment variable holding the key, not the key itself.
                "apiKey": _KEY_ENV,
                "compat": {"supportsDeveloperRole": False, "supportsStore": False},
                "models": [{"id": model, "reasoning": True, "contextWindow": 1_000_000, "maxTokens": 64_000}],
            }
        }
    }


def hazardous_inputs(run_dir: Path, stage_dir: Path) -> list[str]:
    """Return candidate files for PRs whose head would make the judge load project code."""
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    keys = [pr["key"] for pr in manifest.get("prs", []) if "judge_hazard" in pr]
    return [key for key in keys if (stage_dir / f"{key}.json").exists()]


def summarize(path: Path) -> str:
    results = json.loads(path.read_text(encoding="utf-8"))
    overall = results.get("micro", {}).get("overall") or results.get("overall") or {}
    values = [overall.get(metric) for metric in _METRICS]
    return "  ".join(
        f"{metric}={value:.3f}" if value is not None else f"{metric}=n/a"
        for metric, value in zip(_METRICS, values, strict=True)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0] if __doc__ else None)
    parser.add_argument("run_dir", type=Path, help="run directory written by run.py")
    parser.add_argument("--reviewbench", type=Path, required=True, help="ReviewBench checkout with node_modules")
    parser.add_argument("--model", default="claude-sonnet-5", help="judge model; the leaderboard uses claude-sonnet-5")
    parser.add_argument("--stage", choices=sorted(STAGES), action="append", help="stage to score; repeatable")
    parser.add_argument("--concurrency", type=int, default=4, help="pull requests judged concurrently")
    args = parser.parse_args(argv)

    reviewbench, run_dir = args.reviewbench.resolve(), args.run_dir.resolve()
    if not (reviewbench / "node_modules" / ".bin" / "tsx").exists():
        raise SystemExit(f"run `npm ci --ignore-scripts` in {reviewbench} first")
    stages = [stage for stage in STAGES if stage in (args.stage or STAGES)]
    for stage in stages:
        hazards = hazardous_inputs(run_dir, run_dir / STAGES[stage])
        if hazards:
            raise SystemExit(f"{stage}: remove {', '.join(hazards)} first; their heads contain .pi/")

    env = dict(os.environ)
    base_url, key = gateway(env)
    summaries: dict[str, str] = {}
    failed = False
    with tempfile.TemporaryDirectory(prefix="reviewbench-judge-") as agent_dir:
        (Path(agent_dir) / "models.json").write_text(json.dumps(pi_models(base_url, args.model)), encoding="utf-8")
        env.update({"PI_CODING_AGENT_DIR": agent_dir, _KEY_ENV: key})
        for stage in stages:
            candidates = run_dir / STAGES[stage]
            if not any(candidates.glob("*.json")):
                print(f"{stage}: no candidate files", file=sys.stderr)
                continue
            output = run_dir / "scoring" / stage / "results.json"
            print(f"== {stage}", file=sys.stderr, flush=True)
            command = ["npm", "run", "--silent", "judge", "--"]
            command += ["--candidate", os.fspath(candidates), "--provider", _PROVIDER, "--model", args.model]
            command += ["--output", os.fspath(output), "--repo-dir", os.fspath(run_dir / "judge-repos")]
            command += ["--concurrency", str(args.concurrency)]
            if subprocess.run(command, cwd=reviewbench, env=env).returncode != 0:
                print(f"{stage}: judge failed; rerun to resume from its checkpoint", file=sys.stderr)
                failed = True
                continue
            summaries[stage] = summarize(output)
    for stage, summary in summaries.items():
        print(f"{stage:12} {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
