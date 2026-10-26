"""Report judged runs in leaderboard terms and compare them on the pull requests they share.

    uv run python evals/reviewbench/report.py evals/reviewbench/runs/<run> [evals/reviewbench/runs/<baseline> ...]

Only the findings stage, the final aggregated findings, is comparable with the leaderboard.

The leaderboard ranks by grounded F1, the harmonic mean of grounded precision and recall. Micro figures pool every
judged candidate; macro figures average per pull request over those where the metric is defined. A failed pull
request has no findings and is not judged, so it drops out of a run's figures rather than counting as a miss. With
more than one run, the comparison repeats the figures over only the pull requests judged in every run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_STAGES = ("reviewers", "deliberated", "findings")
_SEVERITIES = ("high", "medium", "low")


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _line(name: str, per_pr: dict[str, Any], keys: Sequence[str]) -> str:
    rows = [per_pr[key]["overall"] for key in keys]

    def total(field: str) -> int:
        return sum(int(row[field]) for row in rows)

    def mean(field: str) -> float:
        # The judge reports a metric with no denominator as n/a; such a PR is left out rather than counted as zero.
        defined = [row[field] for row in rows if row[field] is not None]
        return _ratio(sum(defined), len(defined))

    gp = _ratio(total("matched_tp_count"), total("matched_count"))
    gr = _ratio(total("golden_covered_count"), total("golden_tp_count"))
    ap = _ratio(total("matched_tp_count") + total("unmatched_tp_count"), total("candidate_count"))
    ar = _ratio(
        total("golden_covered_count") + total("novel_tp_count"), total("golden_tp_count") + total("novel_tp_count")
    )
    macro_gf1 = _f1(mean("grounded_precision"), mean("grounded_recall"))
    severity = []
    for level in _SEVERITIES:
        found = [per_pr[key]["by_severity"][level] for key in keys if level in per_pr[key]["by_severity"]]
        covered = sum(int(row["golden_covered_count"]) for row in found)
        severity.append(f"{level} {covered}/{sum(int(row['golden_tp_count']) for row in found)}")
    return (
        f"{name:20} {len(rows):3} PRs  GF1 {100 * _f1(gp, gr):5.1f} (GP {100 * gp:5.1f} GR {100 * gr:5.1f})  "
        f"macro GF1 {100 * macro_gf1:5.1f}  "
        f"AP {100 * ap:5.1f} AR {100 * ar:5.1f}  {_ratio(total('candidate_count'), len(rows)):4.1f} findings/PR  "
        f"golden {total('golden_covered_count')}/{total('golden_tp_count')}  {', '.join(severity)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0] if __doc__ else None)
    parser.add_argument("runs", type=Path, nargs="+", help="judged run directories; the first is reported first")
    parser.add_argument("--stage", choices=_STAGES, action="append", help="scored stage; repeatable; default findings")
    args = parser.parse_args(argv)

    for run in args.runs:
        manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
        statuses = [outcome["status"] for outcome in manifest["prs"]]
        unscored = [f"{statuses.count(status)} {status}" for status in ("failed", "degraded") if status in statuses]
        if unscored:
            print(f"{run.name}: {', '.join(unscored)} (failed PRs are not scored; see run.json)")
    for stage in [stage for stage in _STAGES if stage in (args.stage or ["findings"])]:
        per_run: dict[str, dict[str, Any]] = {}
        for run in args.runs:
            results = run / "scoring" / stage / "results.json"
            if not results.exists():
                raise SystemExit(f"{results} does not exist; judge the run first")
            scored = json.loads(results.read_text(encoding="utf-8"))["per_pr"]
            per_run[run.name] = {entry["pr_key"]: entry["metrics"] for entry in scored}
        print(f"== {stage}: every judged PR")
        for name, per_pr in per_run.items():
            print(_line(name, per_pr, sorted(per_pr)))
        if len(per_run) > 1:
            shared = sorted(set.intersection(*(set(per_pr) for per_pr in per_run.values())))
            print(f"== {stage}: the {len(shared)} PRs judged in every run")
            for name, per_pr in per_run.items():
                print(_line(name, per_pr, shared))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
