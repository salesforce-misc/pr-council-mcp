# ReviewBench offline evaluation

[ReviewBench](https://github.com/review-bench/ReviewBench) scores code-review findings against a curated golden set.
Its packaged runner reviews inside a Linux container, and the review sandbox here is macOS Seatbelt. This harness
therefore runs pr-council natively and hands its findings to ReviewBench's judge, which accepts findings from any
source.

`run.py` drives the real review runtime to the preview interrupt and then cancels, so nothing is ever published. It
swaps out only the GitHub adapter:

- base and head come from the benchmark manifest, not the GitHub API;
- each pull request gets its own blobless repository fetched from the benchmark's public mirror, holding only the
  history reachable from its base and head, with no tags and no other pull requests;
- the repository's `origin` is the upstream URL. The mirror rewrite exists only in the fetch process's environment, so
  nothing a reviewer can read names the benchmark;
- the canonical diff is `git diff base...head`, and every `gh` call is refused, so no GitHub API, token, or account is
  used;
- global and system Git config are disabled for fetches, and LFS files stay as pointers.

Reviewer Bash keeps the normal sandbox: read-only, with no network. Reviewers therefore can't reach the upstream pull
request's comments or later fix commits.

## Review

Prerequisites are the usual server prerequisites, a configured `~/.config/localmcp/localmcp.toml`, and a ReviewBench
checkout. The checkout is only read; nothing in it is executed during review.

```sh
uv run python evals/reviewbench/run.py --reviewbench ../ReviewBench --pr 0          # one test-set PR
uv run python evals/reviewbench/run.py --reviewbench ../ReviewBench --jobs 2        # the 25-PR test set
uv run python evals/reviewbench/run.py --reviewbench ../ReviewBench --set full      # all 219 PRs
```

Models, iterations, and per-PR timeout default to the configured values. Override them with `--quality`,
`--security`, `--deliberation`, `--aggregation`, `--iterations`, and `--timeout-minutes`. Each run writes to a fresh
directory beneath `evals/reviewbench/runs/`, which is ignored by Git:

| Path | Contents |
|---|---|
| `findings/` | Final aggregated findings: the judge input for the system score. |
| `stages/reviewers/` | Every successful reviewer finding before deliberation, in the judge format. |
| `stages/deliberated/` | Findings deliberation kept or revised, with revisions applied, before aggregation. |
| `artifacts/` | Reviewer, deliberation, and aggregation state, plus candidate-to-finding ID maps. |
| `run.json` | ReviewBench revision, models, iterations, and per-PR status. |
| `pr-council.log` | The runtime's structured log. |

A failed pull request has no findings file and drops out of scoring rather than counting as a miss. A degraded one
completed without some of its configured reviewers, usually after exhausting rate-limit retries, so its findings
don't measure the configuration. Both make the run exit nonzero; check `run.json` before you compare runs.

## Judge

Judging runs ReviewBench's own code (Node 20+). Install its dependencies without running install scripts:

```sh
cd ../ReviewBench
npm ci --ignore-scripts
```

`judge.py` runs ReviewBench's unmodified judge through the `openai_compatible` endpoint in your localmcp config,
over OpenAI chat completions, with the `llm_api_key` secret. It resolves both at run time: the endpoint goes into a
pi `models.json` in a private temporary directory that is removed afterwards, and the key reaches the judge only
through its environment. That environment is otherwise an allowlist of what node and npm need to run and reach the
endpoint (`PATH`, `HOME`, locale, proxy, and CA-certificate variables), so the caller's tokens and other credentials
never reach the judge's agent. Each stage is scored separately:

```sh
uv run python evals/reviewbench/judge.py --reviewbench ../ReviewBench evals/reviewbench/runs/<run>
```

The judge model defaults to `claude-sonnet-5`, the leaderboard's judge. The leaderboard calls it through the
Anthropic Messages API, so thinking settings reach it differently here. Results go to `scoring/<stage>/` beneath the
run directory. A judge that fails partway resumes from its checkpoint when you run the same command again.

Comparing the stages shows where the pipeline gains or loses findings. Deliberation removes or rewrites findings and
aggregation merges them, so recall lost between stages is a true positive the pipeline discarded or rewrote out of
recognition, and precision should rise. Recall gained at the deliberated stage comes from corrected claims. The
harness scores every aggregated finding regardless of `min_severity`.
`results.details.json` lists each candidate in file order; the matching list in `artifacts/<pr>.json` under
`candidate_ids` maps those candidates back to finding IDs.

The judge's classifier runs a pi agent session inside a checkout of each pull request's head. pi loads project
extensions, which are executable code, from `.pi/` in that directory. `run.py` reports any head with a root `.pi`
entry, and `judge.py` refuses to score a stage that still contains one. None of the 25 test-set PRs has one.
Repository `AGENTS.md` and `CLAUDE.md` files also reach the classifier as context; this affects leaderboard judging
equally.

The leaderboard scores all 219 pull requests three times, and a different judge model shifts the scores. Read
offline numbers as comparisons between configurations of this system, not as leaderboard positions.

## Make targets

The Makefile wraps review, judging, and reporting. These targets call the configured models and the judge, so they
are never part of `make ci`:

```sh
make eval REVIEWBENCH=../ReviewBench                                # the 25-PR test set
make eval EVAL_RUN=my-change EVAL_BASELINE=previous-run             # name the run; compare with an earlier one
make eval-pr EVAL_RUN=my-change EVAL_BASELINE=previous-run          # only EVAL_PR, for quick iteration
make eval EVAL_PRS="3 5 16 21"                                      # chosen test-set indices
make eval EVAL_SET=full                                             # all 219 PRs
make eval-report EVAL_RUN=my-change EVAL_BASELINE="run-a run-b"     # report judged runs again
```

`eval` reviews the whole of `EVAL_SET` (default `test`), or only its `EVAL_PRS` indices when set, judges the findings
stage, and reports. The run still gets judged and reported when some pull requests fail or are degraded, but `make`
then exits nonzero.

The test set is the baseline: its per-PR results show where the system falls short. `eval-pr` runs `eval` on `EVAL_PR`
alone, the pull request the system currently handles worst, which makes a cheap targeted check while tuning. One pull
request is a noisy signal, so confirm a gain on the test set before relying on it, and change the default when another
pull request becomes the weakest.

| Variable | Default | Meaning |
|---|---|---|
| `REVIEWBENCH` | `../ReviewBench` | ReviewBench checkout |
| `EVAL_SET` | `test` | `test` (25 PRs) or `full` (219 PRs) |
| `EVAL_PRS` | none | set indices to review instead of the whole set |
| `EVAL_PR` | `12` | the single set index `eval-pr` reviews |
| `EVAL_RUN` | UTC timestamp | directory name beneath `runs/` |
| `EVAL_BASELINE` | none | earlier run names to compare against, by name beneath `runs/` |
| `EVAL_STAGES` | `findings` | stages to judge and report |
| `EVAL_JOBS` | `1` | pull requests reviewed concurrently |
| `EVAL_JUDGE_CONCURRENCY` | `2` | pull requests judged concurrently |
| `EVAL_ARGS` | none | extra `run.py` arguments, such as `--aggregation <model>` |

`report.py` prints each run's grounded F1, precision, and recall, both pooled across pull requests and averaged per
pull request. It also prints accuracy figures and golden coverage by severity. With baselines, it repeats the figures
over the pull requests judged in every run, so runs with different failures still compare like for like.
