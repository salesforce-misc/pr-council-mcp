# Experiments

A log of review-quality experiments: what changed, what was measured, and what we concluded. Scores are ReviewBench
findings-stage results from [`reviewbench/`](reviewbench/README.md). Releases records each major and minor release's
25-PR benchmark; the dated sections below it are experiments.

Single-PR scores are noisy. The same code has scored up to 40 GF1 points apart on one PR, so a single-PR difference
is a lead to confirm on the 25-PR test set, not a result.

## Measurements

Scores judge the findings a run would publish, after deliberation and aggregation. ReviewBench's golden set lists
each PR's known issues, labeled true or false positive, and its judge matches each finding to golden issues where it
can.

- **GF1** (grounded F1): the harmonic mean of GP and GR, and the headline score.
- **GP** (grounded precision): of the findings matched to a golden issue, the share whose issue is a true positive.
  A finding that matches a known false positive lowers it. Findings that match nothing are left out; the judge
  classifies them separately for ReviewBench's augmented metrics, which this log does not report.
- **GR** (grounded recall): the share of the golden true positives that at least one finding matched.
- **Golden (high / med / low)**: the golden true positives found out of those present, in total and by severity.
- **Findings**: the number of findings the run produced, every one of which is judged.
- **Tokens in / out**: model input and output tokens across reviewers, deliberation, and aggregation. Judging is
  excluded.
- **Reviewer tool calls**: `Bash` calls made by all reviewers across every iteration.
- **Time**: the operation's wall time from preparation through aggregation. Judging is excluded.

## Releases

Every major and minor release is benchmarked on the 25-PR test set with that release's default review settings, newest
first. Scores pool every judged finding across the 25 PRs; findings, tokens, reviewer tool calls, and time are totals.

| Version | Run                   | GF1  | GP   | GR   | Golden (high / med / low)       | Findings | Tokens in / out | Reviewer tool calls | Time   | Failed / degraded PRs |
| ------- | --------------------- | ---- | ---- | ---- | ------------------------------- | -------- | --------------- | ------------------- | ------ | --------------------- |
| 0.6.0   | `benchmark-25-v0.6.0` | 61.8 | 87.8 | 47.7 | 168/352 (20/36, 91/134, 57/182) | 280      | 49.07M / 1.61M  | 2,177               | 2.68 h | 0 / 1                 |

### 0.6.0

Defaults: the mixed matrix (`claude-opus-5-5` and `gpt-6-sol` for both quality and security, `claude-opus-5-5`
deliberation, `claude-sonnet-5` aggregation), 3 iterations, `medium` reasoning effort for every role, and 12 reviewer
tool calls per iteration, reviewing one PR at a time.

Code: [`f96269a`][f96269a] with the version set to 0.6.0. Since the previous 25-PR benchmark, `benchmark-25-ee5f5c4` at
[`ee5f5c4`][ee5f5c4], [#9] added per-role reasoning effort, removed the output cap and the operation-wide tool budget,
and moved to localmcplib 0.4.0, which streams every response under a response timeout. [#10] moved to localmcplib
0.5.0, which retries each failed model call instead of rerunning the reviewer pass and lowers the default response
timeout to 300 s, and reports retries for every role.

Findings:

- GF1 rose from 58.0 to 61.8 against `benchmark-25-ee5f5c4` (GP 86.1 to 87.8, GR 43.8 to 47.7). That run failed two
  PRs in aggregation, `vanyastaff/nebula#224` and `superhighfives/pika#200`; their baseline scores come from
  `benchmark-25-rerun-21k`, which reran them at the same code with a 21,000-token output cap instead of 8,000.
- 0.6.0 found 14 more golden issues, 168 against 154: 13 more medium, one more low, and the same 20 of 36 high.
- 11 PRs scored higher, 6 lower, and 8 the same; most differences are within single-PR noise. The largest gain,
  `PaulStSmith/figlet-comment-generator#32` (28.6 to 77.8, 5 more golden issues), is partly because the baseline lost
  its `gpt-6-sol` security reviewer to a provider error. Without that PR, GF1 rose from 58.9 to 61.3 and coverage by 9
  golden issues, 8 of them medium.
- The largest drop, `simnaut/emdash#2` (25.0 to 0.0), matched none of its 6 golden issues, although the judge
  classified 15 of its 16 findings as real issues outside the golden set.
- Outside the golden set, the judge classified 123 findings as real issues, against 126.
- No PR failed, against two. One was degraded, as in the baseline: in `renedierking/LANdalf#88`, a `gpt-6-sol` quality
  reviewer's request exceeded the model's context window after 22 tool calls, which is not retried. The PR still scored
  the baseline's 66.7.
- Four model calls were retried, and all recovered without rerunning a reviewer pass: two rate-limited
  `claude-opus-5-5` reviewer calls, and two responses that exceeded the response timeout, one from an aggregator and
  one from a `claude-opus-5-5` reviewer.
- Cost was unchanged: 49.07M input tokens against 49.06M, 1.61M output against 1.68M, 2,177 reviewer tool calls against
  2,165, and 2.68 h against 2.79 h.

Conclusion: 0.6.0 scores 61.8 GF1, 3.8 points above the previous 25-PR benchmark at the same cost, with no failed PRs.
It is the baseline for the next release.

## 2026-10-25

### Reviewer model matrix

Does a mixed Claude and GPT reviewer matrix find more than either model alone?

Each run reviewed the 25-PR test set with 3 iterations, no reasoning effort sent, 12 reviewer tool calls per
iteration, a 300-call operation budget, and output capped at 8,000 tokens per call. The table shows PR 3
(`cmik/apilix#12`). The single-model runs had one reviewer per disposition, the mixed run two. Deliberation used
`gpt-6.1-sol` in the GPT run and `claude-opus-5-5` in the others.

Code: [`2fa4c61`][2fa4c61] with the then-uncommitted changes later merged in [#8]. `test-25-mixed-rerun` also had
#8's aggregator fix, which records each merge as its own tool call; without it, `test-25-mixed` had failed PR 3 in
aggregation.

| Run                   | Reviewer models                | Aggregation       | GF1  | GP    | GR   | Golden (high / med / low) | Findings | Tokens in / out | Reviewer tool calls | Time  |
| --------------------- | ------------------------------ | ----------------- | ---- | ----- | ---- | ------------------------- | -------- | --------------- | ------------------- | ----- |
| `test-25-opus`        | `claude-opus-5-5`              | `claude-sonnet-5` | 50.0 | 100.0 | 33.3 | 12/36 (6/11, 6/13, 0/12)  | 12       | 1.02M / 60k     | 47                  | 375 s |
| `test-25-gpt`         | `gpt-6.1-sol`                  | `gpt-6-luna`      | 53.1 | 100.0 | 36.1 | 13/36 (5/11, 8/13, 0/12)  | 25       | 1.99M / 32k     | 48                  | 255 s |
| `test-25-mixed-rerun` | `claude-opus-5-5`, `gpt-6-sol` | `claude-sonnet-5` | 64.2 | 100.0 | 47.2 | 17/36 (7/11, 8/13, 2/12)  | 21       | 2.30M / 85k     | 112                 | 466 s |

Findings:

- The mixed matrix found 4-5 more golden issues than either single model, and the only low-severity ones. It matched
  or beat both at every severity.
- The single-model runs cost less mostly because they ran half as many reviewers. Reviewer count is therefore
  confounded with the model mix.
- The GPT run used `gpt-6.1-sol` and `gpt-6-luna`, not the mixed matrix's `gpt-6-sol`, so it is not the mixed matrix
  without Opus.

Conclusion: across the 23 PRs that every matrix scored, GF1 was 61.5 mixed, 53.4 Opus-only, and 46.5 GPT-only.
Experiments since use the mixed matrix only.

## 2026-10-26

### Reviewer reasoning effort

Does raising reviewer reasoning effort find enough more to justify its cost?

Each run reviews PR 3 of the 25-PR test set (`cmik/apilix#12`: general code and documentation changes, 36 golden
issues, 11 high) with the mixed matrix (`claude-opus-5-5` and `gpt-6-sol` for both quality and security,
`claude-opus-5-5` deliberation, `claude-sonnet-5` aggregation), 3 iterations, 30 reviewer tool calls per iteration, no
operation-wide tool budget, and no output cap. Deliberation and aggregation run at `medium`; only reviewer effort
differs.

Code: [`bd9c380`][bd9c380] with the then-uncommitted changes later merged in [#9]: per-role reasoning effort, no output
cap, 30 reviewer tool calls per iteration with no operation budget, and a localmcplib, later released as 0.4.0, that
streams every response and adds the response timeout. The earlier PR 3 runs cited below ran at [`2fa4c61`][2fa4c61] (64.2) and [`ee5f5c4`][ee5f5c4] (56.0).

| Run                 | Reviewer effort | GF1  | GP    | GR   | Golden (high / med / low) | Findings | Tokens in / out | Reviewer tool calls | Time   |
| ------------------- | --------------- | ---- | ----- | ---- | ------------------------- | -------- | --------------- | ------------------- | ------ |
| `pr3-effort-high-2` | `high`          | 62.3 | 91.7  | 47.2 | 17/36 (8/11, 6/13, 3/12)  | 30       | 9.67M / 235k    | 221                 | 1226 s |
| `pr3-effort-medium` | `medium`        | 61.5 | 100.0 | 44.4 | 16/36 (7/11, 8/13, 1/12)  | 27       | 5.45M / 153k    | 191                 | 1064 s |

Findings:

- `high` and `medium` scored the same. `high` found one more golden issue (more high and low, fewer medium), and one of
  its findings matched a known false positive.
- `medium` used 43% fewer tokens and 13% less time. Most of the saving is `gpt-6-sol`, whose tokens fell from 5.89M to
  2.18M; `claude-opus-5-5` fell from 3.94M to 3.36M.
- Both scored within noise of two earlier runs of PR 3 (56.0 and 64.2 GF1) that sent no effort, allowed 12 tool calls
  per iteration, and capped output, at about 2.2M tokens and 110 reviewer tool calls. Even at `medium`, these settings
  cost 2.5x those tokens without a clear gain. A PR 12 run at `high` also moved within noise (44.4 against 35.3) at
  2.3x the tokens.
- The first `high` attempt failed when one reviewer call stalled mid-stream until the harness's one-hour limit
  cancelled it. localmcplib now bounds each response with `[llm].response_timeout` (default 600 s) and retries one
  that exceeds it. Neither run reached it; `pr3-effort-medium` retried one reviewer pass after a provider error
  mid-stream.

Conclusion: `medium` matches `high` at 57% of its tokens, so the reviewer default stays `medium`. Most of the remaining
cost over the earlier runs likely comes with the larger tool budget; the reviewer tool budget experiment below
measures it.

### Review iterations

How do more review iterations trade findings for cost?

All three runs use the `pr3-effort-medium` setup: PR 3, the mixed matrix, reviewer effort `medium`, 30 reviewer tool
calls per iteration, no operation-wide tool budget, and no output cap. Only the number of iterations each reviewer
model runs differs.

Code: [`bd9c380`][bd9c380] with the same uncommitted changes as the reasoning-effort experiment.

| Run                       | Iterations | GF1  | GP    | GR   | Golden (high / med / low) | Findings | Tokens in / out | Reviewer tool calls | Time   |
| ------------------------- | ---------- | ---- | ----- | ---- | ------------------------- | -------- | --------------- | ------------------- | ------ |
| `pr3-effort-medium-iter1` | 1          | 56.0 | 100.0 | 38.9 | 14/36 (7/11, 7/13, 0/12)  | 20       | 2.04M / 101k    | 66                  | 1359 s |
| `pr3-effort-medium`       | 3          | 61.5 | 100.0 | 44.4 | 16/36 (7/11, 8/13, 1/12)  | 27       | 5.45M / 153k    | 191                 | 1064 s |
| `pr3-effort-medium-iter6` | 6          | 66.7 | 100.0 | 50.0 | 18/36 (7/11, 9/13, 2/12)  | 27       | 9.39M / 244k    | 340                 | 1322 s |

Findings:

- Coverage grew with iterations and was nearly nested. Three iterations found every golden issue that one did, plus
  one medium and one low. Six found all of three's but one low, plus one medium and two lows. Every run found the same
  7 high issues.
- GF1 rose about 5 points per step: 56.0, 61.5, 66.7. Each step alone is within single-PR noise; the consistent
  direction and nested coverage are a stronger signal than any one difference.
- Six iterations used 1.72x the tokens of three (9.63M against 5.60M) and 1.78x the reviewer tool calls, but took only
  24% longer, with no retries. Reviewers account for most of the increase (8.90M against 5.20M): `gpt-6-sol` used 2.15x
  its tokens and `claude-opus-5-5` 1.44x. Deliberation rose from 0.34M to 0.65M tokens for 52 reviewer findings
  against 48, and aggregation produced 27 findings in both.
- Each step cost 3.5-4.0M more tokens for two more golden issues, roughly 2M per issue at either step.
- One iteration took longer than three because one `claude-opus-5-5` reviewer call stalled. The response timeout ended
  it after 600 s and the whole reviewer pass restarted, adding about 11 minutes. Its token and tool-call counts include
  the discarded attempt, so a clean run would cost slightly less. Its aggregation also took 285 s against 118 s, with
  0.20M tokens against 0.06M; the cause was not investigated.
- At about 2.1M tokens, one iteration costs the same as the earlier 3-iteration runs at 12 tool calls per iteration
  (about 2.2M), which scored 56.0 and 64.2.

Conclusion: each step from one to three to six iterations found about two more golden issues, all medium or low, at
roughly 2M tokens apiece; high-severity coverage never changed. One PR cannot confirm the trend, so the default stays
at 3 iterations; a 25-PR run at 6 would test it.

### Reviewer tool budget

Does allowing reviewers 30 tool calls per iteration, rather than 12, find more?

Both runs use the `pr3-effort-medium` setup: PR 3, the mixed matrix, reviewer effort `medium`, 3 iterations, no
operation-wide tool budget, and no output cap. Only `[pr_review.limits].reviewer_tool_calls_per_iteration` differs.
`run.json` does not record limits; the 12-call run read a copy of the config with that one limit added.

Code: [`bd9c380`][bd9c380] with the same uncommitted changes as the reasoning-effort experiment. By the 12-call run,
localmcplib also had the review fixes from its [#6][localmcplib-6]; none of them change requests at these settings,
and no call failed or retried.

| Run                         | Tool calls per iteration | GF1  | GP    | GR   | Golden (high / med / low) | Findings | Tokens in / out | Reviewer tool calls | Time   |
| --------------------------- | ------------------------ | ---- | ----- | ---- | ------------------------- | -------- | --------------- | ------------------- | ------ |
| `pr3-effort-medium-tools12` | 12                       | 61.5 | 100.0 | 44.4 | 16/36 (7/11, 7/13, 2/12)  | 19       | 2.87M / 95k     | 115                 | 499 s  |
| `pr3-effort-medium`         | 30                       | 61.5 | 100.0 | 44.4 | 16/36 (7/11, 8/13, 1/12)  | 27       | 5.45M / 153k    | 191                 | 1064 s |

Findings:

- The two budgets scored the same. They shared 14 of their 16 golden issues, including all 7 high ones. Each found
  two the other missed: 30 calls a medium and a low, 12 calls two lows. Every high or critical finding at either
  budget matched a golden issue.
- GF1 does not credit findings outside the golden set, and that hides a difference. The judge classified all 14 such
  findings at 30 calls as real, and 8 of 9 at 12 calls. Six issues appear in both runs. The 30-call run alone found 8
  more, 4 medium and 4 low by published severity: a Git token in the remote URL leaking through errors and saved
  tracking config (which the judge rated high), open tabs not reconciled on snapshot restore, failed disk writes not
  handled before later reads prefer the disk copy, and Git sync failing in browser mode; the lows were an unreachable
  modal, missing tests, unused snapshot diff code, and a documentation error. The 12-call run alone found two low
  issues, and its other finding was judged a false positive.
- 12 calls used 53% of the tokens (2.97M against 5.60M) and 47% of the time. The review phase took 309 s against
  822 s. `claude-opus-5-5` used 1.35M tokens against 3.36M, and `gpt-6-sol` 1.57M against 2.18M. The 30-call total
  includes one `claude-opus-5-5` reviewer pass that was retried after a provider error, so a clean run would cost
  somewhat less.
- Reviewers pace themselves to the stated budget rather than running out of it. The prompt gives the budget and asks
  for a result before it is exhausted. At 12, iterations made 5-12 calls, usually 10, and only 2 of 12 used all 12.
  At 30, they made 6-22, and none came near 30. The budget therefore sets how far reviewers explore, not only where
  they stop.
- The 12-call run cost 24-37% more than the two earlier 12-call runs of PR 3 (2.17M and 2.39M tokens) with similar
  reviewer tool calls (115 against 106 and 112). It scored between them (56.0 and 64.2). The gap likely comes from
  what those runs lacked: reasoning effort on every role and uncapped output. Deliberation alone used 0.41M tokens
  against 0.20-0.26M.
- One iteration at 30 calls (2.14M tokens) found 14 golden issues; three at 12 calls (2.97M) found 16. At similar cost,
  spreading calls across iterations found more.

Conclusion: on PR 3, 30 tool calls per iteration cost 1.9x the tokens and 2.1x the time of 12 for the same score and
golden count, but also found about 8 more real issues outside the golden set, 4 of them medium. That gain did not
justify roughly doubling cost and time, so the default returns to 12, the budget behind the 25-PR benchmark results.
The ceiling stays at 100 so experiments can still raise it through config. The next 25-PR run should compare findings
outside the golden set as well as GF1. Because iterations appear to add more than calls per iteration, a PR 3 run at 6 iterations and 12 calls,
which should cost about as much as 3 at 30, would test whether those tokens buy more as iterations.

[2fa4c61]: https://github.com/salesforce-misc/pr-council-mcp/commit/2fa4c617a26b0810fe548c1b7eedb12248f2a243
[#8]: https://github.com/salesforce-misc/pr-council-mcp/pull/8
[#9]: https://github.com/salesforce-misc/pr-council-mcp/pull/9
[bd9c380]: https://github.com/salesforce-misc/pr-council-mcp/commit/bd9c38090c25b4a762c5713591eaa0108e3a0d1c
[#10]: https://github.com/salesforce-misc/pr-council-mcp/pull/10
[ee5f5c4]: https://github.com/salesforce-misc/pr-council-mcp/commit/ee5f5c489b5727bc1be9fe5fe99445211dfda56a
[f96269a]: https://github.com/salesforce-misc/pr-council-mcp/commit/f96269acd681d75f0393c8efcd426316646c4f2b
[localmcplib-6]: https://github.com/salesforce-misc/localmcplib/pull/6
