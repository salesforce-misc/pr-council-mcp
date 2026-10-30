# pr-council-mcp

## Purpose

`pr-council-mcp` is a standalone stdio MCP server for durable, multi-model pull-request reviews. It is built on
`fastmcp`, `pydantic`, LangChain provider adapters, LangGraph, SQLite, and `uv`. The actual entrypoint is
`pr_council/server.py`, run with `uv run python -m pr_council.server`; the installed `pr-council-mcp` console
script invokes the same entrypoint.

A review is a long-lived, checkpointed operation. The calling agent starts it, polls it, presents an exact
revision-bound preview, and publishes only after explicit human approval. Publication creates inline comments and a
`COMMENT`-only GitHub summary; it never approves or rejects the pull request.

## Module layout

The package lives at `src/pr_council/` as a src-layout distribution.

Keep application code separated by responsibility:

- **`agents/<domain>/`** contains model-driven behavior: prompts, structured-output bindings, model tool selection,
  and judgment supplied by an LLM.
- **`workflows/<domain>/`** contains bounded coordination across agents, durable state, retries, approval gates,
  cancellation, and side effects.
- **Root domain packages** such as **`review/`** contain deterministic models, validation, transformations, durable
  stores, and external-service adapters. They must not own prompts or end-to-end workflow control.
- **`tools/`** is the MCP boundary. Tools validate/adapt MCP inputs, resolve the lifespan-owned runtime, and invoke
  domain workflows; business logic does not belong in tool functions.

The current modules are organized as follows:

- **`pr_council/config.py`** — the application schema over localmcplib's shared, versioned TOML document and XDG
  paths. It owns only frozen PR-review configuration models and rejects unknown keys in this server's effective view.
- **`localmcp.secrets`** and **`localmcp.llm`** own shared secret resolution, model capabilities, transport routing,
  and owned model lifetimes. The application supplies its effective endpoint, logical `llm_api_key`, and role model
  IDs.
- **`localmcp.sandbox`** owns the generic model-facing Bash tool and its platform enforcement: Seatbelt on macOS, and
  bubblewrap and seccomp on Linux, behind a shared check of each command's Bash syntax and command names. Agents
  import only the portable `localmcp.sandbox` interface. The review graph constructs its repository-specific root
  grants and exclusions.
- **`pr_council/review/`** — deterministic PR-review contracts:
  - `git.py` owns bounded `git`/`gh` execution, PR URL parsing, managed clone/worktree operations, local-source
    validation, immutable endpoint pinning, GitHub publication, and trusted Git-metadata validation.
  - `models.py` owns the shared review, finding, preview, status, and error models.
  - `diff.py` contains pure unified-diff/commentability helpers.
  - `policy.py` owns deterministic normalization, rendering, reconciliation, and payload hashing.
  - `locks.py` owns per-repository async locks.
  - `store.py` owns the durable operation catalog beside LangGraph's checkpoint database.
- **`pr_council/agents/review/`** — the reviewer, closed-set deliberator, and aggregator. Role-specific prompt text
  is colocated with its owning agent module. `common.py` owns the shared safety prompt builder, tool budgets, and usage
  accounting; `recording.py` owns the incremental-recording agent loop shared by the deliberator and aggregator.
  Every agent is built with `localmcp.agents.create_agent`, whose `localmcp.retry.ModelCallRetryMiddleware` retries
  each failed model call with rate-limit backoff, so a transient fault never restarts an agent run or repeats its
  tool calls. Repository content, diffs, tool output, prior findings, and caller context are always untrusted data.
- **`localmcp.workflows.runtime.WorkflowRuntime`** owns state-directory permissions, store/checkpointer lifecycle,
  operation leases and heartbeats, background-task tracking, and recovery scheduling. Domain subclasses own graph
  construction, cancellation mapping, and domain-resource cleanup.
- **`pr_council/workflows/registry.py`** — deterministic bootstrap imports for registered workflow-runtime
  subclasses. A runtime whose module is never imported cannot register itself.
- **`pr_council/workflows/review/graph.py`** — the checkpointed review LangGraph: preparation, parallel review,
  deliberation, aggregation, preview interrupt, stale-state checks, publication, and cleanup.
- **`pr_council/workflows/review/runtime.py`** — the review domain runtime. It builds the graph and collaborators,
  validates and persists start requests, resumes recoverable operations, and exposes owned lifecycle methods.
- **`pr_council/tools/review.py`** — the five `pr_council_` lifecycle tools and their module-level `TOOLS` list.
  `tools/__init__.py` exposes the package-level `TOOLS` registered by `server.py`. Operational diagnostics do not
  belong in the model-facing tool surface; use logs or the underlying service instead.
- **`localmcp.observability`** owns stdio-safe logging and optional, failure-isolated Langfuse tracing and MCP
  middleware. It is not exposed as an MCP diagnostic tool.
- **`pr_council/server.py`** — a thin `localmcp.STDIOServer` declaration. `localmcp.server` owns config/secret/model
  composition, the process-global runtime, FastMCP registration, lifespan, logging, telemetry, and stdio serving.
- **`pr_council/server_setup.py`** — this application's first-run shared config template and selected-model credential
  checks at review start.

## Agent interaction design

The calling agent owns the conversation. MCP tools expose durable capabilities rather than conducting interviews:

- Gather, edit, and confirm arbitrary review context in normal conversation, then pass the finalized snapshot to
  `pr_council_start`.
- Return explicit structured state and previews. The calling agent retains opaque IDs, polls operations, and presents
  human-readable results without asking the user to copy implementation identifiers.
- Keep read-only preparation separate from publication. A preview can be fetched repeatedly; only
  `pr_council_commit` may publish it.
- Treat tool-call arguments as workflow guidance, not proof of human authorization. MCP clients may enforce their own
  tool-call approval policy, but the server still requires revision and payload-hash binding.
- Revalidate externally sourced identities and revision state immediately before publication.

Do not add MCP elicitation for context gathering or iterative editing. If server-verifiable human authorization ever
becomes mandatory, use an authenticated out-of-band mechanism rather than a boolean or confirmation phrase supplied
by a model.

### Durable lifecycle compatibility

Ordinary MCP tools are the portable baseline. Do not make MCP Tasks a required dependency until all target clients
support them; Tasks may later be an optional transport over the same internal operation model.

The lifecycle is:

- `pr_council_start` validates inputs, creates the durable operation, and returns its opaque ID and label.
- `pr_council_get` reports state, progress, errors, preview readiness, and terminal results.
- `pr_council_preview` returns the exact revision-bound publication payload without writing.
- `pr_council_commit` resumes the checkpointed graph with the approved revision and payload hash.
- `pr_council_cancel` requests cooperative cancellation before publication begins.

Every operation uses its `operation_id` as the LangGraph `thread_id`. The operation catalog is the query and
coordination plane: it owns caller binding, status, request/state snapshots, preview/result data, and leases. The
LangGraph checkpointer is the execution plane: it owns node progress and resumable graph state. Do not collapse these
two stores into one abstraction.

Operations are bound to the local OS UID; possession of an ID alone is not authorization. Runtime startup recovers
nonterminal operations. Multiple independent operations may run concurrently, while preparation/publication for the
same repository uses a per-repository lock. Consequential writes must remain replay-safe or protected by durable
state at the side-effect boundary.

## Pull-request review contract

### Source preparation

`pr_council_start` accepts a GitHub PR URL and an explicit `managed` or `local` source mode. Managed mode remains the
default. The runtime stores managed data beneath
`<XDG_STATE_HOME or ~/.local/state>/localmcp/pr-council-mcp/review/`:

- `repos/` contains managed blobless repository clones;
- `worktrees/` contains operation-specific detached linked worktrees;
- `locks/` contains repository lock state;
- `operations.sqlite3` is the durable operation catalog;
- `checkpoints.sqlite3` is the LangGraph checkpoint store.

Managed preparation obtains the advertised base/head SHAs, fetches the PR head, pins both endpoints under an
operation-scoped ref namespace, hydrates the exact three-dot comparison, and creates a detached worktree at the head.
The trusted Git adapter validates the linked worktree's `.git` marker and common metadata directory before either is
granted to a model sandbox.

Local preparation does not clone, fetch, update refs, create a worktree, or clean up caller files. The requested path
defaults to the server startup working directory and must resolve beneath it, as must the repository's common Git
metadata. It must be the repository root, have a clean worktree and submodules, point `HEAD` at the advertised PR head,
and already contain the PR base/head objects and merge base. Existing ignored paths are excluded from model sandbox
authority; tracked symlinks must resolve inside the repository; and Git alternates outside the repository/common
metadata roots are unsupported. These properties and the ignored-path snapshot are revalidated before preview and
publication. Local repositories are caller-owned and must never be removed or mutated during cleanup.
Local metadata validation reads every non-command config scope, including per-worktree `config.worktree`, and
rejects executable, include-based, or credential-bearing repository configuration unless
`[pr_review.allowed_local_git_config]` permits the value; model sandboxes deny hooks, reflogs, and transient Git
state while retaining the minimal repository config needed by Git. Each submodule's metadata and configuration is
validated before any recursive command such as `git status` can run its filters. Every trusted `git` and `gh` call
drops relative `PATH` entries and directories inside the repository it runs in or the local-source checkout, so
neither can supply the programs it runs.

Local mode intentionally reviews the caller-owned live checkout so it can avoid a second full worktree for very large
trees. The server and checkout owner therefore share one OS-UID trust boundary: callers must keep the checkout stable
while agents run. Revalidation detects persistent or accidental drift before preview/publication, but cannot exclude
a same-UID process making and restoring a transient edit. Callers that do not accept that boundary must use managed
mode.

Both modes separately obtain GitHub's canonical PR diff for inline-comment validation. In local mode, agents inspect
the validated local worktree and base/head objects; the canonical remote patch is not their source view. A second
local validation after obtaining that patch closes the local-mutation window while a separate endpoint check detects
PR-side movement.

The preview is bound to base SHA, head SHA, revision, and payload hash. Commit rechecks both PR endpoints, the managed
fetched head or local-source snapshot, authenticated GitHub identity, duplicate-publication state, and exact preview
payload before posting. If either endpoint or the local snapshot moves, the operation becomes stale instead of
publishing comments against a different revision.

### Agent sandbox

Reviewer and deliberator agents receive exactly one bounded `Bash` tool for source inspection. Its profile contains
two read-only roots: the selected source tree first (therefore `cwd`) and its validated common Git metadata directory
second. This permits ordinary local `git`, `rg`, `sed`, `awk`, and other system inspection utilities without
maintaining a bespoke set of read/list/search/diff tools.

The platform sandbox, not command curation, is the security boundary:

- writes, outbound network, shared-memory IPC, and access outside declared roots are denied;
- local-mode ignored paths are explicitly denied within the otherwise read-only source root;
- the caller environment and credentials are not inherited;
- Git global/system config and terminal prompting are disabled;
- command length, wall time, CPU time, output bytes, open files, process creation, and per-pass calls are bounded;
- the model can read shared clone history, objects, refs, and branches through the Git metadata root;
- command names must be granted tools or supported builtins, but on Linux that is only a shell-level check: a granted
  tool such as `find` or `awk` can launch other programs, including repository code, inside the sandbox;
- prompts additionally prohibit executing repository code, package managers, language runtimes, tests, or builds.

The generic sandbox must remain ignorant of GitHub, repository-cache layout, worktree markers, and review SHAs. The
trusted review layer constructs root grants and embeds validated base/head literals in agent prompts.

### Review roles and follow-ups

Quality and security reviewer matrices execute independently for the configured number of refinement iterations.
Reviewer and deliberator prompts share one severity rubric and one scope: a renamed or moved file is in scope for the
effects of the move even when no line changes, such as relative references that no longer resolve, references to its
old path, or content that does not suit its new location. A defect in unchanged content that the move does not affect
predates the PR and is out of scope, as it would be for a human reading the diff. Each disposition then receives a
closed-set deliberation pass that keeps, revises, or removes every finding. Removal is only for a disproven defect. A
revision corrects the title, body, severity, and evidence of a real but misstated defect while its ID and location
stay fixed; the reviewer's original remains in the reviewer results. Deliberation does not remove duplicates, so
aggregation can credit every model that found an issue. The aggregator merges findings that describe the same defect,
one that a single change at one location fixes; findings in different places that each need their own change stay
separate. Findings it does not merge are published exactly as deliberated, and it cannot introduce new source-finding
IDs. Deterministic validation derives sensitive metadata, including merged severity, from source findings rather than
trusting model output.

The deliberator and aggregator are recording agents: each decision, prior-finding assessment, or merge is its own
tool call, validated as it arrives against a closed ID set, so no single response carries the whole result and a
rejected call records nothing. When a response reaches the output limit, its final call is rejected for the model to
record again; a response cut off before any call, or a deliberator reply that stops with IDs unrecorded, is answered
with what remains, a bounded number of times. A failure that outlasts its model-call retries keeps what was already
recorded; a failure before anything is recorded, such as a rejected credential or model, fails the operation instead.
The graph then closes deliberation over the supplied IDs: an undecided finding is kept as written and an unassessed
prior finding becomes `cannot_verify`, both logged. The summary warns when findings went undeliberated or aggregation
ended early. A reviewer pass that fails after its model-call retries fails that reviewer job rather than running
again, and the summary reports the degraded coverage.

The top-level summary is rendered in code from counts of the published findings and follow-up assessments; no model
writes it. `[pr_review].min_severity` (default `info`), or the start tool's per-operation override, sets the lowest
severity that is published. Findings below it are listed in the preview with a suppression note and excluded from the
comments, summary counts, payload hash, and the completed result that later follow-ups use as their baseline. The
exception is a suppressed finding that carries forward an unresolved prior finding: an earlier review published it,
so it stays in that baseline, and the summary reports how many unresolved issues fall below the threshold.

`auto` mode uses the latest completed review by the same local owner for the same PR as a baseline when available.
`initial` forbids a baseline. `follow_up` requires one. Follow-ups compare context snapshots and require each relevant
disposition to assess every prior finding before publication.

## Config and secret contract

`localmcplib` is pinned to an exact version in `pyproject.toml`. It owns the shared config document, secret
resolution, and model catalog this server builds on, so upgrades are deliberate: bump the pin, re-lock, and adapt this
repository to any shared config, secret, catalog, or API changes in the same change — including `config.py`,
`server_setup.py`'s first-run template, tests, and this document.

- Config file: `~/.config/localmcp/localmcp.toml`, or `$XDG_CONFIG_HOME/localmcp/localmcp.toml` when
  `XDG_CONFIG_HOME` is an absolute path.
- If the file is absent, startup creates a native-backend base document with `OPENAI_API_KEY` and
  `ANTHROPIC_API_KEY` secret aliases. Missing credential values do not prevent startup; starting a review reports
  which selected model needs a key and how to provide it.
- State root: `~/.local/state/localmcp/pr-council-mcp`, or beneath an absolute `$XDG_STATE_HOME`.
- `schema_version`, `[observability]`, `[llm]` (including `[llm.models]` capability overrides), and `[secrets]` are
  shared. Application-specific `[pr_review]` configuration should live beneath `[server.pr-council-mcp]`.
- Unknown keys in this server's effective application view are rejected with `ConfigError`.
- `LOCALMCP_LOG_LEVEL` overrides `[observability].log_level`; unset values fall back to the file and invalid
  values degrade to `WARNING`.
- Endpoint URLs are configuration, not secrets. The `openai_compatible` LLM endpoint lives in `[llm].base_url`.
- Secret declarations contain ordered environment aliases. Resolution checks those aliases first and then the shared
  OS-keyring service `localmcp`, using the logical secret name as the account. The `openai_compatible` backend uses
  `llm_api_key`; the `native` backend uses `openai_api_key` and `anthropic_api_key` as required by the selected
  models. GitHub authentication is owned by `gh`, using its credential store or `GH_TOKEN`/`GITHUB_TOKEN`.
  `[pr_review.github_accounts]` optionally maps `host`, `host/owner`, or `host/owner/repo` to a `gh` login. A mapped
  operation records the login in its request snapshot, verifies it at start, and passes that account's token to each
  `gh` and networked `git` call (SSH remotes rewritten to HTTPS, `gh` as the only credential helper) instead of
  inherited token variables. Never switch `gh`'s active account; unmapped hosts keep the inherited environment.
- Never log secret values. Logs may identify only the logical secret name and resolution source.

`[llm].backend` is explicitly either `openai_compatible` or `native`; it is never inferred from available secrets.
The compatible backend shares one endpoint and `llm_api_key` across model dialects. The native backend routes each
model through its registered provider and corresponding personal credential. Switching backends must not require
changes to `[pr_review.models]`. The model catalog is advisory: model IDs are resolved through the configured
factory's registry (the built-in catalog plus shared `[llm.models]` overrides), unlisted IDs use inferred capabilities,
and the provider remains the authority on whether a model exists. Review start still rejects models the selected
backend cannot route or whose credential is missing.

`[pr_review.models]` owns the quality/security reviewer lists and the deliberation/aggregation role models. The start
tool accepts optional per-operation overrides; the runtime validates, deduplicates, bounds, and persists the resolved
selection so later config changes cannot alter an in-flight operation. The resolved `min_severity` and whether it came
from config or an override are persisted the same way.

`[pr_review.reasoning_effort]` sets one effort per role: `review` for every reviewer, and `deliberation` and
`aggregation`, each defaulting to `medium`. Effort is configuration only; the start tool does not override it.
At start, each selected model receives its role's effort when its catalog spec accepts one and none otherwise, and that
per-model resolution is persisted with the model selection. The provider remains the authority on which levels a model
supports.

`[pr_review.allowed_local_git_config]` maps extra local-source Git config keys to exact values or a full-match
`pattern`. Section and variable names are lowercased while subsections keep their case, as Git compares them. Known
keys whose values Git runs as programs require exact values. Keys accepted by default cannot be listed, so their own
checks, such as remote URL validation, always apply, and `include`, `includeIf`, and `core.hooksPath` are never
allowable. Validation and configuration errors name the key without its subsection, which could carry a credential,
and never echo its value. The allowlist is read from current configuration, not persisted per operation: it governs
a security check, so a tightened allowlist must apply to in-flight operations when they revalidate.

`[pr_review.limits]` bounds per-pass Bash calls, matrix size, model concurrency, prepared repository size, command
timeout, attempts per model call, and retry delays. Every Bash result reports the remaining budget. Model output is
deliberately uncapped: localmcplib requests each model's maximum, because a response truncated mid-reasoning must be
repeated anyway.

Langfuse is optional and environment-configured. It requires `LOCALMCP_LANGFUSE_ENABLED=true`,
`LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`; the client is always installed. Langfuse
failures must never change application behavior.
Tool inputs and outputs are captured only when `LOCALMCP_LANGFUSE_CAPTURE_PAYLOADS=true` is explicitly set.

## MCP tool naming

Keep MCP names in the `pr_council_<verb>_<noun>` form. The model-facing surface is intentionally limited to the five
review lifecycle tools. Do not add general server-info, health, telemetry, or diagnostic tools when logs, MCP
initialization metadata, or the underlying service provide the same information.

Tools are the decorated functions themselves and each module exposes a `TOOLS` list. `tools/__init__.py` explicitly
aggregates those lists; `server.py` registers every function in package `TOOLS` without wrappers. Tool collaborators
must be resolved through module globals at call time so tests can monkeypatch the boundary cleanly.

## Observability

This is a stdio MCP server: stdout and stderr carry the JSON-RPC stream. A stray write corrupts the protocol.

- Logging is structured JSON and file-only at
  `~/.local/state/localmcp/pr-council-mcp/logs/pr-council-mcp.log`, or beneath `$XDG_STATE_HOME`.
- `configure_logging()` installs only a file handler and falls back to a null handler if the file cannot be opened;
  it never attaches a stream handler or corrupts stdio.
- Import-time setup neutralizes structlog's stdout default, stdlib `logging.lastResort`, logging exception tracebacks,
  FastMCP's stderr handler, and the default warnings path.
- A fatal startup or serving error exits nonzero silently; diagnostics go to the file log.
- Never `print`, attach a stream handler, or log credentials or untrusted full payloads.
- Langfuse tracing is off by default and failure-isolated. It traces MCP tools and workflow/model spans but exposes no
  MCP diagnostic endpoint.

## Running

```bash
uv sync
uv run python -m pr_council.server   # stdio MCP server
uv run pytest
make ci                                 # format-check, lint, strict mypy, tests + coverage
```

Runtime prerequisites are macOS or Linux, Python 3.12+, `git`, authenticated `gh`, and `rg`. Linux also requires
bubblewrap 0.6.3 or newer, Bash, libseccomp, and `file` and `git` from the system or a supported Nix profile, plus
unprivileged user and cgroup namespaces. The checked-in `.mcp.json` and `opencode.jsonc` start the module through
`uv` when the client is launched from the repository root. Machine-local configuration that contains credentials must
remain untracked.
