import pytest
from localmcp.config import ServerConfig
from pydantic import ValidationError

from pr_council.config import (
    ConfigError,
    LocalGitConfigPattern,
    PRReviewLimitsConfig,
    PRReviewModelsConfig,
    parse_config,
)
from pr_council.review.models import Severity


def test_empty_application_config_uses_defaults() -> None:
    config = parse_config(ServerConfig(name="pr-council-mcp", values={}))

    assert config.pr_review.models.quality == ["claude-opus-5-5", "gpt-6-sol"]
    assert config.pr_review.models.security == ["claude-opus-5-5", "gpt-6-sol"]
    assert config.pr_review.models.deliberation == "claude-opus-5-5"
    assert config.pr_review.models.aggregation == "claude-sonnet-5"
    assert config.pr_review.allowed_hosts == ["github.com"]
    assert config.pr_review.min_severity == Severity.INFO
    assert config.pr_review.limits.reviewer_tool_calls_per_iteration == 12
    assert config.pr_review.limits.deliberator_tool_calls_per_disposition == 20
    effort = config.pr_review.reasoning_effort
    assert (effort.review, effort.deliberation, effort.aggregation) == ("medium", "medium", "medium")


def test_parse_config_reads_min_severity() -> None:
    config = parse_config(ServerConfig(name="pr-council-mcp", values={"pr_review": {"min_severity": "medium"}}))

    assert config.pr_review.min_severity == Severity.MEDIUM


def test_parse_config_rejects_unknown_min_severity() -> None:
    with pytest.raises(ConfigError, match="invalid application configuration"):
        parse_config(ServerConfig(name="pr-council-mcp", values={"pr_review": {"min_severity": "severe"}}))


def test_parse_config_reads_reasoning_effort_and_rejects_unknown_levels() -> None:
    config = parse_config(
        ServerConfig(name="pr-council-mcp", values={"pr_review": {"reasoning_effort": {"review": "max"}}})
    )

    assert config.pr_review.reasoning_effort.review == "max"
    assert config.pr_review.reasoning_effort.deliberation == "medium"
    with pytest.raises(ConfigError, match="invalid application configuration"):
        parse_config(
            ServerConfig(name="pr-council-mcp", values={"pr_review": {"reasoning_effort": {"review": "extreme"}}})
        )


@pytest.mark.parametrize("key", ["operation_tool_calls", "model_output_tokens_per_call"])
def test_parse_config_rejects_retired_limits(key: str) -> None:
    with pytest.raises(ConfigError, match=key):
        parse_config(ServerConfig(name="pr-council-mcp", values={"pr_review": {"limits": {key: 100}}}))


def test_parse_config_accepts_enterprise_github_host() -> None:
    config = parse_config(
        ServerConfig(
            name="pr-council-mcp",
            values={"pr_review": {"allowed_hosts": ["github.com", "github.enterprise.example"]}},
        )
    )

    assert config.pr_review.allowed_hosts == ["github.com", "github.enterprise.example"]


@pytest.mark.parametrize("host", ["https://example.com", "example.com:443", "example.com/path", "bad..host"])
def test_parse_config_rejects_invalid_allowed_host(host: str) -> None:
    with pytest.raises(ConfigError, match="allowed_hosts must contain valid hostnames"):
        parse_config(ServerConfig(name="pr-council-mcp", values={"pr_review": {"allowed_hosts": [host]}}))


def test_parse_config_accepts_application_only_server_config() -> None:
    config = parse_config(
        ServerConfig(
            name="pr-council-mcp",
            values={"pr_review": {"models": {"quality": ["gpt-5.6"]}}},
        )
    )

    assert config.pr_review.models.quality == ["gpt-5.6"]
    assert set(config.model_dump()) == {"pr_review"}


def test_unknown_application_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="invalid application configuration"):
        parse_config(ServerConfig(name="pr-council-mcp", values={"unknown": "boom"}))


def test_review_limits_and_model_ids_remain_bounded() -> None:
    with pytest.raises(ValidationError, match="rate_limit_base_delay_seconds"):
        PRReviewLimitsConfig(rate_limit_base_delay_seconds=10, rate_limit_max_delay_seconds=5)
    with pytest.raises(ValidationError, match="model IDs"):
        PRReviewModelsConfig(quality=["bad model id"])


def test_github_accounts_are_normalized_and_bound_to_allowed_hosts() -> None:
    config = parse_config(
        ServerConfig(
            name="pr-council-mcp",
            values={"pr_review": {"github_accounts": {"GitHub.com": "host_user", "github.com/Acme/Repo": "repo-user"}}},
        )
    )

    assert config.pr_review.github_accounts == {"github.com": "host_user", "github.com/acme/repo": "repo-user"}


@pytest.mark.parametrize(
    ("accounts", "message"),
    [
        ({"github.com/acme/repo/extra": "user"}, "keys must be"),
        ({"github.com/..": "user"}, "keys must be"),
        ({"github.com": "not a login"}, "not a valid GitHub login"),
        ({"github.com/acme": "a", "github.com/ACME": "b"}, "duplicate key"),
        ({"ghe.example": "user"}, "must be listed in allowed_hosts: ghe.example"),
    ],
)
def test_github_accounts_reject_invalid_entries(accounts: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(ServerConfig(name="pr-council-mcp", values={"pr_review": {"github_accounts": accounts}}))


def test_allowed_local_git_config_normalizes_keys_and_accepts_exact_values_or_patterns() -> None:
    config = parse_config(
        ServerConfig(
            name="pr-council-mcp",
            values={
                "pr_review": {
                    "allowed_local_git_config": {
                        "Filter.LFS.Clean": ["git-lfs clean -- %f"],
                        "filter.lfs.clean": ["git-lfs clean -- %f"],
                        "Tool.Origin.Mode": ["fast"],
                        "tool.last-run": {"pattern": "[0-9]{1,12}"},
                        "tool.empty": [""],
                    }
                }
            },
        )
    )

    # Git ignores case in section and variable names but not in subsections.
    rules = config.pr_review.allowed_local_git_config
    assert set(rules) == {"filter.LFS.clean", "filter.lfs.clean", "tool.Origin.mode", "tool.last-run", "tool.empty"}
    assert rules["filter.LFS.clean"] == ["git-lfs clean -- %f"]
    assert rules["tool.last-run"] == LocalGitConfigPattern(pattern="[0-9]{1,12}")
    assert rules["tool.empty"] == [""]


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ({"nodot": ["x"]}, "is not a Git configuration key"),
        ({"tool.": ["x"]}, "is not a Git configuration key"),
        ({"tool.1mode": ["x"]}, "is not a Git configuration key"),
        ({"include.path": ["extra.cfg"]}, 'cannot allow "include.path"'),
        ({"includeIf.gitdir:/work/.path": ["extra.cfg"]}, "cannot allow"),
        ({"core.hooksPath": ["/dev/null"]}, "cannot allow"),
        ({"filter.lfs.clean": {"pattern": "git-lfs .*"}}, "runs a command and must list exact values"),
        ({"core.fsmonitor": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"pager.log": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"mergetool.Tool.cmd": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"remote.origin.uploadPack": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"submodule.child.update": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"gpg.ssh.defaultKeyCommand": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"imap.tunnel": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"sendemail.sendmailCmd": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"sendemail.work.sendmailCmd": {"pattern": ".*"}}, "runs a command and must list exact values"),
        ({"hook.lint.command": {"pattern": ".*"}}, "runs a command and must list exact values"),
        # Built-in keys keep their own checks, such as remote URL transport and credential validation.
        ({"Remote.Origin.URL": ["ext::sh -c true"]}, "is accepted by default and cannot be listed"),
        ({"filter.lfs.required": ["true"]}, "is accepted by default and cannot be listed"),
        ({"tool.mode": []}, "at least 1 item"),
        ({"tool.mode": ["x"] * 33}, "at most 32 items"),
        ({"tool.mode": ["a" * 1025]}, "at most 1024 characters"),
        ({"tool.mode": ["a\0b"]}, "cannot contain NUL"),
        ({"tool.mode": {"pattern": "("}}, "not a valid regular expression"),
        ({"tool.mode": {"pattern": ""}}, "at least 1 character"),
        ({"tool.mode": ["a"], "Tool.Mode": ["b"]}, "duplicate key"),
        ({"tool.Sub.mode": ["a"], "TOOL.Sub.MODE": ["b"]}, "duplicate key"),
    ],
)
def test_allowed_local_git_config_rejects_invalid_rules(rules: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(ServerConfig(name="pr-council-mcp", values={"pr_review": {"allowed_local_git_config": rules}}))


def test_allowed_local_git_config_errors_never_echo_subsections_or_input() -> None:
    rules = {"credential.https://user:secret-token@example.invalid.helper": {"pattern": ".*"}}

    with pytest.raises(ConfigError, match='"credential.<subsection>.helper" runs a command') as error:
        parse_config(ServerConfig(name="pr-council-mcp", values={"pr_review": {"allowed_local_git_config": rules}}))
    assert "secret-token" not in str(error.value)
