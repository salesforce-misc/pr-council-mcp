"""Application schema over the shared localmcp configuration document."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from localmcp.config import ServerConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pr_council.errors import PrCouncilError
from pr_council.review.models import Severity

SERVER_NAME = "pr-council-mcp"
_MODEL_ID = re.compile(r"^[A-Za-z0-9._:/-]+$")
_HOSTNAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$")
_GITHUB_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_GITHUB_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,38})$")
# Git config keys as normalize_git_config_key returns them: section, optional subsection, variable.
_GIT_CONFIG_KEY = re.compile(r"^[a-z0-9-]+(?:\.[^\n\0]+)?\.[a-z][a-z0-9-]*$")
# Keys local-source validation accepts without an allowlist entry, with remote URLs checked for executable transports
# and credentials. An entry could not relax those checks, so these keys cannot be listed.
SAFE_LOCAL_GIT_CONFIG_KEY = re.compile(
    r"^(?:"
    r"core\.(?:repositoryformatversion|filemode|bare|logallrefupdates|ignorecase|precomposeunicode|symlinks|worktree|"
    r"untrackedcache|sparsecheckout|sparsecheckoutcone)|"
    r"extensions\.(?:objectformat|worktreeconfig|partialclone)|"
    r"index\.(?:version|skiphash|sparse)|"
    r"fetch\.(?:all|parallel)|"
    r"http\.postbuffer|"
    r"filter\..+\.required|"
    r"remote\..+\.(?:url|pushurl|fetch|mirror|promisor|partialclonefilter|tagopt)|"
    r"branch\..+\.(?:remote|merge|rebase)|"
    r"submodule\..+\.(?:url|active)"
    r")$"
)
# Their values name configuration or programs whose content validation never reads, so pinning a value pins nothing.
_UNALLOWABLE_GIT_CONFIG_KEY = re.compile(r"^(?:include\.[^\n\0]+|includeif\.[^\n\0]+|core\.hookspath)$")
# Known keys whose values Git runs as programs. They must list exact values; this guards against an over-broad
# pattern but is not a security boundary, since keys absent from the allowlist are denied anyway.
_GIT_COMMAND_KEY = re.compile(
    r"^(?:filter\..+\.(?:clean|smudge|process)|diff\.(?:.+\.)?(?:textconv|command|external)|merge\..+\.driver|"
    r"(?:difftool|mergetool|man|browser)\..+\.(?:cmd|path)|guitool\..+\.cmd|trailer\..+\.(?:cmd|command)|"
    r"tar\..+\.command|credential\.(?:.+\.)?helper|gpg\.(?:.+\.)?program|gpg\.ssh\.defaultkeycommand|"
    r"remote\..+\.(?:uploadpack|receivepack|vcs)|submodule\..+\.update|alias\..+|pager\..+|sequence\.editor|"
    r"interactive\.difffilter|uploadpack\.packobjectshook|instaweb\.httpd|imap\.tunnel|hook\..+\.command|"
    r"sendemail\.(?:.+\.)?(?:smtpserver|sendmailcmd|tocmd|cccmd|headercmd)|"
    r"core\.(?:alternaterefscommand|askpass|editor|fsmonitor|gitproxy|pager|sshcommand))$"
)
DEFAULT_ALLOWED_HOSTS = ("github.com",)
ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]
LocalGitConfigValue = Annotated[str, Field(max_length=1024)]


def normalize_git_config_key(key: str) -> str:
    """Lowercase a Git config key's section and variable names, which Git compares without case, but not its
    subsection, which Git compares exactly."""
    section, _, rest = key.partition(".")
    subsection, dot, variable = rest.rpartition(".")
    return f"{section.lower()}.{subsection}{dot}{variable.lower()}"


def git_config_key_label(key: str) -> str:
    """Name a Git config key for an error without its subsection, which may carry a credential."""
    section, _, rest = key.partition(".")
    subsection, dot, variable = rest.rpartition(".")
    return f"{section}.<subsection>.{variable}" if dot else key


class ConfigError(PrCouncilError):
    """Raised when the shared document or application view is invalid."""


class PRReviewLimitsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer_tool_calls_per_iteration: int = Field(default=12, ge=1, le=100)
    deliberator_tool_calls_per_disposition: int = Field(default=20, ge=1, le=20)
    models_per_disposition: int = Field(default=10, ge=1, le=10)
    concurrent_model_calls: int = Field(default=4, ge=1, le=20)
    repository_size_mib: int = Field(default=5_120, ge=100, le=102_400)
    source_tool_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    model_max_attempts: int = Field(default=3, ge=1, le=8)
    rate_limit_base_delay_seconds: float = Field(default=2.0, ge=0.0, le=30.0)
    rate_limit_max_delay_seconds: float = Field(default=90.0, ge=1.0, le=300.0)

    @model_validator(mode="after")
    def _check_rate_limit_delays(self) -> PRReviewLimitsConfig:
        if self.rate_limit_base_delay_seconds > self.rate_limit_max_delay_seconds:
            raise ValueError("rate_limit_base_delay_seconds must not exceed rate_limit_max_delay_seconds")
        return self


class PRReviewModelsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    quality: list[str] = Field(
        default_factory=lambda: ["claude-opus-5-5", "gpt-6-sol"],
        min_length=1,
        max_length=10,
    )
    security: list[str] = Field(
        default_factory=lambda: ["claude-opus-5-5", "gpt-6-sol"],
        min_length=1,
        max_length=10,
    )
    deliberation: str = "claude-opus-5-5"
    aggregation: str = "claude-sonnet-5"

    @field_validator("quality", "security")
    @classmethod
    def _check_roster_charset(cls, value: list[str]) -> list[str]:
        for entry in value:
            if not _MODEL_ID.fullmatch(entry):
                raise ValueError("model IDs may contain only letters, numbers, dot, underscore, colon, slash, and dash")
        return value

    @field_validator("deliberation", "aggregation")
    @classmethod
    def _check_role_charset(cls, value: str) -> str:
        if not _MODEL_ID.fullmatch(value):
            raise ValueError("model IDs may contain only letters, numbers, dot, underscore, colon, slash, and dash")
        return value


class PRReviewReasoningEffortConfig(BaseModel):
    """Per-role reasoning effort, applied to each selected model that accepts one.

    The provider remains the authority on which levels a model supports.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    review: ReasoningEffort = "medium"
    deliberation: ReasoningEffort = "medium"
    aggregation: ReasoningEffort = "medium"


class LocalGitConfigPattern(BaseModel):
    """A regular expression that a local-source Git config value must match in full."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: str = Field(min_length=1, max_length=256)

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"pattern is not a valid regular expression: {exc}") from exc
        return value


class PRReviewConfig(BaseModel):
    # Errors name allowed_local_git_config keys only through git_config_key_label, never by echoing input.
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    allowed_hosts: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_HOSTS), min_length=1, max_length=20)
    models: PRReviewModelsConfig = Field(default_factory=PRReviewModelsConfig)
    reasoning_effort: PRReviewReasoningEffortConfig = Field(default_factory=PRReviewReasoningEffortConfig)
    limits: PRReviewLimitsConfig = Field(default_factory=PRReviewLimitsConfig)
    # Findings below this severity are listed in the preview but never published.
    min_severity: Severity = Severity.INFO
    # gh account per "host", "host/owner", or "host/owner/repo"; the most specific match wins.
    github_accounts: dict[str, str] = Field(default_factory=dict, max_length=100)
    # Extra repository Git config a local-source checkout may carry, by key: exact values, or a full-match pattern.
    allowed_local_git_config: dict[
        str, Annotated[list[LocalGitConfigValue], Field(min_length=1, max_length=32)] | LocalGitConfigPattern
    ] = Field(default_factory=dict, max_length=128)

    @field_validator("allowed_hosts")
    @classmethod
    def _check_allowed_hosts(cls, value: list[str]) -> list[str]:
        hosts = [host.lower() for host in value]
        if any(len(host) > 253 or not _HOSTNAME.fullmatch(host) for host in hosts):
            raise ValueError("allowed_hosts must contain valid hostnames without schemes, ports, or paths")
        return list(dict.fromkeys(hosts))

    @field_validator("github_accounts")
    @classmethod
    def _check_github_accounts(cls, value: dict[str, str]) -> dict[str, str]:
        accounts: dict[str, str] = {}
        for key, login in value.items():
            parts = key.lower().split("/")
            if len(parts) > 3 or any(part in {".", ".."} or not _GITHUB_SEGMENT.fullmatch(part) for part in parts[1:]):
                raise ValueError('github_accounts keys must be "host", "host/owner", or "host/owner/repo"')
            if not _GITHUB_LOGIN.fullmatch(login):
                raise ValueError(f'github_accounts value for "{key}" is not a valid GitHub login')
            normalized = "/".join(parts)
            if normalized in accounts:
                raise ValueError(f'github_accounts has duplicate key "{key}"')
            accounts[normalized] = login
        return accounts

    @field_validator("allowed_local_git_config")
    @classmethod
    def _check_allowed_local_git_config(
        cls, value: dict[str, list[str] | LocalGitConfigPattern]
    ) -> dict[str, list[str] | LocalGitConfigPattern]:
        rules: dict[str, list[str] | LocalGitConfigPattern] = {}
        for key, rule in value.items():
            normalized = normalize_git_config_key(key)
            label = git_config_key_label(key)
            if len(normalized) > 256 or not _GIT_CONFIG_KEY.fullmatch(normalized):
                raise ValueError(f'allowed_local_git_config key "{label}" is not a Git configuration key')
            if _UNALLOWABLE_GIT_CONFIG_KEY.fullmatch(normalized):
                raise ValueError(f'allowed_local_git_config cannot allow "{label}"')
            if SAFE_LOCAL_GIT_CONFIG_KEY.fullmatch(normalized):
                raise ValueError(f'allowed_local_git_config key "{label}" is accepted by default and cannot be listed')
            if isinstance(rule, LocalGitConfigPattern) and _GIT_COMMAND_KEY.fullmatch(normalized):
                raise ValueError(f'allowed_local_git_config key "{label}" runs a command and must list exact values')
            if isinstance(rule, list) and any("\0" in item for item in rule):
                raise ValueError(f'allowed_local_git_config values for "{label}" cannot contain NUL')
            if normalized in rules:
                raise ValueError(f'allowed_local_git_config has duplicate key "{label}"')
            rules[normalized] = rule
        return rules

    @model_validator(mode="after")
    def _check_account_hosts(self) -> PRReviewConfig:
        unknown = sorted({key.split("/")[0] for key in self.github_accounts} - set(self.allowed_hosts))
        if unknown:
            raise ValueError(f"github_accounts hosts must be listed in allowed_hosts: {', '.join(unknown)}")
        return self


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pr_review: PRReviewConfig = Field(default_factory=PRReviewConfig)


def parse_config(config: ServerConfig) -> Config:
    """Validate the application-owned part of this server's effective config."""
    try:
        return Config.model_validate(config.values)
    except ValidationError as exc:
        raise ConfigError(f"invalid application configuration: {exc}") from exc
