"""Application schema over the shared localmcp configuration document."""

from __future__ import annotations

import re

from localmcp.config import ServerConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pr_council.errors import PrCouncilError

SERVER_NAME = "pr-council-mcp"
_MODEL_ID = re.compile(r"^[A-Za-z0-9._:/-]+$")


class ConfigError(PrCouncilError):
    """Raised when the shared document or application view is invalid."""


class PRReviewLimitsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer_tool_calls_per_iteration: int = Field(default=12, ge=1, le=20)
    deliberator_tool_calls_per_disposition: int = Field(default=8, ge=1, le=20)
    operation_tool_calls: int = Field(default=300, ge=1, le=1_000)
    model_output_tokens_per_call: int = Field(default=8_000, ge=256, le=32_000)
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
        default_factory=lambda: ["claude-opus-4-8", "gpt-5.6", "gemini-3.1-pro-preview"],
        min_length=1,
        max_length=10,
    )
    security: list[str] = Field(
        default_factory=lambda: ["claude-opus-4-8", "gpt-5.6", "gemini-3.1-pro-preview"],
        min_length=1,
        max_length=10,
    )
    deliberation: str = "claude-opus-4-8"
    aggregation: str = "claude-haiku-4-5-20251001"

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


class PRReviewConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    models: PRReviewModelsConfig = Field(default_factory=PRReviewModelsConfig)
    limits: PRReviewLimitsConfig = Field(default_factory=PRReviewLimitsConfig)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pr_review: PRReviewConfig = Field(default_factory=PRReviewConfig)


def parse_config(config: ServerConfig) -> Config:
    """Validate the application-owned part of this server's effective config."""
    try:
        return Config.model_validate(config.values)
    except ValidationError as exc:
        raise ConfigError(f"invalid application configuration: {exc}") from exc
