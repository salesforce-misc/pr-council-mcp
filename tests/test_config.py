import pytest
from localmcp.config import ServerConfig
from pydantic import ValidationError

from pr_council.config import ConfigError, PRReviewLimitsConfig, PRReviewModelsConfig, parse_config


def test_empty_application_config_uses_defaults() -> None:
    config = parse_config(ServerConfig(name="pr-council-mcp", values={}))

    assert config.pr_review.models.quality == ["claude-opus-4-8", "gpt-5.6", "gemini-3.1-pro-preview"]
    assert config.pr_review.models.security == ["claude-opus-4-8", "gpt-5.6", "gemini-3.1-pro-preview"]


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
