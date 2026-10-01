import pytest
from localmcp.config import ServerConfig
from pydantic import ValidationError

from pr_council.config import ConfigError, PRReviewLimitsConfig, PRReviewModelsConfig, parse_config


def test_empty_application_config_uses_defaults() -> None:
    config = parse_config(ServerConfig(name="pr-council-mcp", values={}))

    assert config.pr_review.models.quality == ["claude-opus-5-5", "gpt-6-sol"]
    assert config.pr_review.models.security == ["claude-opus-5-5", "gpt-6-sol"]
    assert config.pr_review.models.deliberation == "claude-opus-5-5"
    assert config.pr_review.models.aggregation == "claude-sonnet-5"
    assert config.pr_review.allowed_hosts == ["github.com"]


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
