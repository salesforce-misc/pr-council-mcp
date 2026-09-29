"""Credential checks for models resolved through the configured registry."""

from __future__ import annotations

import pytest
from localmcp.llm import (
    DEFAULT_MODEL_REGISTRY,
    ConfiguredModelFactory,
    LLMBackendConfig,
    ModelConfigurationError,
    ModelSpec,
)
from localmcp.secrets import SecretCatalog, SecretRef, SecretResolver

from pr_council.server_setup import _CredentialCheckedModelFactory

_CATALOG = SecretCatalog(
    {
        "openai_api_key": SecretRef(name="openai_api_key", env_vars=("OPENAI_API_KEY",)),
        "anthropic_api_key": SecretRef(name="anthropic_api_key", env_vars=("ANTHROPIC_API_KEY",)),
    }
)


def _factory(env: dict[str, str], models: dict[str, ModelSpec] | None = None) -> _CredentialCheckedModelFactory:
    delegate = ConfiguredModelFactory(
        DEFAULT_MODEL_REGISTRY.overlay(models or {}),
        LLMBackendConfig(backend="native"),
        openai_api_key=env.get("OPENAI_API_KEY"),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY"),
    )
    return _CredentialCheckedModelFactory(
        delegate,
        _CATALOG,
        SecretResolver(env, keyring_getter=lambda _service, _account: None),
        openai_secret_name="openai_api_key",
        anthropic_secret_name="anthropic_api_key",
        llm_secret_name="llm_api_key",
    )


def test_uncatalogued_model_checks_its_inferred_provider_credential() -> None:
    factory = _factory({"OPENAI_API_KEY": "openai"})

    with pytest.raises(ModelConfigurationError, match='anthropic API key for model "claude-opus-9" is missing'):
        factory.validate("claude-opus-9")
    assert factory.validate("gpt-9").native_provider == "openai"


def test_model_override_selects_the_credential_to_check() -> None:
    models = {"house-model": ModelSpec(native_provider="anthropic")}

    with pytest.raises(ModelConfigurationError, match='anthropic API key for model "house-model" is missing'):
        _factory({"OPENAI_API_KEY": "openai"}, models).validate("house-model")
    spec = _factory({"ANTHROPIC_API_KEY": "anthropic"}, models).validate("house-model")
    assert spec.native_provider == "anthropic"


def test_override_can_reroute_a_catalogued_model() -> None:
    models = {"gpt-5.6": ModelSpec(native_provider="anthropic")}

    with pytest.raises(ModelConfigurationError, match='anthropic API key for model "gpt-5.6" is missing'):
        _factory({"OPENAI_API_KEY": "openai"}, models).validate("gpt-5.6")
