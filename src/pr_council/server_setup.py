"""Application bootstrap and credential checks for the shared stdio server."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import localmcp
from localmcp.config import LocalMCPPaths
from localmcp.llm import (
    DEFAULT_MODEL_REGISTRY,
    ConfiguredModelFactory,
    LLMBackendConfig,
    ModelConfigurationError,
    ModelSpec,
    OwnedChatModel,
)
from localmcp.secrets import SecretBackendError, SecretCatalog, SecretNotFoundError, SecretResolver

from pr_council.config import Config
from pr_council.workflows.review.runtime import ReviewRuntime

_INITIAL_CONFIG = """schema_version = 1

[llm]
backend = "native"

[secrets.openai_api_key]
env_vars = ["OPENAI_API_KEY"]

[secrets.anthropic_api_key]
env_vars = ["ANTHROPIC_API_KEY"]
"""


def _ensure_config(env: Mapping[str, str], home: Path) -> None:
    """Install the initial shared document without replacing a concurrent writer."""
    path = LocalMCPPaths.from_environment(env, home=home).config_file
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".localmcp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_INITIAL_CONFIG)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            pass
    finally:
        os.unlink(temporary_path)


class _CredentialCheckedModelFactory(ConfiguredModelFactory):
    """Check selected models' credentials before a review is queued."""

    def __init__(
        self,
        delegate: ConfiguredModelFactory,
        catalog: SecretCatalog,
        resolver: SecretResolver,
        *,
        openai_secret_name: str,
        anthropic_secret_name: str,
        llm_secret_name: str,
    ) -> None:
        self.config = delegate.config
        self._checked_delegate = delegate
        self._catalog = catalog
        self._resolver = resolver
        self._openai_secret_name = openai_secret_name
        self._anthropic_secret_name = anthropic_secret_name
        self._llm_secret_name = llm_secret_name

    def validate(self, model_id: str, *, reasoning_effort: str | None = None) -> ModelSpec:
        spec = DEFAULT_MODEL_REGISTRY.resolve(model_id)
        if self.config.backend == "native":
            if spec.native_provider is None:
                return self._checked_delegate.validate(model_id, reasoning_effort=reasoning_effort)
            provider: str = spec.native_provider
            logical_name = self._anthropic_secret_name if provider == "anthropic" else self._openai_secret_name
        else:
            provider = "gateway"
            logical_name = self._llm_secret_name
        reference = self._catalog.get(logical_name)
        if reference is None:
            raise ModelConfigurationError(
                f'{provider} API key for model "{model_id}" is not configured; '
                f"add [secrets.{logical_name}] to localmcp.toml, then set a declared env_vars alias "
                "or store the account in the localmcp OS keyring"
            )
        try:
            self._resolver.require(reference)
        except SecretBackendError as exc:
            aliases = ", ".join(reference.env_vars) or "a declared env_vars alias"
            raise ModelConfigurationError(
                f'{provider} API key for model "{model_id}" could not be read from the keyring; '
                f"set {aliases} or fix the keyring"
            ) from exc
        except SecretNotFoundError as exc:
            service = "localmcp" if reference.server is None else f"localmcp:{reference.server}"
            aliases = ", ".join(reference.env_vars)
            setup = f"set {aliases} " if aliases else f"add an env_vars alias to [secrets.{logical_name}] "
            raise ModelConfigurationError(
                f'{provider} API key for model "{model_id}" is missing; '
                f'{setup}or store account "{reference.name}" in the "{service}" OS keyring service'
            ) from exc
        return self._checked_delegate.validate(model_id, reasoning_effort=reasoning_effort)

    def create(
        self,
        model_id: str,
        *,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> OwnedChatModel:
        self.validate(model_id, reasoning_effort=reasoning_effort)
        return self._checked_delegate.create(
            model_id,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )


class PrCouncilServer(localmcp.STDIOServer[Config, ReviewRuntime]):
    def load_config(self, *, env: Mapping[str, str], home: Path) -> Config:
        _ensure_config(env, home)
        return super().load_config(env=env, home=home)

    async def start_runtime(self, *, env: Mapping[str, str], home: Path) -> Config:
        _ensure_config(env, home)
        return await super().start_runtime(env=env, home=home)

    def run(self) -> None:
        try:
            _ensure_config(os.environ, Path.home())
        except Exception as exc:
            raise SystemExit(1) from exc
        super().run()

    def _model_factory(
        self,
        config: LLMBackendConfig,
        catalog: SecretCatalog,
        resolver: SecretResolver,
    ) -> ConfiguredModelFactory:
        delegate = super()._model_factory(config, catalog, resolver)
        return _CredentialCheckedModelFactory(
            delegate,
            catalog,
            resolver,
            openai_secret_name=self.openai_secret_name,
            anthropic_secret_name=self.anthropic_secret_name,
            llm_secret_name=self.llm_secret_name,
        )
