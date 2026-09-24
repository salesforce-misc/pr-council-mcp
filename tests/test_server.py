"""Product-specific assertions over the shared localmcp server composition."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP

import pr_council.server as server
import pr_council.tools.review as review_tool
from pr_council.config import Config
from pr_council.review.models import OperationStatus
from pr_council.tools import TOOLS

_AWAIT_TIMEOUT = 5.0


@dataclass
class _FakeRecord:
    id: str
    label: str
    status: OperationStatus
    preview: dict[str, Any] | None = None
    commit_ready: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = "2026-01-01T00:00:00Z"
    updated_at: str = "2026-01-01T00:00:00Z"


class _FakeRuntime:
    def __init__(self, *, start_record: _FakeRecord, get_record: _FakeRecord) -> None:
        self._start_record = start_record
        self._get_record = get_record
        self.start_kwargs: dict[str, Any] = {}
        self.get_calls: list[str] = []

    async def start_operation(self, **kwargs: Any) -> _FakeRecord:
        self.start_kwargs = kwargs
        return self._start_record

    async def get_owned(self, operation_id: str) -> _FakeRecord:
        self.get_calls.append(operation_id)
        return self._get_record


@pytest.fixture
def _hermetic_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    async def start_runtime(*, env: Mapping[str, str], home: Path) -> Config:
        return Config()

    async def stop_runtime() -> None:
        return None

    monkeypatch.setattr(server.server, "start_runtime", start_runtime)
    monkeypatch.setattr(server.server, "stop_runtime", stop_runtime)


def test_server_declares_shared_stdio_server_and_tools() -> None:
    assert isinstance(server.server.mcp, FastMCP)
    assert server.server.name == "pr-council-mcp"
    assert server.server.tools == tuple(TOOLS)


@pytest.mark.parametrize("backend", ["openai_compatible", "native"])
def test_server_accepts_shared_llm_and_returns_only_application_config(tmp_path: Path, backend: str) -> None:
    config_home = tmp_path / "config"
    config_path = config_home / "localmcp/localmcp.toml"
    config_path.parent.mkdir(parents=True)
    base_url = 'base_url = "https://gateway.example/v1"\n' if backend == "openai_compatible" else ""
    config_path.write_text(
        f"""schema_version = 1

[llm]
backend = "{backend}"
{base_url}
[server.pr-council-mcp.pr_review.models]
quality = ["gpt-5.6"]
""",
        encoding="utf-8",
    )

    config = server.server.load_config(
        env={"XDG_CONFIG_HOME": str(config_home)},
        home=tmp_path,
    )

    assert config.pr_review.models.quality == ["gpt-5.6"]
    assert set(config.model_dump()) == {"pr_review"}


def test_main_delegates_to_localmcp(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(server.localmcp, "main", calls.append)

    server.main()

    assert calls == [server.server]


def test_console_script_resolves_to_server_main() -> None:
    matches = [entry for entry in entry_points(group="console_scripts") if entry.name == "pr-council-mcp"]
    assert len(matches) == 1
    assert matches[0].value == "pr_council.server:main"


@pytest.mark.usefixtures("_hermetic_lifespan")
async def test_registered_mcp_surface_contains_only_lifecycle_tools() -> None:
    async with Client(server.server.mcp) as client:
        listed = await asyncio.wait_for(client.list_tools(), _AWAIT_TIMEOUT)

    assert {tool.name for tool in listed} == {
        "pr_council_start",
        "pr_council_get",
        "pr_council_preview",
        "pr_council_commit",
        "pr_council_cancel",
    }


@pytest.mark.usefixtures("_hermetic_lifespan")
async def test_start_then_get_roundtrips_operation_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    start_record = _FakeRecord(id="op-123", label="demo", status=OperationStatus.QUEUED)
    get_record = _FakeRecord(id="op-123", label="demo", status=OperationStatus.READY)
    fake = _FakeRuntime(start_record=start_record, get_record=get_record)
    monkeypatch.setattr(review_tool, "review_runtime", lambda: fake)

    async with Client(server.server.mcp) as client:
        started = await asyncio.wait_for(
            client.call_tool("pr_council_start", {"pr_url": "https://github.com/o/r/pull/1"}),
            _AWAIT_TIMEOUT,
        )
        fetched = await asyncio.wait_for(
            client.call_tool("pr_council_get", {"operation_id": started.data["operation_id"]}),
            _AWAIT_TIMEOUT,
        )

    assert started.data == {"operation_id": "op-123", "label": "demo", "status": "queued"}
    assert fetched.data["operation_id"] == "op-123"
    assert fetched.data["status"] == "ready"
    assert fake.start_kwargs["pr_url"] == "https://github.com/o/r/pull/1"
    assert fake.get_calls == ["op-123"]
