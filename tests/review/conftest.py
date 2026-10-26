"""Shared fixtures for the review-agent tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted turns in order; an exception entry is raised from that model call."""

    script: list[Any] = []
    requests: list[list[Any]] = []
    bound: list[dict[str, Any]] = []

    def bind_tools(self, tools, **kwargs):
        self.bound.append({"tools": [getattr(tool, "name", None) for tool in tools], **kwargs})
        return self

    def _generate(self, messages, *args, **kwargs):
        self.requests.append(list(messages))
        turn = self.script.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        self.messages = iter([turn])
        return super()._generate(messages, *args, **kwargs)


class RunRecorder(BaseCallbackHandler):
    """Counts model calls and records the name of the outermost run."""

    def __init__(self) -> None:
        self.starts = 0
        self.run_names: list[str | None] = []

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.starts += 1

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("parent_run_id") is None:
            self.run_names.append(kwargs.get("name"))


@pytest.fixture
def scripted_model() -> Callable[..., ScriptedModel]:
    def build(*turns: Any) -> ScriptedModel:
        return ScriptedModel(messages=iter([]), script=list(turns), requests=[], bound=[])

    return build


@pytest.fixture
def run_recorder() -> RunRecorder:
    return RunRecorder()
