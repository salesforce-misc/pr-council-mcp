"""Record an agent's result one validated tool call at a time.

One submission of a whole result can exceed a model's per-call output limit, and a truncated submission loses
everything it carried. A recording agent instead records each item through its own tool call. Its recorder checks
every call as it arrives: a rejected call records nothing, and its error goes back to the model to correct. The run
ends when the model replies without a tool call, and the caller assembles its result from what was recorded.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, ClassVar

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ToolCallRequest, hook_config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.runtime import Runtime
from langgraph.types import Command
from localmcp.retry import is_retryable_error
from pydantic import BaseModel

from pr_council.agents.review.common import (
    ResponseUsageCallback,
    ToolBudget,
    Usage,
    agent_recursion_limit,
    extract_callback_usage,
)

_CUT_OFF_REPLY = "Your last reply reached the output limit before it made a tool call, so nothing was recorded."
_CUT_OFF_CALL = "Not recorded: this call was cut off at the output limit. Record it again."


class OutputLimitError(Exception):
    """The model kept reaching its output limit without making a tool call."""


def reached_output_limit(message: AIMessage) -> bool:
    """Whether a response stopped at its output limit, as reported by Anthropic, Chat Completions, or Responses."""
    metadata = message.response_metadata
    return (
        metadata.get("stop_reason") == "max_tokens"
        or metadata.get("finish_reason") == "length"
        or (metadata.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    )


class ClosedSet[T]:
    """An ordered, closed set of IDs, each of which can be recorded once."""

    def __init__(self, ids: Iterable[str], *, source: str, recorded: str):
        self.ids = list(dict.fromkeys(ids))
        self.records: dict[str, T] = {}
        self._known = set(self.ids)
        self._source = source
        self._recorded = recorded

    def check(self, ids: Sequence[str]) -> None:
        """Reject unknown, repeated, or already recorded IDs with an error the model can act on."""
        if unknown := [item for item in ids if item not in self._known]:
            raise ToolException(f"Not recorded: {unknown} are not {self._source} IDs.")
        if len(set(ids)) != len(ids):
            raise ToolException("Not recorded: list each ID once.")
        if done := [item for item in ids if item in self.records]:
            raise ToolException(f"Not recorded: {done} were already {self._recorded}.")

    def record(self, ids: Sequence[str], value: T) -> None:
        """Record ``value`` for IDs that passed :meth:`check`."""
        for item in ids:
            self.records[item] = value

    def open(self) -> list[str]:
        return [item for item in self.ids if item not in self.records]

    def recorded(self) -> list[T]:
        return [self.records[item] for item in self.ids if item in self.records]


def recording_tool(
    record: Callable[..., Awaitable[str]], *, name: str, description: str, schema: type[BaseModel]
) -> BaseTool:
    """Expose a recorder method as a tool whose rejections go back to the model rather than ending the run."""
    return StructuredTool.from_function(
        coroutine=record,
        name=name,
        description=description,
        args_schema=schema,
        handle_tool_error=True,
    )


class Recorder(ABC):
    """Checks and keeps what an agent records through its tools."""

    #: Whether the model must record every item. A model that stops early is then reminded of what remains.
    required: ClassVar[bool] = False
    tools: list[BaseTool]
    #: The number of recording calls a complete run may need, which sizes the run's step limit.
    items: int

    @abstractmethod
    def remaining(self) -> str | None:
        """Tell the model what it can still record, or return None when nothing remains."""

    @abstractmethod
    def has_records(self) -> bool:
        """Whether anything has been recorded, which a failed run can keep."""


class RecordingMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Keep a recording run going after a turn that was cut off, stopped early, or whose tool calls did not parse.

    A response that reaches the output limit can end in a cut-off call whose arguments still parse with truncated
    text, so that call is answered with an error rather than recorded. When reasoning uses the whole limit, the
    response has no tool call at all and would end the run as though everything were recorded; while anything remains,
    it is dropped and the model is reminded instead. A required recorder's model that stops with items unrecorded is
    told what remains. Both reminders are bounded by consecutive turns without a tool call. Tool calls whose arguments
    did not parse would also end the run when no other call parsed, or be replayed verbatim to providers that reject
    them. As empty-argument calls, the tools node answers them with a validation error and the model can retry.
    """

    def __init__(self, recorder: Recorder, *, max_idle_turns: int = 3):
        super().__init__()
        self._recorder = recorder
        self._max_idle_turns = max_idle_turns
        self._idle_turns = 0
        self._cut_off_calls: set[str] = set()

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        message = state["messages"][-1] if state["messages"] else None
        if not isinstance(message, AIMessage):
            return None
        truncated = reached_output_limit(message)
        if message.tool_calls or message.invalid_tool_calls:
            self._idle_turns = 0
            if truncated and message.tool_calls and not message.invalid_tool_calls:
                self._cut_off_calls.add(message.tool_calls[-1]["id"] or "")
            if not message.invalid_tool_calls:
                return None
            fallback = self._recorder.tools[0].name
            unparsed = [
                tool_call(name=call["name"] or fallback, args={}, id=call["id"] or f"invalid_{index}")
                for index, call in enumerate(message.invalid_tool_calls)
            ]
            update = {"tool_calls": [*message.tool_calls, *unparsed], "invalid_tool_calls": []}
            return {"messages": [message.model_copy(update=update)]}
        remaining = self._recorder.remaining()
        if remaining is None or not (truncated or self._recorder.required):
            return None
        self._idle_turns += 1
        if self._idle_turns >= self._max_idle_turns:
            if truncated:
                raise OutputLimitError(f"{self._idle_turns} consecutive responses reached the output limit")
            return None
        if truncated:
            return {
                "messages": [RemoveMessage(id=message.id or ""), HumanMessage(content=f"{_CUT_OFF_REPLY} {remaining}")],
                "jump_to": "model",
            }
        return {"messages": [HumanMessage(content=remaining)], "jump_to": "model"}

    async def aafter_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call_id = request.tool_call["id"] or ""
        if call_id in self._cut_off_calls:
            return ToolMessage(content=_CUT_OFF_CALL, tool_call_id=call_id, status="error")
        return await handler(request)


async def run_recording_agent(
    model: BaseChatModel,
    recorder: Recorder,
    *,
    name: str,
    run_name: str,
    system_prompt: str,
    payload: dict[str, object],
    usage: Usage,
    tools: Sequence[BaseTool] = (),
    tool_budget: ToolBudget | None = None,
    langfuse_callback: Any | None = None,
    on_incomplete: Callable[[Exception], None] | None = None,
) -> None:
    """Run a recording agent until the model stops, keeping whatever ``recorder`` recorded.

    Transient transport and API faults (429, 5xx, connection, timeout) propagate to the caller's
    call_with_rate_limit_retry for reset-aware backoff. Any other failure ends the run with what was already recorded
    and goes to ``on_incomplete``. A failure before anything was recorded, such as a rejected credential or model, has
    nothing to keep and propagates as well. Token usage and ``tool_budget`` calls accumulate into ``usage`` either way,
    so spend from an attempt that re-raises is still reported when the caller retries with the same accumulator.
    """
    # Unforced tools, as for the reviewers: forced tool_choice is not supported by every model and gateway.
    agent = create_agent(
        model,
        [*tools, *recorder.tools],
        system_prompt=system_prompt,
        middleware=[RecordingMiddleware(recorder)],
        name=name,
    )
    usage_callback = ResponseUsageCallback()
    callbacks: list[Any] = [usage_callback]
    if langfuse_callback is not None:
        callbacks.append(langfuse_callback)
    budget = tool_budget.max_calls if tool_budget is not None else 0
    config: RunnableConfig = {
        "run_name": run_name,
        "recursion_limit": agent_recursion_limit(recorder.items + budget),
        "callbacks": callbacks,
    }
    try:
        await agent.ainvoke({"messages": [HumanMessage(content=json.dumps(payload))]}, config=config)
    except Exception as exc:
        if is_retryable_error(exc) or not recorder.has_records():
            raise
        if on_incomplete is not None:
            on_incomplete(exc)
    finally:
        tool_calls = tool_budget.used if tool_budget is not None else 0
        usage.add(extract_callback_usage(usage_callback, tool_calls=tool_calls))
