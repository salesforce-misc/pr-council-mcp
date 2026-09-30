from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from pr_council.agents.review.common import ResponseUsageCallback, extract_callback_usage


def _response(*, reported: bool) -> LLMResult:
    message = AIMessage(
        content="",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12} if reported else None,
        response_metadata={"model_name": "model"},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_usage_without_responses_is_complete():
    usage = extract_callback_usage(ResponseUsageCallback())

    assert usage.complete is True
    assert usage.total_tokens == 0


def test_usage_is_complete_only_when_every_response_reported_it():
    callback = ResponseUsageCallback()
    callback.on_llm_end(_response(reported=True))
    assert extract_callback_usage(callback).complete is True

    callback.on_llm_end(_response(reported=False))
    usage = extract_callback_usage(callback, tool_calls=3)

    assert usage.complete is False
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens, usage.tool_calls) == (10, 2, 12, 3)
