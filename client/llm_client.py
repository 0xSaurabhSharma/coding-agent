from typing import Any, AsyncGenerator
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
import asyncio
import os 
from dotenv import load_dotenv

from client.response import TokenUsage, StreamEvent, TextDelta, StreamEventType, ToolCallDelta, ToolCall, parse_tool_call_arguments

load_dotenv()

class LLMClient:
    def __init__(self) -> None:
        self._client : AsyncOpenAI | None = None 
        self._max_attempt: int = 3

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                # base_url=os.getenv("OPENROUTER_BASE_URL"),
                # api_key=os.getenv("OPENROUTER_API_KEY")
                base_url=os.getenv("OPENROUTER_BASE_URL"),
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            return self._client
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _build_tools(self, tools: list[dict[str,Any]]) -> list[dict[str,Any]]:
        # converting every schema into openai schema
        return [ {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties":{}})
            }
        }
            for tool in tools
        ]

    async def chat_completion(
            self, 
            messages: list[dict[str, Any]], 
            tools: list[dict[str, Any]] | None = None,
            stream: bool = True
        ) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs = {
            # groq
            # "model": "openai/gpt-oss-20b", 
            # openrouter
            # "model": "google/gemma-3-27b-it:free",
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "messages": messages,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(self._max_attempt+1):
            try:
                if stream :
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else: 
                    event = await self._non_stream_response(client, kwargs)
                    yield event 

                return
            
            except RateLimitError as e:
                # wait time backoff | we will wait till it 
                if attempt < self._max_attempt:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Rate Limit Exceeded: {e}"
                    )
                    return 
            
            except APIConnectionError as e:
                if attempt < self._max_attempt:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        error=f"API Connection Error: {e}",
                        type=StreamEventType.ERROR
                    )
                    return
            
            except APIError as e:
                # not retrying as llm api itself is not working properly
                yield StreamEvent(
                    error=f"API Connection Error: {e}",
                    type=StreamEventType.ERROR
                )
                return


            

    async def _stream_response(
            self,
            client: AsyncOpenAI,
            kwargs: dict[str, Any]
        ) -> AsyncGenerator[StreamEvent, None]:
            response = await client.chat.completions.create(**kwargs)
            # print("========== _stream_response ==========")
            # print(response)

            usege: TokenUsage | None = None 
            finish_reason: str | None = None
            tool_calls: dict[int,dict[str,Any]] ={}

            async for chunk in response:
                # print('------------------ chunk in response ----------------------')
                # print(chunk)
                
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens = chunk.usage.prompt_tokens,
                        completion_tokens = chunk.usage.completion_tokens,
                        total_tokens = chunk.usage.total_tokens,
                        cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens
                    )
                
                if not chunk.choices:
                    continue 

                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta.content:
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        text_delta=TextDelta(content=delta.content),
                        # finish_reason=finish_reason,
                        # usage=usage
                    )

                # print(delta)
                # print("----> ",delta.tool_calls, " delta.tool_calls")
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index

                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": tc_delta,
                                "name" : "",
                                "arguments": ""
                            }
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls[idx]["name"] = tc_delta.function.name
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    tool_call_delta = ToolCallDelta(
                                        call_id=tool_calls[idx]['id'],
                                        name=tc_delta.function.name
                                    )
                                )
                        
                            if tc_delta.function.arguments:
                                tool_calls[idx]['arguments'] += tc_delta.function.arguments
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_DELTA,
                                    tool_call_delta = ToolCallDelta(
                                        call_id=tool_calls[idx]['id'],
                                        name=tc_delta.function.name,
                                        arguments_delta=tc_delta.function.arguments,
                                    )
                                )

            for idx, tc in tool_calls.items():
                yield StreamEvent(
                    type=StreamEventType.TOOL_CALL_COMPLETE,
                    tool_call=ToolCall(
                        call_id=tc["id"],
                        name=tc["name"],
                        arguments=parse_tool_call_arguments(tc["arguments"])
                    )
                )


            
            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                # text_delta=TextDelta(content=delta.content),
                finish_reason=finish_reason,
                usage=usage
            )

    async def _non_stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> StreamEvent:

        response = await client.chat.completions.create(**kwargs)
        # print(response)
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []

        text_delta = None 
        # llm can return anything like text, toolcall, mcp, audio, image, resoning, type, error; so we create new schema of what types of responses we can get
        if message.content:
            text_delta = TextDelta(content = message.content)

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    call_id = tc.id, 
                    name = tc.function.name,
                    arguments = parse_tool_call_arguments(tc.function.arguments)
                ))

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens = response.usage.prompt_tokens,
                completion_tokens = response.usage.completion_tokens,
                total_tokens = response.usage.total_tokens,
                cached_tokens = response.usage.prompt_tokens_details.cached_tokens
            )
        
        return StreamEvent(
            type= StreamEventType.MESSAGE_COMPLETE,
            text_delta = text_delta,
            finish_reason = choice.finish_reason,
            usage = usage,
        )