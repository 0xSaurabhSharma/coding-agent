from typing import Any, AsyncGenerator
from openai import AsyncOpenAI
import os 
from dotenv import load_dotenv

from client.response import TokenUsage, StreamEvent, TextDelta, EventType

load_dotenv()

class LLMClient:
    def __init__(self) -> None:
        self._client : AsyncOpenAI | None = None 

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=os.getenv("OPENROUTER_BASE_URL"),
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_completion(
            self, 
            messages: list[dict[str, Any]], 
            stream: bool = True
        ) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs = {
            "model": "google/gemma-3-27b-it:free",
            # "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "messages": messages,
            "stream": stream
        }

        if stream :
            async for event in self._stream_response(client, kwargs):
                yield event
        else: 
            event = await self._non_stream_response(client, kwargs)
            yield event 

        return

    async def _stream_response(
            self,
            client: AsyncOpenAI,
            kwargs: dict[str, Any]
        ) -> AsyncGenerator[StreamEvent, None]:
            response = await client.chat.completions.create(**kwargs)

            usege: TokenUsage | None = None 
            finish_reason: str | None = None

            async for chunk in response:
                
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
                        type=EventType.TEXT_DELTA,
                        text_delta=TextDelta(content=delta.content),
                        # finish_reason=finish_reason,
                        # usage=usage
                    )
            
            yield StreamEvent(
                type=EventType.MESSAGE_COMPLETE,
                # text_delta=TextDelta(content=delta.content),
                finish_reason=finish_reason,
                usage=usage
            )

    async def _non_stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None 
        # llm can return anything like text, toolcall, mcp, audio, image, resoning, type, error; so we create new schema of what types of responses we can get
        if message.content:
            text_delta = TextDelta(content = message.content)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens = response.usage.prompt_tokens,
                completion_tokens = response.usage.completion_tokens,
                total_tokens = response.usage.total_tokens,
                cached_tokens = response.usage.prompt_tokens_details.cached_tokens
            )
        
        return StreamEvent(
            type= EventType.MESSAGE_COMPLETE,
            text_delta = text_delta,
            finish_reason = choice.finish_reason,
            usage = usage,
        )