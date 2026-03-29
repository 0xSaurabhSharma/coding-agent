from __future__ import annotations
from typing import AsyncGenerator
from client.response import StreamEventType
from client.llm_client import LLMClient
from agent.events import AgentEventType, AgentEvent
from context.manager import ContextManager

class Agent:
    def __init__(self):
        self.client = LLMClient()
        self.context_manager = ContextManager()

    async def run(self, message: str | None = None):
        yield AgentEvent.agent_start(message)
        self.context_manager.add_user_message(message)
        # Add user message to context

        final_response: str | None = None

        # before agent hooks 
        # after agent hooks 
        async for event in self._agentic_loop(message):
            # print(event)
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        # when agent's execution stops 
        yield AgentEvent.agent_end(final_response)


    async def _agentic_loop (self, message: str ) -> AsyncGenerator[AgentEvent, None]:

        # context manager & sessions but for now, lets keep it simple
        # messages = [
        #     {
        #         "role": "user",
        #         "content": message or "test invoke, ans in three word only: "
        #     }
        # ]
        responseText = ""

        async for event in self.client.chat_completion(self.context_manager.get_messages(), True):
            
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    responseText += content
                    yield AgentEvent.text_delta(content)

            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown error occured.")

        
        self.context_manager.add_assistant_message(responseText or None)
        if responseText:
            yield AgentEvent.text_complete(content=responseText)

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(
            self,
            exc_type,
            exc_val,
            exc_tb,
        ) -> Agent:
        if self.client:
            await self.client.close()
            self.client = None
        