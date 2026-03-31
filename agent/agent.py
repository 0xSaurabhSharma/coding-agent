from __future__ import annotations
from typing import AsyncGenerator
from client.response import StreamEventType, ToolCall, ToolResultMessage
from client.llm_client import LLMClient
from agent.events import AgentEventType, AgentEvent
from context.manager import ContextManager
from tools.registry import create_default_registry

class Agent:
    def __init__(self):
        self.client = LLMClient()
        self.context_manager = ContextManager()
        self.tool_registry = create_default_registry()

    async def run(self, message: str | None = None):
        yield AgentEvent.agent_start(message)
        self.context_manager.add_user_message(message)
        # Add user message to context
        # print('---> Agent.run()')

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
        # print("---> Agent._agentic_loop()")
        # context manager & sessions but for now, lets keep it simple
        # messages = [
        #     {
        #         "role": "user",
        #         "content": message or "test invoke, ans in three word only: "
        #     }
        # ]
        responseText = ""
        tool_calls: list[ToolCall] = []
        tool_schemas = self.tool_registry.get_schemas()
        tool_call_result: list[ToolResultMessage] = []

        async for event in self.client.chat_completion(
                self.context_manager.get_messages(),
                tools=tool_schemas if tool_schemas else None,
                # stream=False
            ):
            # print("---> <agent loop> ",event)
            
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    responseText += content
                    yield AgentEvent.text_delta(content)
            elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                if event.tool_call:
                    tool_calls.append(event.tool_call)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown error occured.")

        # print("---> <response-text>")
        # print(responseText)
        self.context_manager.add_assistant_message(responseText or None)
        if responseText:
            # print('check')
            yield AgentEvent.text_complete(content=responseText)

        # executing all tc in tool_calls
        for tc in tool_calls:
            # telling ui by event
            yield AgentEvent.tool_call_start(
                call_id=tc["call_id"],
                name=tc["name"],
                arguments=tc["arguments"]
            )

            # execution
            result = await self.tool_registry.invoke(tc.name, tc.arguments, Path.cwd)
            yield AgentEvent.tool_call_complete(
                tc.call_id,
                tc.name,
                result
            )

            # adding tool_call results to context
            tool_call_result.append(ToolResultMessage(
                tool_call_id=tc.call_id, 
                content=result.to_model_output(),
                is_error= not result.success
            ))
        
        for tc in tool_call_result:
            self.context_manager.add_tool_message(
                tc.tool_call_id,
                tc.content
            )


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
