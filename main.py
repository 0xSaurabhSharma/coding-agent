import asyncio
import click
import sys
from typing import Any
from agent.agent import Agent
from agent.events import AgentEventType, AgentEvent
from ui.tui import TUI, get_console


console = get_console()


class CLI: 
    def __init__(self):
        self.agent: Agent | None = None
        self.tui = TUI()

    async def run_single(self, message: str) -> str:
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    def _get_tool_kind(self, tool_name: str) -> str | None:
        tool = self.agent.tool_registry.get_tool_by_name(tool_name)
        tool_kind = None
        if not tool:
            tool_kind = None
        tool_kind = tool.kind.value
        return tool_kind

    async def _process_message(self, message):
        if not self.agent:
            return None
        
        assistant_streaming = False
        final_response: str | None = None
        # arr = []
        
        async for event in self.agent.run(message):
            # it will return events: start, delta, end, error, complete
            # print(event)
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content","")
                # arr.append(content)
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming = True
                self.tui.stream_assistant_delta(content)

            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content", "")
                if assistant_streaming:
                    self.tui.close_assistant()
                assistant_streaming = False 

            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown error occured.")
                console.print(f"\n[error]Error: {error}[/error]")
                # not closing agent streaming as it can be resolved in tool calls

            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_start(
                    event.data.get('call_id', ''),
                    tool_name,
                    tool_kind,
                    event.data.get('arguments', {})
                )
            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_complete(
                    event.data.get('call_id', ''),
                    tool_name,
                    tool_kind,
                    event.data.get('success', False),
                    event.data.get('output', ''),
                    event.data.get('error'),
                    event.data.get('metadata'),
                    event.data.get('truncated', False),
                )
            
        
        # print(arr)
        return final_response



@click.command()
@click.argument("prompt", required=False)
def main(prompt: str | None):

    print('--initiated--')    
    
    cli = CLI()
    # messages = [
    #     {
    #         "role": "user",
    #         "content": prompt or "test invoke, ans in one word only: "
    #     }
    # ]
    
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            print('--err--')
            sys.exit(1)
    print('--done--')

main()