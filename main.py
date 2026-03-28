import asyncio
import click
from typing import Any


class CLI: 
    def __init__(self):
        pass 

    def run_single(self):
        pass 

@click.command()
@click.argument("prompt", required=False)
def main(prompt: str | None):

    print('--initiated--')    
    print(prompt)
    messages = [
        {
            "role": "user",
            "content": prompt or "test invoke, ans in one word only: "
        }
    ]
    asyncio.run(run(messages))
    print('--done--')

main()