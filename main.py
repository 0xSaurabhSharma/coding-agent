from client.llm_client import LLMClient
import asyncio

async def main():
    client = LLMClient()
    messages = [
        {
            "role": "user",
            "content": "count to 5: "
        }
    ]
    # res = await client.chat_completion(messages, False)

    print('---')
    async for event in client.chat_completion(messages, True):
        print(event)

    print('---')
    # print(res)

asyncio.run(main())


