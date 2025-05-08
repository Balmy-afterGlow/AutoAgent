import asyncio
from collections import defaultdict
from typing import AsyncGenerator, Callable, Dict, Any, Optional, Union, List
from litellm import Message
from autoagent.logger import MetaChainLogger
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    Choices,
    ChatCompletionMessageToolCall,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from rich import print
import json

tool_call_format = lambda index, name, id: {
    "index": index,
    "function": {"arguments": "", "name": name},
    "id": id,
    "type": "function",
}


def print_callback(chunk: ModelResponseStream, last_type: str) -> str:
    delta = chunk.choices[0].delta
    current_type = "none"

    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        current_type = "reasoning"
    elif delta.tool_calls is not None:
        current_type = "tool_call"
    elif delta.content is not None:
        current_type = "content"

    if current_type != "none" and current_type != last_type:
        if current_type == "reasoning":
            print("[grey58]\n[推理过程开始][/]", end="\n\n", flush=True)
        elif current_type == "tool_call":
            print(
                f"[grey58]\n\n[工具调用开始] [/]",
                end="",
                flush=True,
            )
        elif current_type == "content":
            print("[grey58]\n\n[AI答复][/]", end="\n\n", flush=True)
        last_type = current_type

    if current_type == "reasoning":
        assert delta.reasoning_content is not None
        print(f"[grey58]{delta.reasoning_content}[/]", end="", flush=True)
    elif current_type == "tool_call":
        assert delta.tool_calls is not None
        tool_name = delta.tool_calls[0].function.name
        if tool_name is not None:
            print(
                f"[grey58]正在调用 {tool_name}...[/]",
                end="",
                flush=True,
            )
    elif current_type == "content":
        assert delta.content is not None
        print(f"[grey58]{delta.content}[/]", end="", flush=True)

    return last_type


def chunk_callback(chunk: ModelResponseStream, tool_calls: list) -> tuple[str, str]:
    delta = chunk.choices[0].delta

    reasoning_content = ""
    content = ""

    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning_content += delta.reasoning_content
    if delta.content is not None:
        content += delta.content
    if delta.tool_calls is not None:
        for tool_call in delta.tool_calls:
            tool_calls[tool_call.index]["function"][
                "arguments"
            ] += tool_call.function.arguments

    return reasoning_content, content


class LiteLLMStreamHandler:
    def __init__(self, logger: MetaChainLogger):
        self.logger = logger
        self.full_message: Message = Message(role="assistant", content="")
        self.print_callback: Callable[[ModelResponseStream, str], str] | None = None
        self.chunk_callback: (
            Callable[[ModelResponseStream, list], tuple[str, str]] | None
        ) = None

    def set_print_callback(self, callback: Callable[[ModelResponseStream, str], str]):
        self.print_callback = callback

    def set_chunk_callback(
        self, callback: Callable[[ModelResponseStream, list], tuple[str, str]]
    ):
        self.chunk_callback = callback

    def init_handler(self):
        self.full_message = Message(role="assistant", content="")
        self.set_print_callback(print_callback)
        self.set_chunk_callback(chunk_callback)

    async def process_async_stream(self, async_generator: CustomStreamWrapper):
        try:
            last_type = "none"
            tool_calls = []
            reasoning_content = ""
            content = ""
            async for chunk in async_generator:
                if self.chunk_callback:
                    if chunk.choices[0].delta.tool_calls is not None:
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            tool_calls.append(
                                tool_call_format(
                                    tool_call.index,
                                    tool_call.function.name,
                                    tool_call.id,
                                )
                            )
                    reasoning_content_chunk, content_chunk = self.chunk_callback(
                        chunk, tool_calls
                    )
                    reasoning_content += reasoning_content_chunk
                    content += content_chunk

                if self.print_callback:
                    last_type = self.print_callback(chunk, last_type)

            self.full_message = Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )
            return self.full_message

        except Exception as e:
            print(f"处理流时出错: {e}")
            raise e

    def process_sync_stream(self, sync_generator: CustomStreamWrapper):
        try:
            last_type = "none"
            tool_calls = []
            reasoning_content = ""
            content = ""
            is_first_chunk = True
            for chunk in sync_generator:
                if is_first_chunk:
                    self.logger.info(
                        "The information returned by the model:\n",
                        "The first chunk:",
                        chunk.model_dump_json(indent=2),
                        "\n...",
                        title="Model Output (From Stream)",
                    )

                    if chunk.choices[0].delta.tool_calls is not None:
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            tool_calls.append(
                                tool_call_format(
                                    tool_call.index,
                                    tool_call.function.name,
                                    tool_call.id,
                                )
                            )

                    is_first_chunk = False

                if self.chunk_callback:

                    reasoning_content_chunk, content_chunk = self.chunk_callback(
                        chunk, tool_calls
                    )
                    reasoning_content += reasoning_content_chunk
                    content += content_chunk

                if self.print_callback:
                    last_type = self.print_callback(chunk, last_type)

            print("\n")

            self.full_message = Message(
                role="assistant",
                content=content,
                tool_calls=[
                    ChatCompletionMessageToolCall(**tool_call)
                    for tool_call in tool_calls
                ],
                reasoning_content=reasoning_content,
            )
            return self.full_message

        except Exception as e:
            print(f"处理流时出错: {e}")
            raise e

    async def process_stream(self, generator: CustomStreamWrapper):
        # if hasattr(generator, "__aiter__"):
        #     return await self.process_async_stream(generator)
        # else:

        # 将同步代码加入线程池
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_sync_stream, generator)
