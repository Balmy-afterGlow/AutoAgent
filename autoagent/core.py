# Standard library imports
import asyncio
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union
from datetime import datetime

# Local imports
import os

from httpx._transports import default

from autoagent.stream_handler import LiteLLMStreamHandler
from .util import function_to_json, debug_print, merge_chunk, pretty_print_messages
from .types import (
    Agent,
    AgentFunction,
    Response,
    Result,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message
from litellm import completion, acompletion
from litellm.utils import supports_function_calling
from litellm.types.utils import (
    ModelResponse,
    Choices,
    StreamingChoices,
    ModelResponseStream,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from pathlib import Path
from .logger import MetaChainLogger, LoggerManager
from httpx import RemoteProtocolError, ConnectError
from litellm.exceptions import APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import AsyncOpenAI
import litellm
import inspect
from constant import (
    MC_MODE,
    FN_CALL,
    API_BASE_URL,
    NOT_SUPPORT_SENDER,
    ADD_USER,
    NON_FN_CALL,
    COMPLETION_MODEL,
)
from autoagent.fn_call_converter import (
    convert_tools_to_description,
    convert_non_fncall_messages_to_fncall_messages,
    SYSTEM_PROMPT_SUFFIX_TEMPLATE,
    convert_fn_messages_to_non_fn_messages,
    interleave_user_into_messages,
    remove_reasoning_content_in_messages,
)
from rich import print


# litellm.set_verbose=True
# client = AsyncOpenAI()
def should_retry_error(exception):
    if MC_MODE is False:
        print(f"Caught exception: {type(exception).__name__} - {str(exception)}")

    # 匹配更多错误类型
    if isinstance(exception, (APIError, RemoteProtocolError, ConnectError)):
        return True

    # 通过错误消息匹配
    error_msg = str(exception).lower()
    return any(
        [
            "connection error" in error_msg,
            "server disconnected" in error_msg,
            "eof occurred" in error_msg,
            "timeout" in error_msg,
            "event loop is closed" in error_msg,  # 添加事件循环错误
            "anthropicexception" in error_msg,  # 添加 Anthropic 相关错误
        ]
    )


__CTX_VARS_NAME__ = "context_variables"
logger = LoggerManager.get_logger()


def adapt_tools_for_gemini(tools):
    """为 Gemini 模型适配工具定义，确保所有 OBJECT 类型参数都有非空的 properties"""

    adapted_tools = []
    for tool in tools:
        adapted_tool = copy.deepcopy(tool)

        # 检查参数
        if "parameters" in adapted_tool["function"]:
            params = adapted_tool["function"]["parameters"]

            # 处理顶层参数
            if params.get("type") == "object":
                if "properties" not in params or not params["properties"]:
                    params["properties"] = {
                        "dummy": {
                            "type": "string",
                            "description": "Dummy property for Gemini compatibility",
                        }
                    }

            # 处理嵌套参数
            if "properties" in params:
                for prop_name, prop in params["properties"].items():
                    if isinstance(prop, dict) and prop.get("type") == "object":
                        if "properties" not in prop or not prop["properties"]:
                            prop["properties"] = {
                                "dummy": {
                                    "type": "string",
                                    "description": "Dummy property for Gemini compatibility",
                                }
                            }

        adapted_tools.append(adapted_tool)
    return adapted_tools


class MetaChain:
    def __init__(self, log_path: Union[str, None, MetaChainLogger] = None):
        """
        log_path: path of log file, None
        """
        if logger:
            self.logger = logger
        elif isinstance(log_path, MetaChainLogger):
            self.logger = log_path
        else:
            self.logger = MetaChainLogger(log_path=log_path)
        # if self.logger.log_path is None: self.logger.info("[Warning] Not specific log path, so log will not be saved", "...", title="Log Path", color="light_cyan3")
        # else: self.logger.info("Log file is saved to", self.logger.log_path, "...", title="Log Path", color="light_cyan3")

    def _preprocess_tools_and_history(
        self,
        agent: Agent,
        history: list,
        context_variables: dict,
        model_override: str | None,
    ):
        # NOTE: preprocess llm_model:
        llm_model = model_override or agent.model

        # NOTE: preprocess tools:
        tools_schema = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools_schema:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)
        tools_description = convert_tools_to_description(tools_schema)
        if "gemini" in llm_model.lower():
            tools_schema = adapt_tools_for_gemini(tools_schema)

        # NOTE: preprocess history:
        if agent.examples:
            examples = (
                agent.examples(context_variables)
                if callable(agent.examples)
                else agent.examples
            )
            history = examples + history

        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )

        history_with_header = [
            {
                "role": "system",
                "content": instructions
                + "\n[IMPORTANT] You MUST use the tools provided to complete the task."
                + f"\n[IMPORTANT] The agent you are currently working on is {agent.name}. Each agent has an independent set of tool lists, and tools cannot be accessed across agents!\n"
                + SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(
                    description=tools_description, agent=agent.name
                ),
            }
        ] + history

        NO_SENDER_MODE = False
        for not_sender_model in NOT_SUPPORT_SENDER:
            if not_sender_model in llm_model:
                NO_SENDER_MODE = True
                break
        if NO_SENDER_MODE:
            for message in history_with_header:
                if "sender" in message:
                    del message["sender"]

        return llm_model, tools_schema, history_with_header

    def _construct_request_body(
        self,
        model: str,
        messages: list,
        tools: list | None = None,
        tool_choice: str | None = None,
        stream: bool = False,
        parallel_tool_calls: bool = False,
    ) -> dict:
        llm_request = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if tools:
            llm_request["tools"] = tools

        if tool_choice:
            llm_request["tool_choice"] = tool_choice

        if tools and model.startswith("gpt"):
            llm_request["parallel_tool_calls"] = parallel_tool_calls

        return llm_request

    def _reorganize_message_to_include_tool_calls(
        self,
        message: Message,
        agent: str,
        tools_schema: list,
        is_native_support_for_functions: bool,
    ):
        # NOTE: Format the standard response result into the tool_calls field to simulate the function call request of the native output
        assistant_message = [
            {
                "role": "assistant",
                "content": message.content,
            }
        ]

        assistant_message_with_tool_calls = (
            convert_non_fncall_messages_to_fncall_messages(
                agent, assistant_message, tools_schema
            )
        )

        if "tool_calls" in assistant_message_with_tool_calls[0]:
            converted_tool_calls = [
                ChatCompletionMessageToolCall(**tool_call)
                for tool_call in assistant_message_with_tool_calls[0]["tool_calls"]
            ]
        elif is_native_support_for_functions:
            converted_tool_calls = message.tool_calls
        else:
            converted_tool_calls = None

        message_be_return = Message(
            role="assistant",
            content=assistant_message_with_tool_calls[0]["content"],
            tool_calls=converted_tool_calls,
        )

        if hasattr(message, "reasoning_content"):
            message_be_return.reasoning_content = message.reasoning_content

        return message_be_return

    def _handle_function_result(self, result) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    self.logger.info(
                        error_message, title="Handle Function Result Error", color="red"
                    )
                    raise TypeError(error_message)

    def _handle_tool_calls(
        self,
        agent: Agent,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        handle_mm_func: Callable[[], str] | None = None,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                self.logger.info(
                    f"Tool {name} not found in function map. You are recommended to use `run_tool` to run this tool.",
                    title="Tool Call Error",
                )
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": f"Error: Tool {name} not found. You are recommended to use `run_tool` to run this tool.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            func = function_map[name]
            if __CTX_VARS_NAME__ in inspect.signature(func).parameters.keys():
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self._handle_function_result(raw_result)

            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": f"{result.value}\n{result.task_context if result.task_context else ''}",
                }
            )

            self.logger.print_message_block(partial_response.messages[-1])
            if result.image:
                assert (
                    handle_mm_func
                ), f"handle_mm_func is not provided, but an image is returned by tool call {name}({tool_call.function.arguments})"
                partial_response.messages.append(
                    {
                        "role": "user",
                        "content": [
                            # {"type":"text", "text":f"After take last action `{name}({tool_call.function.arguments})`, the image of current page is shown below. Please take next action based on the image, the current state of the page as well as previous actions and observations."},
                            {
                                "type": "text",
                                "text": handle_mm_func(
                                    name, tool_call.function.arguments
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{result.image}"
                                },
                            },
                        ],
                    }
                )

            # update是字典内置方法，用于将另一个字典或键值对的可迭代对象合并到当前字典中
            # 已有的键覆盖，未有的键添加
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent
            elif name != "case_resolved" and name != "case_not_resolved":
                partial_response.agent = agent

        return partial_response

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str | None,
        stream: bool,
    ) -> tuple[ModelResponse | CustomStreamWrapper, list]:
        llm_model, tools_schema, history_with_header = (
            self._preprocess_tools_and_history(
                agent=agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
            )
        )

        if FN_CALL:
            assert (
                supports_function_calling(model=llm_model) == True
            ), f"Model {llm_model} does not support function calling, please set `FN_CALL=False` to use non-function calling mode"

            history_with_header = remove_reasoning_content_in_messages(
                history_with_header
            )

            llm_request = self._construct_request_body(
                model=llm_model,
                messages=history_with_header,
                tools=tools_schema,
                tool_choice=agent.tool_choice,
                stream=stream,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

        else:
            assert (
                agent.tool_choice == "required"
            ), f"Non-function calling mode MUST use tool_choice = 'required' rather than {agent.tool_choice}"

            # NOTE: Remove the tool role from the conversation history, and make user and assistant appear alternately when necessary
            if NON_FN_CALL:
                history_with_header = convert_fn_messages_to_non_fn_messages(
                    history_with_header
                )

            if ADD_USER:
                history_with_header = interleave_user_into_messages(history_with_header)

            llm_request = self._construct_request_body(
                model=llm_model, messages=history_with_header, stream=stream
            )

        self.logger.info(
            "The information send to the model:",
            json.dumps(history_with_header, indent=2, default=str),
            title="Model Input",
        )

        llm_response = completion(**llm_request)

        if not stream:
            self.logger.info(
                "The information returned by the model:",
                (
                    llm_response.model_dump_json(indent=2)
                    if isinstance(llm_response, ModelResponse)
                    else json.dumps(llm_response, indent=2, default=str)
                ),
                title="Model Output",
            )

        return llm_response, tools_schema

    # NOTE: HKUDS original plan, its runnable example code is in ’repl‘ directory
    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str | None = None,
        max_turns: float = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.copy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
            )

            yield {"delim": "start"}
            for chunk in completion:
                assert isinstance(chunk, ModelResponseStream)
                assert isinstance(chunk.choices[0], StreamingChoices)
                delta = json.loads(chunk.choices[0].delta.to_json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None

            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self._handle_tool_calls(
                active_agent, tool_calls, active_agent.functions, context_variables
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str | None = None,
        stream: bool = False,
        max_turns: float = float("inf"),
    ):
        active_agent = agent
        context_variables = copy.copy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)
        is_native_support_for_functions = False

        self.logger.print_message_block(history[-1])

        while len(history) - init_len < max_turns and active_agent:

            llm_response, tools_schema = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
            )

            if stream:
                assert isinstance(llm_response, CustomStreamWrapper)
                stream_handler = LiteLLMStreamHandler(self.logger)
                stream_handler.init_handler()

                # try:
                #     loop = asyncio.get_event_loop()
                # except RuntimeError:
                #     loop = asyncio.new_event_loop()
                #     asyncio.set_event_loop(loop)
                # message = loop.run_until_complete(
                #     stream_handler.process_stream(llm_response)
                # )
                message = asyncio.run(stream_handler.process_stream(llm_response))
            else:
                assert isinstance(llm_response, ModelResponse)
                assert isinstance(llm_response.choices[0], Choices)
                message = llm_response.choices[0].message

            if message.tool_calls is not None:
                is_native_support_for_functions = True

            message = self._reorganize_message_to_include_tool_calls(
                message,
                active_agent.name,
                tools_schema,
                is_native_support_for_functions,
            )

            if not stream:
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    print(
                        f"[grey58]\n[推理过程开始]\n\n{message.reasoning_content}[/]",
                    )
                if message.content is not None and message.content != "":
                    print(f"[grey58]\n[AI答复]\n\n{message.content}[/]")
                if message.tool_calls is not None:
                    print(
                        f"[grey58]\n[工具调用开始] 正在调用 {message.tool_calls[0].function.name}...[/]"
                    )
                print()
            elif not is_native_support_for_functions:
                if message.tool_calls is not None:
                    print(
                        f"[grey58][工具调用开始] 正在调用 {message.tool_calls[0].function.name}...[/]"
                    )
                print()

            if hasattr(message, "reasoning_content") and message.reasoning_content:
                self.logger.info(
                    "The reasoning content:",
                    message.reasoning_content,
                    title="Model Reasoning",
                )

            message.sender = active_agent.name  # type: ignore

            self.logger.print_message_block(message)

            history.append(message.to_dict())

            task_status = ""
            if message.tool_calls:
                assert isinstance(message.tool_calls[0].function.name, str)
                task_status = (message.tool_calls[0].function.name).replace("_", " ")
                partial_response = self._handle_tool_calls(
                    active_agent,
                    message.tool_calls,
                    active_agent.functions,
                    context_variables,
                    handle_mm_func=active_agent.handle_mm_func,
                )
            else:
                task_status = "no_tool_calls"
                partial_response = Response(messages=[])
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if task_status == "additional_inquiry":
                break
            elif not partial_response.agent:
                self.logger.info(f"Ending turn with {task_status}.", title="End Turn")
                break
            active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=10, max=180),
        retry=should_retry_error,
        before_sleep=lambda retry_state: print(
            f"Retrying... (attempt {retry_state.attempt_number})"
        ),
    )
    async def get_chat_completion_async(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> Message:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        if agent.examples:
            examples = (
                agent.examples(context_variables)
                if callable(agent.examples)
                else agent.examples
            )
            history = examples + history

        messages = [{"role": "system", "content": instructions}] + history
        # debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        if FN_CALL:
            create_model = model_override or agent.model
            assert (
                litellm.supports_function_calling(model=create_model) == True
            ), f"Model {create_model} does not support function calling, please set `FN_CALL=False` to use non-function calling mode"

            create_params = {
                "model": create_model,
                "messages": messages,
                "tools": tools or None,
                "tool_choice": agent.tool_choice,
                "stream": stream,
            }
            NO_SENDER_MODE = False
            for not_sender_model in NOT_SUPPORT_SENDER:
                if not_sender_model in create_model:
                    NO_SENDER_MODE = True
                    break

            if NO_SENDER_MODE:
                messages = create_params["messages"]
                for message in messages:
                    if "sender" in message:
                        del message["sender"]
                create_params["messages"] = messages

            if tools and create_params["model"].startswith("gpt"):
                create_params["parallel_tool_calls"] = agent.parallel_tool_calls
            completion_response = await acompletion(**create_params)
        else:
            create_model = model_override or agent.model
            assert (
                agent.tool_choice == "required"
            ), f"Non-function calling mode MUST use tool_choice = 'required' rather than {agent.tool_choice}"
            last_content = messages[-1]["content"]
            tools_description = convert_tools_to_description(tools)
            messages[-1]["content"] = (
                last_content
                + "\n[IMPORTANT] You MUST use the tools provided to complete the task.\n"
                + SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=tools_description)
            )
            NO_SENDER_MODE = False
            for not_sender_model in NOT_SUPPORT_SENDER:
                if not_sender_model in create_model:
                    NO_SENDER_MODE = True
                    break

            if NO_SENDER_MODE:
                for message in messages:
                    if "sender" in message:
                        del message["sender"]
            create_params = {
                "model": create_model,
                "messages": messages,
                "stream": stream,
                "base_url": API_BASE_URL,
            }
            completion_response = await acompletion(**create_params)
            last_message = [
                {
                    "role": "assistant",
                    "content": completion_response.choices[0].message.content,
                }
            ]
            converted_message = convert_non_fncall_messages_to_fncall_messages(
                agent.name, last_message, tools
            )
            converted_tool_calls = [
                ChatCompletionMessageToolCall(**tool_call)
                for tool_call in converted_message[0]["tool_calls"]
            ]
            completion_response.choices[0].message = litellmMessage(
                content=converted_message[0]["content"],
                role="assistant",
                tool_calls=converted_tool_calls,
            )

        # response = await acompletion(**create_params)
        # response = await client.chat.completions.create(**create_params)
        return completion_response

    async def run_async(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = True,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        assert stream == False, "Async run does not support stream"
        active_agent = agent
        enter_agent = agent
        context_variables = copy.copy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        self.logger.info(
            "Receiveing the task:",
            history[-1]["content"],
            title="Receive Task",
            color="green",
        )

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = await self.get_chat_completion_async(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message: Message = completion.choices[0].message
            message.sender = active_agent.name
            # debug_print(debug, "Received completion:", message.model_dump_json(indent=4), log_path=log_path, title="Received Completion", color="blue")
            # self.logger.pretty_print_messages(message)
            self.logger.print_message_block(message)
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if enter_agent.tool_choice != "required":
                if (
                    not message.tool_calls and active_agent.name == enter_agent.name
                ) or not execute_tools:
                    self.logger.info("Ending turn.", title="End Turn", color="red")
                    break
            else:
                if (
                    message.tool_calls
                    and message.tool_calls[0].function.name == "case_resolved"
                ) or not execute_tools:
                    self.logger.info(
                        "Ending turn with case resolved.", title="End Turn", color="red"
                    )
                    partial_response = self._handle_tool_calls(
                        message.tool_calls,
                        active_agent.functions,
                        context_variables,
                        debug,
                        handle_mm_func=active_agent.handle_mm_func,
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    break
                elif (
                    message.tool_calls
                    and message.tool_calls[0].function.name == "case_not_resolved"
                ) or not execute_tools:
                    self.logger.info(
                        "Ending turn with case not resolved.",
                        title="End Turn",
                        color="red",
                    )
                    partial_response = self._handle_tool_calls(
                        message.tool_calls,
                        active_agent.functions,
                        context_variables,
                        debug,
                        handle_mm_func=active_agent.handle_mm_func,
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    break
                elif (not message.tool_calls) or not execute_tools:
                    self.logger.info(
                        "Ending turn with no tool calls.", title="End Turn", color="red"
                    )
                    break

            # if (message.tool_calls and message.tool_calls[0].function.name == "case_resolved") or not execute_tools:
            #     debug_print(debug, "Ending turn.", log_path=log_path, title="End Turn", color="red")
            #     break

            # handle function calls, updating context_variables, and switching agents
            if message.tool_calls:
                partial_response = self._handle_tool_calls(
                    message.tool_calls,
                    active_agent.functions,
                    context_variables,
                    debug,
                    handle_mm_func=active_agent.handle_mm_func,
                )
            else:
                partial_response = Response(messages=[message])
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
