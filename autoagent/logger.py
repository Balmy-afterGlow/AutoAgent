from datetime import datetime
from rich.console import Console
from rich.markup import escape
import json
from typing import List
from constant import DEBUG, DEFAULT_LOG, LOG_PATH, MC_MODE
from pathlib import Path
from litellm.types.utils import ModelResponse, Choices
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

BAR_LENGTH = 60


class MetaChainLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.console = Console()
        self.debug = DEBUG

    def _write_log(self, message: str):
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def _warp_args(self, args_json_format: str):
        args_dict = json.loads(args_json_format)
        args_str = ""
        for k, v in args_dict.items():
            args_str += f"{repr(k)}={repr(v)}, "
        # 去除最后一个逗号和空格
        return args_str[:-2]

    def _wrap_title(self, title: str):
        single_len = (BAR_LENGTH - len(title)) // 2
        return f"{'-'*single_len} {title} {'-'*single_len}"

    def info(self, *args: str, **kwargs: str):
        timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
        message = "\n".join(map(str, args))
        title = kwargs.get("title", "INFO")
        log_str = f"{self._wrap_title(title)}\n[{timestamp}]\n{message}\n"
        if self.debug:
            # escape用于避免rich解析标签渲染富文本，原样输出字符串
            # highlight=True用于高亮输出，当字符串中包含代码、文件路径、URL、数字等可识别的结构化内容时，rich 会尝试用不同颜色标记它们e
            # emoji=True控制是否将 :emoji_name: 转换为实际的 Unicode 表情符号
            self.console.print(f"[grey58]{escape(log_str)}", highlight=True, emoji=True)
        if self.log_path:
            self._write_log(log_str)

    # 只能对具有__dict__属性的对象进行递归，后来发现message的类型是基于pydantic模型的，有特定的json化方案，所以此函数不再使用
    def _recursive_vars(self, message):
        if not hasattr(message, "__dict__"):
            return message
        res = {}
        for key, value in vars(message).items():
            if hasattr(value, "__dict__"):
                res[key] = self._recursive_vars(value)
            elif isinstance(value, (list, tuple)):
                res[key] = [self._recursive_vars(item) for item in value]
            elif isinstance(value, dict):
                res[key] = {k: self._recursive_vars(v) for k, v in value.items()}
            else:
                res[key] = value
        return res

    def print_message_block(self, message):
        match message["role"]:
            case "user":
                self.info(
                    "Receiveing the task:", message["content"], title="User Message"
                )
            case "assistant":
                # message_dict = self._recursive_vars(message)
                # json_str = json.dumps(message_dict, indent=2, default=str)
                self.info(
                    f"{message["sender"]}:",
                    message["content"],
                    "\n",
                    "original message:",
                    message.to_json(),
                    title="Assistant Message",
                )

                # - 如果message["tool_calls"]不存在，返回空列表[]
                # - 如果message["tool_calls"]为None，也会返回空列表[]
                # - 只有明确设置了非None值时才会返回原值
                tool_calls = message.get("tool_calls") or []
                for tool_call in tool_calls:
                    f = tool_call["function"]
                    name, args = f["name"], f["arguments"]
                    arg_str = self._warp_args(args)
                    self.info(
                        f"{message["sender"]} call tool:",
                        f"{name}({arg_str})",
                        title="Tool Call",
                    )
            case "tool":
                self.info(
                    f"tool execution: {message['name']}",
                    "Result:",
                    "\n+++++ the start +++++\n",
                    f"{message['content']}",
                    "\n+++++ the end +++++",
                    title="Tool Execution",
                )


class LoggerManager:
    _instance = None
    _logger: MetaChainLogger | None = None

    # 保证单例模式
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LoggerManager()
        return cls._instance

    @classmethod
    def get_logger(cls):
        return cls.get_instance()._logger

    @classmethod
    def set_logger(cls, new_logger):
        cls.get_instance()._logger = new_logger


if DEFAULT_LOG:
    if LOG_PATH is None:
        log_dir = Path(f'logs/res_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        log_dir.mkdir(
            parents=True, exist_ok=True
        )  # recursively create all necessary parent directories
        log_path = str(log_dir / "agent.log")
        # logger = MetaChainLogger(log_path=log_path)
        LoggerManager.set_logger(MetaChainLogger(log_path=log_path))
    else:
        # logger = MetaChainLogger(log_path=LOG_PATH)
        LoggerManager.set_logger(MetaChainLogger(log_path=LOG_PATH))
    # logger.info("Log file is saved to", logger.log_path, "...", title="Log Path", color="light_cyan3")
    LoggerManager.get_logger().info(
        "Log file is saved to",
        LoggerManager.get_logger().log_path,
        "...",
        title="Log Path",
        color="light_cyan3",
    )
else:
    # logger = None
    LoggerManager.set_logger(None)
logger = LoggerManager.get_logger()


def set_logger(new_logger):
    LoggerManager.set_logger(new_logger)


# if __name__ == "__main__":
#     logger = MetaChainLogger(log_path="test.log")
#     logger.pretty_print_messages({"role": "assistant", "content": "Hello, world!", "tool_calls": [{"function": {"name": "test", "arguments": {"url": "https://www.google.com", "query": "test"}}}], "sender": "test_agent"})

#     logger.pretty_print_messages({"role": "tool", "name": "test", "content": "import requests\n\nurl = 'https://www.google.com'\nquery = 'test'\n\nresponse = requests.get(url)\nprint(response.text)", "sender": "test_agent"})
#     logger.info("test content", color="red", title="test")
