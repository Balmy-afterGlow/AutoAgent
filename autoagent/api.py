"""
AutoAgent API 模块
为上层应用提供标准化的接口，支持实时获取执行过程中的详细信息
"""

import json
import os
import asyncio
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from queue import Queue
import threading

from .core import MetaChain
from .types import Agent, Response
from .logger import LoggerManager, MetaChainLogger
from .environment.docker_env import DockerEnv, DockerConfig
from .environment.local_env import LocalEnv
from .environment.browser_env import BrowserEnv
from .environment.markdown_browser import RequestsMarkdownBrowser
from .agents import get_orchestrator_agent
from constant import COMPLETION_MODEL, DOCKER_WORKPLACE_NAME


@dataclass
class AgentEvent:
    """智能体事件数据结构"""

    event_type: str
    timestamp: str
    data: Dict[str, Any]

    def to_dict(self):
        return asdict(self)


class AutoAgentAPI:
    """AutoAgent API 类，为上层应用提供统一接口"""

    def __init__(
        self,
        container_name: str = "auto_agent",
        port: int = 12347,
        local_env: bool = False,
        model: str = None,
    ):
        """
        初始化 AutoAgent API

        Args:
            container_name: 容器名称
            port: 端口号
            local_env: 是否使用本地环境
            model: 使用的模型名称
        """
        self.container_name = container_name
        self.port = port
        self.local_env = local_env
        self.model = model or COMPLETION_MODEL

        # 事件队列和回调
        self.event_queue = Queue()
        self.event_callbacks: List[Callable[[AgentEvent], None]] = []

        # 环境和智能体
        self.context_variables = {}
        self.agents = {}
        self.current_agent = None
        self.metachain = None

        # 初始化状态
        self.is_initialized = False

    def add_event_callback(self, callback: Callable[[AgentEvent], None]):
        """添加事件回调函数"""
        self.event_callbacks.append(callback)

    def remove_event_callback(self, callback: Callable[[AgentEvent], None]):
        """移除事件回调函数"""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)

    def _on_event(self, event_type: str, data: Dict[str, Any]):
        """内部事件处理函数"""
        event = AgentEvent(
            event_type=event_type, timestamp=datetime.now().isoformat(), data=data
        )

        # 添加到队列
        self.event_queue.put(event)

        # 调用所有回调函数
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Event callback error: {e}")

    def get_events(self, max_events: int = None) -> List[AgentEvent]:
        """获取事件队列中的事件"""
        events = []
        count = 0

        while not self.event_queue.empty() and (
            max_events is None or count < max_events
        ):
            events.append(self.event_queue.get())
            count += 1

        return events

    def _get_docker_config(self):
        """获取 Docker 配置"""
        import os
        from evaluation.utils import check_port_available

        # 端口检查逻辑
        port = self.port
        while not check_port_available(port):
            port += 1

        local_root = os.path.join(
            os.getcwd(), "workspace_meta_showcase", f"showcase_{self.container_name}"
        )
        os.makedirs(local_root, exist_ok=True)

        return DockerConfig(
            workplace_name=DOCKER_WORKPLACE_NAME,
            container_name=self.container_name,
            communication_port=port,
            conda_path="/root/miniconda3",
            local_root=local_root,
            test_pull_name="autoagent_mirror",
            git_clone=False,
            logger=LoggerManager.get_logger(),
        )

    def _create_environment(self, docker_config: DockerConfig):
        """创建环境"""
        if self.local_env:
            code_env = LocalEnv(docker_config)
        else:
            code_env = DockerEnv(docker_config)
            code_env.init_container()

        web_env = BrowserEnv(
            browsergym_eval_env=None,
            local_root=docker_config.local_root,
            workplace_name=docker_config.workplace_name,
        )

        file_env = RequestsMarkdownBrowser(
            viewport_size=1024 * 5,
            local_root=docker_config.local_root,
            workplace_name=docker_config.workplace_name,
            downloads_folder=os.path.join(
                docker_config.local_root, docker_config.workplace_name, "downloads"
            ),
        )

        return code_env, web_env, file_env

    def initialize(self):
        """初始化 AutoAgent 环境"""
        if self.is_initialized:
            return

        # 设置日志
        log_path = os.path.join(
            os.getcwd(),
            "logs",
            self.container_name,
            f"{self.model.split('/')[1] if '/' in self.model else self.model}",
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log",
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        LoggerManager.set_logger(MetaChainLogger(log_path))

        # 创建配置和环境
        docker_config = self._get_docker_config()
        code_env, web_env, file_env = self._create_environment(docker_config)

        # 设置上下文变量
        self.context_variables = {
            "working_dir": docker_config.workplace_name,
            "code_env": code_env,
            "web_env": web_env,
            "file_env": file_env,
        }

        # 初始化智能体
        orchestrator_agent = get_orchestrator_agent(self.model)
        self.agents = {orchestrator_agent.name.replace(" ", "_"): orchestrator_agent}

        for agent_name, call_func in orchestrator_agent.agent_teams.items():
            agent_instance = call_func("", "{}").agent
            self.agents[agent_name.replace(" ", "_")] = agent_instance

        self.current_agent = orchestrator_agent

        # 创建 MetaChain 实例，传入事件回调
        self.metachain = MetaChain(
            log_path=LoggerManager.get_logger(), event_callback=self._on_event
        )

        self.is_initialized = True

        # 发送初始化完成事件
        self._on_event(
            "initialization_complete",
            {
                "container_name": self.container_name,
                "model": self.model,
                "available_agents": list(self.agents.keys()),
            },
        )

    def process_query(
        self, query: str, agent_name: str = None, upload_files: List[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户查询

        Args:
            query: 用户查询
            agent_name: 指定的智能体名称（可选）
            upload_files: 上传的文件列表（可选）

        Returns:
            包含执行结果的字典
        """
        if not self.is_initialized:
            self.initialize()

        # 选择智能体
        if agent_name and agent_name in self.agents:
            agent = self.agents[agent_name]
        else:
            agent = self.current_agent

        # 处理上传文件
        if upload_files:
            file_info = "\n\nUser uploaded files:\n" + "\n".join(upload_files)
            query = query + file_info

        # 构建消息
        messages = [{"role": "user", "content": query}]

        # 发送查询开始事件
        self._on_event(
            "query_start",
            {
                "query": query,
                "agent_name": agent.name,
                "upload_files": upload_files or [],
            },
        )

        try:
            # 执行查询
            response = self.metachain.run(
                agent=agent,
                messages=messages,
                context_variables=self.context_variables,
                stream=False,
            )

            # 更新当前智能体
            if response.agent:
                self.current_agent = response.agent

            # 提取最终结果
            model_answer_raw = (
                response.messages[-1]["content"] if response.messages else ""
            )

            # 解析结果
            if model_answer_raw.startswith("Case resolved"):
                import re

                model_answer = re.findall(
                    r"<solution>(.*?)</solution>", model_answer_raw, re.DOTALL
                )
                if len(model_answer) == 0:
                    model_answer = model_answer_raw
                else:
                    model_answer = model_answer[0]
            else:
                model_answer = model_answer_raw

            result = {
                "success": True,
                "result": model_answer,
                "raw_result": model_answer_raw,
                "agent_name": response.agent.name if response.agent else agent.name,
                "messages": response.messages,
                "context_variables": response.context_variables,
            }

            # 发送查询完成事件
            self._on_event(
                "query_complete",
                {
                    "query": query,
                    "result": model_answer,
                    "agent_name": response.agent.name if response.agent else agent.name,
                    "success": True,
                },
            )

            return result

        except Exception as e:
            error_result = {"success": False, "error": str(e), "agent_name": agent.name}

            # 发送查询错误事件
            self._on_event(
                "query_error",
                {"query": query, "error": str(e), "agent_name": agent.name},
            )

            return error_result

    def get_available_agents(self) -> List[str]:
        """获取可用的智能体列表"""
        if not self.is_initialized:
            self.initialize()
        return list(self.agents.keys())

    def get_current_agent(self) -> str:
        """获取当前智能体名称"""
        return self.current_agent.name if self.current_agent else None

    def reset_session(self):
        """重置会话状态"""
        if self.is_initialized:
            orchestrator_agent = get_orchestrator_agent(self.model)
            self.current_agent = orchestrator_agent

            # 发送会话重置事件
            self._on_event("session_reset", {"agent_name": orchestrator_agent.name})


# 便利的全局函数
def create_agent_api(
    container_name: str = "auto_agent",
    port: int = 12347,
    local_env: bool = False,
    model: str = None,
) -> AutoAgentAPI:
    """创建 AutoAgent API 实例的便利函数"""
    return AutoAgentAPI(container_name, port, local_env, model)


def process_query_simple(
    query: str, container_name: str = "auto_agent", model: str = None
) -> Dict[str, Any]:
    """简单的查询处理函数，适用于一次性查询"""
    api = create_agent_api(container_name=container_name, model=model)
    return api.process_query(query)
