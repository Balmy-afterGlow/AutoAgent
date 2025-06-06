# from autoagent.agents.coding_agent import get_coding_agent
# from autoagent.agents.tool_retriver_agent import get_tool_retriver_agent
# from autoagent.agents.agent_check_agent import get_agent_check_agent
# from autoagent.agents.tool_check_agent import get_tool_check_agent
# from autoagent.agents.github_agent import get_github_agent
# from autoagent.agents.programming_triage_agent import get_programming_triage_agent
# from autoagent.agents.plan_agent import get_plan_agent

# import os
# import importlib
# from autoagent.registry import registry

# # 获取当前目录下的所有 .py 文件
# current_dir = os.path.dirname(__file__)
# for file in os.listdir(current_dir):
#     if file.endswith('.py') and not file.startswith('__'):
#         module_name = file[:-3]
#         importlib.import_module(f'autoagent.agents.{module_name}')

# # 导出所有注册的 agent 创建函数
# globals().update(registry.agents)

# __all__ = list(registry.agents.keys())

import os
import importlib
from autoagent.registry import registry


def import_agents_recursively(base_dir: str, base_package: str):
    """Recursively import all agents in .py files

    Args:
        base_dir: the root directory to start searching
        base_package: the base name of the Python package
    """
    for root, dirs, files in os.walk(base_dir):
        # get the relative path to the base directory
        rel_path = os.path.relpath(root, base_dir)

        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                # build the module path
                if rel_path == ".":
                    # in the root directory
                    module_path = f"{base_package}.{file[:-3]}"
                else:
                    # in the subdirectory
                    package_path = rel_path.replace(os.path.sep, ".")
                    module_path = f"{base_package}.{package_path}.{file[:-3]}"

                try:
                    importlib.import_module(module_path)
                except Exception as e:
                    print(f"Warning: Failed to import {module_path}: {e}")


# get the current directory and import all agents
# 动态导入当前目录下的所有agents
# 这里动态导入的目的是为了自动执行模块级代码（装饰器/全局语句），执行模块才会使用到装饰器注册到registry中
current_dir = os.path.dirname(__file__)
import_agents_recursively(current_dir, "autoagent.agents")

# export all agent creation functions
# 注入当前模块的全局作用域，动态导入的模块内容不会自动成为全局名称
globals().update(registry.agents)
globals().update(registry.plugin_agents)

__all__ = list(registry.agents.keys())
