"""
AutoAgent API 使用示例
展示如何在上层应用中集成 AutoAgent 并实时获取执行过程信息
"""

from autoagent.api import AutoAgentAPI, AgentEvent
import json
import time
from typing import Dict, Any


class AutoAgentEventHandler:
    """事件处理器示例"""

    def __init__(self):
        self.events_log = []

    def handle_event(self, event: AgentEvent):
        """处理 AutoAgent 事件"""
        self.events_log.append(event)

        # 根据事件类型进行不同处理
        if event.event_type == "task_start":
            print(f"🚀 任务开始: {event.data.get('user_query', '')}")
            print(f"📋 使用智能体: {event.data.get('agent_name', '')}")

        elif event.event_type == "ai_thinking_start":
            print(
                f"🤔 {event.data.get('agent_name', '')} 开始思考... (第 {event.data.get('turn', '')} 轮)"
            )

        elif event.event_type == "ai_response":
            agent_name = event.data.get("agent_name", "")
            content = event.data.get("content", "")
            has_tools = event.data.get("has_tool_calls", False)

            print(f"💭 {agent_name} 回复: {content}")
            if has_tools:
                tool_calls = event.data.get("tool_calls", [])
                print(f"🔧 准备调用工具: {[tc['name'] for tc in tool_calls]}")

        elif event.event_type == "tool_call_start":
            tool_name = event.data.get("tool_name", "")
            agent_name = event.data.get("agent_name", "")
            print(f"⚡ {agent_name} 开始调用工具: {tool_name}")

        elif event.event_type == "tool_call_complete":
            tool_name = event.data.get("tool_name", "")
            agent_name = event.data.get("agent_name", "")
            result = (
                event.data.get("tool_result", "")[:100] + "..."
                if len(event.data.get("tool_result", "")) > 100
                else event.data.get("tool_result", "")
            )
            print(f"✅ {agent_name} 完成工具调用: {tool_name}")
            print(f"📄 工具结果: {result}")

        elif event.event_type == "agent_switch":
            from_agent = event.data.get("from_agent", "")
            to_agent = event.data.get("to_agent", "")
            reason = event.data.get("reason", "")
            print(f"🔄 智能体切换: {from_agent} -> {to_agent} ({reason})")

        elif event.event_type == "task_complete":
            agent_name = event.data.get("agent_name", "")
            turns = event.data.get("total_turns", 0)
            print(f"🎉 任务完成! 智能体: {agent_name}, 总轮数: {turns}")

        elif event.event_type == "query_complete":
            result = (
                event.data.get("result", "")[:200] + "..."
                if len(event.data.get("result", "")) > 200
                else event.data.get("result", "")
            )
            print(f"📋 最终结果: {result}")

        print("-" * 50)

    def get_events_summary(self) -> Dict[str, Any]:
        """获取事件摘要"""
        event_types = {}
        for event in self.events_log:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        return {
            "total_events": len(self.events_log),
            "event_types": event_types,
            "timeline": [
                (event.timestamp, event.event_type) for event in self.events_log
            ],
        }


def example_basic_usage():
    """基本使用示例"""
    print("=== AutoAgent API 基本使用示例 ===\n")

    # 创建事件处理器
    event_handler = AutoAgentEventHandler()

    # 创建 API 实例
    api = AutoAgentAPI(
        container_name="demo_agent",
        port=12350,
        local_env=False,
    )

    # 添加事件回调
    api.add_event_callback(event_handler.handle_event)

    # 初始化（可选，process_query 会自动初始化）
    print("📦 初始化 AutoAgent...")
    api.initialize()

    # 处理查询
    print("🔍 处理查询...")
    result = api.process_query("hello, what can you do?")

    print("\n=== 查询结果 ===")
    print(f"成功: {result['success']}")
    print(f"结果: {result.get('result', result.get('error', ''))}")
    print(f"智能体: {result['agent_name']}")

    # 获取事件摘要
    print("\n=== 事件摘要 ===")
    summary = event_handler.get_events_summary()
    print(f"总事件数: {summary['total_events']}")
    print(f"事件类型分布: {summary['event_types']}")


def example_weather_query():
    """天气查询示例"""
    print("\n=== 天气查询示例 ===\n")

    event_handler = AutoAgentEventHandler()

    api = AutoAgentAPI(container_name="weather_agent", port=12351, local_env=False)

    api.add_event_callback(event_handler.handle_event)

    # 处理天气查询
    result = api.process_query("tell me today's weather in Beijing")

    print("\n=== 查询结果 ===")
    print(f"成功: {result['success']}")
    print(f"结果: {result.get('result', result.get('error', ''))}")

    # 展示详细的事件信息
    print("\n=== 详细事件日志 ===")
    for event in event_handler.events_log:
        print(
            f"[{event.timestamp}] {event.event_type}: {json.dumps(event.data, ensure_ascii=False, indent=2)}"
        )


def example_real_time_monitoring():
    """实时监控示例"""
    print("\n=== 实时监控示例 ===\n")

    # 创建一个实时事件监控器
    class RealTimeMonitor:
        def __init__(self):
            self.current_task = None
            self.tool_calls = []
            self.agent_switches = []

        def monitor_event(self, event: AgentEvent):
            if event.event_type == "task_start":
                self.current_task = event.data.get("user_query", "")
                print(f"📝 开始监控任务: {self.current_task}")

            elif event.event_type == "tool_call_start":
                tool_info = {
                    "tool": event.data.get("tool_name", ""),
                    "agent": event.data.get("agent_name", ""),
                    "start_time": event.timestamp,
                }
                self.tool_calls.append(tool_info)
                print(f"🔧 工具调用: {tool_info['agent']} -> {tool_info['tool']}")

            elif event.event_type == "agent_switch":
                switch_info = {
                    "from": event.data.get("from_agent", ""),
                    "to": event.data.get("to_agent", ""),
                    "time": event.timestamp,
                }
                self.agent_switches.append(switch_info)
                print(f"🔄 智能体切换: {switch_info['from']} -> {switch_info['to']}")

            elif event.event_type == "task_complete":
                print(f"✅ 任务完成!")
                print(f"📊 统计信息:")
                print(f"   - 工具调用次数: {len(self.tool_calls)}")
                print(f"   - 智能体切换次数: {len(self.agent_switches)}")
                print(
                    f"   - 使用的工具: {list(set([tc['tool'] for tc in self.tool_calls]))}"
                )

    monitor = RealTimeMonitor()

    api = AutoAgentAPI(container_name="monitor_agent", port=12352, local_env=False)

    api.add_event_callback(monitor.monitor_event)

    # 执行一个复杂任务
    result = api.process_query(
        "Search for the latest AI research papers and summarize the top 3"
    )

    print(f"\n📋 最终结果: {result.get('result', result.get('error', ''))}")


if __name__ == "__main__":
    # 运行示例
    try:
        example_basic_usage()
        time.sleep(2)

        example_weather_query()
        time.sleep(2)

        example_real_time_monitoring()

    except KeyboardInterrupt:
        print("\n❌ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback

        traceback.print_exc()
