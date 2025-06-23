# AutoAgent API 集成指南

## 概述

AutoAgent API 提供了一套完整的接口，让上层应用能够轻松集成 AutoAgent 框架的功能，并实时获取智能体执行过程中的详细信息。

## 功能特性

- ✅ **实时事件监控**: 获取智能体调用、工具使用、任务状态等详细信息
- ✅ **多智能体协作**: 支持智能体之间的切换和协作
- ✅ **灵活的部署方式**: 支持本地环境和 Docker 环境
- ✅ **事件回调机制**: 支持自定义事件处理逻辑
- ✅ **Web API 接口**: 提供 HTTP/WebSocket 接口（可选）
- ✅ **会话管理**: 支持多会话并发处理

## 快速开始

### 1. 基本使用

```python
from autoagent.api import AutoAgentAPI

# 创建 API 实例
api = AutoAgentAPI(
    container_name="my_agent",
    port=12347,
    local_env=True,  # 使用本地环境
    model="gpt-4o-2024-08-06"
)

# 处理用户查询
result = api.process_query("hello, what can you do?")

print(f"结果: {result['result']}")
print(f"成功: {result['success']}")
print(f"智能体: {result['agent_name']}")
```

### 2. 事件监控

```python
from autoagent.api import AutoAgentAPI, AgentEvent

def handle_event(event: AgentEvent):
    print(f"事件类型: {event.event_type}")
    print(f"时间戳: {event.timestamp}")
    print(f"数据: {event.data}")

# 创建 API 实例并添加事件回调
api = AutoAgentAPI(container_name="monitor_agent", local_env=True)
api.add_event_callback(handle_event)

# 处理查询，事件会实时回调
result = api.process_query("tell me today's weather")
```

## 事件类型说明

### 任务级事件

| 事件类型 | 说明 | 数据字段 |
|---------|------|---------|
| `task_start` | 任务开始 | `agent_name`, `user_query`, `context_variables` |
| `task_complete` | 任务完成 | `agent_name`, `final_result`, `total_turns` |
| `task_status` | 任务状态变化 | `status`, `agent_name`, `reason` |

### AI 思考和回复事件

| 事件类型 | 说明 | 数据字段 |
|---------|------|---------|
| `ai_thinking_start` | AI 开始思考 | `agent_name`, `turn` |
| `ai_response` | AI 回复内容 | `agent_name`, `content`, `has_tool_calls`, `tool_calls` |

### 工具调用事件

| 事件类型 | 说明 | 数据字段 |
|---------|------|---------|
| `tool_call_start` | 工具调用开始 | `agent_name`, `tool_name`, `tool_args`, `tool_call_id` |
| `tool_call_complete` | 工具调用完成 | `agent_name`, `tool_name`, `tool_result`, `task_context` |
| `tool_call_error` | 工具调用错误 | `agent_name`, `tool_name`, `tool_args`, `error` |

### 智能体管理事件

| 事件类型 | 说明 | 数据字段 |
|---------|------|---------|
| `agent_switch` | 智能体切换 | `from_agent`, `to_agent`, `reason` |
| `initialization_complete` | 初始化完成 | `container_name`, `model`, `available_agents` |
| `session_reset` | 会话重置 | `agent_name` |

### 查询级事件

| 事件类型 | 说明 | 数据字段 |
|---------|------|---------|
| `query_start` | 查询开始 | `query`, `agent_name`, `upload_files` |
| `query_complete` | 查询完成 | `query`, `result`, `agent_name`, `success` |
| `query_error` | 查询错误 | `query`, `error`, `agent_name` |

## 高级使用

### 1. 自定义事件处理器

```python
class MyEventHandler:
    def __init__(self):
        self.events = []
        self.tool_calls = []
        self.agent_switches = []
    
    def handle_event(self, event: AgentEvent):
        self.events.append(event)
        
        if event.event_type == "tool_call_complete":
            self.tool_calls.append({
                "tool": event.data["tool_name"],
                "agent": event.data["agent_name"],
                "result": event.data["tool_result"][:100]  # 截断长结果
            })
        
        elif event.event_type == "agent_switch":
            self.agent_switches.append({
                "from": event.data["from_agent"],
                "to": event.data["to_agent"],
                "reason": event.data["reason"]
            })
    
    def get_summary(self):
        return {
            "total_events": len(self.events),
            "tool_calls": len(self.tool_calls),
            "agent_switches": len(self.agent_switches),
            "tools_used": list(set([tc["tool"] for tc in self.tool_calls]))
        }

# 使用自定义处理器
handler = MyEventHandler()
api = AutoAgentAPI(container_name="custom_agent", local_env=True)
api.add_event_callback(handler.handle_event)

result = api.process_query("研究最新的 AI 技术发展")
summary = handler.get_summary()
print(f"执行摘要: {summary}")
```

### 2. 多会话管理

```python
from autoagent.api import AutoAgentAPI
import threading

class MultiSessionManager:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
    
    def create_session(self, session_id: str, **kwargs):
        with self.lock:
            if session_id not in self.sessions:
                api = AutoAgentAPI(
                    container_name=f"session_{session_id}",
                    **kwargs
                )
                self.sessions[session_id] = api
                return True
            return False
    
    def process_query(self, session_id: str, query: str):
        if session_id in self.sessions:
            return self.sessions[session_id].process_query(query)
        else:
            raise ValueError(f"Session {session_id} not found")
    
    def close_session(self, session_id: str):
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

# 使用多会话管理器
manager = MultiSessionManager()

# 创建多个会话
manager.create_session("user1", local_env=True, port=12350)
manager.create_session("user2", local_env=True, port=12351)

# 并发处理查询
result1 = manager.process_query("user1", "hello")
result2 = manager.process_query("user2", "what's the weather?")
```

## Web API 使用（可选）

如果需要通过 HTTP/WebSocket 接口使用 AutoAgent，首先安装额外依赖：

```bash
pip install -r requirements_web_api.txt
```

### 启动 Web API 服务

```python
from autoagent.web_api import create_web_api

# 创建并启动 Web API
web_api = create_web_api(host='0.0.0.0', port=5000, debug=True)
web_api.run()
```

### HTTP API 接口

#### 1. 创建智能体实例

```bash
curl -X POST http://localhost:5000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "container_name": "my_agent",
    "local_env": true,
    "model": "gpt-4o-2024-08-06"
  }'
```

#### 2. 处理查询

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "query": "hello, what can you do?"
  }'
```

#### 3. 获取智能体信息

```bash
curl http://localhost:5000/api/agents/user123/info
```

### WebSocket 实时事件

```javascript
// 前端 JavaScript 示例
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Connected to AutoAgent WebSocket');
    
    // 订阅特定会话的事件
    socket.emit('subscribe_session', {session_id: 'user123'});
});

socket.on('agent_event', (data) => {
    console.log('Event:', data.event.event_type);
    console.log('Data:', data.event.data);
    
    // 根据事件类型更新 UI
    if (data.event.event_type === 'tool_call_start') {
        showToolCallIndicator(data.event.data.tool_name);
    } else if (data.event.event_type === 'ai_response') {
        displayAIResponse(data.event.data.content);
    }
});
```

## 配置选项

### AutoAgentAPI 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `container_name` | str | "auto_agent" | 容器名称 |
| `port` | int | 12347 | 通信端口 |
| `local_env` | bool | False | 是否使用本地环境 |
| `model` | str | None | 使用的模型名称 |

### 环境变量配置

在使用前，确保设置了必要的环境变量：

```bash
# 模型配置
export COMPLETION_MODEL="gpt-4o-2024-08-06"
export API_BASE_URL="https://api.openai.com/v1"

# 功能开关
export FN_CALL=True
export STREAM=False
export MC_MODE=True
```

## 故障排除

### 常见问题

1. **端口冲突**: 如果指定端口被占用，API 会自动寻找下一个可用端口
2. **环境初始化失败**: 检查 Docker 服务是否运行（如果使用 Docker 环境）
3. **模型调用错误**: 验证 API 密钥和模型配置是否正确

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 创建 API 实例时启用调试
api = AutoAgentAPI(container_name="debug_agent", local_env=True)

# 添加调试事件处理器
def debug_handler(event):
    print(f"DEBUG: {event.event_type} - {event.data}")

api.add_event_callback(debug_handler)
```

## 最佳实践

1. **事件处理器要轻量**: 避免在事件回调中执行耗时操作
2. **适当的会话管理**: 及时清理不用的会话实例
3. **错误处理**: 总是检查返回结果的 `success` 字段
4. **资源监控**: 监控 Docker 容器或本地进程的资源使用情况
5. **异步处理**: 对于长时间运行的任务，考虑使用异步模式

## 示例项目

完整的示例代码请参考：
- `examples/api_usage_example.py` - 基本使用示例
- `autoagent/web_api.py` - Web API 实现
- `examples/frontend_integration/` - 前端集成示例（如果存在）
