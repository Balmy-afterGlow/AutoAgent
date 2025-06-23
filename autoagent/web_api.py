"""
AutoAgent Web API
提供 HTTP 接口用于与上层应用集成，支持 WebSocket 实时事件推送
"""

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import json
import threading
from typing import Dict, Any
import uuid

from autoagent.api import AutoAgentAPI, AgentEvent


class AutoAgentWebAPI:
    """AutoAgent Web API 服务"""

    def __init__(self, host="0.0.0.0", port=5000, debug=False):
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "autoagent_secret_key"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self.host = host
        self.port = port
        self.debug = debug

        # 存储活跃的 AutoAgent 实例
        self.active_agents: Dict[str, AutoAgentAPI] = {}

        # 设置路由
        self._setup_routes()
        self._setup_socketio_events()

    def _setup_routes(self):
        """设置 HTTP 路由"""

        @self.app.route("/api/health", methods=["GET"])
        def health_check():
            """健康检查"""
            return jsonify(
                {"status": "healthy", "message": "AutoAgent Web API is running"}
            )

        @self.app.route("/api/agents", methods=["POST"])
        def create_agent():
            """创建新的智能体实例"""
            try:
                data = request.get_json()
                session_id = data.get("session_id", str(uuid.uuid4()))
                container_name = data.get("container_name", "auto_agent")
                port = data.get("port", 12347)
                local_env = data.get("local_env", True)
                model = data.get("model", None)

                # 创建 AutoAgent 实例
                agent_api = AutoAgentAPI(
                    container_name=f"{container_name}_{session_id}",
                    port=port,
                    local_env=local_env,
                    model=model,
                )

                # 添加 WebSocket 事件回调
                def event_callback(event: AgentEvent):
                    self.socketio.emit(
                        "agent_event",
                        {"session_id": session_id, "event": event.to_dict()},
                    )

                agent_api.add_event_callback(event_callback)

                # 存储实例
                self.active_agents[session_id] = agent_api

                return jsonify(
                    {
                        "success": True,
                        "session_id": session_id,
                        "message": "Agent created successfully",
                    }
                )

            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/agents/<session_id>", methods=["DELETE"])
        def delete_agent(session_id):
            """删除智能体实例"""
            if session_id in self.active_agents:
                del self.active_agents[session_id]
                return jsonify(
                    {"success": True, "message": "Agent deleted successfully"}
                )
            else:
                return jsonify({"success": False, "error": "Agent not found"}), 404

        @self.app.route("/api/query", methods=["POST"])
        def process_query():
            """处理查询请求"""
            try:
                data = request.get_json()
                session_id = data.get("session_id")
                query = data.get("query", "")
                agent_name = data.get("agent_name", None)
                upload_files = data.get("upload_files", None)

                if not session_id or session_id not in self.active_agents:
                    return jsonify(
                        {
                            "success": False,
                            "error": "Invalid session_id or agent not found",
                        }
                    ), 400

                agent_api = self.active_agents[session_id]

                # 处理查询
                result = agent_api.process_query(
                    query=query, agent_name=agent_name, upload_files=upload_files
                )

                return jsonify(result)

            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/agents/<session_id>/info", methods=["GET"])
        def get_agent_info(session_id):
            """获取智能体信息"""
            if session_id not in self.active_agents:
                return jsonify({"success": False, "error": "Agent not found"}), 404

            agent_api = self.active_agents[session_id]

            return jsonify(
                {
                    "success": True,
                    "session_id": session_id,
                    "current_agent": agent_api.get_current_agent(),
                    "available_agents": agent_api.get_available_agents(),
                    "is_initialized": agent_api.is_initialized,
                }
            )

        @self.app.route("/api/agents/<session_id>/reset", methods=["POST"])
        def reset_agent(session_id):
            """重置智能体会话"""
            if session_id not in self.active_agents:
                return jsonify({"success": False, "error": "Agent not found"}), 404

            agent_api = self.active_agents[session_id]
            agent_api.reset_session()

            return jsonify(
                {"success": True, "message": "Agent session reset successfully"}
            )

        @self.app.route("/api/agents/<session_id>/events", methods=["GET"])
        def get_events(session_id):
            """获取事件历史"""
            if session_id not in self.active_agents:
                return jsonify({"success": False, "error": "Agent not found"}), 404

            agent_api = self.active_agents[session_id]
            max_events = request.args.get("max_events", type=int)

            events = agent_api.get_events(max_events=max_events)

            return jsonify(
                {
                    "success": True,
                    "events": [event.to_dict() for event in events],
                    "count": len(events),
                }
            )

    def _setup_socketio_events(self):
        """设置 WebSocket 事件"""

        @self.socketio.on("connect")
        def handle_connect():
            """客户端连接"""
            print(f"Client connected: {request.sid}")
            emit("connected", {"message": "Connected to AutoAgent WebSocket"})

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """客户端断开连接"""
            print(f"Client disconnected: {request.sid}")

        @self.socketio.on("subscribe_session")
        def handle_subscribe(data):
            """订阅特定会话的事件"""
            session_id = data.get("session_id")
            if session_id:
                # 这里可以实现更细粒度的会话订阅逻辑
                emit("subscribed", {"session_id": session_id})

    def run(self):
        """启动 Web API 服务"""
        print(f"Starting AutoAgent Web API on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug)


# 便利函数
def create_web_api(host="0.0.0.0", port=5000, debug=False) -> AutoAgentWebAPI:
    """创建 Web API 实例"""
    return AutoAgentWebAPI(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # 创建并启动 Web API
    web_api = create_web_api(port=5000, debug=True)
    web_api.run()
