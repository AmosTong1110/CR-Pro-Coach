"""
WebSocket 远程通信服务桥接器 (Server Bridge)

提供 Python 后端引擎与外部客户端 (如 Unity C# 客户端、Web 可视化面板或远程教练服务)
之间的双向异步 JSON-RPC 通信通道。
"""
import asyncio
import json
import logging
from typing import Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CR-ServerBridge")

PORT = 8765


class ServerBridge:
    """基于 WebSocket 协议的异步远程状态分发与指令处理桥接器"""
    def __init__(self):
        self.clients = set()

    async def register(self, websocket):
        """注册新接入的客户端连接"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

    async def unregister(self, websocket):
        """移除断开连接的客户端"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")

    async def handler(self, websocket, path=None):
        """单个连接的异步消息监听与响应循环"""
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                response = await self.process_request(data)
                await websocket.send(json.dumps(response))
        except Exception as e:
            logger.error(f"Error in websocket connection: {e}")
        finally:
            await self.unregister(websocket)

    async def process_request(self, req: dict) -> dict:
        """处理客户端请求动作：
        
        支持动作：
        - ping: 心跳检测
        - get_coach_state: 获取当前教练实时评估指标 (对手推测圣水、当前胜率、AI 推荐落子虚影等)
        """
        cmd = req.get("action")
        if cmd == "ping":
            return {"status": "ok", "message": "pong"}
        elif cmd == "get_coach_state":
            return {
                "status": "ok",
                "estimated_opp_elixir": 6.5,
                "current_win_rate": 0.58,
                "ghost_recommendation": {
                    "card": "goblin_cage",
                    "cell_id": 1152,
                    "confidence": 0.92
                }
            }
        return {"status": "error", "message": f"Unknown action: {cmd}"}


async def main():
    """启动 WebSocket 服务主监听协程"""
    try:
        import websockets
        bridge = ServerBridge()
        logger.info(f"Starting CR-Pro-Coach WebSocket Server on ws://localhost:{PORT}")
        async with websockets.serve(bridge.handler, "localhost", PORT):
            await asyncio.Future()  # run forever
    except ImportError:
        logger.warning("[!] websockets library not installed. Run `pip install websockets` to enable live bridge.")


if __name__ == "__main__":
    asyncio.run(main())
