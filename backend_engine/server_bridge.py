"""
WebSocket Server Bridge for CR-Pro-Coach.
Provides bidirectional JSON-RPC communication between the Python backend engine
and the Unity (C#) client interface.
"""
import asyncio
import json
import logging
from typing import Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CR-ServerBridge")

PORT = 8765


class ServerBridge:
    def __init__(self):
        self.clients = set()

    async def register(self, websocket):
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

    async def unregister(self, websocket):
        self.clients.remove(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")

    async def handler(self, websocket, path=None):
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
