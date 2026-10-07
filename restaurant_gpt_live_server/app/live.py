import asyncio
import json
import logging
import uuid
from typing import Any

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from .config import Settings
from .prompts import backend_instructions, live_instructions
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class LiveCallManager:
    API_BASE = "https://api.openai.com/v1"
    LIVE_ATTACH_BASE = "wss://api.openai.com/v1/live/sessions"

    def __init__(self, settings: Settings, tools: ToolRegistry):
        self.settings = settings
        self.tools = tools
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self.http.aclose()

    def _session_config(self) -> dict[str, Any]:
        return {
            "type": "live",
            "model": self.settings.live_model,
            "instructions": live_instructions(self.settings),
            "audio": {
                "output": {
                    "voice": self.settings.live_voice,
                }
            },
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": self.settings.backend_model,
                    "instructions": backend_instructions(self.settings),
                    "tools": self.tools.live_tool_schemas(),
                    "tool_choice": "auto",
                    # Keeping this false makes custom tool-result handling simple and
                    # prevents a response from waiting on multiple local functions.
                    "parallel_tool_calls": False,
                    "max_output_tokens": 700,
                },
            },
        }

    async def accept_call(self, session_id: str) -> None:
        url = f"{self.API_BASE}/live/sessions/{session_id}/accept"
        payload = {"session": self._session_config()}

        response = await self.http.post(url, json=payload)
        if response.is_error:
            body = response.text[:2000]
            raise RuntimeError(
                f"Live accept failed ({response.status_code}): {body}"
            )

        logger.info("Accepted GPT-Live SIP session %s", session_id)

    async def handle_incoming_call(self, session_id: str) -> None:
        try:
            await self.accept_call(session_id)
            await self.run_sideband(session_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Call handler failed for session %s", session_id)

    async def run_sideband(self, session_id: str) -> None:
        url = f"{self.LIVE_ATTACH_BASE}/{session_id}/attach"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        processed_tool_call_ids: set[str] = set()

        try:
            async with websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_size=8 * 1024 * 1024,
            ) as ws:
                logger.info("Attached sideband to %s", session_id)

                # An attached sideband belongs to an already-running session; don't send
                # session.start. We can immediately append a one-time greeting request.
                if self.settings.greet_on_connect:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.instructions.append",
                                "event_id": f"greeting_{uuid.uuid4().hex}",
                                "delegation_id": None,
                                "content": (
                                    f"Greet the caller now in English. Introduce yourself "
                                    f"as the automated assistant for "
                                    f"{self.settings.restaurant_name}, say you can help "
                                    f"with questions about the food, and ask how you can "
                                    f"help. Then pause and listen."
                                ),
                            }
                        )
                    )

                async for raw_message in ws:
                    event = json.loads(raw_message)
                    event_type = event.get("type")

                    if event_type == "session.closed":
                        logger.info("GPT-Live session closed: %s", session_id)
                        break

                    if event_type == "error":
                        logger.error(
                            "GPT-Live sideband error for %s: %s",
                            session_id,
                            json.dumps(event, ensure_ascii=False)[:4000],
                        )
                        continue

                    if self.settings.log_transcripts and event_type in {
                        "session.input_transcript.delta",
                        "session.output_transcript.delta",
                    }:
                        logger.info(
                            "%s %s: %s",
                            session_id,
                            event_type,
                            event.get("delta", ""),
                        )

                    if event_type != "response.event":
                        continue

                    nested = event.get("event") or {}
                    if nested.get("type") != "response.output_item.done":
                        continue

                    item = nested.get("item") or {}
                    if item.get("type") != "function_call":
                        continue

                    call_id = item.get("call_id")
                    tool_name = item.get("name")
                    arguments_json = item.get("arguments", "{}")

                    if not call_id or not tool_name:
                        logger.warning("Malformed function-call item: %s", item)
                        continue

                    # Sideband/replay safety: execute each function call once.
                    if call_id in processed_tool_call_ids:
                        continue
                    processed_tool_call_ids.add(call_id)

                    logger.info(
                        "Executing tool %s for Live session %s",
                        tool_name,
                        session_id,
                    )

                    output = await self.tools.execute_json(
                        name=tool_name,
                        arguments_json=arguments_json,
                    )

                    await ws.send(
                        json.dumps(
                            {
                                "type": "response.item.create",
                                "event_id": f"tool_result_{uuid.uuid4().hex}",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": output,
                                },
                            }
                        )
                    )

                    # Explicitly resume the delegated Responses backend after the tool
                    # result is supplied.
                    await ws.send(
                        json.dumps(
                            {
                                "type": "response.create",
                                "event_id": f"continue_{uuid.uuid4().hex}",
                            }
                        )
                    )

        except ConnectionClosed as exc:
            logger.info(
                "Sideband connection closed for %s (code=%s)",
                session_id,
                getattr(exc, "code", None),
            )

    async def hangup(self, session_id: str) -> None:
        response = await self.http.post(
            f"{self.API_BASE}/live/sessions/{session_id}/hangup",
            content=b"",
        )
        if response.is_error:
            raise RuntimeError(
                f"Hangup failed ({response.status_code}): {response.text[:1000]}"
            )
