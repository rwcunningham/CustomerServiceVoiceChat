import asyncio
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .knowledge import KnowledgeBase
from .live import LiveCallManager
from .tools import ToolRegistry


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(default="restaurant-kb", min_length=1, max_length=300)
    replace_source: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=8)


def require_admin(settings: Settings, token: str | None) -> None:
    if not token or token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    kb = KnowledgeBase(settings)
    kb.init_db()

    if settings.auto_bootstrap_kb and kb.stats()["total_chunks"] == 0:
        bundled_path = Path(settings.bundled_kb_path)
        if bundled_path.exists():
            logging.getLogger(__name__).info(
                "Knowledge DB is empty; bootstrapping %s", bundled_path
            )
            text = bundled_path.read_text(encoding="utf-8")
            result = await kb.ingest_page_summary_document(
                text=text,
                source=settings.bundled_kb_source,
                replace_source=True,
            )
            logging.getLogger(__name__).info("KB bootstrap complete: %s", result)

    tools = ToolRegistry(kb)
    calls = LiveCallManager(settings, tools)
    webhook_client = OpenAI(api_key=settings.openai_api_key)

    app.state.settings = settings
    app.state.kb = kb
    app.state.tools = tools
    app.state.calls = calls
    app.state.webhook_client = webhook_client
    app.state.call_tasks = set()
    app.state.seen_webhook_ids = set()

    yield

    tasks = list(app.state.call_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    await calls.close()


app = FastAPI(
    title="Restaurant GPT-Live Call Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    stats = request.app.state.kb.stats()
    return {
        "ok": True,
        "service": "restaurant-gpt-live",
        "knowledge_base": stats,
    }


@app.post("/webhooks/openai")
async def openai_webhook(request: Request) -> dict[str, bool]:
    raw = await request.body()
    if len(raw) > 1_048_576:
        raise HTTPException(status_code=413, detail="Webhook body too large")

    settings: Settings = request.app.state.settings
    client: OpenAI = request.app.state.webhook_client

    try:
        # Verify the ORIGINAL body bytes before parsing JSON. We intentionally ignore
        # the SDK's typed return object and parse the verified payload ourselves so the
        # handler is not coupled to a particular generated event class.
        client.webhooks.unwrap(
            raw,
            request.headers,
            secret=settings.openai_webhook_secret,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Rejected invalid OpenAI webhook: %s", exc
        )
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event = json.loads(raw)

    event_id = event.get("id")
    event_type = event.get("type")

    # Basic in-process webhook deduplication. For horizontally scaled production
    # deployments, replace this with Redis or another shared idempotency store.
    if event_id:
        seen: set[str] = request.app.state.seen_webhook_ids
        if event_id in seen:
            return {"ok": True}
        seen.add(event_id)
        # Bound memory for a long-running single-process starter deployment.
        if len(seen) > 10_000:
            seen.clear()
            seen.add(event_id)

    if event_type in {"live.transport.incoming", "live.call.incoming"}:
        data = event.get("data") or {}
        session_id = data.get("session_id")
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="Incoming Live webhook has no data.session_id",
            )

        task = asyncio.create_task(
            request.app.state.calls.handle_incoming_call(session_id)
        )
        tasks: set[asyncio.Task] = request.app.state.call_tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    elif event_type == "realtime.call.incoming":
        # A pending SIP call may emit both Live and Realtime events. This project
        # intentionally chooses GPT-Live, so DO NOT accept the Realtime event too.
        logging.getLogger(__name__).info(
            "Ignoring realtime.call.incoming because this service uses GPT-Live"
        )

    return {"ok": True}


@app.post("/admin/knowledge/ingest")
async def ingest_knowledge(
    body: IngestRequest,
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(request.app.state.settings, x_admin_token)
    return await request.app.state.kb.ingest_text(
        text=body.text,
        source=body.source,
        replace_source=body.replace_source,
    )


@app.post("/admin/knowledge/search")
async def search_knowledge(
    body: SearchRequest,
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(request.app.state.settings, x_admin_token)
    return await request.app.state.kb.search(
        query=body.query,
        top_k=body.top_k,
    )


@app.get("/admin/knowledge/stats")
async def knowledge_stats(
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(request.app.state.settings, x_admin_token)
    return request.app.state.kb.stats()
