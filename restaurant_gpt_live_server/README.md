# Restaurant GPT-Live Call Server

A FastAPI service for a restaurant phone assistant using OpenAI GPT-Live over SIP.

## Architecture

```text
Caller
  |
  v
Twilio phone number
  |
  v
Your existing TwiML / SIP routing
  |
  v
OpenAI SIP endpoint for your Project
  |
  +---- SIP audio ----> GPT-Live 1
  |
  +---- OpenAI webhook ----> this FastAPI server
                              |
                              +-- accepts/configures the Live session
                              +-- attaches a sideband WebSocket
                              +-- runs private custom tools
                              +-- owns the local RAG knowledge base

GPT-Live 1
  |
  +-- delegates factual work to a Responses backend (default: gpt-6-luna)
        |
        +-- function call: search_knowledge_base
              |
              v
         SQLite + OpenAI embeddings
```

The phone audio does **not** pass through this Python server in this architecture.
Twilio sends SIP media to OpenAI. The server handles call authorization/configuration,
private tools, RAG, and later customer-service business logic.

## What is implemented

- Inbound GPT-Live SIP call acceptance
- `gpt-live-1` model selection and voice selection
- Restaurant-specific Live conversation prompt
- Responses delegation for backend reasoning/tool selection
- Private `search_knowledge_base` function tool
- Local SQLite knowledge store
- OpenAI embedding-based semantic retrieval
- Admin ingestion/search endpoints
- CLI ingestion for a text knowledge document
- OpenAI webhook signature verification
- Sideband WebSocket tool execution
- One-time spoken greeting
- A shared `ToolRegistry` designed to become the business-logic layer behind a future
  MCP server
- Ordering is intentionally disabled until real order/customer-service operations exist

## 1. Configure secrets

Copy the example:

```bash
cp .env.example .env
```

Put your **OpenAI Project API key** here:

```dotenv
OPENAI_API_KEY=sk-proj-your-project-key
```

Do not put the key in source code or commit `.env`.

You also need the **webhook signing secret** from the webhook endpoint you create in
the OpenAI Project:

```dotenv
OPENAI_WEBHOOK_SECRET=whsec_your-webhook-signing-secret
```

Finally, create a long random admin token:

```dotenv
ADMIN_TOKEN=some-long-random-secret
```

These are different secrets:
- `OPENAI_API_KEY` authorizes this server to accept/control the Live call and create
  embeddings.
- `OPENAI_WEBHOOK_SECRET` proves that an incoming webhook actually came from OpenAI.

## 2. Install and run

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

For real calls, this server must be reachable through public HTTPS, for example through
your deployed cloud service or a secure development tunnel.


### Docker

```bash
docker build -t restaurant-gpt-live .
docker run --rm -p 8000:8000 --env-file .env restaurant-gpt-live
```

If you deploy the SQLite version to a cloud container, attach persistent storage for
`/app/data`. Otherwise the knowledge base can disappear when the container is replaced.

## 3. Configure the OpenAI Project webhook

Your Twilio/SIP routing is separate from this step.

In the same OpenAI Project used by the SIP address, create a webhook endpoint pointing
to:

```text
https://YOUR_PUBLIC_HOST/webhooks/openai
```

Subscribe to:

```text
live.transport.incoming
```

The server also understands the older `live.call.incoming` event during migration.

A SIP invite can produce both a Live event and a Realtime event. This server deliberately
ignores `realtime.call.incoming` and accepts the Live event so the call runs on
`gpt-live-1`.

## 4. Ingest the restaurant knowledge document

Once you have a UTF-8 text document:

```bash
python -m app.ingest /path/to/restaurant_knowledge.txt --source restaurant-about-food
```

By default, re-ingesting the same source label replaces that source's prior chunks.

Or POST text to the admin endpoint:

```bash
curl -X POST http://127.0.0.1:8000/admin/knowledge/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: YOUR_ADMIN_TOKEN" \
  -d '{
    "source": "restaurant-about-food",
    "replace_source": true,
    "text": "Your knowledge-base text goes here..."
  }'
```

Test retrieval before calling:

```bash
curl -X POST http://127.0.0.1:8000/admin/knowledge/search \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: YOUR_ADMIN_TOKEN" \
  -d '{"query":"What oil are the fries cooked in?","top_k":5}'
```


## Bundled McDonald's knowledge base

This project now includes the supplied source at:

```text
data/McDonalds_About_Our_Food_Summary.txt
```

The document is parsed page-by-page so each retrieval chunk retains its source page
title and URL. This avoids mixing supplier stories, nutrition qualifications, and
food-quality claims into one generic chunk stream.

With `AUTO_BOOTSTRAP_KB=true` (the default), the first server launch detects an
empty SQLite knowledge database and automatically embeds this document. So after `.env`
contains your valid Project API key, you can simply run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The log should report a page-aware bootstrap with `sections_found: 26`. You can also
force a manual rebuild at any time with:

```bash
python -m app.bootstrap_kb
```

The source itself says it is a summary of accessible pages rather than independent
verification. It also documents gaps: detailed `/product/` pages, the general FAQ
answer set, some dynamic nutrition results, corporate reports, and linked external
certification material are outside its scope. The backend prompt therefore preserves
those limits rather than inventing missing facts.

## 5. Place a test call

Your existing path should now be:

```text
Twilio number -> TwiML/SIP -> OpenAI Project SIP endpoint
```

On the OpenAI side:

1. OpenAI receives the SIP call.
2. OpenAI POSTs `live.transport.incoming` to `/webhooks/openai`.
3. This server verifies the webhook.
4. This server accepts `POST /v1/live/sessions/{session_id}/accept`.
5. The accepted session is configured with `gpt-live-1`, the restaurant prompt, the
   selected voice, and Responses delegation.
6. This server attaches to
   `wss://api.openai.com/v1/live/sessions/{session_id}/attach`.
7. GPT-Live talks to the caller while the Responses backend requests private RAG
   lookups through `search_knowledge_base`.
8. The server executes the search locally, returns the function result, and tells the
   delegated response to continue.

## Prompt behavior

The Live frontend is instructed to:

- sound like a concise restaurant phone assistant
- delegate factual restaurant questions before answering
- avoid guessing
- say ordering is not enabled yet
- be careful with allergy/cross-contact statements

The backend is separately instructed to search the knowledge base before making factual
restaurant claims.

Keeping those prompts separate is intentional: GPT-Live handles the conversation;
the backend handles factual retrieval/reasoning.

## RAG implementation

This starter uses:

- `text-embedding-3-small`
- 512-dimensional embeddings
- about 320 words per chunk with 60-word overlap
- SQLite persistence
- in-process NumPy cosine similarity

That is deliberately simple and easy to inspect. It is appropriate for a restaurant
knowledge corpus. If this grows into tens of thousands of chunks or multiple tenants,
keep the `KnowledgeBase.search()` interface and swap the implementation for pgvector,
Qdrant, Pinecone, Weaviate, or another vector store.

## Adding future order/customer-service actions

Add each real business operation to `ToolRegistry` rather than directly to the Live
WebSocket code. Examples later might include:

- `lookup_menu_item`
- `create_order_draft`
- `add_item_to_order`
- `quote_order_total`
- `confirm_order`
- `lookup_order`
- `cancel_order`
- `transfer_to_store`

For consequential actions, implement server-side authorization, validation, and explicit
confirmation before mutation.

`ToolRegistry` is also the boundary a future MCP server should wrap, so the same business
logic can be exposed through MCP without duplicating it.

## Production hardening still recommended

Before handling real customer transactions:

- move webhook idempotency to Redis/database if running multiple app instances
- add a real customer/order data store
- add authentication/authorization for customer-specific data
- implement explicit confirmation before purchases, cancellations, or refunds
- add structured audit logs for actions, not secret credentials
- define retention/privacy policy for transcripts and call data
- add monitoring and alerting
- use a managed vector DB if the KB becomes large
- add tests/evals for allergen, nutrition, pricing, order confirmation, interruption,
  and tool-failure cases
