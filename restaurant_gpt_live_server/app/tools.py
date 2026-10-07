import json
from typing import Any

from .knowledge import KnowledgeBase


class ToolRegistry:
    """
    One tool layer shared by GPT-Live today and an MCP adapter later.

    Add future order/customer-service business functions here. The Live adapter and
    future MCP server can both call the same execute() method, keeping business logic
    out of transport-specific code.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def live_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "search_knowledge_base",
                "description": (
                    "Search the restaurant's private knowledge base for verified facts "
                    "about food, ingredients, preparation, nutrition, allergens, sourcing, "
                    "restaurant information, and policies. Use this before answering "
                    "restaurant factual questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A concise semantic search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8,
                            "description": "How many passages to retrieve; normally 5.",
                        },
                    },
                    "required": ["query", "top_k"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_knowledge_base":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return {"error": "query must not be empty"}
            top_k = int(arguments.get("top_k", 5))
            return await self.kb.search(query=query, top_k=top_k)

        return {"error": f"Unknown tool: {name}"}

    async def execute_json(self, name: str, arguments_json: str) -> str:
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid tool arguments: {exc}"})

        result = await self.execute(name, arguments)
        return json.dumps(result, ensure_ascii=False)
