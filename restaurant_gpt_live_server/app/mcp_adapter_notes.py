"""
MCP-ready boundary.

Do NOT run this module as an MCP server yet. The transport-specific MCP layer is
intentionally deferred until the actual customer-service operations are defined.

The important architectural choice is already in place:

    ToolRegistry.execute(name, arguments)

is the single business-tool entry point. A future MCP server should expose wrappers
around those same functions instead of duplicating order/customer-service logic.

For example, a future MCP tool named `lookup_order` can call:

    await registry.execute("lookup_order", {"order_id": ...})

The GPT-Live session can then either:
1. keep using local function tools through Responses delegation, or
2. be configured to use a remote MCP server when that workflow is ready.

This file is documentation-by-code so the MCP boundary is explicit from day one.
"""
