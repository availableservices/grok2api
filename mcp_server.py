"""
Grok2API MCP Server

Exposes Grok2API as MCP tools for use with Claude and other MCP clients.

Usage:
    uv run mcp_server.py

Environment variables:
    GROK2API_BASE_URL  - Base URL of the Grok2API server (default: http://localhost:8000)
    GROK2API_API_KEY   - API key for authentication (default: empty)
"""

import os
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("GROK2API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("GROK2API_API_KEY", "")

mcp = FastMCP(
    "Grok2API",
    instructions=(
        "Tools for interacting with Grok AI through the Grok2API proxy server. "
        "Supports chat completions, image generation, video generation, and model listing."
    ),
)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


async def _post(endpoint: str, payload: dict) -> Any:
    url = f"{BASE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload, headers=_auth_headers())
        resp.raise_for_status()
        return resp.json()


async def _get(endpoint: str) -> Any:
    url = f"{BASE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_auth_headers())
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_models() -> dict:
    """
    List all available Grok models.

    Returns a dictionary with a 'data' list where each item contains:
    - id: Model identifier
    - object: Always 'model'
    - owned_by: Owner identifier
    """
    return await _get("/v1/models")


@mcp.tool()
async def chat_completion(
    model: str,
    message: str,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """
    Send a chat message to a Grok model and get a completion.

    Args:
        model: Model ID to use (e.g. 'grok-3', 'grok-3-mini'). Use list_models to see available models.
        message: The user message to send.
        system_prompt: Optional system prompt to set context/behaviour.
        stream: Whether to use streaming (returns non-streaming response regardless for MCP).
        thinking: Whether to enable chain-of-thought thinking output.
        reasoning_effort: Reasoning intensity: none/minimal/low/medium/high/xhigh.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.

    Returns:
        OpenAI-compatible chat completion response dict with 'choices' array.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,  # MCP always uses non-streaming
    }

    if thinking is not None:
        payload["thinking"] = thinking
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    return await _post("/v1/chat/completions", payload)


@mcp.tool()
async def multi_turn_chat(
    model: str,
    messages: list[dict],
    system_prompt: Optional[str] = None,
    thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """
    Send a multi-turn conversation to a Grok model.

    Args:
        model: Model ID to use. Use list_models to see available models.
        messages: List of message dicts with 'role' (user/assistant) and 'content' keys.
                  Example: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "How are you?"}]
        system_prompt: Optional system prompt prepended to the conversation.
        thinking: Whether to enable chain-of-thought thinking output.
        reasoning_effort: Reasoning intensity: none/minimal/low/medium/high/xhigh.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.

    Returns:
        OpenAI-compatible chat completion response dict with 'choices' array.
    """
    all_messages = []
    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})
    all_messages.extend(messages)

    payload: dict[str, Any] = {
        "model": model,
        "messages": all_messages,
        "stream": False,
    }

    if thinking is not None:
        payload["thinking"] = thinking
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    return await _post("/v1/chat/completions", payload)


@mcp.tool()
async def generate_image(
    prompt: str,
    model: str = "grok-imagine-1.0",
    n: int = 1,
    size: str = "1024x1024",
    response_format: Optional[str] = None,
) -> dict:
    """
    Generate images using Grok's image generation model.

    Args:
        prompt: Text description of the image to generate.
        model: Model ID (default: 'grok-imagine-1.0').
        n: Number of images to generate (1-10).
        size: Image dimensions. Options: 1024x1024, 1280x720 (16:9), 720x1280 (9:16),
              1792x1024 (3:2), 1024x1792 (2:3).
        response_format: Response format ('url' or 'b64_json'). Leave empty to use server default.

    Returns:
        OpenAI-compatible image generation response with 'data' list containing image URLs or base64.
    """
    payload: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "n": n,
        "size": size,
    }
    if response_format:
        payload["response_format"] = response_format

    return await _post("/v1/images/generations", payload)


@mcp.tool()
async def generate_video(
    prompt: str,
    size: str = "1792x1024",
    seconds: int = 6,
    quality: str = "standard",
) -> dict:
    """
    Generate a video from a text prompt using Grok's video model.

    Args:
        prompt: Text description of the video to generate.
        size: Video dimensions. Options: 1792x1024 (3:2), 1280x720 (16:9),
              1024x1792 (2:3), 720x1280 (9:16), 1024x1024 (1:1).
        seconds: Video length in seconds (6-30).
        quality: Video quality: 'standard' (480p) or 'high' (720p).

    Returns:
        Response containing video URL or HTML embed link.
    """
    payload: dict[str, Any] = {
        "prompt": prompt,
        "model": "grok-imagine-1.0-video",
        "size": size,
        "seconds": seconds,
        "quality": quality,
    }
    return await _post("/v1/videos/generations", payload)


@mcp.tool()
async def get_server_info() -> dict:
    """
    Get information about the Grok2API server configuration including version and available features.

    Returns:
        Server info dict with base_url and connection status.
    """
    try:
        models = await _get("/v1/models")
        model_ids = [m["id"] for m in models.get("data", [])]
        return {
            "status": "connected",
            "base_url": BASE_URL,
            "available_models": model_ids,
            "has_api_key": bool(API_KEY),
        }
    except Exception as exc:
        return {
            "status": "error",
            "base_url": BASE_URL,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
