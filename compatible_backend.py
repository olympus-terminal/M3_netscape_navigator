"""Tool-free structured transformations through a recording model proxy."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import aiohttp

from config import LLM_BASE_URL, LLM_ENABLE_THINKING, LLM_MAX_TOKENS, LLM_MODEL

log = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """An endpoint failed, or returned an incomplete/unsafe transformation."""


async def _request(method: str, path: str, timeout_seconds: float, body=None):
    if not LLM_MODEL:
        raise ModelError("LLM_MODEL must explicitly identify the served model")
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds, connect=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method, LLM_BASE_URL + path, json=body, allow_redirects=False,
            ) as response:
                if response.status != 200:
                    # Raw server errors may echo prompts or credentials. The
                    # recording proxy retains the detailed diagnostic locally.
                    raise ModelError(f"Model endpoint returned HTTP {response.status}; check proxy ledger")
                result = await response.json(content_type=None)
                if not isinstance(result, dict):
                    raise ModelError("Model endpoint returned a non-object response")
                return result
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        raise ModelError(f"Model endpoint request failed ({type(exc).__name__})") from exc


async def preflight(timeout_seconds: float = 15) -> tuple[str, str]:
    data = await _request("GET", "/models", timeout_seconds)
    models = data.get("data")
    if not isinstance(models, list) or LLM_MODEL not in {
        item.get("id") for item in models if isinstance(item, dict)
    }:
        raise ModelError(f"Configured model {LLM_MODEL!r} is not advertised by the endpoint")
    log.info("Model preflight passed: %s via %s", LLM_MODEL, LLM_BASE_URL)
    return LLM_MODEL, "recording proxy"


async def transform(
    instruction: str, payload: str, *, timeout_seconds: float,
    output_schema: Path | None = None,
) -> str:
    if not instruction.strip() or timeout_seconds <= 0:
        raise ValueError("A nonempty instruction and positive timeout are required")
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": payload},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": LLM_ENABLE_THINKING},
    }
    if output_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.stem.replace(".", "_"),
                "strict": True,
                "schema": json.loads(output_schema.read_text(encoding="utf-8")),
            },
        }
    result = await _request("POST", "/chat/completions", timeout_seconds, body)
    if result.get("model") != LLM_MODEL:
        raise ModelError("Response model does not match the configured model")
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ModelError("Expected exactly one model completion")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ModelError("Model response has no assistant message")
    if message.get("tool_calls") or message.get("function_call"):
        raise ModelError("Model attempted a disallowed tool call")
    if choice.get("finish_reason") != "stop":
        raise ModelError(f"Model completion is incomplete: {choice.get('finish_reason')!r}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ModelError("Model returned empty content")
    if output_schema is not None:
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelError("Model returned invalid structured JSON") from exc
    log.info("Model call complete: model=%s usage=%s", result["model"], result.get("usage"))
    return content.strip()
