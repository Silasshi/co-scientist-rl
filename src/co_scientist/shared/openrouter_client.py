"""Async OpenRouter API client for the co-scientist multi-turn pipeline.

Used for student model (Gemini Flash) and PRM judge (Qwen3-235B) calls.
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 30.0
RETRYABLE_STATUS = {429, 500, 502, 503}


class OpenRouterClient:
    """Async HTTP client for OpenRouter chat completions."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY env var "
                "or pass api_key directly."
            )
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._total_tokens = 0

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request with retry logic.

        Returns the assistant's response text.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": [],           # explicitly disable tool use
            "web_search": False,   # explicitly disable web search
        }

        delay = BASE_DELAY
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.post(OPENROUTER_URL, json=payload)

                if response.status_code in RETRYABLE_STATUS:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else delay
                    wait = min(wait, MAX_DELAY)
                    logger.warning(
                        "OpenRouter %d (attempt %d/%d), retrying in %.1fs",
                        response.status_code,
                        attempt + 1,
                        MAX_RETRIES + 1,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, MAX_DELAY)
                    continue

                response.raise_for_status()
                data = response.json()

                # Track token usage
                usage = data.get("usage", {})
                self._total_tokens += usage.get("total_tokens", 0)

                choices = data.get("choices", [])
                if not choices:
                    logger.warning("OpenRouter returned empty choices for model=%s", model)
                    return ""

                choice = choices[0]
                if choice.get("finish_reason") == "length":
                    logger.warning(
                        "OpenRouter response truncated (max_tokens=%d) for model=%s",
                        max_tokens,
                        model,
                    )

                content = choice.get("message", {}).get("content", "")
                return content

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "OpenRouter timeout (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    e,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)

            except httpx.HTTPStatusError as e:
                if e.response.status_code in RETRYABLE_STATUS:
                    logger.warning(
                        "OpenRouter HTTP %d (attempt %d/%d)",
                        e.response.status_code,
                        attempt + 1,
                        MAX_RETRIES + 1,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, MAX_DELAY)
                    last_error = e
                else:
                    raise

        raise RuntimeError(
            f"OpenRouter request failed after {MAX_RETRIES + 1} attempts: {last_error}"
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all calls."""
        return self._total_tokens

    def reset_token_counter(self):
        """Reset the token counter (call at batch boundaries)."""
        self._total_tokens = 0

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
