"""DeepSeek chat-completion adapter with native JSON output."""

import json
import os
from pathlib import Path
from typing import Any, Optional

from .llm import _normalize_messages, parse_json_output

_SYSTEM_PROMPT = (
    "Return exactly one valid JSON object and no Markdown or surrounding text. "
    "Follow the JSON shape and requirements in the user prompt."
)

_EMPTY_RESPONSE_RETRY = (
    "The previous response was empty. Return the required JSON object now. "
    "Follow the original field requirements exactly, do not add new facts, and "
    "do not return Markdown or any text outside the JSON object."
)


def load_deepseek_api_key(api_key: Optional[str] = None) -> str:
    """Return an explicit key, an environment key, or a key from project .env."""
    if api_key is not None:
        return str(api_key).strip()

    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Reading DEEPSEEK_API_KEY from .env requires python-dotenv. "
            "Install requirements-deepseek.txt."
        ) from exc

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()


class DeepSeekLLM:
    """Generate validated JSON through DeepSeek's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 1024,
    ) -> None:
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("A DeepSeek model name is required.")
        self.api_key = load_deepseek_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "Set DEEPSEEK_API_KEY in the project .env file or environment."
            )
        self.base_url = str(base_url).strip()
        if not self.base_url:
            raise ValueError("A DeepSeek base URL is required.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.max_tokens = max_tokens
        self.client: Any = None

    def generate(self, prompt: Any) -> str:
        messages = _normalize_messages(prompt)
        if messages[0]["role"] == "system":
            messages[0]["content"] = "%s\n\n%s" % (
                _SYSTEM_PROMPT,
                messages[0]["content"],
            )
        else:
            messages.insert(0, {"role": "system", "content": _SYSTEM_PROMPT})
        self._load()
        finish_reason = None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            # max_tokens=self.max_tokens,
            stream=False,
        )
        try:
            choice = response.choices[0]
            content = choice.message.content
            print(f"DeepSeek response content: {content}")
            finish_reason = getattr(choice, "finish_reason", None)
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "DeepSeek returned an unexpected response shape."
            ) from exc

        if finish_reason == "length":
            raise RuntimeError(
                "DeepSeek JSON output was truncated. Increase " "DEEPSEEK_MAX_TOKENS."
            )
        if isinstance(content, str) and content.strip():
            try:
                parsed = parse_json_output(content)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "DeepSeek did not return a valid JSON object."
                ) from exc
            return json.dumps(parsed, ensure_ascii=False)

        detail = " (finish_reason=%s)" % finish_reason if finish_reason else ""
        raise RuntimeError("DeepSeek returned empty JSON content%s." % detail)

    def _load(self) -> None:
        if self.client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The DeepSeek backend requires the OpenAI Python package. "
                "Install requirements-deepseek.txt."
            ) from exc
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)


__all__ = ["DeepSeekLLM", "load_deepseek_api_key"]
