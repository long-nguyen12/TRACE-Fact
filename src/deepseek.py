"""DeepSeek chat-completion adapter with native JSON output."""

import json
import os
from typing import Any, Optional

from .llm import parse_json_output


_SYSTEM_PROMPT = (
    "Return exactly one valid JSON object and no Markdown or surrounding text. "
    "Follow the JSON shape and requirements in the user prompt."
)


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
        self.api_key = str(
            api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        ).strip()
        if not self.api_key:
            raise ValueError("Set the DEEPSEEK_API_KEY environment variable.")
        self.base_url = str(base_url).strip()
        if not self.base_url:
            raise ValueError("A DeepSeek base URL is required.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.max_tokens = max_tokens
        self.client: Any = None

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self._load()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            stream=False,
        )
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek returned an unexpected response shape.") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek returned empty JSON content.")
        try:
            parsed = parse_json_output(content)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("DeepSeek did not return a valid JSON object.") from exc
        return json.dumps(parsed, ensure_ascii=False)

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


__all__ = ["DeepSeekLLM"]
