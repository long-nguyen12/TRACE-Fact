"""Language-model wrappers and JSON output parsing."""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_messages(prompt: Any) -> List[Dict[str, str]]:
    """Normalize a string or role-based prompt into validated chat messages."""
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or not prompt:
        raise TypeError("Prompt must be a string or a non-empty message list.")

    messages: List[Dict[str, str]] = []
    for index, message in enumerate(prompt):
        if not isinstance(message, dict):
            raise TypeError("Prompt message at index %d must be an object." % index)
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Prompt message at index %d has an invalid role." % index)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Prompt message at index %d must have non-empty text content." % index
            )
        messages.append({"role": role, "content": content})
    return messages


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    """Flatten chat messages for legacy caller-supplied generation callbacks."""
    return "\n\n".join(message["content"] for message in messages)


def parse_json_output(output: str, *, allow_top_level_array: bool = False) -> Any:
    """Parse a JSON object, or an allowed array, from an optional JSON fence."""
    if not isinstance(output, str):
        raise TypeError("Model output must be a string containing a JSON object.")

    candidate = output.strip()
    fenced = _JSON_FENCE.fullmatch(candidate)
    if fenced:
        candidate = fenced.group("body").strip()

    try:
        parsed = json.loads(candidate)
    except RecursionError as exc:
        raise ValueError("Model JSON is nested too deeply.") from exc
    if not isinstance(parsed, dict) and not (
        allow_top_level_array and isinstance(parsed, list)
    ):
        expected = (
            "a JSON object or array" if allow_top_level_array else "a JSON object"
        )
        raise ValueError(f"Model output must be {expected}.")
    return parsed


class LLM:
    """Generate text with Transformers or a caller-supplied function."""

    def __init__(
        self,
        model: Any,
        generate_fn: Optional[Callable[[Any], str]] = None,
        *,
        device_map: Any = "auto",
        dtype: Any = "auto",
        max_new_tokens: int = 512,
        local_files_only: bool = False,
    ) -> None:
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("A Hugging Face model ID or local path is required.")
        if generate_fn is not None and not callable(generate_fn):
            raise TypeError("generate_fn must be callable.")
        self.uses_default_generator = generate_fn is None
        self.generate_fn = (
            generate_fn
            if generate_fn is not None
            else _HuggingFaceTextGenerator(
                model=self.model,
                device_map=device_map,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                local_files_only=local_files_only,
            )
        )

    def generate(self, prompt: Any) -> str:
        messages = _normalize_messages(prompt)
        generator_input = (
            messages if self.uses_default_generator else _messages_to_text(messages)
        )
        output = self.generate_fn(generator_input)
        if not isinstance(output, str):
            raise TypeError("LLM generate_fn must return a string.")
        return output


class _HuggingFaceTextGenerator:
    """Lazy local inference adapter for Transformers chat/causal models."""

    def __init__(
        self,
        model: str,
        device_map: Any,
        dtype: Any,
        max_new_tokens: int,
        local_files_only: bool,
    ) -> None:
        self.model_name = model
        self.device_map = device_map
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.generation_kwargs = {
            "do_sample": False,
            # "max_new_tokens": max_new_tokens,
            "num_beams": 1,
        }
        self.tokenizer = None
        self.model = None
        self.torch = None

    def __call__(self, prompt: Any) -> str:
        self._load()
        tokenizer = self.tokenizer
        model = self.model
        torch = self.torch

        messages = _normalize_messages(prompt)
        if getattr(tokenizer, "chat_template", None):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer(_messages_to_text(messages), return_tensors="pt")

        inputs = _move_inputs(inputs, model)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(**inputs, **self.generation_kwargs)
        # sequences = getattr(generated, "sequences", generated)

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        output_ids = generated[0][len(inputs.input_ids[0]) :].tolist()
        print(tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n"))
        return tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend requires PyTorch and Transformers. "
                "Activate an environment with PyTorch, then install "
                "requirements-huggingface.txt there."
            ) from exc

        model_location = _resolve_model_location(self.model_name, self.local_files_only)
        load_kwargs = {}
        if self.local_files_only:
            load_kwargs["local_files_only"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_location, **load_kwargs)
        model_kwargs = {}
        model_kwargs.update(load_kwargs)
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        if self.dtype is not None:
            model_kwargs["dtype"] = self.dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_location, **model_kwargs
        )
        self.model.eval()
        self.torch = torch


def _resolve_model_location(model_name: str, local_files_only: bool) -> str:
    """Resolve a cached model ID to a directory before Transformers sees it."""
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path.resolve())
    if not local_files_only:
        return model_name
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Model %r is not available in the local Hugging Face cache. "
            "Download it first or set local_files_only=False." % model_name
        ) from exc


def _move_inputs(inputs: Any, model: Any) -> Any:
    """Move a Transformers BatchEncoding to the model's input device."""
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
