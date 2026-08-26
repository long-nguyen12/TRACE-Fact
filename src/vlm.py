"""Vision-language-model wrappers."""

from pathlib import Path
from typing import Any, Callable, Optional

from .llm import _move_inputs, _resolve_model_location


class VLM:
    """Generate image-conditioned text with Transformers or a supplied function."""

    def __init__(
        self,
        model: Any,
        generate_fn: Optional[Callable[[Any, str], str]] = None,
        *,
        device_map: Any = "auto",
        dtype: Any = "auto",
        max_new_tokens: int = 512,
        local_files_only: bool = False,
    ) -> None:
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("A Hugging Face model ID or local path is required.")
        self.generate_fn = (
            generate_fn
            if generate_fn is not None
            else _HuggingFaceVisionGenerator(
                model=self.model,
                device_map=device_map,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                local_files_only=local_files_only,
            )
        )

    def generate(self, image: Any, prompt: str) -> str:
        if not callable(self.generate_fn):
            raise RuntimeError(
                f"No generate_fn callable was supplied for VLM model {self.model!r}."
            )
        output = self.generate_fn(image, prompt)
        if not isinstance(output, str):
            raise TypeError("VLM generate_fn must return a string.")
        return output


class _HuggingFaceVisionGenerator:
    """Lazy local inference adapter for multimodal Transformers chat models."""

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
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "repetition_penalty": 1.1,
        }
        self.processor = None
        self.model = None
        self.torch = None

    def __call__(self, image: Any, prompt: str) -> str:
        self._load()
        processor = self.processor
        model = self.model
        torch = self.torch

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.is_file():
                raise FileNotFoundError("Image file does not exist: %s" % image_path)
            image_content = {"type": "image", "path": str(image_path.resolve())}
        else:
            image_content = {"type": "image", "image": image}

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _move_inputs(inputs, model)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(**inputs, **self.generation_kwargs)
        sequences = getattr(generated, "sequences", generated)
        sequence = sequences[0]
        if not getattr(getattr(model, "config", None), "is_encoder_decoder", False):
            sequence = sequence[input_length:]
        return processor.decode(sequence, skip_special_tokens=True).strip()

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend requires PyTorch, Transformers, and "
                "Pillow. Activate an environment with PyTorch, then install "
                "requirements-huggingface.txt there."
            ) from exc

        model_location = _resolve_model_location(
            self.model_name, self.local_files_only
        )
        load_kwargs = {}
        if self.local_files_only:
            load_kwargs["local_files_only"] = True
        self.processor = AutoProcessor.from_pretrained(
            model_location, **load_kwargs
        )
        model_kwargs = {}
        model_kwargs.update(load_kwargs)
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        if self.dtype is not None:
            model_kwargs["dtype"] = self.dtype
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_location, **model_kwargs
        )
        self.model.eval()
        self.torch = torch
