"""Extraction of visual observations and separate semantic inferences."""

from pathlib import Path
from typing import Any, Dict, List

from .llm import parse_json_output


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "image_analysis.txt"


class ImageAnalyzer:
    def __init__(self, vlm: Any) -> None:
        self.vlm = vlm
        self.prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def analyze(self, image: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "description": "",
            "objects": [],
            "text": [],
            "observations": [],
            "inferences": [],
            "relations": [],
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]

        try:
            raw_output = self.vlm.generate(image, self.prompt)
            result["raw_output"] = raw_output
        except Exception as exc:
            errors.append(f"Model generation failed: {type(exc).__name__}: {exc}")
            return result

        try:
            parsed = parse_json_output(raw_output)
        except (TypeError, ValueError) as exc:
            errors.append(f"Model output could not be parsed: {type(exc).__name__}: {exc}")
            return result

        description = parsed.get("description")
        if isinstance(description, str):
            result["description"] = description.strip()
        else:
            errors.append("Model output field 'description' must be a string.")

        for field in ("objects", "text", "observations", "inferences"):
            result[field] = self._normalize_strings(parsed.get(field), field, errors)

        relations = parsed.get("relations", [])
        if not isinstance(relations, list):
            errors.append("Model output field 'relations' must be a list.")
        else:
            for index, relation in enumerate(relations):
                if (
                    not isinstance(relation, (list, tuple))
                    or len(relation) != 3
                    or any(not isinstance(part, str) or not part.strip() for part in relation)
                ):
                    errors.append(
                        f"Model output relation at index {index} must contain three non-empty strings."
                    )
                    continue
                result["relations"].append([part.strip() for part in relation])

        return result

    @staticmethod
    def _normalize_strings(value: Any, field: str, errors: List[str]) -> List[str]:
        if not isinstance(value, list):
            errors.append(f"Model output field '{field}' must be a list.")
            return []

        normalized: List[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(
                    f"Model output field '{field}' at index {index} must be a non-empty string."
                )
                continue
            item = item.strip()
            if item not in normalized:
                normalized.append(item)
        return normalized
