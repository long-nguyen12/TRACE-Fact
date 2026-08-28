"""Extraction of explicit factual statements from text evidence."""

from typing import Any, Dict, List

from .llm import parse_json_output
from .prompt import TEXT_ANALYSIS_SYSTEM, TEXT_ANALYSIS_USER, render_prompt


class TextAnalyzer:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def analyze(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "facts": [],
            "entities": [],
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]

        if not isinstance(text, str):
            errors.append("Text evidence must be a string.")
            return result

        prompt = render_prompt(
            TEXT_ANALYSIS_SYSTEM,
            TEXT_ANALYSIS_USER,
            EVIDENCE_TEXT=text,
        )
        try:
            raw_output = self.llm.generate(prompt)
            result["raw_output"] = raw_output
        except Exception as exc:
            errors.append(f"Model generation failed: {type(exc).__name__}: {exc}")
            return result

        try:
            parsed = parse_json_output(raw_output)
        except (TypeError, ValueError) as exc:
            errors.append(f"Model output could not be parsed: {type(exc).__name__}: {exc}")
            return result

        for field in ("facts", "entities"):
            result[field] = self._normalize_strings(parsed.get(field), field, errors)
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
