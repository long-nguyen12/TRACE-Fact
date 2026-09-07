"""Extraction of explicit factual statements from text evidence."""

from typing import Any, Dict, List

from .model_output import generate_json, string_list
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
        parsed = generate_json(result, self.llm.generate, prompt)
        if parsed is None:
            return result

        for field in ("facts", "entities"):
            result[field] = string_list(parsed.get(field), field, errors)
        return result
