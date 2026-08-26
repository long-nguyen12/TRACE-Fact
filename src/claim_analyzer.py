"""Claim decomposition into a compact, inspectable representation."""

from pathlib import Path
from typing import Any, Dict, List

from .llm import parse_json_output


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "claim_analysis.txt"


class ClaimAnalyzer:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def analyze(self, claim: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "claim": claim,
            "atoms": [],
            "entities": [],
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]

        if not isinstance(claim, str):
            errors.append("Claim must be a string.")
            return result

        prompt = f"{self.prompt.rstrip()}\n\nCLAIM TO EXTRACT:\n{claim}"
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

        atoms = parsed.get("atoms")
        if not isinstance(atoms, list):
            errors.append("Model output field 'atoms' must be a list.")
        else:
            for index, atom in enumerate(atoms):
                if not isinstance(atom, dict):
                    errors.append(f"Model output atom at index {index} must be an object.")
                    continue
                text = atom.get("text")
                if not isinstance(text, str) or not text.strip():
                    errors.append(
                        f"Model output atom at index {index} must contain non-empty text."
                    )
                    continue
                result["atoms"].append(
                    {"id": f"C{len(result['atoms']) + 1}", "text": text.strip()}
                )

        result["entities"] = self._normalize_strings(
            parsed.get("entities"), "entities", errors
        )
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
