"""Claim decomposition into a compact, inspectable representation."""

from typing import Any, Dict, List

from .model_output import generate_json, string_list
from .prompt import CLAIM_ANALYSIS_SYSTEM, CLAIM_ANALYSIS_USER, render_prompt


class ClaimAnalyzer:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

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

        prompt = render_prompt(
            CLAIM_ANALYSIS_SYSTEM,
            CLAIM_ANALYSIS_USER,
            CLAIM_TEXT=claim,
        )
        parsed = generate_json(result, self.llm.generate, prompt)
        if parsed is None:
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

        result["entities"] = string_list(parsed.get("entities"), "entities", errors)
        return result
