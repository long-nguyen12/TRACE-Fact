"""Extraction of visual observations and separate semantic inferences."""

from typing import Any, Dict, List

from .model_output import generate_json, string_list
from .prompt import IMAGE_ANALYSIS_SYSTEM, IMAGE_ANALYSIS_USER, render_prompt


class ImageAnalyzer:
    def __init__(self, vlm: Any) -> None:
        self.vlm = vlm

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

        prompt = render_prompt(IMAGE_ANALYSIS_SYSTEM, IMAGE_ANALYSIS_USER)
        parsed = generate_json(result, self.vlm.generate, image, prompt)
        if parsed is None:
            return result

        description = parsed.get("description")
        if isinstance(description, str):
            result["description"] = description.strip()
        else:
            errors.append("Model output field 'description' must be a string.")

        for field in ("objects", "text", "observations", "inferences"):
            result[field] = string_list(parsed.get(field), field, errors)

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
