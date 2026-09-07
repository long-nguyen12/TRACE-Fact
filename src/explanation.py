"""Generation of concise explanations grounded in supplied evidence identifiers."""

from typing import Any, Dict, List, Set

from .fact_checker import (
    _analysis_input,
    _ground_label,
    _normalize_label,
    _normalize_reasoning,
)
from .model_output import generate_json, json_text
from .prompt import EXPLANATION_SYSTEM, EXPLANATION_USER, render_prompt


def _normalize_citations(
    value: Any, allowed_ids: Set[str], errors: List[str]
) -> List[str]:
    if not isinstance(value, list):
        errors.append("Model output field 'citations' must be a list.")
        return []

    citations: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            errors.append("Explanation contains a non-string citation identifier.")
            continue
        evidence_id = item.strip()
        if evidence_id not in allowed_ids:
            errors.append(
                f"Explanation references unknown evidence id {evidence_id!r}."
            )
            continue
        if evidence_id not in citations:
            citations.append(evidence_id)
    return citations


class ExplanationGenerator:
    """Explain an existing verdict without exposing evaluation-only fields."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def generate(
        self, evidence: Dict[str, Any], prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "explanation": "",
            "citations": [],
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]
        if not isinstance(evidence, dict):
            errors.append("Evidence must be a dictionary.")
            return result
        if not isinstance(prediction, dict):
            errors.append("Prediction must be a dictionary.")
            return result

        model_input, atoms, allowed_ids, input_errors = _analysis_input(evidence)
        errors.extend(input_errors)
        if not atoms:
            errors.append("Evidence did not contain a claim or any valid claim atoms.")
            return result

        prediction_label = _normalize_label(prediction.get("label"), errors)
        prediction_reasoning = _normalize_reasoning(
            prediction.get("reasoning"), atoms, allowed_ids, errors
        )
        normalized_prediction = {
            "label": _ground_label(
                prediction_label, prediction_reasoning, errors
            ),
            "reasoning": prediction_reasoning,
        }
        model_input["prediction"] = normalized_prediction
        image_evidence = {
            "observations": model_input["image_observations"],
            "inferences": model_input["image_inferences"],
        }
        provenance_evidence = {
            "facts": model_input["provenance_facts"],
            "sources": model_input["provenance_sources"],
        }
        prompt = render_prompt(
            EXPLANATION_SYSTEM,
            EXPLANATION_USER,
            PREDICTION_JSON=json_text(normalized_prediction),
            CLAIM_COMPONENTS_JSON=json_text(model_input["claim_components"]),
            IMAGE_EVIDENCE_JSON=json_text(image_evidence),
            TEXT_EVIDENCE_JSON=json_text(model_input["text_facts"]),
            PROVENANCE_FACTS_JSON=json_text(provenance_evidence),
            CONSISTENCY_COMPARISONS_JSON=json_text(model_input["consistency"]),
        )
        parsed = generate_json(result, self.llm.generate, prompt)
        if parsed is None:
            return result

        explanation = parsed.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            result["explanation"] = explanation.strip()
        else:
            errors.append("Model output field 'explanation' must be a non-empty string.")
        result["citations"] = _normalize_citations(
            parsed.get("citations"), allowed_ids, errors
        )
        if allowed_ids and result["explanation"] and not result["citations"]:
            errors.append(
                "Explanation had no valid grounding citations; discarded explanation."
            )
            result["explanation"] = ""
        return result


__all__ = ["ExplanationGenerator"]
