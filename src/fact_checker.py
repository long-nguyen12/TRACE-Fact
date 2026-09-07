"""Final evidence-grounded fact-checking decision."""

import math
from typing import Any, Dict, List, Set, Tuple

from .consistency import (
    _as_items,
    _collect_image_evidence,
    _collect_text_facts,
    _container_id,
    _filter_evidence_ids,
    _normalize_consistency,
    _normalize_status,
    _unknown,
)
from .model_output import generate_json, json_text
from .prompt import FACT_CHECK_SYSTEM, FACT_CHECK_USER, render_prompt


_LABELS = {
    "support": "support",
    "supported": "support",
    "refute": "refute",
    "refuted": "refute",
    "nei": "not_enough_information",
    "not_enough_information": "not_enough_information",
}


def _normalize_label(value: Any, errors: List[str]) -> str:
    if isinstance(value, str):
        label = _LABELS.get(value.strip().lower().replace("-", "_").replace(" ", "_"))
        if label:
            return label
    errors.append("Prediction label must be support, refute, or not_enough_information.")
    return "not_enough_information"


def _normalize_confidence(value: Any, errors: List[str]) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append("Prediction confidence must be a number from 0 to 1.")
        return None
    try:
        confidence = float(value)
    except (OverflowError, ValueError):
        confidence = math.nan
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        errors.append("Prediction confidence must be between 0 and 1.")
        return None
    return confidence


def _normalize_reasoning(
    value: Any,
    allowed_ids: Set[str],
    errors: List[str],
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        errors.append("Model output field 'reasoning' must be an object.")
        return _unknown("No valid reasoning was returned.")

    status = _normalize_status(value.get("status"), "Reasoning", errors)
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("Reasoning reason must be a non-empty string.")
        reason = "No valid reason was returned."
    evidence_ids = _filter_evidence_ids(
        value.get("evidence_ids"), allowed_ids, "Reasoning", errors
    )
    if status in {"support", "contradict"} and not evidence_ids:
        errors.append(f"Reasoning {status} status has no valid evidence.")
        return _unknown("The returned status had no valid grounding evidence.")
    return {
        "status": status,
        "reason": reason.strip(),
        "evidence_ids": evidence_ids,
    }


def _ground_label(
    label: str, reasoning: Dict[str, Any], errors: List[str]
) -> str:
    status = reasoning.get("status")
    if label == "support" and status != "support":
        errors.append(
            "Support verdict requires grounded supporting evidence; changed verdict "
            "to not_enough_information."
        )
        return "not_enough_information"
    if label == "refute" and status != "contradict":
        errors.append(
            "Refute verdict requires grounded contradictory evidence; changed verdict "
            "to not_enough_information."
        )
        return "not_enough_information"
    return label


def _collect_provenance(
    value: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    facts: List[Dict[str, Any]] = []
    sources: List[Dict[str, str]] = []
    used_ids: Set[str] = set()
    for index, item in enumerate(_as_items(value), start=1):
        if not isinstance(item, dict):
            continue
        container_id = _container_id(item, "P", index)
        source_ids = set()
        for source_index, source in enumerate(_as_items(item.get("sources")), start=1):
            if not isinstance(source, dict):
                continue
            source_id = str(source.get("id", "")).strip()
            prefix = f"{container_id}.S"
            if not source_id.startswith(prefix) or not source_id[len(prefix) :].isdigit():
                source_id = f"{prefix}{source_index}"
            if source_id in source_ids:
                continue
            source_ids.add(source_id)
            sources.append(
                {
                    "id": source_id,
                    "url": str(source.get("url", "")),
                    "title": str(source.get("title", "")),
                    "date": str(source.get("date", "")),
                    "caption": str(source.get("caption", "")),
                }
            )

        for fact in _as_items(item.get("facts")):
            if not isinstance(fact, dict) or not str(fact.get("text", "")).strip():
                continue
            fact_id = str(fact.get("id", "")).strip()
            prefix = f"{container_id}.F"
            suffix = len(facts) + 1
            if not fact_id.startswith(prefix) or not fact_id[len(prefix) :].isdigit():
                fact_id = f"{prefix}{suffix}"
            while fact_id in used_ids:
                suffix += 1
                fact_id = f"{prefix}{suffix}"
            valid_sources = [
                source_id
                for source_id in _as_items(fact.get("source_ids"))
                if source_id in source_ids
            ]
            if not valid_sources:
                continue
            used_ids.add(fact_id)
            facts.append(
                {
                    "id": fact_id,
                    "text": str(fact["text"]).strip(),
                    "source_ids": list(dict.fromkeys(valid_sources)),
                }
            )
    return facts, sources


def _analysis_input(
    evidence: Dict[str, Any],
) -> Tuple[Dict[str, Any], Set[str], List[str]]:
    errors: List[str] = []
    facts = _collect_text_facts(evidence.get("text_analysis", []))
    observations, inferences = _collect_image_evidence(
        evidence.get("image_analysis", [])
    )
    provenance_facts, provenance_sources = _collect_provenance(
        evidence.get("provenance", [])
    )
    image_ids = {item["id"] for item in observations + inferences}
    text_ids = {item["id"] for item in facts}
    consistency = evidence.get("consistency")
    if consistency:
        consistency = _normalize_consistency(
            consistency, image_ids, text_ids, errors
        )
    else:
        consistency = {}
    model_input = {
        "claim": evidence["claim"],
        "image_observations": observations,
        "image_inferences": inferences,
        "text_facts": facts,
        "provenance_facts": provenance_facts,
        "provenance_sources": provenance_sources,
        "consistency": consistency,
    }
    allowed_ids = image_ids | text_ids | {
        item["id"] for item in provenance_facts
    }
    return model_input, allowed_ids, errors


class FactChecker:
    """Choose one verdict using only sanitized structured evidence."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def verify(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "label": "not_enough_information",
            "confidence": 0.0,
            "reasoning": _unknown("No valid reasoning was returned."),
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]
        model_input, allowed_ids, input_errors = _analysis_input(evidence)
        errors.extend(input_errors)
        prompt = render_prompt(
            FACT_CHECK_SYSTEM,
            FACT_CHECK_USER,
            CLAIM_TEXT=model_input["claim"],
            IMAGE_EVIDENCE_JSON=json_text(
                {
                    "observations": model_input["image_observations"],
                    "inferences": model_input["image_inferences"],
                }
            ),
            TEXT_EVIDENCE_JSON=json_text(model_input["text_facts"]),
            PROVENANCE_FACTS_JSON=json_text(
                {
                    "facts": model_input["provenance_facts"],
                    "sources": model_input["provenance_sources"],
                }
            ),
            CONSISTENCY_JSON=json_text(model_input["consistency"]),
        )
        parsed = generate_json(result, self.llm.generate, prompt)
        if parsed is None:
            return result

        before_label = len(errors)
        label = _normalize_label(parsed.get("label"), errors)
        label_is_valid = len(errors) == before_label
        reasoning = _normalize_reasoning(
            parsed.get("reasoning"), allowed_ids, errors
        )
        result["label"] = _ground_label(label, reasoning, errors)
        result["confidence"] = _normalize_confidence(
            parsed.get("confidence"), errors
        )
        result["reasoning"] = reasoning
        if not label_is_valid or result["label"] != label:
            result["confidence"] = None
        return result


__all__ = ["FactChecker"]
