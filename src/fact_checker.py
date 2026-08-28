"""Final evidence-grounded fact-checking decision."""

import json
import math
from typing import Any, Dict, List, Sequence, Set, Tuple

from .consistency import (
    _as_items,
    _collect_image_evidence,
    _collect_text_facts,
    _container_id,
    _filter_evidence_ids,
    _normalize_claim_atoms,
    _normalize_comparisons,
    _normalize_status,
)
from .llm import parse_json_output
from .prompt import FACT_CHECK_SYSTEM, FACT_CHECK_USER, render_prompt


_LABELS = {"support", "refute", "not_enough_information"}
_LABEL_ALIASES = {
    "support": "support",
    "supported": "support",
    "refute": "refute",
    "refuted": "refute",
    "nei": "not_enough_information",
    "not_enough_information": "not_enough_information",
}


def _first_present(mapping: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return []


def _normalize_label(value: Any, errors: List[str]) -> str:
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        label = _LABEL_ALIASES.get(key)
        if label in _LABELS:
            return label
    errors.append(
        "Prediction label must be one of: support, refute, "
        "not_enough_information (or the specified supported/refuted/NEI aliases)."
    )
    return "not_enough_information"


def _normalize_confidence(value: Any, errors: List[str], fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append("Prediction confidence must be a number from 0 to 1 or null.")
        return fallback
    try:
        confidence = float(value)
    except (OverflowError, ValueError):
        errors.append("Prediction confidence must be a finite number from 0 to 1.")
        return fallback
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        errors.append("Prediction confidence must be between 0 and 1.")
        return fallback
    return confidence


def _ground_label(
    label: str, reasoning: Sequence[Dict[str, Any]], errors: List[str]
) -> str:
    statuses = [item.get("status") for item in reasoning]
    if label == "support" and (not statuses or any(status != "support" for status in statuses)):
        errors.append(
            "Support verdict requires grounded support reasoning for every claim atom; "
            "changed verdict to not_enough_information."
        )
        return "not_enough_information"
    if label == "refute" and "contradict" not in statuses:
        errors.append(
            "Refute verdict requires at least one grounded contradiction; changed "
            "verdict to not_enough_information."
        )
        return "not_enough_information"
    return label


def _analysis_input(evidence: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]], Set[str], List[str]]:
    errors: List[str] = []
    claim = evidence.get("claim") if isinstance(evidence.get("claim"), str) else ""
    claim_analysis = evidence.get("claim_analysis")
    if not isinstance(claim_analysis, dict):
        claim_analysis = {"claim": claim, "atoms": []}
    atoms = _normalize_claim_atoms(claim_analysis, claim)

    text_analysis = _first_present(evidence, ("text_analysis", "texts"))
    image_analysis = _first_present(evidence, ("image_analysis", "images"))
    facts = _collect_text_facts(text_analysis)
    observations, inferences = _collect_image_evidence(image_analysis)
    text_ids = {item["id"] for item in facts}
    image_ids = {item["id"] for item in observations + inferences}
    provenance_facts, provenance_sources = _collect_provenance(
        evidence.get("provenance", [])
    )
    provenance_ids = {item["id"] for item in provenance_facts}

    consistency_value = evidence.get("consistency")
    # ``[]`` is the pipeline's explicit representation for a disabled
    # consistency stage, not a malformed comparison response.
    if consistency_value is None or consistency_value == []:
        comparisons: List[Dict[str, Any]] = []
    else:
        comparisons = _normalize_comparisons(
            consistency_value,
            atoms,
            image_ids,
            text_ids,
            errors,
            "No valid consistency result was supplied.",
        )

    model_input = {
        "claim": claim or (claim_analysis.get("claim") or ""),
        "claim_components": atoms,
        "image_observations": observations,
        "image_inferences": inferences,
        "text_facts": facts,
        "provenance_facts": provenance_facts,
        "provenance_sources": provenance_sources,
        "consistency": comparisons,
    }
    return (
        model_input,
        atoms,
        image_ids.union(text_ids).union(provenance_ids),
        errors,
    )


def _collect_provenance(value: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
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
            expected = "%s.S" % container_id
            if not source_id.startswith(expected) or not source_id[len(expected) :].isdigit():
                source_id = "%s.S%d" % (container_id, source_index)
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
            if not isinstance(fact, dict):
                continue
            text = str(fact.get("text", "")).strip()
            if not text:
                continue
            fact_id = str(fact.get("id", "")).strip()
            expected = "%s.F" % container_id
            if (
                not fact_id.startswith(expected)
                or not fact_id[len(expected) :].isdigit()
                or fact_id in used_ids
            ):
                fact_id = "%s.F1" % container_id
            suffix = 1
            while fact_id in used_ids:
                suffix += 1
                fact_id = "%s.F%d" % (container_id, suffix)
            valid_source_ids = []
            for source_id in _as_items(fact.get("source_ids")):
                source_id = str(source_id).strip()
                if source_id in source_ids and source_id not in valid_source_ids:
                    valid_source_ids.append(source_id)
            if not valid_source_ids:
                continue
            used_ids.add(fact_id)
            facts.append(
                {
                    "id": fact_id,
                    "text": text,
                    "source_ids": valid_source_ids,
                }
            )
    return facts, sources


def _normalize_reasoning(
    value: Any,
    atoms: Sequence[Dict[str, str]],
    allowed_ids: Set[str],
    errors: List[str],
) -> List[Dict[str, Any]]:
    by_atom: Dict[str, Dict[str, Any]] = {}
    valid_atom_ids = {atom["id"] for atom in atoms}
    if not isinstance(value, list):
        errors.append("Model output field 'reasoning' must be a list.")
        value = []

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"Reasoning item at index {index} must be an object.")
            continue
        atom_id = item.get("claim_atom_id")
        if not isinstance(atom_id, str) or atom_id not in valid_atom_ids:
            errors.append(f"Reasoning item at index {index} has an unknown claim_atom_id.")
            continue
        if atom_id in by_atom:
            errors.append(f"Duplicate reasoning returned for claim atom {atom_id}.")
            continue
        by_atom[atom_id] = item

    reasoning: List[Dict[str, Any]] = []
    for atom in atoms:
        item = by_atom.get(atom["id"])
        if item is None:
            errors.append(f"No reasoning was returned for claim atom {atom['id']}.")
            reasoning.append(
                {
                    "claim_atom_id": atom["id"],
                    "status": "unknown",
                    "reason": "No valid reasoning was returned.",
                    "evidence_ids": [],
                }
            )
            continue

        status = _normalize_status(
            item.get("status"), f"Reasoning {atom['id']}", errors
        )
        reason_value = item.get("reason")
        if isinstance(reason_value, str) and reason_value.strip():
            reason = reason_value.strip()
        else:
            errors.append(f"Reasoning {atom['id']} reason must be a non-empty string.")
            reason = "No valid reason was returned."
        evidence_ids = _filter_evidence_ids(
            item.get("evidence_ids"),
            allowed_ids,
            f"Reasoning {atom['id']}",
            errors,
        )
        if status in {"support", "contradict"} and not evidence_ids:
            errors.append(
                f"Reasoning {atom['id']} {status} status has no valid grounding evidence."
            )
            status = "unknown"
            reason = "The returned status had no valid grounding evidence."

        reasoning.append(
            {
                "claim_atom_id": atom["id"],
                "status": status,
                "reason": reason,
                "evidence_ids": evidence_ids,
            }
        )
    return reasoning


class FactChecker:
    """Choose one normalized verdict using only sanitized structured evidence."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def verify(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "label": "not_enough_information",
            "confidence": 0.0,
            "reasoning": [],
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]
        if not isinstance(evidence, dict):
            errors.append("Evidence must be a dictionary.")
            return result

        model_input, atoms, allowed_ids, input_errors = _analysis_input(evidence)
        errors.extend(input_errors)
        if not atoms:
            errors.append("Evidence did not contain a claim or any valid claim atoms.")
            return result

        image_evidence = {
            "observations": model_input["image_observations"],
            "inferences": model_input["image_inferences"],
        }
        provenance_evidence = {
            "facts": model_input["provenance_facts"],
            "sources": model_input["provenance_sources"],
        }
        prompt = render_prompt(
            FACT_CHECK_SYSTEM,
            FACT_CHECK_USER,
            CLAIM_COMPONENTS_JSON=json.dumps(
                model_input["claim_components"], ensure_ascii=False, indent=2
            ),
            IMAGE_EVIDENCE_JSON=json.dumps(
                image_evidence, ensure_ascii=False, indent=2
            ),
            TEXT_EVIDENCE_JSON=json.dumps(
                model_input["text_facts"], ensure_ascii=False, indent=2
            ),
            PROVENANCE_FACTS_JSON=json.dumps(
                provenance_evidence, ensure_ascii=False, indent=2
            ),
            CONSISTENCY_COMPARISONS_JSON=json.dumps(
                model_input["consistency"], ensure_ascii=False, indent=2
            ),
        )
        try:
            raw_output = self.llm.generate(prompt)
            result["raw_output"] = raw_output
        except Exception as exc:
            errors.append(f"Model generation failed: {type(exc).__name__}: {exc}")
            result["reasoning"] = _normalize_reasoning([], atoms, allowed_ids, errors)
            return result

        try:
            parsed = parse_json_output(raw_output)
        except (TypeError, ValueError) as exc:
            errors.append(
                f"Model output could not be parsed: {type(exc).__name__}: {exc}"
            )
            result["reasoning"] = _normalize_reasoning([], atoms, allowed_ids, errors)
            return result

        label_error_count = len(errors)
        normalized_label = _normalize_label(parsed.get("label"), errors)
        label_was_valid = len(errors) == label_error_count
        result["label"] = normalized_label
        result["confidence"] = _normalize_confidence(
            parsed.get("confidence"), errors
        )
        result["reasoning"] = _normalize_reasoning(
            parsed.get("reasoning"), atoms, allowed_ids, errors
        )
        result["label"] = _ground_label(
            normalized_label, result["reasoning"], errors
        )
        if not label_was_valid or result["label"] != normalized_label:
            result["confidence"] = None
        return result


__all__ = ["FactChecker"]
