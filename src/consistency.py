"""Claim-to-evidence consistency analysis with grounded evidence references."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .llm import parse_json_output


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "consistency.txt"
_STATUSES = {"support", "contradict", "unknown"}


def _as_items(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return []


def _payload(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    analysis = item.get("analysis")
    if isinstance(analysis, dict):
        return analysis
    return item


def _container_id(item: Any, prefix: str, index: int) -> str:
    if isinstance(item, dict):
        for key in ("evidence_id", "id"):
            value = item.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate.startswith(prefix) and candidate[len(prefix) :].isdigit():
                    return candidate
    return f"{prefix}{index}"


def _normalize_claim_atoms(claim_analysis: Any, claim: str = "") -> List[Dict[str, str]]:
    analysis = claim_analysis if isinstance(claim_analysis, dict) else {}
    atoms: List[Dict[str, str]] = []
    for item in _as_items(analysis.get("atoms")):
        text: Optional[str] = None
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text = item["text"]
        if isinstance(text, str) and text.strip():
            atoms.append({"id": f"C{len(atoms) + 1}", "text": text.strip()})

    fallback = claim
    if not fallback and isinstance(analysis.get("claim"), str):
        fallback = analysis["claim"]
    if not atoms and isinstance(fallback, str) and fallback.strip():
        atoms.append({"id": "C1", "text": fallback.strip()})
    return atoms


def _normalize_leaf_items(
    value: Any,
    container_id: str,
    marker: str,
    used_ids: Set[str],
) -> List[Dict[str, str]]:
    leaves: List[Dict[str, str]] = []
    for item in _as_items(value):
        text: Optional[str] = None
        candidate_id: Optional[str] = None
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            if isinstance(item.get("text"), str):
                text = item["text"]
            if isinstance(item.get("id"), str) and item["id"].strip():
                candidate_id = item["id"].strip()

        if not isinstance(text, str) or not text.strip():
            continue

        fallback_id = f"{container_id}.{marker}{len(leaves) + 1}"
        expected_prefix = f"{container_id}.{marker}"
        valid_candidate = bool(
            candidate_id
            and candidate_id.startswith(expected_prefix)
            and candidate_id[len(expected_prefix) :].isdigit()
        )
        leaf_id = (
            candidate_id
            if valid_candidate and candidate_id not in used_ids
            else fallback_id
        )
        suffix = len(leaves) + 1
        while leaf_id in used_ids:
            suffix += 1
            leaf_id = f"{container_id}.{marker}{suffix}"
        used_ids.add(leaf_id)
        leaves.append({"id": leaf_id, "text": text.strip()})
    return leaves


def _collect_text_facts(text_analysis: Any) -> List[Dict[str, str]]:
    facts: List[Dict[str, str]] = []
    used_ids: Set[str] = set()
    for index, item in enumerate(_as_items(text_analysis), start=1):
        container_id = _container_id(item, "T", index)
        facts.extend(
            _normalize_leaf_items(
                _payload(item).get("facts"), container_id, "F", used_ids
            )
        )
    return facts


def _collect_image_evidence(
    image_analysis: Any,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    observations: List[Dict[str, str]] = []
    inferences: List[Dict[str, str]] = []
    used_ids: Set[str] = set()
    for index, item in enumerate(_as_items(image_analysis), start=1):
        container_id = _container_id(item, "I", index)
        analysis = _payload(item)
        observations.extend(
            _normalize_leaf_items(
                analysis.get("observations"), container_id, "O", used_ids
            )
        )
        # VLM-transcribed image text is directly observed visual evidence too.
        observations.extend(
            _normalize_leaf_items(
                analysis.get("text"), container_id, "TXT", used_ids
            )
        )
        inferences.extend(
            _normalize_leaf_items(
                analysis.get("inferences"), container_id, "INF", used_ids
            )
        )
    return observations, inferences


def _normalize_status(value: Any, context: str, errors: List[str]) -> str:
    if isinstance(value, str):
        status = value.strip().lower()
        if status in _STATUSES:
            return status
    errors.append(
        f"{context} status must be one of: support, contradict, unknown."
    )
    return "unknown"


def _filter_evidence_ids(
    value: Any,
    allowed_ids: Set[str],
    context: str,
    errors: List[str],
) -> List[str]:
    if not isinstance(value, list):
        errors.append(f"{context} evidence_ids must be a list.")
        return []

    result: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{context} contains a non-string evidence identifier.")
            continue
        evidence_id = item.strip()
        if evidence_id not in allowed_ids:
            errors.append(f"{context} references unknown evidence id {evidence_id!r}.")
            continue
        if evidence_id not in result:
            result.append(evidence_id)
    return result


def _unknown_modality(reason: str) -> Dict[str, Any]:
    return {"status": "unknown", "reason": reason, "evidence_ids": []}


def _normalize_modality(
    value: Any,
    allowed_ids: Set[str],
    context: str,
    no_evidence_reason: str,
    errors: List[str],
) -> Dict[str, Any]:
    if not allowed_ids:
        return _unknown_modality(no_evidence_reason)
    if not isinstance(value, dict):
        errors.append(f"{context} result must be an object.")
        return _unknown_modality("No valid consistency result was returned.")

    status = _normalize_status(value.get("status"), context, errors)
    reason_value = value.get("reason")
    if isinstance(reason_value, str) and reason_value.strip():
        reason = reason_value.strip()
    else:
        errors.append(f"{context} reason must be a non-empty string.")
        reason = "No valid reason was returned."
    evidence_ids = _filter_evidence_ids(
        value.get("evidence_ids"), allowed_ids, context, errors
    )

    if status in {"support", "contradict"} and not evidence_ids:
        errors.append(f"{context} {status} status has no valid grounding evidence.")
        status = "unknown"
        reason = "The returned status had no valid grounding evidence."

    return {"status": status, "reason": reason, "evidence_ids": evidence_ids}


def _comparison_items(value: Any) -> List[Any]:
    if isinstance(value, dict):
        return _as_items(value.get("comparisons"))
    return _as_items(value)


def _normalize_comparisons(
    value: Any,
    atoms: Sequence[Dict[str, str]],
    image_ids: Set[str],
    text_ids: Set[str],
    errors: List[str],
    missing_reason: str,
) -> List[Dict[str, Any]]:
    by_atom: Dict[str, Dict[str, Any]] = {}
    valid_atom_ids = {atom["id"] for atom in atoms}
    for index, item in enumerate(_comparison_items(value)):
        if not isinstance(item, dict):
            errors.append(f"Comparison at index {index} must be an object.")
            continue
        atom_id = item.get("claim_atom_id")
        if not isinstance(atom_id, str) or atom_id not in valid_atom_ids:
            errors.append(f"Comparison at index {index} has an unknown claim_atom_id.")
            continue
        if atom_id in by_atom:
            errors.append(f"Duplicate comparison returned for claim atom {atom_id}.")
            continue
        by_atom[atom_id] = item

    records: List[Dict[str, Any]] = []
    for atom in atoms:
        item = by_atom.get(atom["id"])
        if item is None:
            errors.append(f"No comparison was returned for claim atom {atom['id']}.")
            image = _unknown_modality(
                "No image evidence was supplied."
                if not image_ids
                else missing_reason
            )
            text = _unknown_modality(
                "No text evidence was supplied." if not text_ids else missing_reason
            )
        else:
            image = _normalize_modality(
                item.get("image"),
                image_ids,
                f"Comparison {atom['id']} image",
                "No image evidence was supplied.",
                errors,
            )
            text = _normalize_modality(
                item.get("text"),
                text_ids,
                f"Comparison {atom['id']} text",
                "No text evidence was supplied.",
                errors,
            )
        records.append(
            {
                "claim_atom_id": atom["id"],
                "claim_atom": atom["text"],
                "image": image,
                "text": text,
            }
        )
    return records


class ConsistencyChecker:
    """Compare each claim atom with extracted image and text evidence."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def compare(
        self,
        claim_analysis: Dict[str, Any],
        image_analysis: Any,
        text_analysis: Any,
    ) -> Dict[str, Any]:
        atoms = _normalize_claim_atoms(claim_analysis)
        observations, inferences = _collect_image_evidence(image_analysis)
        facts = _collect_text_facts(text_analysis)
        image_ids = {item["id"] for item in observations + inferences}
        text_ids = {item["id"] for item in facts}

        result: Dict[str, Any] = {
            "comparisons": [],
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]
        if not atoms:
            errors.append("Claim analysis did not contain a claim or any valid atoms.")
            return result

        model_input = {
            "claim_components": atoms,
            "image_observations": observations,
            "image_inferences": inferences,
            "text_facts": facts,
        }
        prompt = (
            f"{self.prompt.rstrip()}\n\nINPUT:\n"
            f"{json.dumps(model_input, ensure_ascii=False, indent=2)}"
        )
        try:
            raw_output = self.llm.generate(prompt)
            result["raw_output"] = raw_output
        except Exception as exc:
            errors.append(f"Model generation failed: {type(exc).__name__}: {exc}")
            result["comparisons"] = _normalize_comparisons(
                [],
                atoms,
                image_ids,
                text_ids,
                errors,
                "Consistency analysis was unavailable.",
            )
            return result

        try:
            parsed = parse_json_output(raw_output)
        except (TypeError, ValueError) as exc:
            errors.append(
                f"Model output could not be parsed: {type(exc).__name__}: {exc}"
            )
            parsed = {}

        comparisons = parsed.get("comparisons")
        if not isinstance(comparisons, list):
            errors.append("Model output field 'comparisons' must be a list.")
            comparisons = []
        result["comparisons"] = _normalize_comparisons(
            comparisons,
            atoms,
            image_ids,
            text_ids,
            errors,
            "No valid consistency result was returned.",
        )
        return result


__all__ = ["ConsistencyChecker"]
