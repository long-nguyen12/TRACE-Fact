"""Compare one complete claim with extracted image and text evidence."""

from typing import Any, Dict, List, Optional, Set, Tuple

from .model_output import generate_json, json_text
from .prompt import CONSISTENCY_SYSTEM, CONSISTENCY_USER, render_prompt


_STATUSES = {"support", "contradict", "unknown"}


def _as_items(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _container_id(item: Any, prefix: str, index: int) -> str:
    if isinstance(item, dict):
        candidate = str(item.get("evidence_id", "")).strip()
        if candidate.startswith(prefix) and candidate[len(prefix) :].isdigit():
            return candidate
    return f"{prefix}{index}"


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
            text = item.get("text")
            candidate_id = item.get("id")

        if not isinstance(text, str) or not text.strip():
            continue
        expected = f"{container_id}.{marker}"
        valid_id = bool(
            isinstance(candidate_id, str)
            and candidate_id.startswith(expected)
            and candidate_id[len(expected) :].isdigit()
            and candidate_id not in used_ids
        )
        suffix = len(leaves) + 1
        leaf_id = candidate_id if valid_id else f"{expected}{suffix}"
        while leaf_id in used_ids:
            suffix += 1
            leaf_id = f"{expected}{suffix}"
        used_ids.add(leaf_id)
        leaves.append({"id": leaf_id, "text": text.strip()})
    return leaves


def _collect_text_facts(text_analysis: Any) -> List[Dict[str, str]]:
    facts: List[Dict[str, str]] = []
    used_ids: Set[str] = set()
    for index, item in enumerate(_as_items(text_analysis), start=1):
        if isinstance(item, dict):
            facts.extend(
                _normalize_leaf_items(
                    item.get("facts"),
                    _container_id(item, "T", index),
                    "F",
                    used_ids,
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
        if not isinstance(item, dict):
            continue
        evidence_id = _container_id(item, "I", index)
        observations.extend(
            _normalize_leaf_items(
                item.get("observations"), evidence_id, "O", used_ids
            )
        )
        observations.extend(
            _normalize_leaf_items(item.get("text"), evidence_id, "TXT", used_ids)
        )
        inferences.extend(
            _normalize_leaf_items(
                item.get("inferences"), evidence_id, "INF", used_ids
            )
        )
    return observations, inferences


def _normalize_status(value: Any, context: str, errors: List[str]) -> str:
    if isinstance(value, str) and value.strip().lower() in _STATUSES:
        return value.strip().lower()
    errors.append(f"{context} status must be support, contradict, or unknown.")
    return "unknown"


def _filter_evidence_ids(
    value: Any,
    allowed_ids: Set[str],
    context: str,
    errors: List[str],
) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        errors.append(f"{context} evidence_ids must be a list.")
        return []

    result = []
    for item in value:
        evidence_id = item.strip() if isinstance(item, str) else ""
        if evidence_id not in allowed_ids:
            errors.append(f"{context} references unknown evidence id {evidence_id!r}.")
        elif evidence_id not in result:
            result.append(evidence_id)
    return result


def _unknown(reason: str) -> Dict[str, Any]:
    return {"status": "unknown", "reason": reason, "evidence_ids": []}


def _normalize_modality(
    value: Any,
    allowed_ids: Set[str],
    context: str,
    errors: List[str],
) -> Dict[str, Any]:
    if not allowed_ids:
        return _unknown(f"No {context.lower()} evidence was supplied.")
    if not isinstance(value, dict):
        errors.append(f"{context} result must be an object.")
        return _unknown("No valid consistency result was returned.")

    status = _normalize_status(value.get("status"), context, errors)
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append(f"{context} reason must be a non-empty string.")
        reason = "No valid reason was returned."
    evidence_ids = _filter_evidence_ids(
        value.get("evidence_ids"), allowed_ids, context, errors
    )
    if status in {"support", "contradict"} and not evidence_ids:
        errors.append(f"{context} {status} status has no valid evidence.")
        return _unknown("The returned status had no valid grounding evidence.")
    return {
        "status": status,
        "reason": reason.strip(),
        "evidence_ids": evidence_ids,
    }


def _normalize_consistency(
    value: Any,
    image_ids: Set[str],
    text_ids: Set[str],
    errors: List[str],
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        errors.append("Consistency result must be an object.")
        value = {}
    return {
        "image": _normalize_modality(value.get("image"), image_ids, "Image", errors),
        "text": _normalize_modality(value.get("text"), text_ids, "Text", errors),
    }


class ConsistencyChecker:
    """Compare the complete claim with each evidence modality."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def compare(
        self,
        claim: str,
        image_analysis: Any,
        text_analysis: Any,
    ) -> Dict[str, Any]:
        observations, inferences = _collect_image_evidence(image_analysis)
        facts = _collect_text_facts(text_analysis)
        image_ids = {item["id"] for item in observations + inferences}
        text_ids = {item["id"] for item in facts}
        result: Dict[str, Any] = {
            "image": _unknown("Consistency analysis was unavailable."),
            "text": _unknown("Consistency analysis was unavailable."),
            "raw_output": None,
            "errors": [],
        }
        errors: List[str] = result["errors"]
        prompt = render_prompt(
            CONSISTENCY_SYSTEM,
            CONSISTENCY_USER,
            CLAIM_TEXT=claim,
            IMAGE_EVIDENCE_JSON=json_text(
                {"observations": observations, "inferences": inferences}
            ),
            TEXT_EVIDENCE_JSON=json_text(facts),
        )
        parsed = generate_json(result, self.llm.generate, prompt)
        result.update(
            _normalize_consistency(parsed, image_ids, text_ids, errors)
        )
        return result


__all__ = ["ConsistencyChecker"]
