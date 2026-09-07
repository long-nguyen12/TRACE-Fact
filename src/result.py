"""Build and save the compact user-facing pipeline result."""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


def build_result(details: Mapping[str, Any]) -> Dict[str, Any]:
    prediction = details.get("prediction", {})
    prediction = prediction if isinstance(prediction, Mapping) else {}
    explanation = details.get("explanation", {})
    explanation = explanation if isinstance(explanation, Mapping) else {}

    reasoning = prediction.get("reasoning", {})
    reasoning = reasoning if isinstance(reasoning, Mapping) else {}
    reason = str(reasoning.get("reason", "")).strip()
    prediction_ids = [
        str(item).strip()
        for item in reasoning.get("evidence_ids", []) or []
        if str(item).strip()
    ]

    explanation_text = str(explanation.get("explanation", "")).strip()
    citations = [
        str(item).strip()
        for item in explanation.get("citations", []) or []
        if str(item).strip()
    ]
    if (
        not explanation_text
        or error_list(explanation)
        or not set(citations).issubset(prediction_ids)
    ):
        explanation_text = reason
        citations = prediction_ids

    evidence, sources = _cited_evidence(
        citations,
        details.get("text_analysis", []),
        details.get("image_analysis", []),
        details.get("provenance", []),
    )
    ground_truth = details.get("ground_truth", {})
    ground_truth = ground_truth if isinstance(ground_truth, Mapping) else {}
    return {
        "claim_id": details.get("claim_id", ""),
        "claim": details.get("claim", ""),
        "verdict": prediction.get("label", "not_enough_information"),
        "confidence": prediction.get("confidence"),
        "explanation": explanation_text,
        "evidence": evidence,
        "provenance_sources": sources,
        "warnings": friendly_errors(details.get("errors", [])),
        "dataset_reference": {
            "verdict": ground_truth.get("label", ""),
            "explanation": ground_truth.get("ruling_outline", ""),
        },
    }


def _cited_evidence(
    citations: List[str],
    text_analysis: Any,
    image_analysis: Any,
    provenance: Any,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    by_id: Dict[str, Dict[str, str]] = {}
    source_ids: List[str] = []
    sources: Dict[str, Dict[str, str]] = {}

    for item in text_analysis if isinstance(text_analysis, list) else []:
        if not isinstance(item, Mapping):
            continue
        for fact in item.get("facts", []) or []:
            if isinstance(fact, Mapping):
                evidence_id = str(fact.get("id", ""))
                by_id[evidence_id] = {
                    "id": evidence_id,
                    "type": "text fact",
                    "text": str(fact.get("text", "")),
                }

    for item in image_analysis if isinstance(image_analysis, list) else []:
        if not isinstance(item, Mapping):
            continue
        for field, evidence_type in (
            ("observations", "image observation"),
            ("text", "visible image text"),
            ("inferences", "image inference"),
        ):
            for leaf in item.get(field, []) or []:
                if isinstance(leaf, Mapping):
                    evidence_id = str(leaf.get("id", ""))
                    by_id[evidence_id] = {
                        "id": evidence_id,
                        "type": evidence_type,
                        "text": str(leaf.get("text", "")),
                    }

    for item in provenance if isinstance(provenance, list) else []:
        if not isinstance(item, Mapping):
            continue
        for source in item.get("sources", []) or []:
            if isinstance(source, Mapping):
                source_id = str(source.get("id", ""))
                sources[source_id] = {
                    "id": source_id,
                    "title": str(source.get("title", "")),
                    "url": str(source.get("url", "")),
                    "date": str(source.get("date", "")),
                }
        for fact in item.get("facts", []) or []:
            if not isinstance(fact, Mapping):
                continue
            fact_id = str(fact.get("id", ""))
            by_id[fact_id] = {
                "id": fact_id,
                "type": "provenance fact",
                "text": str(fact.get("text", "")),
            }
            if fact_id in citations:
                for source_id in fact.get("source_ids", []) or []:
                    source_id = str(source_id)
                    if source_id not in source_ids:
                        source_ids.append(source_id)

    evidence = [by_id[item] for item in citations if item in by_id]
    cited_sources = [sources[item] for item in source_ids if item in sources]
    return evidence, cited_sources


def friendly_errors(value: Any) -> List[str]:
    errors = value if isinstance(value, list) else ([value] if value else [])
    friendly = []
    for error in errors:
        error = str(error)
        if "DefaultCredentialsError" in error:
            message = (
                "Image provenance is unavailable because Google Cloud "
                "credentials are not configured."
            )
        elif "Explanation references unknown evidence id" in error:
            message = (
                "The generated explanation referenced unavailable evidence; "
                "the validated fact-check reasoning is shown instead."
            )
        else:
            message = error
        if message not in friendly:
            friendly.append(message)
    return friendly


def ground_truth(sample: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "label": sample["label"],
        "raw_label": sample["cleaned_truthfulness"],
        "ruling_outline": sample["ruling_outline"],
    }


def collect_errors(*stages: Any) -> List[str]:
    collected = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for error in error_list(value):
                if error not in collected:
                    collected.append(error)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for stage in stages:
        visit(stage)
    return collected


def error_list(value: Mapping[str, Any]) -> List[str]:
    errors = value.get("errors", [])
    if not isinstance(errors, list):
        return [str(errors)] if errors else []
    return [str(error) for error in errors]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    os.replace(str(temporary), str(path))
