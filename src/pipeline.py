"""Transparent orchestration for MOCHEG fact checking."""

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


LOGGER = logging.getLogger(__name__)


class FactCheckingPipeline:
    """Connect the extraction, comparison, verdict, and explanation stages."""

    _FLAG_NAMES = {
        "use_text",
        "use_image",
        "use_claim_decomposition",
        "use_consistency",
        "generate_explanation",
        "use_provenance",
    }

    def __init__(
        self,
        dataset: Any,
        claim_analyzer: Any,
        image_analyzer: Any,
        text_analyzer: Any,
        consistency_checker: Any,
        fact_checker: Any,
        explanation_generator: Any,
        output_root: Any = "outputs",
        run_id: str = "baseline",
        use_text: bool = True,
        use_image: bool = True,
        use_claim_decomposition: bool = True,
        use_consistency: bool = True,
        generate_explanation: bool = True,
        use_provenance: bool = False,
        provenance_retriever: Any = None,
    ) -> None:
        self.dataset = dataset
        self.claim_analyzer = claim_analyzer
        self.image_analyzer = image_analyzer
        self.text_analyzer = text_analyzer
        self.consistency_checker = consistency_checker
        self.fact_checker = fact_checker
        self.explanation_generator = explanation_generator
        self.provenance_retriever = provenance_retriever
        self.execution_id = "%s_%s" % (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            uuid.uuid4().hex[:8],
        )
        self.output_root = Path(output_root)
        self.run_id = self._safe_name(run_id)
        self.default_flags = {
            "use_text": use_text,
            "use_image": use_image,
            "use_claim_decomposition": use_claim_decomposition,
            "use_consistency": use_consistency,
            "generate_explanation": generate_explanation,
            "use_provenance": use_provenance,
        }

    def run(
        self,
        sample: Mapping[str, Any],
        flags: Optional[Mapping[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Recompute every stage for one claim and save a compact result."""
        claim_id = str(sample.get("claim_id", "")).strip()
        claim = str(sample.get("claim", "")).strip()
        if not claim_id:
            raise ValueError("sample['claim_id'] must be a non-empty value")
        if not claim:
            raise ValueError("sample['claim'] must be a non-empty string")

        active = self._resolve_flags(flags)
        split = self._safe_name(str(sample.get("split", "unspecified")))
        ground_truth = self._ground_truth(sample)

        LOGGER.info("claim_id=%s stage=pipeline status=start", claim_id)
        claim_analysis = self._claim_analysis(
            claim, active["use_claim_decomposition"]
        )
        text_analysis = self._text_analysis(sample, active["use_text"])
        image_analysis = self._image_analysis(sample, active["use_image"])

        if active["use_consistency"]:
            consistency = self.consistency_checker.compare(
                claim_analysis, image_analysis, text_analysis
            )
        else:
            consistency = []

        provenance = self._provenance_analysis(
            sample,
            active["use_provenance"],
        )

        model_evidence = self._model_evidence(
            claim_id,
            claim,
            claim_analysis,
            text_analysis,
            image_analysis,
            consistency,
            provenance,
        )
        prediction = self.fact_checker.verify(model_evidence)

        if active["generate_explanation"]:
            explanation = self.explanation_generator.generate(
                model_evidence, prediction
            )
        else:
            explanation = {
                "explanation": "",
                "citations": [],
                "errors": [],
                "skipped": True,
            }

        # Ground truth is deliberately attached only after every model call.
        details = {
            "claim_id": claim_id,
            "claim": claim,
            "split": str(sample.get("split", "")),
            "claim_analysis": claim_analysis,
            "text_analysis": text_analysis,
            "image_analysis": image_analysis,
            "consistency": consistency,
            "provenance": provenance,
            "prediction": prediction,
            "explanation": explanation,
            "ground_truth": ground_truth,
            "errors": self._collect_errors(
                claim_analysis,
                text_analysis,
                image_analysis,
                consistency,
                provenance,
                prediction,
                explanation,
            ),
        }
        result = self._simple_result(details)
        filename = self._safe_name(
            "%s__%s" % (claim_id, uuid.uuid4().hex[:8])
        ) + ".json"
        result_path = Path(
            self.output_root,
            self.run_id,
            "runs",
            split,
            filename,
        )
        result["result_file"] = result_path.relative_to(
            self.output_root
        ).as_posix()
        self._write_json(result_path, result)
        LOGGER.info("claim_id=%s stage=pipeline status=complete", claim_id)
        return result

    def _simple_result(
        self,
        details: Mapping[str, Any],
    ) -> Dict[str, Any]:
        prediction = details.get("prediction", {})
        prediction = prediction if isinstance(prediction, Mapping) else {}
        explanation = details.get("explanation", {})
        explanation = explanation if isinstance(explanation, Mapping) else {}

        reasons = []
        prediction_ids = []
        reasoning = prediction.get("reasoning", [])
        reasoning = reasoning if isinstance(reasoning, list) else []
        for item in reasoning:
            if not isinstance(item, Mapping):
                continue
            reason = str(item.get("reason", "")).strip()
            if reason and reason not in reasons:
                reasons.append(reason)
            for evidence_id in item.get("evidence_ids", []) or []:
                evidence_id = str(evidence_id).strip()
                if evidence_id and evidence_id not in prediction_ids:
                    prediction_ids.append(evidence_id)

        explanation_text = str(explanation.get("explanation", "")).strip()
        citations = [
            str(item).strip()
            for item in explanation.get("citations", []) or []
            if str(item).strip()
        ]
        explanation_errors = self._error_list(explanation)
        if (
            not explanation_text
            or explanation_errors
            or not set(citations).issubset(prediction_ids)
        ):
            explanation_text = " ".join(reasons)
            citations = prediction_ids

        evidence, provenance_sources = self._cited_evidence(
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
            "provenance_sources": provenance_sources,
            "warnings": self._friendly_errors(details.get("errors", [])),
            "dataset_reference": {
                "verdict": ground_truth.get("label", ""),
                "explanation": ground_truth.get("ruling_outline", ""),
            },
        }

    @staticmethod
    def _cited_evidence(
        citations: List[str],
        text_analysis: Any,
        image_analysis: Any,
        provenance: Any,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        by_id: Dict[str, Dict[str, str]] = {}
        provenance_source_ids: List[str] = []
        provenance_sources: Dict[str, Dict[str, str]] = {}

        for item in text_analysis if isinstance(text_analysis, list) else []:
            if not isinstance(item, Mapping):
                continue
            for fact in item.get("facts", []) or []:
                if isinstance(fact, Mapping):
                    by_id[str(fact.get("id", ""))] = {
                        "id": str(fact.get("id", "")),
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
                        by_id[str(leaf.get("id", ""))] = {
                            "id": str(leaf.get("id", "")),
                            "type": evidence_type,
                            "text": str(leaf.get("text", "")),
                        }
        for item in provenance if isinstance(provenance, list) else []:
            if not isinstance(item, Mapping):
                continue
            for source in item.get("sources", []) or []:
                if isinstance(source, Mapping):
                    source_id = str(source.get("id", ""))
                    provenance_sources[source_id] = {
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
                        if source_id not in provenance_source_ids:
                            provenance_source_ids.append(source_id)

        evidence = [by_id[item] for item in citations if item in by_id]
        sources = [
            provenance_sources[item]
            for item in provenance_source_ids
            if item in provenance_sources
        ]
        return evidence, sources

    @staticmethod
    def _friendly_errors(value: Any) -> List[str]:
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

    def _resolve_flags(
        self, overrides: Optional[Mapping[str, bool]]
    ) -> Dict[str, bool]:
        active = dict(self.default_flags)
        if overrides:
            unknown = set(overrides) - self._FLAG_NAMES
            if unknown:
                raise ValueError("Unknown pipeline flags: %s" % sorted(unknown))
            active.update({key: bool(value) for key, value in overrides.items()})
        if not active["use_text"] and not active["use_image"]:
            raise ValueError("At least one of use_text or use_image must be enabled")
        if active["use_provenance"] and not active["use_image"]:
            raise ValueError("Image provenance requires use_image=True")
        if active["use_provenance"] and self.provenance_retriever is None:
            raise ValueError(
                "use_provenance=True requires a provenance_retriever"
            )
        return active

    def _claim_analysis(
        self,
        claim: str,
        enabled: bool,
    ) -> Dict[str, Any]:
        if not enabled:
            return {
                "claim": claim,
                "atoms": [{"id": "C1", "text": claim}],
                "entities": [],
                "errors": [],
                "decomposition_skipped": True,
            }
        return self._canonical_claim(
            self.claim_analyzer.analyze(claim), claim
        )

    def _text_analysis(
        self,
        sample: Mapping[str, Any],
        enabled: bool,
    ) -> List[Dict[str, Any]]:
        if not enabled:
            return []
        results = []
        for index, (text, source_id) in enumerate(
            self._text_entries(sample), start=1
        ):
            evidence_id = "T%d" % index
            results.append(
                self._canonical_text(
                    self.text_analyzer.analyze(text), evidence_id, source_id
                )
            )
        return results

    def _image_analysis(
        self,
        sample: Mapping[str, Any],
        enabled: bool,
    ) -> List[Dict[str, Any]]:
        if not enabled:
            return []
        results = []
        for index, (image_path, source_id) in enumerate(
            self._image_entries(sample), start=1
        ):
            evidence_id = "I%d" % index
            results.append(
                self._canonical_image(
                    self.image_analyzer.analyze(image_path),
                    evidence_id,
                    source_id,
                    str(image_path),
                )
            )
        return results

    def _provenance_analysis(
        self,
        sample: Mapping[str, Any],
        enabled: bool,
    ) -> List[Dict[str, Any]]:
        if not enabled:
            return []
        results = []
        for index, (image_path, _) in enumerate(
            self._image_entries(sample), start=1
        ):
            provenance_id = "P%d" % index
            image_evidence_id = "I%d" % index
            result = self.provenance_retriever.search_image(
                image_path,
                evidence_id=provenance_id,
                image_evidence_id=image_evidence_id,
            )
            if not isinstance(result, Mapping):
                result = {
                    "evidence_id": provenance_id,
                    "image_evidence_id": image_evidence_id,
                    "image_path": str(image_path),
                    "sources": [],
                    "facts": [],
                    "errors": ["Provenance result was not an object"],
                }
            results.append(dict(result))
        return results

    @staticmethod
    def _text_entries(sample: Mapping[str, Any]) -> Iterable[Tuple[str, str]]:
        values = sample.get("text_evidence", []) or []
        source_ids = sample.get("text_evidence_ids", []) or []
        for index, value in enumerate(values):
            if isinstance(value, Mapping):
                text = value.get("text", "")
                source_id = value.get("source_evidence_id", value.get("id", ""))
            else:
                text = value
                source_id = source_ids[index] if index < len(source_ids) else ""
            text = str(text)
            if text.strip():
                yield text, str(source_id)

    @staticmethod
    def _image_entries(sample: Mapping[str, Any]) -> Iterable[Tuple[str, str]]:
        values = sample.get("images", []) or []
        source_ids = sample.get("image_evidence_ids", []) or []
        for index, value in enumerate(values):
            if isinstance(value, Mapping):
                image_path = value.get("path", "")
                source_id = value.get(
                    "source_evidence_id", value.get("filename", value.get("id", ""))
                )
            else:
                image_path = value
                source_id = source_ids[index] if index < len(source_ids) else Path(str(value)).name
            if str(image_path).strip():
                yield str(image_path), str(source_id)

    @classmethod
    def _canonical_claim(
        cls, result: Any, claim: str
    ) -> Dict[str, Any]:
        output = dict(result) if isinstance(result, Mapping) else {}
        atoms = cls._leaf_records(output.get("atoms", []), "C")
        if not atoms:
            atoms = [{"id": "C1", "text": claim}]
            errors = cls._error_list(output)
            errors.append("claim_analysis: no valid atoms; used the whole claim")
            output["errors"] = errors
        output["claim"] = claim
        output["atoms"] = atoms
        output["entities"] = cls._string_list(output.get("entities", []))
        output.setdefault("errors", [])
        return output

    @classmethod
    def _canonical_text(
        cls, result: Any, evidence_id: str, source_id: str
    ) -> Dict[str, Any]:
        output = dict(result) if isinstance(result, Mapping) else {
            "errors": ["text_analysis: result was not an object"]
        }
        output["evidence_id"] = evidence_id
        output["source_evidence_id"] = source_id
        output["facts"] = cls._leaf_records(
            output.get("facts", []), evidence_id + ".F"
        )
        output["entities"] = cls._string_list(output.get("entities", []))
        output.setdefault("errors", [])
        return output

    @classmethod
    def _canonical_image(
        cls,
        result: Any,
        evidence_id: str,
        source_id: str,
        image_path: str,
    ) -> Dict[str, Any]:
        output = dict(result) if isinstance(result, Mapping) else {
            "errors": ["image_analysis: result was not an object"]
        }
        output["evidence_id"] = evidence_id
        output["source_evidence_id"] = source_id
        output["path"] = image_path
        output["description"] = str(output.get("description", ""))
        output["objects"] = cls._string_list(output.get("objects", []))
        output["text"] = cls._leaf_records(
            output.get("text", []), evidence_id + ".TXT"
        )
        output["observations"] = cls._leaf_records(
            output.get("observations", []), evidence_id + ".O"
        )
        output["inferences"] = cls._leaf_records(
            output.get("inferences", []), evidence_id + ".INF"
        )
        if not isinstance(output.get("relations"), list):
            output["relations"] = []
        output.setdefault("errors", [])
        return output

    @staticmethod
    def _leaf_records(values: Any, prefix: str) -> List[Dict[str, str]]:
        if not isinstance(values, list):
            return []
        records = []
        for value in values:
            if isinstance(value, Mapping):
                text = value.get("text", "")
            else:
                text = value
            text = str(text).strip()
            if text:
                records.append({"id": "%s%d" % (prefix, len(records) + 1), "text": text})
        return records

    @staticmethod
    def _string_list(values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        return [str(value).strip() for value in values if str(value).strip()]

    @staticmethod
    def _model_evidence(
        claim_id: str,
        claim: str,
        claim_analysis: Mapping[str, Any],
        text_analysis: List[Mapping[str, Any]],
        image_analysis: List[Mapping[str, Any]],
        consistency: Any,
        provenance: List[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Create a fresh whitelist-only object for prediction-stage models."""
        return {
            "claim_id": claim_id,
            "claim": claim,
            "claim_analysis": {
                "claim": claim,
                "atoms": claim_analysis.get("atoms", []),
                "entities": claim_analysis.get("entities", []),
            },
            "text_analysis": [
                {
                    "evidence_id": item.get("evidence_id", ""),
                    "source_evidence_id": item.get("source_evidence_id", ""),
                    "facts": item.get("facts", []),
                    "entities": item.get("entities", []),
                }
                for item in text_analysis
            ],
            "image_analysis": [
                {
                    "evidence_id": item.get("evidence_id", ""),
                    "source_evidence_id": item.get("source_evidence_id", ""),
                    "text": item.get("text", []),
                    "observations": item.get("observations", []),
                    "inferences": item.get("inferences", []),
                }
                for item in image_analysis
            ],
            "consistency": consistency,
            "provenance": [
                {
                    "evidence_id": item.get("evidence_id", ""),
                    "image_evidence_id": item.get("image_evidence_id", ""),
                    "first_seen": item.get("first_seen", ""),
                    "location": item.get("location", ""),
                    "event": item.get("event", ""),
                    "people": item.get("people", []),
                    "sources": [
                        {
                            "id": source.get("id", ""),
                            "url": source.get("url", ""),
                            "title": source.get("title", ""),
                            "date": source.get("date", ""),
                            "caption": source.get("caption", ""),
                        }
                        for source in item.get("sources", [])
                        if isinstance(source, Mapping)
                    ],
                    "facts": item.get("facts", []),
                }
                for item in provenance
            ],
        }

    @staticmethod
    def _ground_truth(sample: Mapping[str, Any]) -> Dict[str, Any]:
        supplied = sample.get("ground_truth", {})
        supplied = supplied if isinstance(supplied, Mapping) else {}
        return {
            "label": supplied.get("label", sample.get("label", "")),
            "raw_label": supplied.get(
                "raw_label", sample.get("cleaned_truthfulness", "")
            ),
            "ruling_outline": supplied.get(
                "ruling_outline", sample.get("ruling_outline", "")
            ),
        }

    @classmethod
    def _collect_errors(cls, *stages: Any) -> List[str]:
        collected = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for error in cls._error_list(value):
                    if error not in collected:
                        collected.append(error)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        for stage in stages:
            visit(stage)
        return collected

    @staticmethod
    def _error_list(value: Mapping[str, Any]) -> List[str]:
        errors = value.get("errors", [])
        if not isinstance(errors, list):
            return [str(errors)] if errors else []
        return [str(error) for error in errors]

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            # ASCII escaping keeps results valid if a model emits an
            # otherwise unencodable lone Unicode surrogate.
            json.dump(value, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
        os.replace(str(temporary), str(path))

    @staticmethod
    def _safe_name(value: str) -> str:
        original = value.strip()
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", original) or "unnamed"
        needs_digest = cleaned != original or original != original.casefold()
        if cleaned in {".", ".."}:
            cleaned = "dot"
            needs_digest = True
        if cleaned.endswith("."):
            cleaned = cleaned.rstrip(".") or "dot"
            needs_digest = True
        reserved = {"con", "prn", "aux", "nul"}
        reserved.update("com%d" % number for number in range(1, 10))
        reserved.update("lpt%d" % number for number in range(1, 10))
        if cleaned.split(".", 1)[0].casefold() in reserved:
            cleaned = "_" + cleaned
            needs_digest = True
        if len(cleaned) > 96:
            needs_digest = True
        if needs_digest:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
            cleaned = cleaned[:80] + "__" + digest
        return cleaned
