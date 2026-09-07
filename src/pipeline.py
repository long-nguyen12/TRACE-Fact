"""Transparent orchestration for MOCHEG fact checking."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .result import (
    build_result,
    collect_errors,
    error_list,
    ground_truth,
    write_json,
)

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
    _simple_result = staticmethod(build_result)
    _ground_truth = staticmethod(ground_truth)
    _collect_errors = staticmethod(collect_errors)
    _error_list = staticmethod(error_list)
    _write_json = staticmethod(write_json)

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
        self.output_root = Path(output_root)
        self.run_id = run_id
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
        claim_id = str(sample["claim_id"])
        claim = sample["claim"]
        active = self._resolve_flags(flags)
        split = sample["split"]
        ground_truth = self._ground_truth(sample)

        LOGGER.info("claim_id=%s stage=pipeline status=start", claim_id)
        claim_analysis = self._claim_analysis(claim, active["use_claim_decomposition"])
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
            "split": sample["split"],
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

        filename = f"{claim_id}.json"
        result_path = Path(
            self.output_root,
            self.run_id,
            "runs",
            split,
            filename,
        )
        result["result_file"] = result_path.relative_to(self.output_root).as_posix()
        self._write_json(result_path, result)
        LOGGER.info("claim_id=%s stage=pipeline status=complete", claim_id)
        return result

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
            raise ValueError("use_provenance=True requires a provenance_retriever")
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
        return self._canonical_claim(self.claim_analyzer.analyze(claim), claim)

    def _text_analysis(
        self,
        sample: Mapping[str, Any],
        enabled: bool,
    ) -> List[Dict[str, Any]]:
        if not enabled:
            return []
        results = []
        for index, (text, source_id) in enumerate(self._text_entries(sample), start=1):
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
        for index, (image_path, _) in enumerate(self._image_entries(sample), start=1):
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
        return zip(sample["text_evidence"], sample["text_evidence_ids"])

    @staticmethod
    def _image_entries(sample: Mapping[str, Any]) -> Iterable[Tuple[str, str]]:
        return zip(sample["images"], sample["image_evidence_ids"])

    @classmethod
    def _canonical_claim(cls, result: Any, claim: str) -> Dict[str, Any]:
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
        output = (
            dict(result)
            if isinstance(result, Mapping)
            else {"errors": ["text_analysis: result was not an object"]}
        )
        output["evidence_id"] = evidence_id
        output["source_evidence_id"] = source_id
        output["facts"] = cls._leaf_records(output.get("facts", []), evidence_id + ".F")
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
        output = (
            dict(result)
            if isinstance(result, Mapping)
            else {"errors": ["image_analysis: result was not an object"]}
        )
        output["evidence_id"] = evidence_id
        output["source_evidence_id"] = source_id
        output["path"] = image_path
        output["description"] = str(output.get("description", ""))
        output["objects"] = cls._string_list(output.get("objects", []))
        output["text"] = cls._leaf_records(output.get("text", []), evidence_id + ".TXT")
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
                records.append(
                    {"id": "%s%d" % (prefix, len(records) + 1), "text": text}
                )
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
