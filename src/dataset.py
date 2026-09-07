"""Dependency-free loader for the local MOCHEG dataset."""

import csv
from pathlib import Path


class MochegDataset:
    """Load MOCHEG rows as claim-level dictionaries."""

    SPLITS = ("train", "val", "test")
    LABELS = {
        "supported": "support",
        "refuted": "refute",
        "NEI": "not_enough_information",
    }
    QREL_FILES = {
        ("image", None): "img_evidence_qrels.csv",
        ("text", "article"): "text_evidence_qrels_article_level.csv",
        ("text", "sentence"): "text_evidence_qrels_sentence_level.csv",
    }

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self._splits = {}
        self._claims = {}
        self._qrels = {}

    def load_split(self, split, limit=None):
        """Return claim-level samples, preserving first-appearance order.

        The complete file is aggregated before ``limit`` is applied because
        some training claims recur in noncontiguous CSV blocks.
        """

        split = self._check_split(split)
        if split not in self._splits:
            self._load_split(split)
        samples = self._splits[split]
        if limit is not None:
            samples = samples[:limit]
        return [self._copy(sample) for sample in samples]

    def get_claim(self, claim_id, split=None):
        """Return a claim by ID, optionally constrained to one split."""

        claim_id = str(claim_id)
        if split is not None:
            split = self._check_split(split)
            if split not in self._splits:
                self._load_split(split)
            return self._copy(self._claims[split][claim_id])

        matches = []
        for candidate_split in self.SPLITS:
            if candidate_split not in self._splits:
                self._load_split(candidate_split)
            sample = self._claims[candidate_split].get(claim_id)
            if sample is not None:
                matches.append(sample)
        if not matches:
            raise KeyError(f"Claim {claim_id!r} was not found in any split")
        if len(matches) > 1:
            locations = ", ".join(sample["split"] for sample in matches)
            raise ValueError(
                f"Claim {claim_id!r} occurs in multiple splits ({locations}); "
                "pass split explicitly"
            )
        return self._copy(matches[0])

    def get_images(self, claim_id, split=None):
        """Return split-local gold image paths for a claim."""

        return self.get_claim(claim_id, split=split)["images"]

    def load_qrels(self, split, modality="image", level=None):
        """Load image or article/sentence text qrels as raw string dicts."""

        split = self._check_split(split)
        try:
            filename = self.QREL_FILES[(modality, level)]
        except KeyError as exc:
            raise ValueError("Unsupported qrel modality and level") from exc

        cache_key = (split, modality, level)
        if cache_key in self._qrels:
            return [dict(row) for row in self._qrels[cache_key]]
        path = self.root / split / filename
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self._qrels[cache_key] = rows
        return [dict(row) for row in rows]

    def _load_split(self, split):
        path = self.root / split / "Corpus2.csv"
        samples = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                claim_id = row["claim_id"]
                evidence_id = row["evidence_id"]
                raw_label = row["cleaned_truthfulness"]
                # if raw_label not in self.LABELS:
                #     raise ValueError(
                #         f"Unknown cleaned_truthfulness {raw_label!r} in {path}"
                #     )
                if claim_id not in samples:
                    samples[claim_id] = {
                        "claim_id": claim_id,
                        "claim": row["Claim"],
                        "text_evidence": [],
                        "text_evidence_ids": [],
                        "images": [],
                        "image_evidence_ids": [],
                        "cleaned_truthfulness": raw_label,
                        "label": self.LABELS[raw_label],
                        "ruling_outline": row["ruling_outline"],
                        "origin": row["Origin"],
                        "snopes_url": row["Snopes URL"],
                        "split": split,
                    }
                sample = samples[claim_id]
                evidence = row["Evidence"]
                if evidence.strip():
                    sample["text_evidence"].append(evidence)
                    sample["text_evidence_ids"].append(evidence_id)

        image_dir = self.root / split / "images"
        seen_images = {}
        for row in self.load_qrels(split, modality="image"):
            if row["RELEVANCY"] != "1":
                continue
            claim_id = row["TOPIC"]
            evidence_id = row["evidence_id"]
            seen = seen_images.setdefault(claim_id, set())
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            image_path = image_dir / evidence_id
            samples[claim_id]["image_evidence_ids"].append(evidence_id)
            samples[claim_id]["images"].append(str(image_path))

        self._claims[split] = samples
        self._splits[split] = list(samples.values())

    def _check_split(self, split):
        if split not in self.SPLITS:
            expected = ", ".join(self.SPLITS)
            raise ValueError(f"Unknown split {split!r}; expected one of: {expected}")
        return split

    @staticmethod
    def _copy(sample):
        result = dict(sample)
        for key in (
            "text_evidence",
            "text_evidence_ids",
            "images",
            "image_evidence_ids",
        ):
            result[key] = list(sample[key])
        return result
