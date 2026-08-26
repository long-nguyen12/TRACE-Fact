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
    CORPUS_COLUMNS = {
        "evidence_id",
        "claim_id",
        "Snopes URL",
        "ruling_outline",
        "Origin",
        "Claim",
        "cleaned_truthfulness",
        "Evidence",
    }
    IMAGE_QREL_COLUMNS = {
        "TOPIC",
        "ITERATION",
        "DOCUMENT#",
        "RELEVANCY",
        "evidence_id",
    }
    SENTENCE_QREL_COLUMNS = {
        "TOPIC",
        "ITERATION",
        "DOCUMENT#",
        "RELEVANCY",
    }

    def __init__(self, root):
        root = Path(root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"MOCHEG root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"MOCHEG root is not a directory: {root}")
        self.root = root.resolve()
        self._splits = {}
        self._claims = {}
        self._qrels = {}

    def load_split(self, split, limit=None):
        """Return claim-level samples, preserving first-appearance order.

        The complete file is aggregated before ``limit`` is applied because
        some training claims recur in noncontiguous CSV blocks.
        """

        split = self._check_split(split)
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an integer or None")
            if limit < 0:
                raise ValueError("limit must be non-negative")
        if split not in self._splits:
            self._load_split(split)
        samples = self._splits[split]
        if limit is not None:
            samples = samples[:limit]
        return [self._copy(sample) for sample in samples]

    def get_claim(self, claim_id, split=None):
        """Return a claim by ID, optionally constrained to one split."""

        if isinstance(claim_id, bool) or not isinstance(claim_id, (str, int)):
            raise TypeError("claim_id must be a string or integer")
        claim_id = str(claim_id)
        if not claim_id:
            raise ValueError("claim_id must not be empty")
        if split is not None:
            split = self._check_split(split)
            if split not in self._splits:
                self._load_split(split)
            if claim_id not in self._claims[split]:
                raise KeyError(f"Claim {claim_id!r} was not found in split {split!r}")
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
        if modality == "image":
            if level is not None:
                raise ValueError("level must be None when modality='image'")
            filename = "img_evidence_qrels.csv"
            columns = self.IMAGE_QREL_COLUMNS
        elif modality == "text" and level == "article":
            filename = "text_evidence_qrels_article_level.csv"
            columns = self.IMAGE_QREL_COLUMNS
        elif modality == "text" and level == "sentence":
            filename = "text_evidence_qrels_sentence_level.csv"
            columns = self.SENTENCE_QREL_COLUMNS
        elif modality == "text":
            raise ValueError(
                "level must be 'article' or 'sentence' when modality='text'"
            )
        else:
            raise ValueError("modality must be 'image' or 'text'")

        cache_key = (split, modality, level)
        if cache_key in self._qrels:
            return [dict(row) for row in self._qrels[cache_key]]
        path = self.root / split / filename
        self._require_file(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._check_columns(path, reader.fieldnames, columns)
            rows = []
            for record_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"Malformed CSV row in {path} at record {record_number}"
                    )
                if row["RELEVANCY"] not in {"0", "1"}:
                    raise ValueError(
                        f"Unexpected RELEVANCY {row['RELEVANCY']!r} in {path} "
                        f"at record {record_number}"
                    )
                rows.append(dict(row))
        self._qrels[cache_key] = rows
        return [dict(row) for row in rows]

    def _load_split(self, split):
        path = self.root / split / "Corpus2.csv"
        self._require_file(path)
        samples = {}
        evidence_ids = set()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._check_columns(path, reader.fieldnames, self.CORPUS_COLUMNS)
            for record_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"Malformed CSV row in {path} at record {record_number}"
                    )
                claim_id = row["claim_id"]
                evidence_id = row["evidence_id"]
                if not claim_id.strip() or not evidence_id.strip():
                    raise ValueError(
                        f"Empty claim_id or evidence_id in {path} "
                        f"at record {record_number}"
                    )
                if evidence_id in evidence_ids:
                    raise ValueError(
                        f"Duplicate evidence_id {evidence_id!r} in {path} "
                        f"at record {record_number}"
                    )
                evidence_ids.add(evidence_id)

                raw_label = row["cleaned_truthfulness"]
                if raw_label not in self.LABELS:
                    expected = ", ".join(repr(value) for value in self.LABELS)
                    raise ValueError(
                        f"Unknown cleaned_truthfulness {raw_label!r} in {path} "
                        f"at record {record_number}; expected {expected}"
                    )
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
                repeated = {
                    "claim": row["Claim"],
                    "cleaned_truthfulness": raw_label,
                    "ruling_outline": row["ruling_outline"],
                    "origin": row["Origin"],
                    "snopes_url": row["Snopes URL"],
                }
                for key, value in repeated.items():
                    if sample[key] != value:
                        raise ValueError(
                            f"Conflicting {key} for claim {claim_id!r} in {path} "
                            f"at record {record_number}"
                        )
                evidence = row["Evidence"]
                if evidence.strip():
                    sample["text_evidence"].append(evidence)
                    sample["text_evidence_ids"].append(evidence_id)

        image_dir = self.root / split / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        seen_images = {}
        for row in self.load_qrels(split, modality="image"):
            if row["RELEVANCY"] != "1":
                continue
            claim_id = row["TOPIC"]
            evidence_id = row["evidence_id"]
            if claim_id not in samples:
                raise ValueError(
                    f"Image qrels for {split!r} contain unknown TOPIC {claim_id!r}"
                )
            seen = seen_images.setdefault(claim_id, set())
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            if Path(evidence_id).name != evidence_id:
                raise ValueError(
                    f"Gold image evidence_id must be a filename, got {evidence_id!r}"
                )
            image_path = image_dir / evidence_id
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Gold image {evidence_id!r} for claim {claim_id!r} "
                    f"was not found in {image_dir}"
                )
            samples[claim_id]["image_evidence_ids"].append(evidence_id)
            samples[claim_id]["images"].append(str(image_path))

        self._claims[split] = samples
        self._splits[split] = list(samples.values())

    def _check_split(self, split):
        if not isinstance(split, str):
            raise TypeError("split must be a string")
        if split not in self.SPLITS:
            expected = ", ".join(self.SPLITS)
            raise ValueError(f"Unknown split {split!r}; expected one of: {expected}")
        return split

    @staticmethod
    def _check_columns(path, fieldnames, required):
        if fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"CSV {path} is missing required columns: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _require_file(path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV file: {path}")

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
