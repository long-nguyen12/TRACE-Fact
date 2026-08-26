"""Print compact summaries of a few MOCHEG samples."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MochegDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=str(PROJECT_ROOT / "dataset" / "mocheg")
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    samples = MochegDataset(args.root).load_split(args.split, limit=args.limit)
    summaries = [
        {
            "claim_id": sample["claim_id"],
            "claim": sample["claim"],
            "text_evidence_count": len(sample["text_evidence"]),
            "image_count": len(sample["images"]),
            "label": sample["label"],
            "has_ruling_outline": bool(sample["ruling_outline"].strip()),
        }
        for sample in samples
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
