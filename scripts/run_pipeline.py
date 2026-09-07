"""Run the configured multimodal fact-checking pipeline on a MOCHEG split."""

import argparse
import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.claim_analyzer import ClaimAnalyzer  # noqa: E402
from src.consistency import ConsistencyChecker  # noqa: E402
from src.dataset import MochegDataset  # noqa: E402
from src.deepseek import DeepSeekLLM, load_deepseek_api_key  # noqa: E402
from src.explanation import ExplanationGenerator  # noqa: E402
from src.fact_checker import FactChecker  # noqa: E402
from src.image_analyzer import ImageAnalyzer  # noqa: E402
from src.llm import LLM  # noqa: E402
from src.pipeline import FactCheckingPipeline  # noqa: E402
from src.provenance import (  # noqa: E402
    GoogleVisionWebDetectionProvider,
    ProvenanceRetriever,
    WebCrawler,
)
from src.text_analyzer import TextAnalyzer  # noqa: E402
from src.vlm import VLM  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _build_pipeline() -> FactCheckingPipeline:
    backend = str(config.LLM_BACKEND).strip().lower()
    if backend == "deepseek":
        llm = DeepSeekLLM(
            config.DEEPSEEK_MODEL,
            api_key=load_deepseek_api_key(),
            base_url=config.DEEPSEEK_BASE_URL,
            max_tokens=config.DEEPSEEK_MAX_TOKENS,
        )
    elif backend == "huggingface":
        llm = LLM(
            config.LLM_MODEL,
            device_map=config.HF_DEVICE_MAP,
            dtype=config.HF_DTYPE,
            max_new_tokens=config.HF_MAX_NEW_TOKENS,
            local_files_only=config.HF_LOCAL_FILES_ONLY,
        )
    else:
        raise ValueError("LLM_BACKEND must be 'huggingface' or 'deepseek'")

    if not config.USE_TEXT and not config.USE_IMAGE:
        raise ValueError("At least one of USE_TEXT or USE_IMAGE must be enabled")
    if config.USE_PROVENANCE and not config.USE_IMAGE:
        raise ValueError("USE_PROVENANCE requires USE_IMAGE = True")
    if (
        config.USE_PROVENANCE
        and str(config.PROVENANCE_PROVIDER).lower() != "google_vision"
    ):
        raise ValueError("PROVENANCE_PROVIDER must be 'google_vision'")

    vlm = None
    if config.USE_IMAGE:
        vlm = VLM(
            config.VLM_MODEL,
            device_map=config.HF_DEVICE_MAP,
            dtype=config.HF_DTYPE,
            max_new_tokens=config.HF_MAX_NEW_TOKENS,
            local_files_only=config.HF_LOCAL_FILES_ONLY,
        )

    provenance_retriever = None
    if config.USE_PROVENANCE:
        provider = GoogleVisionWebDetectionProvider(
            max_results=config.PROVENANCE_RIS_MAX_RESULTS
        )
        crawler = WebCrawler(
            timeout=config.PROVENANCE_CRAWL_TIMEOUT,
            max_bytes=config.PROVENANCE_MAX_PAGE_BYTES,
            max_text_chars=config.PROVENANCE_MAX_TEXT_CHARS,
            user_agent=config.PROVENANCE_USER_AGENT,
            respect_robots=config.PROVENANCE_RESPECT_ROBOTS,
        )
        provenance_retriever = ProvenanceRetriever(
            provider=provider,
            crawler=crawler,
            llm=llm,
            top_k=config.PROVENANCE_TOP_K,
        )

    dataset = MochegDataset(config.DATA_ROOT)
    return FactCheckingPipeline(
        dataset=dataset,
        claim_analyzer=ClaimAnalyzer(llm),
        image_analyzer=ImageAnalyzer(vlm) if vlm is not None else None,
        text_analyzer=TextAnalyzer(llm),
        consistency_checker=ConsistencyChecker(llm),
        fact_checker=FactChecker(llm),
        explanation_generator=ExplanationGenerator(llm),
        provenance_retriever=provenance_retriever,
        output_root=config.OUTPUT_ROOT,
        run_id=config.RUN_ID,
        use_text=config.USE_TEXT,
        use_image=config.USE_IMAGE,
        use_claim_decomposition=config.USE_CLAIM_DECOMPOSITION,
        use_consistency=config.USE_CONSISTENCY,
        generate_explanation=config.GENERATE_EXPLANATION,
        use_provenance=config.USE_PROVENANCE,
    )


def main() -> None:
    default_limit = config.MAX_CLAIMS if config.MAX_CLAIMS is not None else 1
    parser = argparse.ArgumentParser(
        description="Run configured text, image, and provenance analysis on MOCHEG."
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="val"
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=default_limit,
        help="number of claims to process (default: config.MAX_CLAIMS or 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        pipeline = _build_pipeline()
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    samples = pipeline.dataset.load_split(args.split, limit=args.limit)
    for sample in samples:
        result = pipeline.run(sample)
        summary = {
            "claim_id": result.get("claim_id"),
            "verdict": result.get("verdict"),
            "confidence": result.get("confidence"),
            "explanation": result.get("explanation"),
            "warnings": result.get("warnings", []),
        }
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
