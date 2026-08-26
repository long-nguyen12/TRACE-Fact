"""Small, explicit experiment configuration for the MOCHEG baseline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset" / "mocheg"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Text LLM backend: "huggingface" or "deepseek".
LLM_BACKEND = "huggingface"

# Local Hugging Face model IDs or directories.
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
HF_DEVICE_MAP = "auto"
HF_DTYPE = "auto"
HF_MAX_NEW_TOKENS = 1024
HF_LOCAL_FILES_ONLY = False

# DeepSeek uses DEEPSEEK_API_KEY from the environment. The VLM remains local.
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MAX_TOKENS = 1024

# Image provenance: Google Cloud Vision Web Detection plus a bounded page crawl.
PROVENANCE_PROVIDER = "google_vision"
PROVENANCE_TOP_K = 5
PROVENANCE_RIS_MAX_RESULTS = 10
PROVENANCE_CRAWL_TIMEOUT = 10.0
PROVENANCE_MAX_PAGE_BYTES = 2_000_000
PROVENANCE_MAX_TEXT_CHARS = 4000
PROVENANCE_USER_AGENT = "TRACE-Fact-Provenance/1.0"
PROVENANCE_RESPECT_ROBOTS = True

# Change this whenever a model, prompt, dtype, or generation setting changes.
RUN_ID = "qwen2.5-1.5b-qwen3-vl-provenance-v1"

USE_TEXT = True
USE_IMAGE = True
USE_CLAIM_DECOMPOSITION = False
USE_CONSISTENCY = True
GENERATE_EXPLANATION = True
USE_PROVENANCE = False

MAX_CLAIMS = 1
