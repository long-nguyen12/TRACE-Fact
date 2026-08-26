# TRACE-Fact MOCHEG Baseline

This repository implements the MOCHEG baseline and the Phase 8 image-provenance
extension in `implementation.md`.

The implemented path is:

```text
MOCHEG gold evidence
  -> claim decomposition
  -> text fact extraction + image observation/inference extraction
  -> per-atom consistency
  -> optional reverse-image search + bounded source-page extraction
  -> support / refute / not_enough_information
  -> evidence-cited explanation
```

The Phase 9 evidence database and benchmark evaluation remain deliberately
separate.

## Dataset handling

The default data root is `dataset/mocheg`. `MochegDataset`:

- groups all `Corpus2.csv` rows by `claim_id`, even when rows are noncontiguous;
- preserves each nonblank `Evidence` value and its source `evidence_id`;
- uses only split-local gold `*-proof-*` images listed as positive qrel
  `evidence_id` values;
- never treats qrel `DOCUMENT#` retrieval candidates as gold evidence;
- maps `supported`, `refuted`, and `NEI` to the three model labels without
  changing the raw label;
- preserves `ruling_outline` and the raw/normalized labels as dataset fields;
  the pipeline excludes them from model inputs and attaches them under
  `ground_truth` only in the final result.

Inspect a real sample without calling a model:

```powershell
python scripts/inspect_dataset.py --split val --limit 1
```

## Model backends

The core still has no required third-party packages. When no `generate_fn` is
supplied, `LLM` and `VLM` use Hugging Face Transformers directly. Model loading
is lazy, so importing the project does not initialize CUDA.

Activate the conda environment that already contains your CUDA-enabled PyTorch,
then install the remaining optional packages into that same environment:

```powershell
conda activate <your-environment>
python -m pip install -r requirements-huggingface.txt
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__)"
```

The final boolean must be `True` for CUDA inference. `device_map="auto"` uses
Accelerate to place model layers on the GPU and, when needed, offload them to
CPU. It does not guarantee that CUDA is available. The optional requirements
pin the tested Transformers 4.57.6 API and expect PyTorch 2.2 or newer.

Set `LLM_MODEL` and `VLM_MODEL` in `config.py` to model IDs or local model
directories, then construct the wrappers directly:

```python
from config import (
    HF_DEVICE_MAP,
    HF_DTYPE,
    HF_LOCAL_FILES_ONLY,
    HF_MAX_NEW_TOKENS,
    LLM_MODEL,
    VLM_MODEL,
)
from src.llm import LLM
from src.vlm import VLM

llm = LLM(
    LLM_MODEL,
    device_map=HF_DEVICE_MAP,
    dtype=HF_DTYPE,
    max_new_tokens=HF_MAX_NEW_TOKENS,
    local_files_only=HF_LOCAL_FILES_ONLY,
)
vlm = VLM(
    VLM_MODEL,
    device_map=HF_DEVICE_MAP,
    dtype=HF_DTYPE,
    max_new_tokens=HF_MAX_NEW_TOKENS,
    local_files_only=HF_LOCAL_FILES_ONLY,
)
```

For example, a small direct-Transformers starting point is:

```python
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct")
vlm = VLM("Qwen/Qwen3-VL-2B-Instruct")
```

Both models load on their first `generate` call and are reused afterward.
The LLM applies the checkpoint's chat template when it has one; the VLM requires
a modern multimodal checkpoint whose `AutoProcessor` supplies a chat template.
Generation is greedy by default, and only newly generated tokens are returned to
the JSON parser.

After first use, both checkpoints remain resident for the lifetime of the
process. Size them for their combined GPU footprint; on a small GPU,
`device_map="auto"` may keep some layers on CPU and run substantially slower.

A model ID downloads into the Hugging Face cache on first use. Passing a complete
local checkpoint directory uses that directory instead. The configured runner
sets `HF_LOCAL_FILES_ONLY = True`, so it never contacts the Hub and fails quickly
when a requested checkpoint is not already cached. Set it to `False` for the
first run if you want Transformers to download a missing model.

The specific `Qwen/Qwen3-VL-8B-Instruct-FP8` model is not a direct Transformers
option: its model card says those FP8 weights currently require vLLM or SGLang.
Use a supported non-FP8 Qwen3-VL checkpoint with this wrapper.

The callable-based interface remains available for other inference stacks:

```python
from src.llm import LLM
from src.vlm import VLM

llm = LLM("my-text-model", generate_fn=my_text_generate)
vlm = VLM("my-vision-model", generate_fn=my_vision_generate)
```

`my_text_generate(prompt)` and `my_vision_generate(image_path, prompt)` must
return strings. Model responses are requested as JSON; malformed output is saved
in `raw_output` with an error instead of aborting the claim.

### DeepSeek API text backend

DeepSeek can replace the local text LLM while the Qwen VLM remains local. Install
the optional client into the same conda environment:

```powershell
conda activate <your-environment>
python -m pip install -r requirements-deepseek.txt
$env:DEEPSEEK_API_KEY = "your-key"
```

Select it in `config.py` and use a fresh run ID:

```python
LLM_BACKEND = "deepseek"
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MAX_TOKENS = 1024
RUN_ID = "deepseek-v4-pro-qwen3-vl-v1"
```

The adapter sends `response_format={"type": "json_object"}` on every request,
validates that the returned content is a JSON object, and never reads the API
key from `config.py`. Setting `LLM_BACKEND = "huggingface"` restores fully local
text inference. With DeepSeek selected, the external API receives the claim,
text evidence, the local VLM's extracted image descriptions, observations,
transcriptions and inferences, and any crawled provenance text used for
extraction. Image pixels are not sent to DeepSeek.

### Image provenance

Provenance uses [Google Cloud Vision Web Detection](https://docs.cloud.google.com/vision/docs/detecting-web)
as a provider-neutral reverse-image-search adapter. It submits each enabled evidence image to Google,
crawls only the top `PROVENANCE_TOP_K` result pages, extracts page title, date,
caption and bounded visible text, and asks the configured text LLM for
source-grounded provenance facts.

Install the optional SDK in the same conda environment:

```powershell
conda activate <your-environment>
python -m pip install -r requirements-provenance.txt
gcloud auth application-default login
```

The Google Cloud project must have billing and the Vision API enabled. Install
the Google Cloud CLI first if `gcloud` is unavailable. As an alternative to
[local ADC login](https://docs.cloud.google.com/docs/authentication/provide-credentials-adc),
set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON file outside this
repository. Then configure:

```python
USE_IMAGE = True
USE_PROVENANCE = True
PROVENANCE_PROVIDER = "google_vision"
PROVENANCE_TOP_K = 5
RUN_ID = "qwen-small-provenance-v1"
```

The crawl is non-recursive, respects `robots.txt` by default, rejects local and
private-network targets and limits response and extracted-text sizes. A
provider, credential, page, or parsing failure is recorded in the claim's
`errors` rather than terminating the run.

Privacy boundary: Google Cloud receives the evidence image bytes, crawled sites
receive the configured user-agent request, and the configured text backend
receives extracted page text. Keep `LLM_BACKEND = "huggingface"` if that text
must remain local after crawling.

## Running from the command line

Set the model names in `config.py`, for example:

```python
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
RUN_ID = "qwen-small-v1"
```

Then run one validation claim from the same conda environment as PyTorch:

```powershell
conda activate <your-environment>
python scripts/run_pipeline.py --split val --limit 1
```

The runner prints one compact JSON summary per claim. Omit `--limit` to use
`MAX_CLAIMS` from `config.py`; when it is `None`, the runner safely defaults to
one claim.

Every stage is recomputed on every run. The pipeline saves only a compact result
under `outputs/<run_id>/runs/<execution_id>/`, with a unique filename for each
claim invocation. Model generation remains greedy, so separately computed
content can still be identical.

## Running the pipeline from Python

Continuing from the configured `llm` and `vlm` wrappers above:

```python
from config import DATA_ROOT, OUTPUT_ROOT
from src.claim_analyzer import ClaimAnalyzer
from src.consistency import ConsistencyChecker
from src.dataset import MochegDataset
from src.explanation import ExplanationGenerator
from src.fact_checker import FactChecker
from src.image_analyzer import ImageAnalyzer
from src.pipeline import FactCheckingPipeline
from src.text_analyzer import TextAnalyzer

dataset = MochegDataset(DATA_ROOT)
sample = dataset.load_split("val", limit=1)[0]

pipeline = FactCheckingPipeline(
    dataset=dataset,
    claim_analyzer=ClaimAnalyzer(llm),
    image_analyzer=ImageAnalyzer(vlm),
    text_analyzer=TextAnalyzer(llm),
    consistency_checker=ConsistencyChecker(llm),
    fact_checker=FactChecker(llm),
    explanation_generator=ExplanationGenerator(llm),
    output_root=OUTPUT_ROOT,
    run_id="my-models-v1",
)

result = pipeline.run(sample)
```

`result` is the compact user-facing record: verdict, confidence, explanation,
cited evidence, provenance sources, warnings, dataset reference, and the saved
`result_file` path.

`config.py` is an editable experiment template, not hidden global state. The
core classes do not auto-load it: pass its paths, run ID, model names, limits,
and flags explicitly from the experiment script so each run remains visible.

Use a descriptive `run_id` for each experiment. Per-run ablations can be
selected explicitly:

```python
text_only = pipeline.run(
    sample,
    flags={"use_text": True, "use_image": False},
)
```

Supported flags are `use_text`, `use_image`, `use_claim_decomposition`,
`use_consistency`, `use_provenance`, and `generate_explanation`. Provenance
requires image analysis and a `ProvenanceRetriever` supplied to the pipeline;
the configured command-line runner creates it automatically.

## Leakage boundary

Prediction-stage prompts are built from a fresh whitelist containing only the
claim and generated evidence structures. The dataset label,
`cleaned_truthfulness`, `Truthfulness`, `ruling_outline`, `Origin`, headline,
fact-check URL, and qrel relevance are never passed to an LLM or VLM. Ground
truth is attached to the final JSON only after prediction and explanation.

Only compact run artifacts are written:

```text
outputs/<run_id>/runs/<execution_id>/<split>/
```

Extracted statements receive stable claim-local IDs: `C1` for claim atoms,
`T1.F1` for text facts, `I1.O1` for visual observations, `I1.TXT1` for visible
image text, `I1.INF1` for visual inferences, and `P1.F1` for a web-source-grounded
provenance fact. Verdict reasoning and explanations may cite only validated leaf
IDs. Provenance facts retain their supporting page IDs such as `P1.S1`.

## Tests

The test suite uses only the Python standard library:

```powershell
python -m unittest discover -s tests -v
```
