# Codex Implementation Plan: Explainable Multimodal Fact-Checking with MOCHEG

## 1. Goal

Build a simple research framework for explainable multimodal fact-checking using MOCHEG.

The system should:

1. Read claims, text evidence, image evidence, and labels from MOCHEG.
2. Decompose each claim into smaller verifiable facts.
3. Extract structured information from image evidence using a VLM.
4. Extract structured information from text evidence using an LLM.
5. Compare the claim against image and text evidence.
6. Retrieve external provenance for images and text when possible.
7. Store retrieved and extracted evidence in a reusable evidence database.
8. Use an LLM to make the final decision:
   - `support`
   - `refute`
   - `not enough information`
9. Generate an explanation that explicitly references the evidence used.
10. Evaluate the final prediction and explanation against MOCHEG.

MOCHEG already provides the main data required for this framework: claims, text evidence, image evidence, `cleaned_truthfulness`, and `ruling_outline`. It also provides retrieval qrels for text and images.

---

# 2. Design Principles

Keep the implementation intentionally simple.

## Rules for Codex

- Use Python.
- One major module per `.py` file.
- Prefer classes and Python objects.
- Avoid Pydantic.
- Avoid complex schemas.
- Avoid dependency injection frameworks.
- Avoid unnecessary inheritance.
- Avoid large configuration systems.
- Avoid microservices.
- Avoid databases initially.
- Use normal Python dictionaries for intermediate results.
- Use JSON/JSONL for saved results.
- Use SQLite only later if evidence storage becomes too large.
- Each class should have one clear responsibility.
- Every module should be runnable/testable independently.
- External model/API calls should be isolated inside small wrapper classes.
- Save outputs after every expensive stage so experiments can resume.
- Do not regenerate a stage if its output already exists unless `force=True`.

Prefer:

```python
class ClaimAnalyzer:
    def analyze(self, claim):
        ...
```

instead of deeply nested schemas such as:

```python
class ClaimSchema(BaseModel):
    entities: List[EntitySchema]
    relations: List[RelationSchema]
    ...
```

Intermediate objects can simply look like:

```python
{
    "claim_id": 123,
    "claim": "...",
    "atoms": [...]
}
```

---

# 3. High-Level Architecture

```text
MOCHEG
  │
  ▼
DatasetLoader
  │
  ▼
ClaimAnalyzer
  │
  ├─────────────┐
  ▼             ▼
TextAnalyzer   ImageAnalyzer
  │             │
  └──────┬──────┘
         ▼
ConsistencyChecker
         │
         ▼
ProvenanceRetriever
         │
         ▼
EvidenceStore
         │
         ▼
FactChecker
         │
         ▼
ExplanationGenerator
         │
         ▼
Evaluator
```

The LLM should not receive only the original claim and raw evidence.

It should receive progressively structured evidence.

```text
raw evidence
      ↓
evidence facts
      ↓
claim/evidence consistency
      ↓
provenance
      ↓
final reasoning
```

---

# 4. Repository Structure

Use the following simple structure.

```text
mocheg-vlm/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evidence_db/
│
├── outputs/
│   ├── claims/
│   ├── images/
│   ├── texts/
│   ├── provenance/
│   ├── consistency/
│   ├── predictions/
│   └── evaluation/
│
├── src/
│   ├── dataset.py
│   ├── claim_analyzer.py
│   ├── image_analyzer.py
│   ├── text_analyzer.py
│   ├── consistency.py
│   ├── provenance.py
│   ├── evidence_store.py
│   ├── fact_checker.py
│   ├── explanation.py
│   ├── evaluator.py
│   ├── llm.py
│   ├── vlm.py
│   └── pipeline.py
│
├── scripts/
│   ├── run_claim_analysis.py
│   ├── run_image_analysis.py
│   ├── run_text_analysis.py
│   ├── run_provenance.py
│   ├── run_fact_checking.py
│   └── run_evaluation.py
│
├── prompts/
│   ├── claim_analysis.txt
│   ├── image_analysis.txt
│   ├── text_analysis.txt
│   ├── consistency.txt
│   ├── fact_check.txt
│   └── explanation.txt
│
├── config.py
├── requirements.txt
└── README.md
```

Do not create a complex Python package hierarchy yet.

One flat `src/` directory is easier to understand during research.

---

# 5. Module 1 — Dataset Loader

File:

```text
src/dataset.py
```

Class:

```python
class MochegDataset:
```

Responsibilities:

- load MOCHEG CSV files;
- associate rows with claims;
- retrieve ground-truth text evidence;
- retrieve associated image evidence;
- expose train/val/test samples.

MOCHEG's `Corpus2.csv` contains fields including:

- `Claim`
- `claim_id`
- `Evidence`
- `evidence_id`
- `cleaned_truthfulness`
- `ruling_outline`
- `Origin`
- `Snopes URL`

and MOCHEG has separate evidence image folders.

Minimal interface:

```python
class MochegDataset:
    def __init__(self, root):
        self.root = root

    def load_split(self, split):
        ...

    def get_claim(self, claim_id):
        ...

    def get_images(self, claim_id):
        ...
```

Return ordinary dictionaries.

Example:

```python
{
    "claim_id": "123",
    "claim": "...",
    "text_evidence": ["...", "..."],
    "images": ["path/a.jpg", "path/b.jpg"],
    "label": "support",
    "ruling_outline": "..."
}
```

Do not create a formal dataset schema.

---

# 6. Module 2 — Claim Analyzer

File:

```text
src/claim_analyzer.py
```

Class:

```python
class ClaimAnalyzer:
```

Purpose:

Convert one complex claim into atomic facts.

Example input:

```text
A photograph shows actor Tom Cruise sitting on top
of the Burj Khalifa without a harness.
```

Possible output:

```python
{
    "claim": "...",
    "atoms": [
        {
            "id": "C1",
            "text": "The photograph contains a person."
        },
        {
            "id": "C2",
            "text": "The person is Tom Cruise."
        },
        {
            "id": "C3",
            "text": "The person is sitting on the Burj Khalifa."
        },
        {
            "id": "C4",
            "text": "The person is not using a harness."
        }
    ],
    "entities": [
        "Tom Cruise",
        "Burj Khalifa"
    ]
}
```

Keep the output small.

Do not force the LLM to populate dozens of fields.

Minimal interface:

```python
class ClaimAnalyzer:
    def __init__(self, llm):
        self.llm = llm

    def analyze(self, claim):
        ...
```

Important instruction to the model:

> Separate directly verifiable facts from assumptions.

---

# 7. Module 3 — Image Analyzer

File:

```text
src/image_analyzer.py
```

Class:

```python
class ImageAnalyzer:
```

Use a VLM such as:

- GPT vision model;
- Gemini;
- Qwen-VL;
- InternVL;
- LLaVA;
- another model selected for experiments.

Keep the model implementation separate in:

```text
src/vlm.py
```

The image analyzer should extract observations rather than immediately classify the claim.

Output should remain simple:

```python
{
    "description": "...",
    "objects": [
        "person",
        "building"
    ],
    "text": [],
    "observations": [
        "A person is sitting on the outer structure of a tall building.",
        "No visible safety harness can be confidently identified."
    ]
}
```

Optionally add relationships:

```python
"relations": [
    ["person", "sitting_on", "building"]
]
```

Do not use a formal scene-graph schema initially.

A plain list is enough.

---

# 8. Critical Rule for Image Analysis

Explicitly distinguish:

```text
OBSERVED
```

from:

```text
INFERRED
```

Example:

Bad:

```text
Tom Cruise is sitting on the Burj Khalifa.
```

Better:

```python
{
    "observations": [
        "A man appears to be sitting on a skyscraper."
    ],
    "inferences": [
        "The skyscraper may be the Burj Khalifa."
    ]
}
```

This reduces VLM hallucination.

Another important distinction:

```text
"No harness is visible"
```

does not mean:

```text
"The person did not use a harness"
```

The framework should preserve this distinction throughout the pipeline.

---

# 9. Module 4 — Text Analyzer

File:

```text
src/text_analyzer.py
```

Class:

```python
class TextAnalyzer:
```

Purpose:

Extract factual statements from evidence paragraphs.

Example:

```python
{
    "facts": [
        "Special safety measures were used.",
        "Special camera mounts were created.",
        "The building is approximately 800 meters tall."
    ],
    "entities": [
        "Burj Khalifa"
    ]
}
```

Interface:

```python
class TextAnalyzer:
    def __init__(self, llm):
        self.llm = llm

    def analyze(self, text):
        ...
```

Do not ask the LLM for the final truthfulness label here.

This component should only understand the evidence.

---

# 10. Module 5 — Consistency Checker

File:

```text
src/consistency.py
```

Class:

```python
class ConsistencyChecker:
```

This is one of the central modules.

Compare each claim atom with:

- image observations;
- image inferences;
- text facts.

For every claim atom return:

```python
{
    "claim_atom": "The person is not using a harness.",
    "image": {
        "status": "unknown",
        "reason": "No harness is visible, but its absence cannot be established."
    },
    "text": {
        "status": "unknown",
        "reason": "The evidence mentions safety measures but does not establish whether a harness was used."
    }
}
```

Only allow three main statuses:

```text
support
contradict
unknown
```

Do not add unnecessary classes or labels.

Interface:

```python
class ConsistencyChecker:
    def __init__(self, llm):
        self.llm = llm

    def compare(self, claim_analysis, image_analysis, text_analysis):
        ...
```

---

# 11. Module 6 — Provenance Retriever

File:

```text
src/provenance.py
```

Class:

```python
class ProvenanceRetriever:
```

This module handles external evidence.

It should support two categories.

## Image provenance

Try to discover:

```text
original image source
earlier appearances
page title
page URL
publication date
image caption
event
location
people mentioned
```

Conceptual interface:

```python
class ProvenanceRetriever:
    def search_image(self, image_path):
        ...

    def search_text(self, query):
        ...
```

Do not tightly couple the system to Google Lens.

Create one simple provider method internally:

```python
def search_image(self, image_path):
    return self.provider.search(image_path)
```

This makes it possible to experiment with:

```text
Google-based image search
Bing Visual Search
TinEye
SerpAPI
custom web search
manual research
```

without changing the rest of the framework.

The output can simply be:

```python
{
    "results": [
        {
            "url": "...",
            "title": "...",
            "date": "...",
            "caption": "...",
            "text": "..."
        }
    ]
}
```

No formal provenance schema is necessary.

---

# 12. Provenance Should Answer Different Questions Than the VLM

The VLM answers:

```text
What is visible?
```

The provenance system answers:

```text
Where did this image come from?

When did it first appear?

What event did it originally represent?

What did the original source say about it?
```

This separation is essential.

Example:

```text
VLM:
There is flooding in the image.

Original source:
Photograph published in Florida in 2018.

Claim:
Photograph shows flooding in Texas in 2026.
```

Result:

```text
visual content:
    support

location:
    contradict

time:
    contradict
```

This is much stronger than global image–claim similarity.

---

# 13. Module 7 — Evidence Store

File:

```text
src/evidence_store.py
```

Class:

```python
class EvidenceStore:
```

Start with JSONL.

Do not start with PostgreSQL, Elasticsearch, Neo4j, or vector databases.

Example files:

```text
data/evidence_db/
    claim_evidence.jsonl
    image_provenance.jsonl
    text_provenance.jsonl
```

Interface:

```python
class EvidenceStore:
    def __init__(self, root):
        self.root = root

    def save(self, item):
        ...

    def load(self, claim_id):
        ...

    def exists(self, claim_id):
        ...
```

One record can contain:

```python
{
    "claim_id": "...",
    "claim": "...",
    "claim_analysis": {...},
    "text_analysis": [...],
    "image_analysis": [...],
    "provenance": {...},
    "consistency": [...]
}
```

This entire dictionary can simply be serialized with:

```python
json.dumps()
```

No schema validation is required initially.

---

# 14. Module 8 — Final Fact Checker

File:

```text
src/fact_checker.py
```

Class:

```python
class FactChecker:
```

The final fact checker receives:

```text
claim
+
claim atoms
+
consistency analysis
+
external provenance
```

It should preferably NOT receive huge raw documents unless necessary.

Interface:

```python
class FactChecker:
    def __init__(self, llm):
        self.llm = llm

    def verify(self, evidence):
        ...
```

Output:

```python
{
    "label": "refute",
    "confidence": 0.88,
    "reasoning": [
        {
            "claim_atom": "C1",
            "status": "support",
            "evidence": ["image_1"]
        },
        {
            "claim_atom": "C2",
            "status": "contradict",
            "evidence": ["provenance_1"]
        }
    ]
}
```

Do not allow arbitrary labels.

Normalize model responses to:

```text
support
refute
not_enough_information
```

These correspond to MOCHEG's truthfulness setting of support, refute, and not enough information.

---

# 15. Module 9 — Explanation Generator

File:

```text
src/explanation.py
```

Class:

```python
class ExplanationGenerator:
```

The explanation should be generated after the prediction.

Input:

```text
claim
prediction
claim atoms
supporting evidence
contradicting evidence
unknown evidence
provenance
```

Output:

```python
{
    "explanation": "...",
    "citations": [
        "text_1",
        "image_1",
        "provenance_2"
    ]
}
```

Prompt rules:

```text
1. Explain the verdict using only supplied evidence.

2. Separate visual observations from external facts.

3. Do not claim that something is absent merely because
   it is not visible.

4. Mention contradictory evidence when it determines
   the verdict.

5. Do not introduce external facts that are absent from
   the evidence.

6. Prefer concise reasoning over long chain-of-thought.

7. Cite the evidence identifier associated with each
   important statement.
```

MOCHEG's `ruling_outline` is especially useful here because it serves as ground truth for explanation generation.

---

# 16. Module 10 — Model Wrappers

## LLM wrapper

File:

```text
src/llm.py
```

Class:

```python
class LLM:
```

Keep it very small.

```python
class LLM:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt):
        ...
```

Do not make separate model classes for every task.

The same wrapper can be reused by:

```text
ClaimAnalyzer
TextAnalyzer
ConsistencyChecker
FactChecker
ExplanationGenerator
```

---

## VLM wrapper

File:

```text
src/vlm.py
```

Class:

```python
class VLM:
```

Minimal interface:

```python
class VLM:
    def __init__(self, model):
        self.model = model

    def generate(self, image, prompt):
        ...
```

This makes model replacement easy.

---

# 17. Module 11 — Pipeline

File:

```text
src/pipeline.py
```

Class:

```python
class FactCheckingPipeline:
```

This module simply connects everything.

Example structure:

```python
class FactCheckingPipeline:
    def __init__(
        self,
        dataset,
        claim_analyzer,
        image_analyzer,
        text_analyzer,
        consistency_checker,
        provenance_retriever,
        evidence_store,
        fact_checker,
        explanation_generator,
    ):
        ...
```

Main method:

```python
def run(self, sample):
    claim = self.claim_analyzer.analyze(sample["claim"])

    texts = [
        self.text_analyzer.analyze(x)
        for x in sample["text_evidence"]
    ]

    images = [
        self.image_analyzer.analyze(x)
        for x in sample["images"]
    ]

    consistency = self.consistency_checker.compare(
        claim,
        images,
        texts
    )

    provenance = self.provenance_retriever.search(...)

    evidence = {
        "claim_id": sample["claim_id"],
        "claim": sample["claim"],
        "claim_analysis": claim,
        "texts": texts,
        "images": images,
        "consistency": consistency,
        "provenance": provenance
    }

    self.evidence_store.save(evidence)

    prediction = self.fact_checker.verify(evidence)

    explanation = self.explanation_generator.generate(
        evidence,
        prediction
    )

    return {
        **evidence,
        "prediction": prediction,
        "explanation": explanation
    }
```

Keep `pipeline.py` boring.

All intelligence should live in the individual modules.

---

# 18. Processing Pipeline

Run the framework in stages rather than repeatedly running the whole pipeline.

## Stage 1

```text
MOCHEG
 ↓
ClaimAnalyzer
 ↓
outputs/claims/
```

Save:

```text
{claim_id}.json
```

---

## Stage 2

```text
MOCHEG images
 ↓
ImageAnalyzer
 ↓
outputs/images/
```

One image should only be analyzed once.

---

## Stage 3

```text
MOCHEG text evidence
 ↓
TextAnalyzer
 ↓
outputs/texts/
```

---

## Stage 4

```text
claim analysis
+
image analysis
+
text analysis
 ↓
ConsistencyChecker
 ↓
outputs/consistency/
```

---

## Stage 5

```text
images
+
claim
 ↓
ProvenanceRetriever
 ↓
outputs/provenance/
```

This step is likely expensive and slow.

Cache everything.

---

## Stage 6

```text
all previous outputs
 ↓
FactChecker
 ↓
outputs/predictions/
```

---

## Stage 7

```text
prediction
+
evidence
 ↓
ExplanationGenerator
 ↓
final output
```

---

# 19. Suggested Final Output

One JSON file per claim is easiest during development.

Example:

```json
{
  "claim_id": "5",
  "claim": "A photograph shows ...",
  "claim_analysis": {},
  "text_analysis": [],
  "image_analysis": [],
  "provenance": {},
  "consistency": [],
  "prediction": {
    "label": "support"
  },
  "explanation": {
    "text": "...",
    "citations": []
  },
  "ground_truth": {
    "label": "support",
    "ruling_outline": "..."
  }
}
```

This is not a data schema.

It is simply the dictionary produced by the pipeline.

---

# 20. Evaluation Module

File:

```text
src/evaluator.py
```

Class:

```python
class Evaluator:
```

Start simple.

## Classification

Calculate:

```text
accuracy
macro F1
support F1
refute F1
NEI F1
```

---

## Retrieval

MOCHEG provides sentence-level text qrels, article-level text qrels, and image qrels.

Later evaluate:

```text
Recall@1
Recall@5
Recall@10
MRR
nDCG
```

---

## Explanation

Initially use:

```text
ROUGE
BERTScore
```

against `ruling_outline`.

Then add a separate LLM-based evaluation for:

```text
faithfulness
evidence coverage
hallucination
correctness
```

---

# 21. Add Evidence Grounding Evaluation

A particularly useful custom metric is:

```text
Evidence Grounding Score
```

For every factual statement generated in the explanation:

```text
Does one of the cited evidence items support this statement?
```

Calculate:

```text
supported explanation statements
--------------------------------
all factual explanation statements
```

Example:

```text
8 factual statements
7 supported by provided evidence

Grounding Score = 7 / 8 = 0.875
```

This fits the main research objective better than only ROUGE.

---

# 22. Experiment Configuration

Do not build YAML configuration initially.

Use:

```text
config.py
```

Example:

```python
DATA_ROOT = "data/raw/MOCHEG"
OUTPUT_ROOT = "outputs"

LLM_MODEL = "..."
VLM_MODEL = "..."

USE_PROVENANCE = True
USE_IMAGE = True
USE_TEXT = True

MAX_CLAIMS = None
```

Simple Python variables are enough.

---

# 23. Command-Line Scripts

Scripts should be intentionally tiny.

Example:

```text
scripts/run_claim_analysis.py
```

```python
from src.dataset import MochegDataset
from src.claim_analyzer import ClaimAnalyzer
from src.llm import LLM

dataset = MochegDataset("data/raw/MOCHEG")
llm = LLM("model")
analyzer = ClaimAnalyzer(llm)

for sample in dataset.load_split("train"):
    result = analyzer.analyze(sample["claim"])
    ...
```

Do not introduce Hydra or other experiment-management frameworks at the beginning.

---

# 24. Prompt Storage

Do not hard-code long prompts inside classes.

Store them under:

```text
prompts/
```

Classes can simply do:

```python
prompt = open("prompts/claim_analysis.txt").read()
```

or use one small helper.

This makes prompt experiments much easier.

---

# 25. JSON Parsing Strategy

LLMs sometimes produce invalid JSON.

Do not introduce complex structured-output frameworks immediately.

Use the simplest possible approach:

```python
text = llm.generate(prompt)

try:
    result = json.loads(text)
except:
    result = {
        "raw_output": text
    }
```

For tasks requiring reliable parsing, prompt the model explicitly:

```text
Return only valid JSON.
Do not use Markdown.
```

Only add stronger structured-output APIs if this becomes a real problem.

---

# 26. Development Order

Do not implement everything simultaneously.

## Phase 1 — Dataset

Implement:

```text
dataset.py
```

Goal:

```text
load one MOCHEG sample correctly
```

Verify:

```python
sample["claim"]
sample["text_evidence"]
sample["images"]
sample["label"]
sample["ruling_outline"]
```

---

## Phase 2 — Basic VLM baseline

Implement:

```text
vlm.py
image_analyzer.py
```

Goal:

```text
image → observations
```

Do not add object detection models yet.

First see what the VLM can extract itself.

---

## Phase 3 — Claim decomposition

Implement:

```text
llm.py
claim_analyzer.py
```

Goal:

```text
claim → atomic claims
```

---

## Phase 4 — Text extraction

Implement:

```text
text_analyzer.py
```

Goal:

```text
evidence paragraph → factual statements
```

---

## Phase 5 — Consistency

Implement:

```text
consistency.py
```

Goal:

```text
claim atom
       ↕
image observations
       ↕
text facts
```

Return:

```text
support / contradict / unknown
```

---

## Phase 6 — Simple final verifier

Implement:

```text
fact_checker.py
```

At this point you already have the first complete experimental system:

```text
claim
+
MOCHEG image evidence
+
MOCHEG text evidence
 ↓
structured extraction
 ↓
consistency
 ↓
LLM
 ↓
prediction
```

Run this before implementing external retrieval.

This becomes your main baseline.

---

# 27. Phase 7 — Explanation

Implement:

```text
explanation.py
```

Compare generated explanation with MOCHEG:

```text
ruling_outline
```

At this stage the framework performs the three tasks MOCHEG is designed around:

```text
evidence understanding
claim verification
explanation generation
```

MOCHEG explicitly describes multimodal evidence retrieval, claim verification, and explanation generation as its intended tasks.

---

# 28. Phase 8 — Provenance Retrieval

Only after the baseline works, add:

```text
provenance.py
```

Start with image evidence.

For each image:

```text
image
 ↓
reverse-image search
 ↓
top K web pages
 ↓
page title
date
caption
surrounding text
 ↓
LLM extraction
 ↓
provenance facts
```

Example:

```python
{
    "first_seen": "2018-09-12",
    "location": "Florida",
    "event": "Hurricane ...",
    "sources": [...]
}
```

Then rerun the fact checker with this additional evidence.

This creates a clean ablation:

```text
without provenance
vs.
with provenance
```

---

# 29. Phase 9 — Evidence Database

Once provenance retrieval works reliably, store its results.

Start with:

```text
JSONL
```

Later, if necessary:

```text
SQLite
```

Only introduce a vector database if you eventually need cross-claim retrieval over thousands of accumulated evidence items.

Do not add it simply because the project uses LLMs.

---

# 30. Important Ablation Experiments

Your implementation should make these experiments easy.

## Experiment A

```text
Text only
```

## Experiment B

```text
Image only
```

## Experiment C

```text
Image + text
```

## Experiment D

```text
Image + text + claim decomposition
```

## Experiment E

```text
Image + text + claim decomposition + consistency
```

## Experiment F

```text
Image + text + structured consistency + provenance
```

## Experiment G

```text
Full framework + evidence-grounded explanation
```

This gives a direct answer to:

> Which component actually improves fact-checking?

---

# 31. Avoid Adding Separate Object Detection Initially

Although the original idea included object detection, do not start with YOLO/DETR/etc.

First:

```text
Image
 ↓
VLM
 ↓
objects + relationships + observations
```

Then evaluate its failures.

Only add a dedicated object detector when experiments show that the VLM systematically misses important objects.

This keeps the first implementation considerably simpler.

The same rule applies to:

```text
OCR
face recognition
geolocation
chart extraction
NER
temporal extraction
```

Use the VLM/LLM first.

Add specialist tools only when an error analysis demonstrates the need.

---

# 32. Error Analysis

Create a simple output field:

```python
"errors": []
```

After running validation data, manually classify failures as:

```text
claim decomposition error

image perception error

entity identification error

OCR error

text interpretation error

retrieval error

provenance error

cross-modal consistency error

reasoning error

hallucinated explanation

insufficient evidence
```

This error analysis should determine which additional tools to add.

Do not add tools before demonstrating that they solve an observed failure class.

---

# 33. Important Research Separation

Keep three concepts separate in code.

## Visual observation

```text
A man is standing beside another man.
```

## Semantic inference

```text
The second man may be a Taliban official.
```

## Provenance-backed fact

```text
The original source identifies the second man as
Abdul Ghani Baradar.
```

Representing these separately makes the explanation substantially more trustworthy.

---

# 34. Final Reasoning Input

The final LLM prompt should resemble:

```text
CLAIM:
...

CLAIM COMPONENTS:
[C1] ...
[C2] ...
[C3] ...

IMAGE OBSERVATIONS:
[I1] ...
[I2] ...

TEXT FACTS:
[T1] ...
[T2] ...

PROVENANCE:
[P1] ...
[P2] ...

CONSISTENCY:
C1 -> support from T1
C2 -> unknown from I1
C3 -> contradiction from P2

Determine the overall verdict.

Allowed labels:
support
refute
not_enough_information

Use only the supplied evidence.
```

This is preferable to:

```text
Here is an image, text and claim. Is it true?
```

because the reasoning process becomes observable and experimentally measurable.

---

# 35. Target Final Pipeline

The eventual system should be:

```text
                       Claim
                         │
                         ▼
                  ClaimAnalyzer
                         │
                   atomic claims
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
    TextAnalyzer                ImageAnalyzer
           │                           │
      text facts               visual facts
           │                           │
           └─────────────┬─────────────┘
                         │
                         ▼
                ConsistencyChecker
                         │
                         ▼
                 initial evidence
                         │
                         ▼
               ProvenanceRetriever
                         │
                 external evidence
                         │
                         ▼
                  EvidenceStore
                         │
                         ▼
                    FactChecker
                         │
                SUPPORT / REFUTE /
                     NEI
                         │
                         ▼
               ExplanationGenerator
                         │
                         ▼
              grounded explanation
```

---

# 36. Recommended First Milestone

Do **not** start with web/reverse-image search.

The first milestone should contain only:

```text
dataset.py
llm.py
vlm.py
claim_analyzer.py
image_analyzer.py
text_analyzer.py
consistency.py
fact_checker.py
explanation.py
pipeline.py
```

and should answer:

> Does explicit claim decomposition and claim–evidence consistency improve VLM-based multimodal fact-checking on MOCHEG?

Once this baseline runs reliably, add:

```text
provenance.py
evidence_store.py
```

and answer the second research question:

> Does external provenance improve fact-checking accuracy and explanation faithfulness?

This produces a much cleaner research progression than implementing every proposed component at once.

---

# 37. Instructions to Give Codex

Implement the project incrementally.

For every module:

1. Create only the requested file.
2. Keep each class small.
3. Use plain Python objects and dictionaries.
4. Do not introduce Pydantic or formal schemas.
5. Do not add abstractions unless they are immediately needed.
6. Prefer readable code over reusable framework code.
7. Keep each major module in one file.
8. Save expensive model outputs to JSON.
9. Reuse cached outputs whenever possible.
10. Include a small `if __name__ == "__main__":` example when useful.
11. Do not implement future modules early.
12. Do not silently modify MOCHEG fields.
13. Keep MOCHEG ground truth separate from model-generated fields.
14. Never expose `cleaned_truthfulness` or `ruling_outline` to the model during prediction.
15. Only use them during evaluation.
16. Make model and API wrappers minimal and replaceable.
17. Handle failure by saving the raw model output rather than crashing the entire experiment.
18. Log `claim_id` for every operation.
19. Process a small number of examples first.
20. Optimize only after the full baseline works.

The most important implementation rule is:

```text
KEEP THE PIPELINE TRANSPARENT.
```

At every stage we should be able to inspect:

```text
what the claim analyzer extracted
what the VLM observed
what the text analyzer extracted
what evidence supports each claim atom
what evidence contradicts it
what remains unknown
what provenance was retrieved
why the final model selected its verdict
which evidence supports the generated explanation
```

That transparency should be treated as part of the research contribution, not merely an implementation detail.