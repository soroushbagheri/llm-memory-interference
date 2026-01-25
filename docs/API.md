# API Reference

## Core Classes

### `InterferenceDetector`

**Purpose:** Detects when prior context interferes with current reasoning.

```python
from src.detector import InterferenceDetector

detector = InterferenceDetector(
    weights={'attention_divergence': 0.4, 'semantic_conflict': 0.35, 'response_inconsistency': 0.25},
    threshold=0.5,
    n_samples=5
)

ids_score, result = detector.detect(
    context=[{"role": "user", "content": "..."}],
    query="Current question",
    llm=llm_adapter
)
```

**Returns:**
- `ids_score` (float): Interference Detection Score [0, 1]
- `result` (InterferenceResult): Detailed results including:
  - `attention_divergence`
  - `semantic_conflict`
  - `response_inconsistency`
  - `interfering_turns`
  - `interfering_tokens`
  - `recommendation`

---

### `CausalTracer`

**Purpose:** Traces which tokens cause interference.

```python
from src.tracer import CausalTracer

tracer = CausalTracer(attribution_method="gradient")

attributions = tracer.trace(
    context=context,
    query="Question",
    response="Model's answer",
    llm=llm_adapter,
    top_k=10
)
```

**Returns:** Dictionary mapping turn indices to (token, attribution_score) pairs.

---

### `InterferenceMitigator`

**Purpose:** Applies intervention strategies.

```python
from src.mitigator import InterferenceMitigator

mitigator = InterferenceMitigator(default_strategy="soft_masking")

result = mitigator.mitigate(
    context=context,
    interfering_turns=[0, 2],
    strategy="soft_masking",
    mask_ratio=0.5
)
```

**Returns:** `MitigationResult` with:
- `modified_context`
- `removed_turns`
- `masked_tokens`
- `metadata`

---

### `LLMAdapter`

**Purpose:** Unified interface to LLM APIs.

```python
from src.llm_adapter import LLMAdapter

# OpenAI
llm = LLMAdapter(model="gpt-4-turbo", temperature=0.7)

# Anthropic
llm = LLMAdapter(model="claude-3-5-sonnet-20241022")

# Mock (no API)
llm = LLMAdapter(model="mock")

# Generate response
response = llm.generate(messages=[{"role": "user", "content": "Hello"}])

# Get embedding
embedding = llm.get_embedding("Some text")
```

---

## Metrics Functions

### `compute_ids()`

```python
from src.metrics import compute_ids

ids = compute_ids(
    attention_divergence=0.6,
    semantic_conflict=0.4,
    response_inconsistency=0.5
)
```

### `evaluate_detection()`

```python
from src.metrics import evaluate_detection

metrics = evaluate_detection(
    predicted_interfering=[0, 2, 5],
    true_interfering=[0, 1, 5],
    total_turns=10
)
# Returns: {precision, recall, f1, accuracy, tp, fp, fn, tn}
```

---

## Utility Functions

### `load_config()`

```python
from src.utils import load_config

config = load_config("configs/default.yaml")
```

### `save_json()` / `load_json()`

```python
from src.utils import save_json, load_json

save_json(data, "output.json")
data = load_json("input.json")
```

---

## Command-Line Interface

### Run Demo

```bash
python experiments/demo.py
```

### Run Benchmark

```bash
python experiments/run_benchmark.py \
    --dataset data/processed/dataset.json \
    --model gpt-4-turbo \
    --output results/benchmark
```

---

**For more examples, see the `examples/` directory.**
