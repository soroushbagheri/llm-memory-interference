# Methodology

## Memory Interference Forensics: Technical Details

### Overview

This document provides a detailed technical description of the memory interference detection, tracing, and mitigation algorithms.

---

## 1. Interference Detection Score (IDS)

### Definition

The **Interference Detection Score (IDS)** is a composite metric that quantifies the degree to which prior conversation context interferes with current reasoning.

### Formula

```
IDS = w₁ · AD + w₂ · SC + w₃ · RI
```

Where:
- **AD**: Attention Divergence [0, 1]
- **SC**: Semantic Conflict [0, 1]
- **RI**: Response Inconsistency [0, 1]
- **w₁, w₂, w₃**: Component weights (default: 0.40, 0.35, 0.25)

### Components

#### 1.1 Attention Divergence (AD)

**Intuition:** Measures how "distracted" the model is by past context.

**Computation:**

1. Embed current query: `q = Embed(query)`
2. Embed each context turn: `c₁, c₂, ..., cₙ = Embed(context)`
3. Compute attention-like weights:
   ```
   αᵢ = softmax(sim(q, cᵢ))
   ```
4. Compute entropy of attention distribution:
   ```
   AD = H(α) / log(n)
   ```
   Where `H(α) = -Σ αᵢ log(αᵢ)`

**Interpretation:**
- **Low AD (< 0.3)**: Attention focused on recent/relevant context
- **High AD (> 0.7)**: Attention spread across many past turns (potential interference)

#### 1.2 Semantic Conflict (SC)

**Intuition:** Detects contradictory or conflicting information in context.

**Computation:**

1. Compute pairwise similarities between all context turns
2. Identify low-similarity pairs (potential conflicts)
3. Aggregate:
   ```
   SC = (1 - mean_sim) · (1 - min_sim)
   ```

**Interpretation:**
- **Low SC (< 0.3)**: Context is coherent
- **High SC (> 0.6)**: Context contains contradictions

#### 1.3 Response Inconsistency (RI)

**Intuition:** Measures whether context causes the model to produce variable/inconsistent outputs.

**Computation:**

1. Generate `k` responses with full context: `R_with = {r₁, ..., rₖ}`
2. Generate `k` responses without context (query only): `R_without = {r₁', ..., rₖ'}`
3. Compute embedding variance:
   ```
   Var_with = Var(Embed(R_with))
   Var_without = Var(Embed(R_without))
   ```
4. Inconsistency ratio:
   ```
   RI = min(Var_with / Var_without, 2.0) / 2.0
   ```

**Interpretation:**
- **Low RI (< 0.3)**: Context stabilizes responses
- **High RI (> 0.6)**: Context introduces variance (interference)

---

## 2. Causal Tracing

### Objective

Identify **which specific tokens** from conversation history are causing interference.

### Gradient-Based Attribution

**Algorithm:**

1. Define loss as distance between generated response and ideal response:
   ```
   L(response, ground_truth) = ||Embed(response) - Embed(ground_truth)||²
   ```

2. Compute gradient with respect to each context token:
   ```
   Attribution(token_i) = ||∇_{token_i} L||₂
   ```

3. Rank tokens by attribution score

**Approximation (when gradients unavailable):**

Use embedding-based projection:
```
Attribution(token) = |⟨Embed(token), direction⟩|
```
Where `direction = Embed(response) - Embed(query)`

### Attention-Based Attribution

When model attention weights are accessible:

```
Attribution(token_i) = Σ_layer Attention_weight(query → token_i)
```

---

## 3. Mitigation Strategies

### 3.1 Soft Masking

**Method:** Replace interfering tokens with mask tokens

**Algorithm:**
```python
for turn in interfering_turns:
    tokens = tokenize(turn.content)
    n_mask = int(len(tokens) * mask_ratio)
    tokens[:n_mask] = [MASK_TOKEN] * n_mask
    turn.content = detokenize(tokens)
```

**When to use:** Minor conflicts, want to preserve some context

### 3.2 Hard Masking

**Method:** Completely remove interfering turns

**Algorithm:**
```python
modified_context = [turn for i, turn in enumerate(context)
                    if i not in interfering_turns]
```

**When to use:** Major contradictions, high IDS (> 0.7)

### 3.3 Context Reordering

**Method:** Move interfering content away from current query

**Algorithm:**
```python
non_interfering = [turn for i, turn in enumerate(context)
                   if i not in interfering_turns]
interfering = [turn for i, turn in enumerate(context)
               if i in interfering_turns]
modified_context = non_interfering + interfering
```

**When to use:** Recency bias effects, structural interference

---

## 4. Evaluation Metrics

### 4.1 Interference Detection Rate (IDR)

```
IDR = TP / (TP + FN)
```

Where:
- TP = True positives (correctly identified interfering turns)
- FN = False negatives (missed interfering turns)

**Target:** IDR > 80%

### 4.2 False Positive Rate (FPR)

```
FPR = FP / (FP + TN)
```

**Target:** FPR < 10%

### 4.3 Accuracy Improvement

```
ΔAcc = Accuracy_after_mitigation - Accuracy_before_mitigation
```

**Target:** ΔAcc > 15%

### 4.4 Computational Overhead

```
Overhead = (Latency_with_detection / Latency_baseline) - 1
```

**Target:** Overhead < 30%

---

## 5. Baseline Comparisons

| Baseline | Description | Expected Performance |
|----------|-------------|-----------------------|
| **No Intervention** | Full context, no filtering | Accuracy: ~70%, FPR: N/A |
| **Fixed Window (N=5)** | Use only last 5 turns | Accuracy: ~75%, IDR: ~45% |
| **Random Masking** | Randomly remove turns | Accuracy: ~72%, IDR: ~50% |
| **Our Method** | IDS + adaptive mitigation | **Accuracy: ~85%, IDR: ~80%** |

---

## 6. Implementation Notes

### Hyperparameters

- **IDS threshold**: 0.5 (tune on validation set)
- **n_samples for RI**: 5 (balance between accuracy and cost)
- **mask_ratio**: 0.5 (50% token masking)
- **Attention divergence weight**: 0.40
- **Semantic conflict weight**: 0.35
- **Response inconsistency weight**: 0.25

### Computational Complexity

- **Detector**: O(n² + k·m) where n=context turns, k=samples, m=model inference
- **Tracer**: O(n·d) where d=embedding dimension
- **Mitigator**: O(n)

### API Costs (Estimated)

- **Detection per query**: ~$0.01 (with GPT-4)
- **Full benchmark (1000 samples)**: ~$10-15

---

## References

1. Attention mechanisms in transformers (Vaswani et al., 2017)
2. Gradient-based attribution (Sundararajan et al., 2017)
3. Self-consistency for LLMs (Wang et al., 2023)
4. Context management in dialogue systems (Zhang et al., 2023)

---

**Last Updated:** January 2026  
**Author:** Soroush Bagheri
