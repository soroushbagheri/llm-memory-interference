# References & Related Work

## Positioning Statement

Memory Interference Forensics addresses a **novel gap** at the intersection of LLM safety, interpretability, and dialogue systems. While related work exists in adjacent areas, **no prior work systematically detects and mitigates context interference in real-time during LLM inference.**

---

## Core Problem: Context Interference in LLMs

### Our Contribution

**What we introduce:**
1. **Interference Detection Score (IDS):** First metric specifically designed to quantify context interference
2. **Real-time mitigation:** Inference-time intervention without model retraining
3. **Causal tracing for interference:** Gradient-based identification of interfering tokens
4. **Two-tier framework:** Combines attention analysis + semantic conflict + behavioral signals

**What makes it novel:**
- Prior work focuses on *what to retrieve* (RAG), we focus on *what to forget*
- Existing methods assume more context = better; we show when less is more
- First system designed specifically for conversational interference patterns

---

## Related Work

### 1. Retrieval-Augmented Generation (RAG)

#### Key Papers

**Lewis et al. (2020) - RAG: Retrieval-Augmented Generation**
- *What they do:* Retrieve relevant documents to augment LLM context
- *Gap:* No mechanism to detect when retrieved context interferes
- *Our difference:* We address harmful context, not missing context

**Adaptive RAG (Zhang et al., 2024)**
- *What they do:* Dynamically adjust retrieval based on query complexity
- *Gap:* Focuses on "how many docs to retrieve," not "which prior context to remove"
- *Our difference:* We handle interference from conversation history, not external retrieval

**Uplift-RAG (Chen et al., 2025)**  
[https://aclanthology.org/2025.findings-emnlp.511/](https://aclanthology.org/2025.findings-emnlp.511/)
- *What they do:* Go beyond fixed top-k retrieval, use preference alignment
- *Gap:* Still additive (what to include), not subtractive (what to exclude)
- *Our difference:* We prune interfering context from existing conversation

---

### 2. LLM Uncertainty & Confidence Calibration

#### Key Papers

**Guo et al. (2017) - On Calibration of Modern Neural Networks**
- *What they do:* Measure and improve confidence calibration in neural nets
- *Gap:* Post-hoc calibration, doesn't detect specific interference sources
- *Our difference:* Real-time detection with token-level attribution

**Kuhn et al. (2023) - Semantic Uncertainty in LLMs**
- *What they do:* Measure uncertainty via semantic consistency across samples
- *Gap:* Identifies *when* model is uncertain, not *why* (interference)
- *Our difference:* We trace uncertainty to specific context segments

**Monte Carlo Temperature Sampling (2025)**  
[https://aclanthology.org/2025.trustnlp-main.21.pdf](https://aclanthology.org/2025.trustnlp-main.21.pdf)
- *What they do:* Robust sampling to handle LLM output variance
- *Gap:* Treats variance as noise, doesn't identify systematic interference
- *Our difference:* We distinguish random variance from context-driven errors

---

### 3. LLM Interpretability & Causal Tracing

#### Key Papers

**Meng et al. (2023) - Locating and Editing Factual Associations**
- *What they do:* Trace which neurons encode specific facts
- *Gap:* Static analysis for single facts, not dynamic conversation interference
- *Our difference:* Real-time tracing for multi-turn dialogue

**Anthropic (2023) - Towards Monosemanticity**
- *What they do:* Decompose model activations into interpretable features
- *Gap:* Focuses on feature discovery, not interference mitigation
- *Our difference:* Applied directly to reducing errors in production systems

**Attention Flow Analysis (Abnar & Zuidema, 2020)**
- *What they do:* Visualize attention propagation across layers
- *Gap:* Descriptive analysis, no intervention
- *Our difference:* Use attention divergence as signal for selective masking

---

### 4. Context Management in Dialogue Systems

#### Key Papers

**Context-Aware Neural Dialogue (Serban et al., 2016)**
- *What they do:* Model dialogue context hierarchically
- *Gap:* Assumes all context is helpful, no conflict detection
- *Our difference:* Explicitly model and mitigate interference

**Long-Form QA with Memory (Karpukhin et al., 2020)**
- *What they do:* Manage long contexts via chunking and retrieval
- *Gap:* Structural solution (sliding window), not content-aware
- *Our difference:* Semantic analysis determines what to keep/remove

---

### 5. Behavioral Biases in LLMs

#### Key Papers

**Jia et al. (2024) - Behavioral Economics in LLM Decision-Making**
- *What they do:* Show LLMs exhibit risk aversion, framing effects
- *Gap:* Evaluation framework, not mitigation system
- *Our difference:* Detect and correct interference that causes biased decisions

**Large Language Models Show Amplified Cognitive Biases (2025)**  
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12207438/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12207438/)
- *What they do:* Document systematic biases in LLM reasoning
- *Gap:* Observational study, no intervention
- *Our difference:* Active mitigation when prior context induces bias

**Consumer Choice Experiments for AI Agents (Cherep et al., 2025)**
- *What they do:* Show LLMs are sensitive to choice architectures (like humans)
- *Gap:* Demonstrates phenomenon, doesn't provide solution
- *Our difference:* System to detect and prevent choice-architecture-induced errors

---

### 6. Satisficing & Bounded Rationality in AI

#### Key Papers

**SITAlign (Chehade et al., 2025) - Satisficing Alignment**
- *What they do:* Apply satisficing thresholds during decoding to reduce compute
- *Gap:* Focuses on token generation, not context filtering
- *Our difference:* Operates on conversation history before generation
- *Complementary:* Could combine SITAlign decoding with our context mitigation

**CLAI (Zhang et al., 2025) - Cognitive Load-Aware Inference**
- *What they do:* Reduce reasoning steps when query is simple
- *Gap:* Addresses computational cost, not correctness from interference
- *Our difference:* Improves accuracy by removing harmful context

---

## What Doesn't Exist (Our Novel Contribution)

### 1. Real-Time Interference Detection

**Existing:** Post-hoc analysis of errors, offline interpretability studies  
**Missing:** System that detects interference during inference  
**We provide:** IDS metric computed in real-time before response generation

### 2. Subtractive Context Management

**Existing:** Retrieval systems decide what to add to context  
**Missing:** Systems that decide what to remove from context  
**We provide:** Selective amnesia via gradient-based causal tracing

### 3. Unified Framework for Interference Types

**Existing:** Separate solutions for semantic, lexical, structural issues  
**Missing:** Single framework handling all interference types  
**We provide:** IDS combines attention + semantics + behavioral signals

### 4. Conversational Interference Benchmark

**Existing:** Benchmarks for hallucination, bias, factuality  
**Missing:** Benchmark specifically for context interference  
**We provide:** 1000-example dataset with labeled interference types

---

## Comparison Table

| Work | Detects Interference? | Real-Time? | Mitigates? | Conversational? |
|------|----------------------|------------|------------|------------------|
| RAG (Lewis 2020) | ❌ | ✅ | ❌ | ❌ |
| Adaptive RAG | ❌ | ✅ | ❌ | ❌ |
| Semantic Uncertainty | Partial | ✅ | ❌ | ❌ |
| Causal Tracing (Meng) | ❌ | ❌ | ❌ | ❌ |
| SITAlign | ❌ | ✅ | ❌ | ❌ |
| CLAI | ❌ | ✅ | ❌ | Partial |
| Behavioral Bias Studies | ❌ | ❌ | ❌ | ❌ |
| **Memory Interference (Ours)** | **✅** | **✅** | **✅** | **✅** |

---

## Key Differentiators

### 1. We Focus on FORGETTING, Not Remembering

**Most LLM research:** "How do we give the model more relevant information?"  
**Our research:** "How do we remove harmful information already in context?"

### 2. We Address Conversational Dynamics

**Most work:** Single-turn QA or document retrieval  
**Our focus:** Multi-turn conversations where history creates interference

### 3. We Provide Actionable Intervention

**Most interpretability:** "Here's why the model failed"  
**Our system:** "Here's why it failed, and we fixed it before responding"

### 4. We Target Production Deployment

**Most research:** Offline analysis or training-time solutions  
**Our design:** Real-time inference-time intervention (works with any LLM)

---

## Inspiration from Cognitive Psychology

### Human Memory Interference

**Proactive Interference:** Old memories interfere with new learning  
→ *LLM equivalent:* Previous "Python = snake" discussion interferes with coding questions

**Retroactive Interference:** New information disrupts recall of old  
→ *LLM equivalent:* Recent context overwrites earlier correct reasoning

**Part-Set Cueing Effect:** Presenting subset of items harms recall of full set  
→ *LLM equivalent:* Partial context can be worse than no context

**Classic Studies:**
- Underwood (1957): Proactive interference in verbal learning
- Wickens (1970): Release from proactive interference via context shift
- Anderson & Neely (1996): Retrieval-induced forgetting

**Our translation:** Apply cognitive psychology principles to LLM context management

---

## Open Questions & Future Directions

### 1. Optimal Forgetting Policy

**Question:** When should we remove context vs. keeping and clarifying?  
**Current:** Threshold-based removal when IDS > 0.7  
**Future:** Reinforcement learning to learn optimal policy

### 2. Cross-Model Generalization

**Question:** Does IDS trained on GPT-4 generalize to Llama/Claude?  
**Current:** Preliminary evidence suggests yes (Exp 5)  
**Future:** Systematic transfer learning study

### 3. Interference in Multi-Modal LLMs

**Question:** Does visual context interfere with text, or vice versa?  
**Current:** Text-only analysis  
**Future:** Extend to GPT-4V, Gemini-Pro-Vision

### 4. User-Specific Interference Patterns

**Question:** Do different users create different interference patterns?  
**Current:** Population-level IDS  
**Future:** Personalized interference detection

---

## Recommended Reading Order

**For understanding our problem:**
1. Cherep et al. (2025) - Consumer choice experiments
2. Jia et al. (2024) - Behavioral biases in LLMs
3. Cognitive psych classics on memory interference

**For understanding our method:**
1. Meng et al. (2023) - Causal tracing foundations
2. Kuhn et al. (2023) - Semantic uncertainty
3. Attention flow analysis papers

**For understanding our positioning:**
1. RAG evolution: Lewis (2020) → Adaptive RAG → Uplift-RAG
2. SITAlign & CLAI for inference-time control context
3. LLM interpretability surveys (Doshi-Velez & Kim, 2017)

---

## Citation

```bibtex
@article{bagheri2026memory,
  title={Memory Interference Forensics: Detecting and Mitigating Context Interference in Large Language Models},
  author={Bagheri, Soroush},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026},
  note={Code: https://github.com/soroushbagheri/llm-memory-interference}
}
```

---

**Last Updated:** January 2026  
**Maintainer:** Soroush Bagheri ([@soroushbagheri](https://github.com/soroushbagheri))
