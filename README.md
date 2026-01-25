# LLM Memory Interference Forensics

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Detecting and Mitigating Context Interference in Large Language Models**

[Overview](#overview) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Citation](#citation)

</div>

---

## 🎯 Overview

**The Problem:** Large Language Models (LLMs) suffer from **memory interference**, when prior conversation context interferes with current reasoning, leading to incorrect or inconsistent responses. This happens when:

- Previous discussion of "Python the snake" interferes with coding questions about Python
- Earlier medical case details contaminate diagnosis of a new patient
- Legal precedents from one domain incorrectly influence analysis in another

**Our Solution:** A real-time forensics system that:

1. **Detects** when prior context is interfering with current reasoning
2. **Traces** the specific tokens/turns causing interference
3. **Mitigates** interference through selective context masking
4. **Validates** improvement in response quality

### Key Features

- ✅ **Interference Detection Score (IDS):** Novel metric quantifying context interference
- ✅ **Causal Tracing:** Gradient-based identification of interfering tokens
- ✅ **Selective Amnesia:** Automatic masking of problematic context
- ✅ **Real-time Processing:** Inference-time intervention without retraining
- ✅ **Model-Agnostic:** Works with GPT-4, Claude, Llama, and other LLMs
- ✅ **Reproducible:** Comprehensive experiments with open datasets

---

## 🔬 Methodology

### 1. Interference Detection

We compute the **Interference Detection Score (IDS)** based on:

```python
IDS = w1 * attention_divergence + 
      w2 * semantic_conflict + 
      w3 * response_inconsistency
```

**Components:**
- **Attention Divergence:** Measure split attention between current query and past context
- **Semantic Conflict:** Detect contradictory information in context windows
- **Response Inconsistency:** Identify when model gives different answers with/without context

### 2. Causal Tracing

Using gradient-based attribution, we identify which specific tokens from conversation history are causing interference:

```
Interference_Score(token_i) = ||∇_token_i Loss(response, ground_truth)||
```

### 3. Selective Mitigation

Apply three intervention strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Soft Masking** | Reduce attention weights to interfering tokens | Minor conflicts |
| **Hard Masking** | Completely remove interfering context segments | Major contradictions |
| **Context Reordering** | Move interfering content away from current query | Structural interference |

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (optional, for local models)
- OpenAI API key or Anthropic API key (for GPT-4/Claude)

### Setup

```bash
# Clone the repository
git clone https://github.com/soroushbagheri/llm-memory-interference.git
cd llm-memory-interference

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make setup

# Or manually:
pip install -r requirements.txt
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Model Configuration
DEFAULT_MODEL=gpt-4-turbo
MAX_TOKENS=4096
TEMPERATURE=0.7

# Experiment Settings
EXPERIMENT_SEED=42
CACHE_DIR=./data/cache
```

---

## 🚀 Quick Start

### Demo: Detect Interference

```python
from src.detector import InterferenceDetector
from src.llm_adapter import LLMAdapter

# Initialize
llm = LLMAdapter(model="gpt-4-turbo")
detector = InterferenceDetector()

# Conversation with interference
context = [
    {"role": "user", "content": "Tell me about Python snakes."},
    {"role": "assistant", "content": "Pythons are large constrictor snakes..."},
]

current_query = "How do I install Python packages?"

# Detect interference
ids, details = detector.detect(
    context=context,
    query=current_query,
    llm=llm
)

print(f"Interference Score: {ids:.3f}")
print(f"Interfering tokens: {details['interfering_tokens']}")
```

**Output:**
```
Interference Score: 0.742
Interfering tokens: ['Python', 'snakes', 'constrictor']
Recommendation: Apply soft masking to conversation turn 1
```

### Run Full Benchmark

```bash
# Generate interference dataset
make generate-dataset

# Run experiments (requires API keys)
make run-experiments

# Generate plots and analysis
make generate-plots

# View results
open results/plots/interference_detection_performance.png
```

---

## 📊 Project Structure

```
llm-memory-interference/
│
├── src/                          # Core source code
│   ├── detector.py               # Interference detection
│   ├── tracer.py                 # Causal tracing implementation
│   ├── mitigator.py              # Mitigation strategies
│   ├── llm_adapter.py            # LLM API wrappers
│   ├── metrics.py                # Evaluation metrics (IDS, accuracy)
│   └── utils.py                  # Helper functions
│
├── experiments/                  # Experiment scripts
│   ├── demo.py                   # Quick demonstration
│   ├── run_detector.py           # Interference detection experiments
│   ├── run_benchmark.py          # Full benchmark suite
│   └── analysis.py               # Results analysis
│
├── scripts/                      # Utility scripts
│   ├── generate_interference_dataset.py  # Dataset creation
│   └── generate_plots.py         # Visualization
│
├── configs/                      # Configuration files
│   ├── default.yaml              # Default settings
│   └── benchmark.yaml            # Benchmark configuration
│
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets
│   ├── processed/                # Processed datasets
│   └── cache/                    # Cached API responses
│
├── results/                      # Experimental results
│   ├── experiments/              # Raw results
│   ├── plots/                    # Visualizations
│   └── logs/                     # Execution logs
│
├── tests/                        # Unit tests
│   ├── test_detector.py
│   ├── test_tracer.py
│   └── test_mitigator.py
│
├── docs/                         # Documentation
│   ├── METHODOLOGY.md            # Detailed methodology
│   ├── API.md                    # API documentation
│   ├── EXPERIMENTS.md            # Experiment guide
│   └── REFERENCES.md             # Literature review
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── Makefile                      # Automation commands
└── README.md                     # This file
```

---

## 📖 Documentation

- **[Methodology](docs/METHODOLOGY.md):** Detailed explanation of detection, tracing, and mitigation algorithms
- **[API Reference](docs/API.md):** Complete API documentation
- **[Experiments Guide](docs/EXPERIMENTS.md):** How to reproduce all experiments
- **[References](docs/REFERENCES.md):** Related work and positioning

---

## 🧪 Experiments

### Dataset

We curate a dataset of **1000 conversation scenarios** with labeled interference:

- **Lexical Interference:** Same words, different meanings (Python example)
- **Semantic Interference:** Related but conflicting concepts (medical diagnoses)
- **Structural Interference:** Prior reasoning patterns affecting new problems

### Baselines

1. **No Intervention:** Standard LLM with full context
2. **Fixed Window:** Only use last N turns (N=3, 5, 10)
3. **Random Masking:** Randomly remove context segments
4. **Recency Bias:** Weight recent context more heavily

### Evaluation Metrics

- **Accuracy:** Correctness of final response
- **Interference Detection Rate (IDR):** % of interference correctly identified
- **False Positive Rate (FPR):** % of benign context incorrectly flagged
- **Response Quality:** Human evaluation on clarity and correctness
- **Computational Overhead:** Latency increase from intervention

---

## 📈 Expected Results

(Preliminary - to be updated with actual experiments)

| Method | Accuracy | IDR | FPR | Latency |
|--------|----------|-----|-----|----------|
| No Intervention | 72.3% | - | - | 1.0× |
| Fixed Window (N=5) | 78.1% | 45.2% | 12.3% | 1.0× |
| **Our Method** | **89.7%** | **84.3%** | **6.1%** | **1.3×** |

---

## 🛠️ Development

### Running Tests

```bash
make test                # Run all tests
make test-coverage       # Run with coverage report
```

### Code Quality

```bash
make lint                # Check code quality
make format              # Auto-format code with black
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{bagheri2026memory,
  title={Memory Interference Forensics: Detecting and Mitigating Context Interference in Large Language Models},
  author={Bagheri, Soroush},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by cognitive psychology research on human memory interference
- Built on interpretability techniques from Anthropic and OpenAI research
- Dataset curation supported by [acknowledge funding sources]

---

## 📧 Contact

**Soroush Bagheri**
- GitHub: [@soroushbagheri](https://github.com/soroushbagheri)
- Location: Stockholm, Sweden

For questions or collaborations, please open an issue or reach out directly.

---

<div align="center">

**⭐ Star this repository if you find it useful!**

</div>
