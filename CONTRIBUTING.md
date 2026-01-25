# Contributing to Memory Interference Forensics

Thank you for your interest in contributing! This project is actively seeking contributions in:

## Areas for Contribution

### 1. Code Contributions
- **New interference types:** Implement detectors for additional interference patterns
- **Mitigation strategies:** Develop new strategies beyond soft/hard masking
- **Model support:** Add adapters for new LLMs (Gemini, Mistral, etc.)
- **Performance optimizations:** Speed up detection or reduce API costs

### 2. Dataset Contributions
- **Domain-specific examples:** Add interference examples from your domain (medical, legal, etc.)
- **Edge cases:** Contribute challenging examples where detection fails
- **Multilingual:** Extend dataset to non-English languages

### 3. Documentation
- **Tutorials:** Write guides for specific use cases
- **API documentation:** Improve docstrings and examples
- **Translations:** Translate README to other languages

### 4. Research Contributions
- **Experiments:** Run experiments and share results
- **Benchmarks:** Compare against other methods
- **Use cases:** Deploy in real applications and report findings

## How to Contribute

### 1. Fork and Clone
```bash
git clone https://github.com/YOUR_USERNAME/llm-memory-interference.git
cd llm-memory-interference
```

### 2. Create Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Write code following existing style (use `make format`)
- Add tests for new features
- Update documentation

### 4. Run Tests
```bash
make test
make lint
```

### 5. Commit and Push
```bash
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### 6. Create Pull Request
- Open PR on GitHub
- Describe changes clearly
- Link related issues

## Code Style

- **Python:** Follow PEP 8, use Black formatter
- **Docstrings:** Google style
- **Type hints:** Required for all functions
- **Tests:** Maintain >80% coverage

## Commit Message Format

```
Type: Brief description (50 chars max)

Detailed explanation of what and why.

Fixes #issue_number
```

**Types:** `Add`, `Fix`, `Update`, `Remove`, `Refactor`, `Docs`, `Test`

## Questions?

Open an issue or reach out to [@soroushbagheri](https://github.com/soroushbagheri).

---

**Thank you for contributing!** 🙏
