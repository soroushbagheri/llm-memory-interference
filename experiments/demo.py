#!/usr/bin/env python
"""Quick demonstration of memory interference detection.

Author: Soroush Bagheri
Date: January 2026
"""

import sys
sys.path.insert(0, '.')

from src.detector import InterferenceDetector
from src.tracer import CausalTracer
from src.mitigator import InterferenceMitigator
from src.llm_adapter import LLMAdapter
import logging

logging.basicConfig(level=logging.INFO)


def main():
    print("="*80)
    print("LLM MEMORY INTERFERENCE FORENSICS - DEMO")
    print("="*80)
    print()
    
    # Initialize components
    print("[1/5] Initializing components...")
    llm = LLMAdapter(model="mock")  # Use mock for demo (no API needed)
    detector = InterferenceDetector(threshold=0.5)
    tracer = CausalTracer()
    mitigator = InterferenceMitigator(default_strategy="soft_masking")
    print("  ✓ Components initialized\n")
    
    # Example 1: Lexical interference (Python snake vs programming)
    print("[2/5] Example 1: Lexical Interference")
    print("-" * 80)
    
    context1 = [
        {"role": "user", "content": "Tell me about Python snakes."},
        {"role": "assistant", "content": "Pythons are large constrictor snakes found in tropical regions. They can grow up to 20 feet long and are non-venomous."},
    ]
    query1 = "How do I install Python packages?"
    
    print(f"Context: {context1[0]['content']}")
    print(f"Query: {query1}")
    print()
    
    # Detect interference
    ids_score, result = detector.detect(context1, query1, llm)
    print(f"IDS Score: {ids_score:.3f}")
    print(f"Attention Divergence: {result.attention_divergence:.3f}")
    print(f"Semantic Conflict: {result.semantic_conflict:.3f}")
    print(f"Response Inconsistency: {result.response_inconsistency:.3f}")
    print(f"Recommendation: {result.recommendation}")
    print()
    
    # Apply mitigation
    if len(result.interfering_turns) > 0:
        mitigation_result = mitigator.mitigate(
            context1, result.interfering_turns, strategy="soft_masking"
        )
        print(f"Mitigation applied: {mitigation_result.strategy}")
        print(f"Modified context: {mitigation_result.modified_context}")
    print("\n")
    
    # Example 2: Medical interference
    print("[3/5] Example 2: Medical Context Interference")
    print("-" * 80)
    
    context2 = [
        {"role": "user", "content": "Patient has fever and cough, diagnosed with flu."},
        {"role": "assistant", "content": "For flu, recommend rest, fluids, and over-the-counter medications."},
        {"role": "user", "content": "New patient: 45-year-old with chest pain and shortness of breath."},
    ]
    query2 = "What should I do?"
    
    print(f"Context: {len(context2)} previous turns")
    print(f"Query: {query2}")
    print()
    
    ids_score2, result2 = detector.detect(context2, query2, llm)
    print(f"IDS Score: {ids_score2:.3f}")
    print(f"Interfering turns: {result2.interfering_turns}")
    print(f"Recommendation: {result2.recommendation}")
    print("\n")
    
    # Example 3: No interference (clean context)
    print("[4/5] Example 3: Clean Context (No Interference)")
    print("-" * 80)
    
    context3 = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    query3 = "What is the population of Paris?"
    
    print(f"Context: {context3[0]['content']}")
    print(f"Query: {query3}")
    print()
    
    ids_score3, result3 = detector.detect(context3, query3, llm)
    print(f"IDS Score: {ids_score3:.3f}")
    print(f"Recommendation: {result3.recommendation}")
    print("\n")
    
    # Summary
    print("[5/5] Summary")
    print("="*80)
    print(f"Example 1 (Lexical): IDS = {ids_score:.3f} - {'INTERFERENCE' if ids_score > 0.5 else 'CLEAN'}")
    print(f"Example 2 (Medical): IDS = {ids_score2:.3f} - {'INTERFERENCE' if ids_score2 > 0.5 else 'CLEAN'}")
    print(f"Example 3 (Clean): IDS = {ids_score3:.3f} - {'INTERFERENCE' if ids_score3 > 0.5 else 'CLEAN'}")
    print()
    print("✓ Demo complete! For real experiments, use experiments/run_benchmark.py")
    print()


if __name__ == "__main__":
    main()
