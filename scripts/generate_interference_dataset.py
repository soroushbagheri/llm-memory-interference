#!/usr/bin/env python3
"""Generate synthetic interference dataset for experiments.

Creates labeled examples of context interference across different types:
- Lexical: Same words, different meanings
- Semantic: Related but conflicting concepts  
- Structural: Prior reasoning patterns affecting new problems

Usage:
    python scripts/generate_interference_dataset.py \
        --n-samples 1000 \
        --output data/processed/interference_v1.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime


@dataclass
class InterferenceExample:
    """Single example of context interference."""
    id: str
    context: List[Dict[str, str]]
    query: str
    ground_truth: str
    has_interference: bool
    interference_type: str  # lexical, semantic, structural, none
    interfering_tokens: List[str]
    interfering_turn_indices: List[int]
    domain: str
    difficulty: str  # easy, medium, hard


class InterferenceDatasetGenerator:
    """Generate synthetic interference examples."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Lexical interference templates (same word, different meanings)
        self.lexical_templates = [
            {
                "context": [
                    {"role": "user", "content": "Tell me about Python snakes."},
                    {"role": "assistant", "content": "Pythons are large constrictor snakes found in Africa, Asia, and Australia. They can grow over 20 feet long and are non-venomous, killing prey by constriction."}
                ],
                "query": "How do I install Python packages using pip?",
                "ground_truth": "Use 'pip install package_name' in your terminal to install Python packages. For example: pip install numpy",
                "interfering_tokens": ["Python", "snakes", "constrictor"],
                "interfering_turns": [0, 1],
                "domain": "technical"
            },
            {
                "context": [
                    {"role": "user", "content": "What is a bank in geography?"},
                    {"role": "assistant", "content": "In geography, a bank is the land alongside a river or stream. River banks can erode over time due to water flow and are important for ecosystems."}
                ],
                "query": "How do I deposit money in a bank?",
                "ground_truth": "Visit a bank branch or ATM, fill out a deposit slip with your account number, and give it to the teller with your cash or check.",
                "interfering_tokens": ["bank", "river", "flow"],
                "interfering_turns": [0, 1],
                "domain": "financial"
            },
            {
                "context": [
                    {"role": "user", "content": "Explain the Apple fruit."},
                    {"role": "assistant", "content": "Apples are edible fruits from the apple tree (Malus domestica). They're rich in fiber and vitamin C, and come in varieties like Granny Smith, Fuji, and Gala."}
                ],
                "query": "What's the latest Apple iPhone model?",
                "ground_truth": "As of 2026, the latest iPhone is the iPhone 16 series, released in September 2024, featuring improved cameras and A18 chip.",
                "interfering_tokens": ["Apple", "varieties", "fruit"],
                "interfering_turns": [0, 1],
                "domain": "technical"
            }
        ]
        
        # Semantic interference templates (related but conflicting concepts)
        self.semantic_templates = [
            {
                "context": [
                    {"role": "user", "content": "Explain type 1 diabetes."},
                    {"role": "assistant", "content": "Type 1 diabetes is an autoimmune condition where the pancreas produces little to no insulin. It typically develops in childhood and requires lifelong insulin therapy."}
                ],
                "query": "What are symptoms of type 2 diabetes?",
                "ground_truth": "Type 2 diabetes symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow-healing wounds. Unlike type 1, it's often linked to obesity and develops gradually in adults.",
                "interfering_tokens": ["diabetes", "insulin", "pancreas"],
                "interfering_turns": [0, 1],
                "domain": "medical"
            },
            {
                "context": [
                    {"role": "user", "content": "What is criminal law?"},
                    {"role": "assistant", "content": "Criminal law deals with crimes against the state or public. It involves prosecution by government, potential imprisonment, and focuses on punishment and deterrence."}
                ],
                "query": "Explain civil law cases.",
                "ground_truth": "Civil law handles disputes between individuals or organizations. It involves lawsuits for damages or specific performance, with remedies like monetary compensation rather than imprisonment.",
                "interfering_tokens": ["law", "prosecution", "punishment"],
                "interfering_turns": [0, 1],
                "domain": "legal"
            }
        ]
        
        # Structural interference templates (reasoning patterns)
        self.structural_templates = [
            {
                "context": [
                    {"role": "user", "content": "Solve: If x + 5 = 12, what is x?"},
                    {"role": "assistant", "content": "To solve x + 5 = 12, subtract 5 from both sides: x = 12 - 5 = 7. Therefore x = 7."}
                ],
                "query": "Solve: If 3x = 21, what is x?",
                "ground_truth": "To solve 3x = 21, divide both sides by 3: x = 21 ÷ 3 = 7. Therefore x = 7.",
                "interfering_tokens": ["subtract", "5", "both sides"],
                "interfering_turns": [0, 1],
                "domain": "educational"
            }
        ]
        
        # Non-interference examples (helpful context)
        self.clean_templates = [
            {
                "context": [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of AI where systems learn from data without explicit programming. Common techniques include supervised learning, unsupervised learning, and reinforcement learning."}
                ],
                "query": "What's the difference between supervised and unsupervised learning?",
                "ground_truth": "Supervised learning uses labeled data to train models (e.g., classification), while unsupervised learning finds patterns in unlabeled data (e.g., clustering).",
                "interfering_tokens": [],
                "interfering_turns": [],
                "domain": "technical"
            }
        ]
    
    def generate_dataset(self, n_samples: int, distribution: Dict[str, float] = None) -> List[InterferenceExample]:
        """Generate dataset with specified distribution of interference types.
        
        Args:
            n_samples: Total number of examples to generate
            distribution: Dict mapping interference types to proportions
                         Default: {"lexical": 0.3, "semantic": 0.4, "structural": 0.3, "none": 0.0}
        
        Returns:
            List of InterferenceExample objects
        """
        if distribution is None:
            distribution = {
                "lexical": 0.30,
                "semantic": 0.40,
                "structural": 0.30,
                "none": 0.0
            }
        
        examples = []
        
        # Calculate target counts
        targets = {k: int(n_samples * v) for k, v in distribution.items()}
        
        # Generate each type
        for itype, count in targets.items():
            if itype == "lexical":
                examples.extend(self._generate_from_templates(self.lexical_templates, count, "lexical"))
            elif itype == "semantic":
                examples.extend(self._generate_from_templates(self.semantic_templates, count, "semantic"))
            elif itype == "structural":
                examples.extend(self._generate_from_templates(self.structural_templates, count, "structural"))
            elif itype == "none":
                examples.extend(self._generate_from_templates(self.clean_templates, count, "none"))
        
        # Shuffle
        random.shuffle(examples)
        
        return examples
    
    def _generate_from_templates(
        self, 
        templates: List[Dict], 
        n_samples: int, 
        interference_type: str
    ) -> List[InterferenceExample]:
        """Generate examples from template set."""
        examples = []
        
        for i in range(n_samples):
            template = random.choice(templates)
            
            # Create unique ID
            uid = hashlib.md5(f"{interference_type}_{i}_{random.random()}".encode()).hexdigest()[:12]
            
            # Assign difficulty (random for now, could be based on features)
            difficulty = random.choice(["easy", "medium", "hard"])
            
            example = InterferenceExample(
                id=f"{interference_type}_{uid}",
                context=template["context"],
                query=template["query"],
                ground_truth=template["ground_truth"],
                has_interference=(interference_type != "none"),
                interference_type=interference_type,
                interfering_tokens=template.get("interfering_tokens", []),
                interfering_turn_indices=template.get("interfering_turns", []),
                domain=template["domain"],
                difficulty=difficulty
            )
            
            examples.append(example)
        
        return examples
    
    def save_dataset(self, examples: List[InterferenceExample], output_path: Path):
        """Save dataset to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            "metadata": {
                "n_examples": len(examples),
                "generation_date": datetime.now().isoformat(),
                "generator_seed": self.seed,
                "version": "1.0"
            },
            "examples": [asdict(ex) for ex in examples]
        }
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✓ Generated {len(examples)} examples")
        print(f"✓ Saved to {output_path}")
        
        # Print statistics
        type_counts = {}
        domain_counts = {}
        for ex in examples:
            type_counts[ex.interference_type] = type_counts.get(ex.interference_type, 0) + 1
            domain_counts[ex.domain] = domain_counts.get(ex.domain, 0) + 1
        
        print("\nDistribution by interference type:")
        for itype, count in sorted(type_counts.items()):
            print(f"  {itype}: {count} ({count/len(examples)*100:.1f}%)")
        
        print("\nDistribution by domain:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count} ({count/len(examples)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate interference detection dataset")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/processed/interference_v1.json", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lexical", type=float, default=0.30, help="Proportion of lexical interference")
    parser.add_argument("--semantic", type=float, default=0.40, help="Proportion of semantic interference")
    parser.add_argument("--structural", type=float, default=0.30, help="Proportion of structural interference")
    
    args = parser.parse_args()
    
    # Validate proportions sum to ~1.0
    total = args.lexical + args.semantic + args.structural
    if abs(total - 1.0) > 0.01:
        print(f"Warning: Proportions sum to {total}, not 1.0. Normalizing...")
        args.lexical /= total
        args.semantic /= total
        args.structural /= total
    
    distribution = {
        "lexical": args.lexical,
        "semantic": args.semantic,
        "structural": args.structural,
        "none": 0.0
    }
    
    print("="*80)
    print("Interference Dataset Generator")
    print("="*80)
    print(f"Generating {args.n_samples} examples with distribution:")
    for itype, prop in distribution.items():
        if prop > 0:
            print(f"  {itype}: {prop*100:.1f}%")
    print()
    
    generator = InterferenceDatasetGenerator(seed=args.seed)
    examples = generator.generate_dataset(args.n_samples, distribution)
    generator.save_dataset(examples, Path(args.output))
    
    print("\n✓ Dataset generation complete!")


if __name__ == "__main__":
    main()
