"""
dataset_loader.py
-----------------
Provides unified dataset loading abstractions for NLP QA datasets.
Handles TriviaQA, HaluEval, and MuSiQue with standardized output schemas.
"""

from typing import List, Dict, Any
from datasets import load_dataset as hf_load

def load_dataset(dataset_name: str, n_samples: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Loads and standardizes datasets for the pipeline.
    
    Returns a unified list of dictionaries containing:
        - id: Unique sample ID
        - question: The query string
        - gold_answer: The human-annotated correct answer or context
        - is_hallucination: (Optional) Ground truth label if the prompt guarantees hallucination
    """
    samples = []
    
    if dataset_name == "trivia_qa":
        # TriviaQA: Open-domain question answering
        ds = hf_load("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
        ds = ds.shuffle(seed=seed)
        
        for i, item in enumerate(ds.select(range(n_samples))):
            samples.append({
                "id": f"tc_{i}",
                "question": item["question"],
                "gold_answer": item["answer"]["value"],
                "is_hallucination": None, # Ground truth not explicitly provided by dataset
            })
            
    elif dataset_name == "halueval":
        # HaluEval: Hallucination Evaluation QA Dataset
        ds = hf_load("pminervini/HaluEval", "qa_samples", split="data")
        ds = ds.shuffle(seed=seed)
        
        for i, item in enumerate(ds.select(range(n_samples))):
            samples.append({
                "id": f"halueval_{i}",
                "question": item["question"],
                "gold_answer": item["knowledge"], # Use knowledge base as reference
                "is_hallucination": item["hallucination"] == "yes",
            })
            
    elif dataset_name == "musique":
        # MuSiQue: Multihop reasoning (higher difficulty, tests deep explanation grounding)
        ds = hf_load("tau/musique", "default", split="validation")
        ds = ds.shuffle(seed=seed)
        
        for i, item in enumerate(ds.select(range(n_samples))):
            samples.append({
                "id": f"musique_{i}",
                "question": item["question"],
                "gold_answer": item["answer"],
                "is_hallucination": None,
            })
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    return samples
