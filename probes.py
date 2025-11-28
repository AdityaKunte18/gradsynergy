"""
Build probe prompts directly from the training dataset instead of fixed math prompts.
"""

import os
from typing import List, Optional, Tuple

import datasets
import pandas as pd


def load_probes_from_dataset(path: str = "data/math500/train.parquet", limit: int = 8) -> List[Tuple[str, Optional[str]]]:
    """
    Load a small set of (prompt, ground_truth) pairs for gradient probing.
    Falls back to downloading HuggingFaceH4/MATH-500 if the parquet is missing.
    """
    probes: List[Tuple[str, Optional[str]]] = []
    if os.path.exists(path):
        df = pd.read_parquet(path)
        for _, row in df.head(limit).iterrows():
            probes.append((str(row.get("prompt", "")), str(row.get("ground_truth", ""))))
        return probes

    # fallback: download and format like train.py
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
    instruction = "Let's think step by step and output the final answer in \\boxed{your answer here}."
    for example in ds.select(range(limit)):
        prompt = f"{example['problem']} {instruction}"
        probes.append((prompt, example["answer"]))
    return probes
