#!/usr/bin/env python3
import argparse
import random
import torch

from train import load_model_and_tok, load_math500
from rewards import compute_component_scores, extract_solution
from text_utils import generate_and_logprobs


def main():
    ap = argparse.ArgumentParser(description="Quickly inspect parsed answers vs ground truth.")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--quantize", type=str, default="4bit")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--train_file", type=str, default="data/math500/train.parquet")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tok = load_model_and_tok(args.model, args.quantize, device=args.device)
    model.eval()

    data, _ = load_math500(args.train_file, None)
    if len(data) == 0:
        print("No data loaded.")
        return
    samples = random.sample(data, k=min(args.max_samples, len(data)))

    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        gt = sample.get("ground_truth", "")
        with torch.no_grad():
            out = generate_and_logprobs(
                model,
                tok,
                prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
        comp = compute_component_scores(
            data_source="math500",
            solution_str=out.text,
            ground_truth=gt,
            extra_info=None,
        )
        parsed = extract_solution(out.text)

        print(f"\n=== sample {idx} ===")
        print(f"correctness={comp['correctness_binary']:.3f} format={comp['format_binary']:.3f} length={comp['length_binary']:.3f}")
        print(f"parsed_solution: {parsed}")
        print(f"ground_truth:    {gt}")
        print(f"output:\n{out.text.strip()}")


if __name__ == "__main__":
    main()
