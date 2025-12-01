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
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"[setup] model={args.model} quantize={args.quantize} device={args.device} "
        f"max_new_tokens={args.max_new_tokens} do_sample={args.do_sample}"
    )
    model, tok = load_model_and_tok(args.model, args.quantize, device=args.device)
    model.eval()

    print(f"[data] loading: {args.train_file}")
    data, _ = load_math500(args.train_file, None)
    if len(data) == 0:
        print("No data loaded.")
        return
    print(f"[data] loaded {len(data)} examples; sampling {min(args.max_samples, len(data))}")
    samples = random.sample(data, k=min(args.max_samples, len(data)))

    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        gt = sample.get("ground_truth", "")
        print(f"[gen] sample {idx} prompt_len={len(prompt)}")
        with torch.no_grad():
            out = generate_and_logprobs(
                model,
                tok,
                prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                do_sample=args.do_sample,
            )
        comp = compute_component_scores(
            data_source="math500",
            solution_str=out.text,
            ground_truth=gt,
            extra_info=None,
        )
        parsed = extract_solution(out.text)
        gen_tokens = out.logprobs.shape[0]
        text_len = len(out.text)

        print(f"\n=== sample {idx} ===")
        print(f"correctness={comp['correctness_binary']:.3f} format={comp['format_binary']:.3f} length={comp['length_binary']:.3f}")
        print(f"gen_tokens={gen_tokens} text_len={text_len}")
        print(f"parsed_solution: {parsed}")
        print(f"ground_truth:    {gt}")
        print(f"output:\n{out.text.strip()}")


if __name__ == "__main__":
    main()
