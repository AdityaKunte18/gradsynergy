# Gradient Synergy Probe (minimal RL-lite w/ per-objective grads)

This repo gives you a tiny, hackable scaffold to **measure gradient synergy/conflict across objectives** while aligning a small LLM for a few epochs. It is intentionally minimal so you can run it on a single GPU and iterate quickly.

## What it does

- Loads a **very small chat LLM** (recommended: `Qwen/Qwen2-0.5B-Instruct` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- Optionally quantizes (4-bit) and applies **LoRA** adapters to keep memory small.
- Runs a **REINFORCE / PPO-lite** training loop for a few epochs on a tiny demo dataset.
- Defines 3 **objectives** (you can enable/disable easily):
  1. **Reasoning/Exactness**: compares final numeric answer to label (toy math set provided).
  2. **Brevity/Format**: rewards short, properly delimited answers.
  3. **Refusal Safety**: rewards safe refusal style on unsafe prompts.
- Computes **per-objective gradients** (last transformer block by default), builds a cosine-similarity matrix, and logs:
  - Pairwise sims, conflict fraction (< -tau), and “net synergy” per objective.
- (Optional) Applies a **weight scheduler** or **gradient surgery** toggle.

## Quick start

1. Create a venv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -U torch transformers accelerate peft bitsandbytes datasets
   ```

2. Put a small GPU on it (12–16GB is plenty with 4-bit). Then run:
   ```bash
   python train.py --model Qwen/Qwen2-0.5B-Instruct --quantize 4bit --epochs 2 --batch_size 2
   ```

   Examples:
   - No quant, CPU/GPU (slower): `--quantize none`
   - TinyLlama: `--model TinyLlama/TinyLlama-1.1B-Chat-v1.0`
   - Turn off safety objective: `--use_obj_safety 0`

3. Logs
   - Metrics & gradient stats in `runs/run_<timestamp>/metrics.jsonl`
   - Cosine matrices as CSV in `runs/.../cosine_stepXXXX.csv`

## Files

- `train.py` — main RL-lite loop
- `objectives.py` — reward functions
- `grad_probe.py` — gradient extraction + similarity
- `utils.py` — helpers (gen, tokenization, logging)
- `toy_data/` — tiny demo datasets

## Notes

- This is a **probe**, not a full PPO library. It’s enough to study synergy trends over a few epochs.
- If 4-bit `bitsandbytes` is flaky on your setup, use `--quantize none` (or `8bit`).
