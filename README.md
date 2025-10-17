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
