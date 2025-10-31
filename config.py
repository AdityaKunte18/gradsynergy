import os, torch
from transformers import set_seed

# ----------------------------
# Global/config (env-overridable)
# ----------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
         else (torch.float16 if DEVICE == "cuda" else torch.float32))

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 24))
TOP_K = int(os.environ.get("TOP_K", 50))
TOP_P = float(os.environ.get("TOP_P", 0.95))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))

SEED = int(os.environ.get("SEED", 42))
set_seed(SEED)
torch.use_deterministic_algorithms(False)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# LoRA
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
LORA_TARGET_MODULES = os.environ.get(
    "LORA_TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")

# Training loop
BATCH_PROBES = int(os.environ.get("BATCH_PROBES", 6))
K_SAMPLES = int(os.environ.get("K_SAMPLES", 2))
NUM_UPDATES = int(os.environ.get("NUM_UPDATES", 60))
LR = float(os.environ.get("LR", 2e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.0))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
CLIP_NORM = float(os.environ.get("CLIP_NORM", 1.0))

# Preferences over [harmlessness, brevity, adherence]
import torch as _torch
PREFERENCES = {
    "Harmless-heavy":  _torch.tensor([0.7, 0.15, 0.15]),
    "Brief-heavy":     _torch.tensor([0.15, 0.7, 0.15]),
    "Adherence-heavy": _torch.tensor([0.15, 0.15, 0.7]),
    "Balanced":        _torch.tensor([1/3, 1/3, 1/3]),
}
ACTIVE_PREF = os.environ.get("ACTIVE_PREF", "Balanced")
assert ACTIVE_PREF in PREFERENCES

# Reward knobs
UNSAFE_LIST = ["kill","hate","bomb","terror","suicide","racist","violent","harm","attack","illegal","drugs"]
SAFETY_PEN_PER_HIT = float(os.environ.get("SAFETY_PEN_PER_HIT", 1.0))
CONCISION_LAMBDA = float(os.environ.get("CONCISION_LAMBDA", 0.02))
ADHERENCE_POS = float(os.environ.get("ADHERENCE_POS", 1.0))
ADHERENCE_NEG = float(os.environ.get("ADHERENCE_NEG", 0.0))
PROBE_ENTROPY_BONUS = float(os.environ.get("PROBE_ENTROPY_BONUS", 1e-3))

# Outputs
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "lora_prefgrad_qwen3")
SAVE_ADAPTER_PATH = os.environ.get("SAVE_ADAPTER_PATH", f"{OUTPUT_PREFIX}_adapter")
FINAL_COS_CSV = "obj_cosine_global_final.csv"
FINAL_COS_PNG = "obj_cosine_global_final.png"
LAYER_LOG_CSV = "layer_cosine_logs.csv"
LAYER_PLOT_PREFIX = "layer_cosine_layer"
