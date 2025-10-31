#!/usr/bin/env python3
# Entry point. Mirrors original behavior/outputs with a cleaner structure.

import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import (
    MODEL_NAME, DEVICE, DTYPE,
    PREFERENCES, ACTIVE_PREF,
    NUM_UPDATES, BATCH_PROBES, K_SAMPLES, GRAD_ACCUM_STEPS,
    LR, WEIGHT_DECAY, CLIP_NORM, PROBE_ENTROPY_BONUS,
    SAVE_ADAPTER_PATH, FINAL_COS_CSV, FINAL_COS_PNG,
    LAYER_LOG_CSV, LAYER_PLOT_PREFIX, DEMO_PROMPTS, TOP_K, TOP_P, MAX_NEW_TOKENS
)
from .probes import PROBES
from .lora_utils import (
    add_lora_adapters, enable_checkpointing_and_freeze_base,
    get_trainable_layer_groups, layer_grad_norms
)
from .sampling import draw_samples
from .reinforce import (
    reinforce_backward_from_samples, objective_grads_per_layer,
    compute_pairwise_cosines_per_layer, compute_global_objective_cosines
)
from .viz import save_cosine_matrix_csv_png, append_layer_log_rows, plot_layer_curves_from_csv

def main():
    w_active = PREFERENCES[ACTIVE_PREF].to("cpu", dtype=torch.float32)
    print(f"Loading MODEL='{MODEL_NAME}' on {DEVICE} (dtype={DTYPE})…")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        torch_dtype=DTYPE if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    enable_checkpointing_and_freeze_base(mdl)
    mdl = add_lora_adapters(mdl)

    lora_params = [p for n, p in mdl.named_parameters() if p.requires_grad]
    optim = torch.optim.AdamW(lora_params, lr=LR, weight_decay=WEIGHT_DECAY)

    layer_groups = get_trainable_layer_groups(mdl)
    if not layer_groups:
        raise RuntimeError("No trainable LoRA layer groups found. Check LORA_TARGET_MODULES and param names.")
    print(f"[info] LoRA layer groups: {list(layer_groups.keys())[:8]}{'...' if len(layer_groups)>8 else ''}")

    with open(LAYER_LOG_CSV, "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv); wcsv.writerow(["step", "layer", "H_B", "H_A", "B_A"])

        print(f"\n==> Training towards '{ACTIVE_PREF}' w={w_active.tolist()} | "
              f"updates={NUM_UPDATES}, batch_probes={BATCH_PROBES}, k_samples={K_SAMPLES}, "
              f"grad_accum={GRAD_ACCUM_STEPS}, lr={LR}, clip={CLIP_NORM}, probe_entropy={PROBE_ENTROPY_BONUS}")

        ema_loss = None; step = 0
        for u in range(NUM_UPDATES):
            samples_with_prompts = draw_samples(mdl, tok, PROBES, BATCH_PROBES, K_SAMPLES)
            samples = [s for (s, _) in samples_with_prompts]
            if len(samples) == 0:
                print("[warn] No samples this update; skipping."); continue

            # TRAINING pass (with baseline centering)
            mdl.zero_grad(set_to_none=True)
            baseline = reinforce_backward_from_samples(mdl, samples, w_active, center_baseline=True, add_entropy=0.0)
            if (u + 1) % GRAD_ACCUM_STEPS == 0:
                if CLIP_NORM and CLIP_NORM > 0: torch.nn.utils.clip_grad_norm_(lora_params, CLIP_NORM)
                optim.step(); optim.zero_grad(set_to_none=True); step += 1

            # PROBE passes
            obj_layer_grads = objective_grads_per_layer(mdl, samples, layer_groups)
            layer_cos = compute_pairwise_cosines_per_layer(obj_layer_grads)
            append_layer_log_rows(wcsv, step, layer_cos)

            # DIAGNOSTICS: grad norms per objective
            mdl.zero_grad(set_to_none=True)
            _ = reinforce_backward_from_samples(mdl, samples, torch.tensor([1.0,0.0,0.0]), center_baseline=False, add_entropy=PROBE_ENTROPY_BONUS)
            normsH = layer_grad_norms(layer_groups)
            mdl.zero_grad(set_to_none=True)
            _ = reinforce_backward_from_samples(mdl, samples, torch.tensor([0.0,1.0,0.0]), center_baseline=False, add_entropy=PROBE_ENTROPY_BONUS)
            normsB = layer_grad_norms(layer_groups)
            mdl.zero_grad(set_to_none=True)
            _ = reinforce_backward_from_samples(mdl, samples, torch.tensor([0.0,0.0,1.0]), center_baseline=False, add_entropy=PROBE_ENTROPY_BONUS)
            normsA = layer_grad_norms(layer_groups)
            first_layers = sorted(set(list(normsH.keys())+list(normsB.keys())+list(normsA.keys())))[:3]
            dbg = ", ".join([f"L{li}:|H|={normsH.get(li,float('nan')):.2e},|B|={normsB.get(li,float('nan')):.2e},|A|={normsA.get(li,float('nan')):.2e}" for li in first_layers])
            print(f"[diag] step={step} grad-norms {dbg}")

            # progress print
            if ema_loss is None: ema_loss = -baseline
            else: ema_loss = 0.9 * ema_loss + 0.1 * (-baseline)
            if (u + 1) % 5 == 0:
                first_layer = next(iter(sorted(layer_cos.keys())))
                ex = layer_cos[first_layer]
                print(f"[upd {u+1:04d}] step={step} baseline={baseline:.4f} ema(-baseline)={ema_loss:.4f} | "
                      f"L{first_layer} cos H-B={ex['H-B']:.3f}, H-A={ex['H-A']:.3f}, B-A={ex['B-A']:.3f}")

    print(f"\nSaving LoRA adapter to: {SAVE_ADAPTER_PATH}")
    mdl.save_pretrained(SAVE_ADAPTER_PATH)

    print("\nComputing final objective cosine matrix (global)…")
    def _draw_fn(batch_probes=8, k_samples=2):
        return draw_samples(mdl, tok, PROBES, batch_probes=min(len(PROBES), batch_probes), k_samples=k_samples)
    M = compute_global_objective_cosines(mdl, tok, _draw_fn)
    names = ["Harmless", "Brevity", "Adherence"]
    head = " " * 14 + "  ".join([f"{n:>10s}" for n in names]); print(head)
    for i, ni in enumerate(names):
        row = [f"{ni:>12s}"] + [f"{M[i,j]:>10.3f}" for j in range(3)]; print("  ".join(row))
    save_cosine_matrix_csv_png(M, names, FINAL_COS_CSV, FINAL_COS_PNG)
    print(f"Saved final cosine matrix → {FINAL_COS_CSV}, {FINAL_COS_PNG}")

    print("Plotting per-layer cosine curves…")
    try:
        plot_layer_curves_from_csv(LAYER_LOG_CSV, LAYER_PLOT_PREFIX)
        print(f"Wrote plots: {LAYER_PLOT_PREFIX}*.png")
    except Exception as e:
        print(f"[warn] plotting layer curves failed: {e}")

    # Quick demo (same as original)
    mdl.eval()
    for dp in DEMO_PROMPTS:
        with torch.no_grad():
            enc = tok(dp, return_tensors="pt").to(DEVICE)
            out = mdl.generate(**enc, do_sample=True, top_k=TOP_K, top_p=TOP_P, temperature=0.7,
                               max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n>> {dp}\n<< {text}")

if __name__ == "__main__":
    main()

