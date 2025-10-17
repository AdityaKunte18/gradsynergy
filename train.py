import os, json, math, time, argparse, random
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import ensure_dir, nowstamp, jsonl_writer, generate_and_logprobs, chat_format, flat_params, moving_avg, clip_and_renorm
from objectives import reward_dict
from grad_probe import cosine_matrix, conflict_fraction, project_nonconflicting

def load_model_and_tok(model_name, quantize="4bit", device="cuda"):
    quantize = quantize.lower()
    bnb_config=None
    load_kwargs = {}
    if "bit" in quantize and quantize!="none":
        nf = 4 if "4" in quantize else 8
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(nf==4),
            load_in_8bit=(nf==8),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        load_kwargs["device_map"] = {"":0} if torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    # LoRA
    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)
    return model, tok

def last_block_named_params(model: nn.Module):
    # heuristic: get last transformer block module and return its parameters
    # works for most HF decoder models (Llama, Qwen, TinyLlama, etc.)
    blocks = None
    for name in ["model.layers", "model.transformer.h", "transformer.h", "transformer.layers", "layers", "model.decoder.layers"]:
        try:
            blocks = eval(f"model.{name}")
            break
        except Exception:
            continue
    if blocks is None:
        # fallback to all trainable params
        return list(model.named_parameters())
    last = blocks[-1]
    return list(last.named_parameters(recurse=True))

def reinforce_loss(logprobs: torch.Tensor, advantage: float):
    # scalar advantage applied to each step (simple)
    if logprobs.numel() == 0:
        return torch.zeros((), device=logprobs.device, dtype=logprobs.dtype, requires_grad=True)
    return -(advantage * logprobs).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--quantize", type=str, default="4bit", choices=["none","8bit","4bit"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tau", type=float, default=-0.1, help="conflict threshold")
    ap.add_argument("--use_obj_reason", type=int, default=1)
    ap.add_argument("--use_obj_brevity", type=int, default=1)
    ap.add_argument("--use_obj_safety", type=int, default=1)
    ap.add_argument("--weight_scheduler", type=int, default=0)
    ap.add_argument("--grad_surgery", type=int, default=0)
    ap.add_argument("--data_path", type=str, default="toy_data")
    ap.add_argument("--use_hf", type=int, default=0, help="Use HF datasets (gsm8k + real-toxicity-prompts)")
    ap.add_argument("--gsm_n", type=int, default=200)
    ap.add_argument("--rtp_n", type=int, default=200)
    ap.add_argument("--outdir", type=str, default="runs")
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = args.device if torch.cuda.is_available() else "cpu"
    model, tok = load_model_and_tok(args.model, args.quantize, device=device)
    model.train()

    # optimizer over LoRA params only
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)


    # Load data: either HF or toy
    if args.use_hf == 1:
        from dataset_loader import make_unified
        data = make_unified(gsm_n=args.gsm_n, rtp_n=args.rtp_n, split="train")
    else:
        def load_jsonl(p):
            with open(p, "r", encoding="utf-8") as f:
                return [json.loads(x) for x in f]
        math_data = load_jsonl(os.path.join(args.data_path, "math_tiny.jsonl"))
        safety_data = load_jsonl(os.path.join(args.data_path, "safety_tiny.jsonl"))
        data = math_data + safety_data

    random.shuffle(data)

    run_dir = os.path.join(args.outdir, f"run_{nowstamp()}")
    ensure_dir(run_dir)
    write = jsonl_writer(os.path.join(run_dir, "metrics.jsonl"))

    # per-objective weights and EMA baselines
    weights = {"reason":1.0, "brevity":1.0, "safety":1.0}
    baselines = {"reason":None, "brevity":None, "safety":None}
    ema_beta = 0.9
    sched_eta = 0.05

    # choose param slice for gradient probe
    probe_params = last_block_named_params(model)

    # training
    step = 0
    for ep in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i+args.batch_size]
            grads = {}
            rewards_batch = []

            # === per-objective gradient passes ===
            for obj_name in ["reason","brevity","safety"]:
                use_flag = (obj_name=="reason" and args.use_obj_reason==1) or \
                           (obj_name=="brevity" and args.use_obj_brevity==1) or \
                           (obj_name=="safety" and args.use_obj_safety==1)
                if not use_flag:
                    continue

                optim.zero_grad(set_to_none=True)
                # simple: compute average advantage across batch for this objective
                batch_adv = 0.0
                out_texts = []
                lp_lists = []
                for sample in batch:
                    out = generate_and_logprobs(model, tok, sample["prompt"], max_new_tokens=args.max_new_tokens, device=device)
                    rdict = reward_dict(sample, out.text, use_reason=args.use_obj_reason==1, use_brevity=args.use_obj_brevity==1, use_safety=args.use_obj_safety==1)
                    r = rdict.get(obj_name, 0.0)
                    out_texts.append(out.text)
                    lp_lists.append(out.logprobs)
                    batch_adv += float(r)
                batch_adv /= max(1, len(batch))

                # baseline & advantage
                baselines[obj_name] = moving_avg(baselines[obj_name], batch_adv, beta=ema_beta)
                adv = batch_adv - (baselines[obj_name] if baselines[obj_name] is not None else 0.0)

                # build a combined loss over the batch for this objective
                # (mean over sequences; reinforce with scalar advantage)
                losses = []
                for lp in lp_lists:
                    loss = reinforce_loss(lp, advantage=adv * weights[obj_name])
                    losses.append(loss)
                if len(losses)==0:
                    continue
                total_loss = torch.stack(losses).mean()
                total_loss.backward(retain_graph=False)

                # flatten grads from probe params
                gvec = []
                for (n,p) in probe_params:
                    if p.grad is not None:
                        gvec.append(p.grad.detach().float().reshape(-1))
                if len(gvec)>0:
                    grads[obj_name] = torch.cat(gvec, dim=0).to(device)
                else:
                    grads[obj_name] = torch.zeros(1, device=device)

                rewards_batch.append({obj_name: batch_adv})

                # (do not step yet; we might do surgery/weighting across objectives)

            if len(grads)==0:
                continue

            # cosine matrix + conflict stats
            keys, M = cosine_matrix(grads)
            conf = conflict_fraction(M, tau=args.tau)

            # optional weight scheduler (EMA of row-means)
            if args.weight_scheduler==1:
                row_means = {k: float(M[i].mean().item()) for i,k in enumerate(keys)}
                for k in row_means:
                    weights[k] = weights.get(k,1.0) * math.exp(sched_eta * row_means[k])
                weights = clip_and_renorm(weights)

            # optional gradient surgery and apply combined grad to probe slice
            if args.grad_surgery==1 and len(keys)>=2:
                # sum with PCGrad-like projections
                # start from first grad
                ordered = [grads[k] for k in keys]
                gsum = ordered[0].clone()
                for g in ordered[1:]:
                    gsum = project_nonconflicting(gsum, g)
                # zero probe grads then stuff gsum back (scaled)
                for (n,p) in probe_params:
                    if p.grad is not None:
                        p.grad.zero_()
                # scatter gsum proportionally across params' sizes
                offset = 0
                for (n,p) in probe_params:
                    if p.grad is None: continue
                    numel = p.grad.numel()
                    chunk = gsum[offset:offset+numel].reshape_as(p.grad)
                    p.grad.copy_(chunk)
                    offset += numel

            # step
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            model.zero_grad(set_to_none=True)

            # log
            log_obj = {
                "step": step, "epoch": ep,
                "weights": {k: float(v) for k,v in weights.items()},
                "conflict_frac": float(conf),
                "cosine_diag": [float(M[i,i].item()) for i in range(len(keys))],
                "keys": keys,
                "rewards_batch": rewards_batch,
            }
            write(log_obj)

            # dump cosine matrix csv occasionally
            if step % 20 == 0:
                import numpy as np, csv
                csv_path = os.path.join(run_dir, f"cosine_step{step:06d}.csv")
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([""]+keys)
                    for i,k in enumerate(keys):
                        w.writerow([k]+[f"{float(M[i,j].item()):+.4f}" for j in range(len(keys))])

            step += 1

    print(f"Done. See logs in: {run_dir}")

if __name__ == "__main__":
    main()
