#!/usr/bin/env python3
"""
Merge an Unsloth-Desktop (MLX-format) LoRA adapter into base Gemma weights.

MLX adapter stores, per targeted Linear module P:
    P.lora_a : (in_features, r)
    P.lora_b : (r, out_features)
and forward is  y = x W^T + scale * (x @ lora_a) @ lora_b .
A transformers Linear weight W has shape (out, in), so:
    delta_W(out,in) = scale * (lora_a @ lora_b).T     # add to W

Usage:
  python scripts/mlx_merge.py --adapter <dir> --out outputs/merged_gemma3_1b_v2   # save merged HF checkpoint
  (or import apply_in_place() to merge a loaded model without saving)
"""
import argparse, json, glob
from pathlib import Path

def _adapter_info(adapter):
    cfg = json.load(open(Path(adapter) / "adapter_config.json"))
    lp = cfg.get("lora_parameters", {})
    base = cfg.get("base_model_name_or_path", "unsloth/gemma-3-1b-it")
    scale = float(lp.get("scale", cfg.get("scale", 2.0)))
    keys = lp.get("keys") or []
    return base, scale, keys

def apply_in_place(model, adapter):
    """Add MLX-LoRA deltas into a loaded transformers model. Returns #modules merged."""
    import torch
    from safetensors import safe_open
    _, scale, keys = _adapter_info(adapter)
    wf = glob.glob(str(Path(adapter) / "*.safetensors"))[0]
    sd = model.state_dict()
    merged = 0
    with safe_open(wf, "pt") as f:
        avail = set(f.keys())
        for p in keys:
            ka, kb = f"{p}.lora_a", f"{p}.lora_b"
            wkey = f"{p}.weight"
            if ka not in avail or kb not in avail or wkey not in sd:
                continue
            a = f.get_tensor(ka).float()          # (in, r)
            b = f.get_tensor(kb).float()          # (r, out)
            delta = (a @ b).T * scale             # (out, in)
            W = sd[wkey]
            W.add_(delta.to(W.dtype).to(W.device))
            merged += 1
    return merged

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, scale, keys = _adapter_info(a.adapter)
    print(f"base={base} scale={scale} targeted_modules={len(keys)}")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    n = apply_in_place(model, a.adapter)
    print(f"merged {n} modules")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
    print(f"saved merged checkpoint -> {a.out}")

if __name__ == "__main__":
    main()
