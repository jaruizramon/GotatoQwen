#!/usr/bin/env python3
"""train_lora.py - deterministic CPU LoRA trainer (no peft).

The LoRA math is written out explicitly (~40 lines of torch), because this is
the exact computation a Go/C port would reimplement, and because peft is a
moving dependency. Emits peft-compatible files (adapter_model.safetensors +
adapter_config.json) so llama.cpp's convert_lora_to_gguf.py can produce a GGUF
adapter for inference.

Discipline (per project design):
  - exactly 2 threads (the "expert pair"); the interactive fleet keeps 4
  - fp32, batch size 1, fixed seed -> bit-reproducible training runs
  - all memory is explicit torch tensors; nothing cached implicitly

Usage:
  train_lora.py --corpus corpus/python.txt --out adapters/python [--epochs 2]
"""
import argparse, json, os, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE = "Qwen/Qwen3-0.6B"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")

torch.set_num_threads(2)
torch.manual_seed(0)


class LoRALinear(nn.Module):
    """y = Wx + (alpha/r) * B(Ax); W frozen. A: [r, in], B: [out, r]."""
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())


def wrap(model: nn.Module, r: int, alpha: float) -> list[str]:
    names = []
    for name, m in list(model.named_modules()):
        if name.endswith(TARGETS) and isinstance(m, nn.Linear) and "." in name:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            setattr(parent, attr, LoRALinear(m, r, alpha))
            names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE,
                    help="HF id (e.g. unsloth/Qwen3.5-2B on the 40GB box)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    text = open(a.corpus, encoding="utf-8", errors="replace").read()
    tok = AutoTokenizer.from_pretrained(a.base_model)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < 32:
        raise SystemExit(f"corpus too small: {len(ids)} tokens")

    # sliding-window samples: deterministic, no shuffling
    samples = [ids[i:i + a.ctx] for i in range(0, len(ids), a.stride)]
    samples = [s for s in samples if len(s) >= 32]
    print(f"[train] {len(ids)} tokens -> {len(samples)} samples (ctx={a.ctx})",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, torch_dtype=torch.float32)
    model.config.use_cache = False
    model.train()
    wrapped = wrap(model, a.rank, a.alpha)
    print(f"[train] LoRA on {len(wrapped)} modules "
          f"(rank {a.rank}, alpha {a.alpha})", flush=True)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in lora_params)
    print(f"[train] trainable params: {n:,} ({n / model.num_parameters() * 100:.2f}%)",
          flush=True)

    opt = torch.optim.AdamW(lora_params, lr=a.lr)
    t0 = time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for s in samples:
            x = torch.tensor([s], dtype=torch.long)
            opt.zero_grad()
            out = model(x, labels=x)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            tot += float(out.loss)
        print(f"[train] epoch {ep + 1}/{a.epochs} loss={tot / len(samples):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- save in peft layout --------------------------------------------
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            tgt = name.rsplit(".", 1)[-1]
            base = f"base_model.model.{name}"
            sd[f"{base}.lora_A.weight"] = m.A.detach().clone().to(torch.float32)
            sd[f"{base}.lora_B.weight"] = m.B.detach().clone().to(torch.float32)
    from safetensors.torch import save_file
    save_file(sd, os.path.join(a.out, "adapter_model.safetensors"))
    cfg = {
        "r": a.rank, "lora_alpha": a.alpha, "lora_dropout": 0.0, "bias": "none",
        "task_type": "CAUSAL_LM", "target_modules": list(TARGETS),
        "base_model_name_or_path": a.base_model,
    }
    json.dump(cfg, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    print(f"[train] saved adapter -> {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
