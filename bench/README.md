# The 40 GB laptop test bench — original Qwen3.8-27B vs the sliced fleet

This bundle runs the architectural A/B that the 15 GB potato could not: the
original 27B, the generalist fleet, and the in-session expert — scored on the
same coding tasks.

## What 40 GB unlocks (vs the 15 GB dev box)

| | 15 GB box | 40 GB laptop |
|---|---|---|
| Qwen3.8-27B Q4_K_M (16.8 GB) | OOM, never ran | **runs, ~2–3 tok/s** (RAM-bandwidth bound) |
| 2B-base expert training (fp32 ≈ 10 GB) | too tight | **fits — the real expertise lever** |
| 27B as local distillation oracle | impossible | **overnight exemplar generation** |
| Fleet 2B/4B/0.6B | 16.5 / 7.4 / 17–20 tok/s | same or faster |

## Expected results (physics, not promises)

- 27B: highest quality, ~2–3 tok/s, ~25–30 GB RSS. The quality champion and
  the distillation source. Never the interactive tier.
- Fleet 4B/2B: 7–16 tok/s. The interactive tier.
- Expert (0.6B or 2B + LoRA): 17–20 tok/s (0.6B) / ~8–10 tok/s (2B), project
  style mimicry. The delegated tier for established stacks.
- The scoreboard tells you whether the expert beats the generalist at equal
  speed — the one genuinely open question.

## Run order

```bash
bash setup.sh            # llama.cpp + models + venv + Go core
bash ab_test.sh          # full A/B on tasks/*.txt -> results/scoreboard.tsv

# then, to make the experts real (the interesting part):
python3 distill.py --lang python --tasks tasks/python.txt --n 8 \
    --corpus-out corpus/python.txt          # 27B generates exemplars (~5-10 min)
python3 train_lora.py --corpus corpus/python.txt --out adapters/python \
    --base-model unsloth/Qwen3.5-2B         # 2B base: real expertise (~30-60 min)
# re-run ab_test.sh; the router (expertd route) now serves the 2B expert
```

## Files

- `setup.sh` — one-shot environment build
- `ab_test.sh` — scoreboard: wall time, tok/s incl. load, gen tok/s per task/model
- `tasks/*.txt` — 5 coding tasks (python/typescript/rust/javascript/generic)
- `distill.py` — 27B → per-language exemplars → corpus (the extend loop)
- `results/` — per-task outputs for quality comparison + `scoreboard.tsv`

## Notes

- Copy from the dev box: `qwen3.8-27b-q4km.gguf` (16.8 GB), the fleet GGUFs,
  `expertd`, `train_lora.py`, `expert-builder.py`, `expert-watcher.py`, `router.py`.
- The 27B wants `llama-server` for repeated use (no per-request reload):
  `llama-server -m models/qwen3.8-27b-q4km.gguf -c 4096 -t 4` then POST /completion.
- Keep context ≤ 4096 on the 27B: the KV cache + DeltaNet state cost ~2.5 GB
  at 4K, but blow up quadratically with context.

## ThinkPad E16 (40 GB) — expected numbers

Run `bash probe.sh` first and paste the output back. The two facts that decide
everything: **RAM type** (DDR4-3200 vs DDR5) and **channel count**.

| Scenario | 27B Q4_K_M decode | Fleet 2B | 2B-expert training |
|---|---|---|---|
| DDR4-3200 dual (~45 GB/s) | ~2.5–3 tok/s | ~25–30 tok/s | ~20–40 min/lang |
| DDR5-5600 dual (~70–80 GB/s) | **~4–4.7 tok/s** | ~35–45 tok/s | ~15–30 min/lang |

Also unlocked on 40 GB (unlike the potato):
- **FP8 (30.9 GB)**: fits (30.9 + 2.4 state + buffers ≈ 35–36 GB) — higher
  fidelity oracle at ~1.5–2 tok/s. Only if the disk has the space.
- **Q6_K (22.9 GB)**: quality middle step, ~2–2.5 tok/s.
- **2B-base LoRA**: `train_lora.py --base-model unsloth/Qwen3.5-2B` — the
  expertise lever the 15 GB box couldn't pull.

Report back: probe.sh output + `results/scoreboard.tsv` from ab_test.sh.
The three decisions that follow: (1) does the expert beat the generalist at
equal speed, (2) is 27B-as-oracle worth an overnight distill run, (3) which
quant/context the E16 can sustain day-to-day.
