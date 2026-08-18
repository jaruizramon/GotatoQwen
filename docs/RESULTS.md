# Results ledger (all measurements from the i7-7700HQ / 15 GB potato)

## The 27B cannot run on 15 GB
- Qwen3.8-27B-FP8 (30.9 GB) downloaded, all 64 layers analyzed
- Q4_K_M GGUF (16.8 GB) built via llama.cpp
- Run attempt: OOM-killed (dmesg) at default 262K context; at `-c 2048` the
  load+repack exceeded 30 minutes. Verdict: not a local-tier model on ≤15 GB.

## Weight-space pruning has no free lunch
- `redundancy.py`: pairwise cosine similarity of down-projection subspaces
  across all 64 layers ≈ 0. Layers are mutually orthogonal in weight space.
- `layer_report.py`: FFN channel saliency is flat — top 25% of channels carry
  ~28% of energy (0.28 at layer 0 → 0.36 at layer 63).
- Conclusion: topics/experts are not separable in the 27B's weights; they
  exist in activations only. "Slicing the LLM" must mean training small
  models from its *output*, not carving its *weights*.

## Fleet speeds (llama.cpp, Q4_K_M, `-t 4`)
| model | generation tok/s |
|---|---|
| Qwen3.5-2B | 16.5 |
| Qwen3.5-4B | 7.4 |
| Qwen3-0.6B + python LoRA | 17–20 |

Thread note: `-t 8` was *slower* than `-t 4` (hyperthreading contention).

## Slicing cost (first-prompt latency budget)
collect ~2 s + LoRA 0.6B/2 epochs ~300 s + convert ~20 s = **326 s**, once.
Adapter: 40 MB on disk. Subsequent loads: milliseconds.

## Deterministic core (Go)
- 3.4 MB static binary, 3 MB RSS vs ~35 MB Python interpreter
- scan 5.1 ms vs 9.0 ms per 10k files; detect ~2 ms vs 48 ms (cold)
- `debug.SetGCPercent(-1)`, zero `:=`, stdlib only
- index signatures byte-identical to the Python prototype
