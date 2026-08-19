# GotatoQwen

**Go + potato + Qwen: slice a 27B-class LLM into stack-aware SLMs that run on
potato hardware.** Per-project experts, generated in-session, stored on disk
(storage is cheap), served by a deterministic no-GC Go core.

The name is the architecture: a **Go**-based deterministic core running
**Qwen**-derived SLM fleets on **potato** PCs.

A dense 27B model reads ~17 GB of weights per token — a 2017 laptop delivers
~1.5 tok/s, a 15 GB box cannot run it at all (measured: OOM). This project
doesn't fight the physics: it routes around it. The big model scaffolds and
distills (API tier, or locally on a 40 GB machine); the potato runs a fleet
of small models (2–4B generalists) plus per-project LoRA experts that are
baked in-session from an index of the project's own code.

```
        API tier (27B) ── scaffolds, plans, distills exemplars
              │  rare, quality-critical
              ▼
   [router]  "indexed model"  ── capability index + language detection
              │  (Go, no GC, ~3 MB RSS)
              ▼
   local fleet:  2B/4B generalists (16.5 / 7.4 tok/s on an i7-7700HQ)
                 + per-project experts: base + LoRA adapter (40 MB, on disk)
                 [expert-watcher] watches the stack; new language →
                 collect corpus → train LoRA in-session (2 threads, nice 10)
                 → publish to index → router serves the expert
```

## Measured ledger (i7-7700HQ, 15 GB — the potato)

| Claim | Result |
|---|---|
| Dense 27B (Q4_K_M, 16.8 GB) on 15 GB RAM | **Cannot run** — OOM at default context; 30+ min load at small ctx |
| Fleet 2B / 4B (Q4, `-t 4`) | **16.5 / 7.4 tok/s** |
| Expert (0.6B + in-session LoRA) | **17–20 tok/s**, adapter 40 MB |
| Slicing cost (first prompt) | **326 s** once (collect + LoRA + convert), then instant |
| Deterministic core | Go, `GOGC=off`, zero `:=`, stdlib-only: **3.4 MB binary, 3 MB RSS**, byte-parity with the Python prototype |
| Pruning the 27B's weights | **No free lunch** — 64-layer redundancy ≈ 0, FFN saliency flat (top 25% ≈ 28% energy). Topics live in activations, not weight slices |

## Status — what is and isn't proven

**Proven**: the physics (27B cannot run on ≤15 GB; the fleet can, at 10–20×
the speed); the pipeline (watcher → collect → LoRA → GGUF adapter → router
dispatch, end-to-end, with index parity between Python and Go); the slicing
latency budget.

**Pending** (see `bench/`, needs a 40 GB machine): the headline claim —
*does a per-project expert beat a same-size generalist at equal speed?*
The A/B scoreboard (`bench/ab_test.sh`) answers it on a real task set.
Also pending: the 27B as a local distillation oracle for the "extend" loop.

## Layout

- `core/expertd` — the deterministic Go core (scan / detect / watch /
  build / oracle / langs / stacks / serve) — no GC, no `:=`, stdlib only
- `core/gglora` — the GoTorch trainer: Go orchestration + hand-written
  AVX2 C kernels (cgo), windowed attention, gradient clipping, NaN abort
- `bench/` — the 40 GB test bundle: setup, probe, A/B scoreboard, task set
- `omp-potato/` — the thin-client fork config (prompt + launcher)

## Quickstart (potato)

```bash
(cd core/expertd && go build -o expertd .)
(cd core/gglora && go build -o gglora .)   # needs gcc + AVX2/FMA
./expertd watch /path/to/your/stack        # daemon: watches, slices experts
./expertd oracle /path/to/your/stack       # 27B analyzes unknown languages
cd /path/to/your/stack && expertd serve    # the gateway (omp-potato connects)
```

## On the 40 GB machine (the ThinkPad)

```bash
bash bench/setup.sh                         # deps + llama.cpp + fleet + Go core
bash bench/testrun.sh                       # servers + scoreboard + demo
# the 27B as the REAL oracle (real language analysis instead of --mock):
llama-server -m Qwen3.8-27B-Q4_K_M.gguf -c 4096 -np 1 --port 8091 &
GOTATO_ORACLE_URL=http://localhost:8091 expertd oracle /path/to/your/stack
# the headline experiment - does a per-stack expert beat a same-size
# generalist? (2B-base LoRA, needs >= 24 GB):
bash bench/ab_test.sh
```

## License

Apache-2.0. Models: Qwen3.8-27B / Qwen3.5 are Apache-2.0.
