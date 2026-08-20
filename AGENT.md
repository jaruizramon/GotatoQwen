# AGENT.md — setup guide for LLM agents (v2: the GoTorch era)

> Point an agent at this file with: "Set up GotatoQwen following AGENT.md"

Read it completely before running anything. This file is the contract: the
agent sets up the machine, builds the fleet, slices a stack, and verifies.

## What this project is

A **fragmented LLM**: small SLMs that delegate to one another. Per-stack
experts (base + LoRA, sliced in-session by the watcher), a 2B brain that
routes every prompt, a 1.7B tool executor with a 2B verifier. No python
anywhere. The Go core (expertd, ~3.4MB, GOGC=off) orchestrates; the trainer
(gglora) is Go + hand-written AVX2 C kernels (cgo).

## 0. The path contract (IMPORTANT)

Every machine-specific path is overridable via environment variables; a
fresh checkout requires **zero code edits**:

| Env var | Default | Purpose |
|---|---|---|
| `GOTATO_FLEET` | `$HOME/slm-fleet` | models, adapters, `index.json`, `languages.json`, `subindex.json`, `sessions/`, `stacks/` (per-stack SLM manifests) |
| `GOTATO_LLAMA_BIN` | `$HOME/llama.cpp/build/bin` | llama.cpp binaries (llama-server) |
| `GOTATO_GGLORA` | `<repo>/core/gglora/gglora` | the GoTorch trainer binary |
| `GOTATO_ORACLE_URL` | unset | the Qwen3.8-27B endpoint (`http://host:port`); without it `expertd oracle` needs `--mock` |
| `GOTATO_APPROVE` | unset | `1` lets the headless gateway approve `write_file`/`run_command` tool calls |
| `GOTATO_EXPERD` | `<repo>/core/expertd/expertd` | the gateway binary (bench scripts) |

## 1. Prerequisites

`uname -a; nproc; free -h; gcc --version; go version`
Required: cmake ≥ 3.24, g++ ≥ 11, gcc (the trainer's AVX2 kernels are cgo),
Go ≥ 1.21, ≥ 20 GB free disk, ≥ 12 GB RAM (potato) / ≥ 40 GB (test box).
AVX2+FMA required for the fast kernels (any Intel Haswell+ / AMD Zen+).
Install: `sudo apt-get install -y cmake g++ make curl` (+ go from go.dev).

## 2. Build

```bash
git clone https://github.com/jaruizramon/GotatoQwen.git
cd GotatoQwen
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" ... # or: bash bench/setup.sh
(cd core/expertd && GOGC=off go build -o expertd .)
(cd core/gglora && GOGC=off go build -o gglora .)   # cgo: gcc + AVX2/FMA
```

**Verify**: `./expertd langs` prints the language catalog; `./gglora train 2>&1 | grep usage` prints usage. Discipline: `GOGC=off` is intentional (no GC); every binding is an explicit `var` (zero `:=` in both packages); stdlib only — **except** `core/gglora/gemm.c`, the AVX2 kernels (the whole point: Go 1.22 emits zero FMA instructions; the C kernels reach ~87 GF/s vs ~10 pure Go). The trainer's per-sample tensors live on an explicit manual heap (`core/gglora/heap.go`): one fixed arena, bump-allocated (`heapAllocF32`), explicitly freed (`heapReset` per sample) — the programmer owns the heap, the runtime never collects.

## 3. The fleet (models)

```bash
mkdir -p "$GOTATO_FLEET" && cd "$GOTATO_FLEET"
hf download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf --local-dir .    # the brain
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir .    # hard fallback
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir .      # expert base
hf download lm-kit/Qwen3-1.7B-Instruct-GGUF Qwen3-1.7B-Q4_K_M.gguf --local-dir .  # tool brain
# 40GB box only - the ORACLE (real language analysis; also distillation):
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --local-dir .
```

## 4. The stack workflow (per stack element)

```bash
# 1) watch the stack: scans, and slices an SLM for every language found
nohup expertd watch /path/to/your/stack > /tmp/watcher.log 2>&1 &

# 2) unknown languages (no catalog entry) get analyzed by the ORACLE:
#    - on the 40GB box: run the 27B and point GOTATO_ORACLE_URL at it
#      (llama-server -m Qwen3.8-27B-Q4_K_M.gguf -c 4096 -np 1 --port 8091)
#    - on the potato: expertd oracle /path/to/stack --mock   (same JSON contract)
expertd oracle /path/to/your/stack

# 3) the watcher then builds the new language's SLM automatically
#    (gglora train, ~1-5 min small corpus / ~60 min 40KB corpus, lr 5e-5)
#    every finished SLM lands in index.json AND the per-stack manifest:
expertd stacks            # per-stack SLM collection (launch recipes)
```

The catalog: `expertd langs`. Language detection, scanning, indexing and
routing ALL read the catalog — a new language registered by the oracle is
instantly sliceable, indexable, routable. (The 27B's weights cannot be
carved into experts — measured: redundancy ≈ 0 across its 64 layers. The
oracle contributes knowledge, not weights.)

## 5. The gateway (the delegation hub)

```bash
# run it FROM the stack directory: the cwd IS the stack (tools + manifest)
cd /path/to/your/stack && GOTATO_APPROVE=1 nohup expertd serve > /tmp/gateway.log 2>&1 &
# backends (llama-server, -np 1 is MANDATORY - see gotcha 4):
#   8082 2B generalist (the brain)  8083 4B   8086 1.7B tool brain
#   expert slices autostart on 8084+ from index.json on demand
```

Routing contract (every request):
1. **The 2B brain classifies the topic first** (temperature 0, thinking
   disabled via chat_template_kwargs) — only picks languages in the stack's
   manifest.
2. The inference **switches to that stack's SLM** (autostarted if needed,
   no consent round-trip — the 2B's decision IS the delegation).
3. If the 2B says "general", the Go index/lexical protocol takes over
   (also manifest-aware).
4. If the routed SLM emits `<tool_call>`, the **1.7B executor** runs the
   tool loop (list_dir / read_file / write_file / run_command) and the
   **2B verifier** checks the write before the answer ships.

Endpoints for omp-potato: `GET /v1/models` (only the stack's SLMs),
`GET /manifest` (the launch recipes), `GET /slms` (the roster), plus
OpenAI-shaped `/v1/chat/completions`.

## 6. The tool loop (correctness machinery)

- `write_file` is approval-gated (`GOTATO_APPROVE=1` headless) and writes
  are confined to the stack root.
- **Pre-write fidelity gate**: a write dropping >50% of the read content is
  rejected BEFORE touching disk (fragments never land).
- **Re-read blocking** + **honesty nudge** (an edit task ending without a
  write gets pushed to write) + **2B verification** (the executor's write is
  checked by a second SLM; a failed verdict loops back for repair).
- The transcript rides in the reply: `[tool] ...`, `[reject] ...`,
  `[verify] 2B: VERIFIED|NO ...`.

## 7. The trainer (gglora)

```
gglora train --base <0.6B gguf> --corpus corpus/<lang>.txt --out adapters/<lang>.gguf \
             --threads N --window 128 --name <lang>
```
- Windowed attention (O(t·w)), parallel backward/adam, RoPE tables, AVX2
  kernels (fwd ~36 GF/s, bwd ~87 GF/s single-thread on a 4-core potato).
- `--window 0` = full attention (parity default); the builder uses 128.
- Gradient clipping (max norm 1.0) + NaN abort are ON: a diverging run
  exits 1 and publishes `error` (never a broken adapter). lr is 5e-5
  (2e-4 diverged on a 46KB corpus).
- `expertd build <lang> <stack>` is the full pipeline (collect → train →
  publish index + stack manifest); `expertd watch` spawns it.

## 8. Test harness

```bash
bash bench/testrun.sh        # servers + scoreboard + escalation demo
bash bench/ab_test.sh        # the A/B: fleet vs the 27B on tasks/*
```
Verify: scoreboard.tsv has per-task tok/s; the demo prints the
out-of-scope escalation. On the 40GB box the headline experiment:
**does a per-stack expert beat a same-size generalist?** (`--base-model
unsloth/Qwen3.5-2B` LoRA training needs ≥ 24 GB RAM — only possible there).

## 9. Known gotchas (each was hit and fixed — do not re-encounter)

1. **`pkill -f <pattern>` can kill your own shell** if the pattern appears
   in your command line — use `pkill -x` for exact process names.
2. **`-np 1` on every llama-server** — this llama.cpp build's auto default
   splits the 4096 ctx into 4 slots of ~1024 and every chained/omp prompt
   dies with "Context size has been exceeded".
3. **cgo's CFLAGS whitelist rejects `-mfma`** — the AVX2 kernels use
   `__attribute__((target("avx2,fma")))` instead; do not add -m flags.
4. **Qwen3-family models think in `reasoning_content`** — the tool loop
   parses BOTH content and reasoning; the brain disables thinking via
   `chat_template_kwargs: {enable_thinking: false}` (the 2B base cannot
   classify without it; with it, use fill-in-the-blank prompts).
5. **The 1.7B embellishes** — it claims edits it never made and writes
   fragments. The guards (pre-write gate, honesty nudge, 2B verifier) are
   the fix, not prompt tweaks. Production: 4B+ instruct as the tool brain.
6. **Go does not auto-vectorize** (0 FMA emitted even with GOAMD64=v3).
   The C kernels exist for this; never "simplify" gemm.c back to Go loops.
7. **LoRA divergence** — big corpora NaN at lr 2e-4; lr 5e-5 + clipping.
   A NaN abort means the build failed SAFELY; check build_<lang>.log.
8. **The 27B needs ≥ 24 GB RAM and `-c 4096`** (its DeltaNet state costs
   ~2.4 GB fixed). On 15 GB machines it OOMs — the oracle is mock-only
   there.
9. **The gateway's cwd is the stack** — start it from the stack dir; the
   tools, the manifest and the model list all resolve from cwd.
10. **Watcher vs manual builds race harmlessly** — the builder re-checks
    index.json before publishing (atomic tmp+rename).

## 10. Machine profiles

| | potato (12-16 GB) | test box (40 GB, 32 threads) |
|---|---|---|
| Fleet | 2B + 4B + 0.6B experts | same |
| 27B oracle | ❌ (OOM) | ✅ `-c 4096 -np 1`, GOTATO_ORACLE_URL |
| 2B-base LoRA | ❌ (needs 24 GB) | ✅ `--base-model unsloth/Qwen3.5-2B` |
| Tool brain | 1.7B (guards compensate) | 4B+ instruct (recommended) |
| Threads | `-t 4` (hyperthreading hurts) | `-t 8..16` |

## 11. Definition of done

```bash
expertd langs                       # catalog lists the stack's languages
expertd stacks                      # per-stack manifests exist (launch recipes)
cat "$GOTATO_FLEET/index.json" | grep '"ready"'    # experts built
curl localhost:8090/manifest        # omp-potato's launch config
curl localhost:8090/v1/models       # the stack's SLMs only
# and the delegation demo: "change X in <stack>" -> 2B brain -> stack SLM
# -> 1.7B tools (transcript visible) -> 2B verifier -> file actually changed
```
