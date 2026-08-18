# AGENT.md — setup guide for LLM agents (and humans)

> Point an agent at this file with: "Set up GotatoQwen following AGENT.md"

This file tells an AI agent how to stand up GotatoQwen on a fresh machine,
end to end, with verification checkpoints and known failure modes. Read it
completely before running anything.

## What "setup" means

The repo provides a *fleet* (quantized Qwen SLMs), a *pipeline* (watcher →
per-project LoRA experts), a *sub-index* (token n-grams → labeled sections →
SLM delegation), a *router with session escalation*, and test harnesses.
Setup = build the toolchain, fetch the models, build the Go core, run one
expert end-to-end, and pass the test harness.

## 0. The path contract (IMPORTANT)

All machine-specific paths are overridable via environment variables; a
fresh checkout requires **zero code edits**:

| Env var | Default | Purpose |
|---|---|---|
| `GOTATO_FLEET` | `/home/pipo/slm-fleet` | models, adapters, `index.json`, `sessions.jsonl`, `subindex.json` |
| `GOTATO_LLAMA` | `/home/pipo/llama.cpp` | llama.cpp checkout (for `convert_lora_to_gguf.py`) |
| `GOTATO_LLAMA_BIN` | `$GOTATO_LLAMA/build/bin` | llama.cpp binaries (llama-cli, llama-server, llama-quantize) |
| `GOTATO_VENV_PY` | `/home/pipo/qwen-venv/bin/python` | the project venv's python |

Set them once, export them, and every component (Go core + Python pipeline +
bench scripts) honors them. The Go core reads them in `applyEnv()` (core/
expertd/main.go); the Python scripts read them via `os.environ.get(...)`.

## 1. Prerequisites (detect, don't assume)

```bash
uname -a; nproc; free -h; df -h /; nvidia-smi 2>/dev/null | head -3
cmake --version; g++ --version | head -1; go version; python3 --version
```

Required: cmake ≥ 3.24, g++ ≥ 11, Go ≥ 1.21, Python ≥ 3.10, ~20 GB free
disk for the potato config (60 GB for the 40 GB config with the 27B), ≥ 12 GB
RAM for the potato config (the 27B itself needs ≥ 24 GB).
Install on Ubuntu/Debian: `sudo apt-get install -y cmake g++ make python3 python3-venv curl`
(Go: https://go.dev/dl if missing.)

## 2. Build llama.cpp (CPU; CUDA/Vulkan optional)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$GOTATO_LLAMA"
cmake -S "$GOTATO_LLAMA" -B "$GOTATO_LLAMA/build" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_NATIVE=OFF
cmake --build "$GOTATO_LLAMA/build" -j"$(nproc)" \
      --target llama-cli llama-server llama-perplexity llama-quantize
export GOTATO_LLAMA_BIN="$GOTATO_LLAMA/build/bin"
```

**Verify**: `"$GOTATO_LLAMA_BIN/llama-cli" --help | head -3` exits 0.
Note: on 4-core/8-thread CPUs use `-t 4` (hyperthreading hurts; measured
16.5 tok/s at t=4 vs 10.8 at t=8 on an i7-7700HQ). The finetune tool
(`llama-finetune`) is broken for this use (GGML_ASSERT in backward graph) —
do not use it; use the Python LoRA trainer.

## 3. Python venv

```bash
python3 -m venv "$HOME/gotato-venv"
"$HOME/gotato-venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$HOME/gotato-venv/bin/pip" install -q transformers safetensors gguf huggingface_hub numpy
export GOTATO_VENV_PY="$HOME/gotato-venv/bin/python"
```

**Verify**: `"$GOTATO_VENV_PY" -c "import torch, transformers, safetensors; print('ok')"`.
PEP 668 ("externally managed environment") is expected on modern distros —
always use the venv, never system pip.

## 4. Fleet models (HF)

```bash
mkdir -p "$GOTATO_FLEET"
"$GOTATO_VENV_PY" -m pip install -q -U "huggingface_hub[hf_transfer]" 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$GOTATO_FLEET"
hf download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf --local-dir .
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir .
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir .
# training base (HF format, for the LoRA trainer):
hf download Qwen/Qwen3-0.6B --include "*.json" "*.safetensors" "*.txt" --exclude "*index*"
```

**Verify**: three .gguf files present; `du -sh "$GOTATO_FLEET"` ≥ 5 GB.
On the 40 GB config also fetch the 27B: `hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf`
(17 GB; needs ≥ 24 GB RAM to run).

## 5. Go core

```bash
cd core/expertd && go mod tidy && GOGC=off go build -o expertd .
cd ../gglora && go mod tidy && GOGC=off go build -o gglora .
```

**Verify**: `./expertd` prints usage with `scan|detect|watch|route|bench|index|resolve|sessions`.
Discipline (do not "fix"): `GOGC=off` is intentional (no garbage collector);
the code uses zero `:=` declarations; stdlib only — no new dependencies.

## 6. One expert end-to-end (the pipeline)

Create a demo stack (a small project with a couple of `.py` files, ≥ 80 bytes
each), then:

```bash
nohup ./expertd watch /path/to/stack > /tmp/watcher.log 2>&1 &
# the watcher spawns the builder: collect corpus -> LoRA (torch, 2 threads) -> convert
# first build takes ~5-6 min on a potato (0.6B base, 2 epochs, small corpus)
```

**Verify**: `cat "$GOTATO_FLEET/index.json"` shows
`{"python": {"status": "ready", "lora": "adapters/python.gguf", ...}}`.
If `status: "error"`, read `"$GOTATO_FLEET/build_python.log"` (the builder
appends its full log there). If the watcher respawns failed builds, wait for
the 300 s error backoff before investigating. The training base for a larger
expert: `pipeline/train_lora.py --base-model unsloth/Qwen3.5-2B` (needs
≥ 24 GB RAM; the potato's 15 GB can only train the 0.6B base).

## 7. Sub-index + session ledger

```bash
./expertd index /path/to/stack        # -> "$GOTATO_FLEET/subindex.json"
./expertd resolve <file|text>         # semantic hit -> {label, lang, score}
./expertd sessions                    # which SLM owns which session
```

**Verify**: `resolve` on a code task prints a top hit with a language; a
nonsense string prints `no hits -> tier-3: streamed LLM`.

## 7.5 The gateway (the llama harness)

`expertd serve` (default :8090) fronts the llama-servers and runs the scope
protocol on EVERY request. Use the gateway instead of hitting llama-server
directly - the raw server has no scope awareness.

```bash
curl http://localhost:8090/completion -d '{"prompt":"...", "n_predict":64, "session":"s1"}'
```

Behavior: in-scope -> forwarded to the owning SLM (X-Gotato-Backend header);
out-of-scope vs session owner -> asks "shall we delegate an SLM for X?";
no slice running -> asks "add a X element to the stack..."; user says
"yes" -> gateway autostarts the slice's llama-server (expert adapter from
index.json) on :8084+, adopts the session owner, and the next task hits it.
Sessions ledger: `expertd sessions`. Backends config: serve.go
defaultServeConfig() - the python expert is :8081, 2B :8082, 4B :8083.

### Context chaining (automatic)

The gateway chains contexts when a session approaches its cap (default 3072
tokens, per-request override `"chain_cap": N`): it summarizes the archived
history with the 2B SLM, seeds a fresh context with [summary + recent turns],
and continues - nothing is lost (the full archive lives in
`$GOTATO_FLEET/sessions/<id>.jsonl`, storage is cheap). Chained responses
carry `X-Gotato-Chain: true` and a visible `[context chained...]` prefix;
the ledger records `context-chain` events. This is orchestration-level, not
a llama.cpp C patch - upstream's native answer is a lossy KV-cache shift.

### The TUI harness (`expertd chat`)

Interactive chat with live SLM visibility: while the backend streams, the
status line shows `Thinking [python-expert · utils] 3.2s` with the SLM tag in
reverse video; the answer prints under a footer with the tag + tok/s; the
terminal bell rings when the delegated SLM switches. Topic suffixes come from
the gateway's `X-Gotato-SLM` header (base · topic, e.g. `python-expert ·
utils`, `rust-slice · rust`). Requires the gateway + at least one backend
running. Streaming is SSE passthrough (`"stream": true`).

### The language bridge (`--bridge zh`)

English in → Chinese for the SLM → English back out. Determinism contract:
(1) fixed layer (templates/messages) is table-driven; (2) free-form uses
greedy (the gateway FORCES temperature:0 on every forward - without it
llama-server's default sampling makes output random) plus a disk translation
cache (`$GOTATO_FLEET/translations/zh/<hash>.txt`), so identical input is
byte-identical output (verified); (3) code blocks (``` fences) are never
translated. The translator is an INSTRUCT model (lm-kit/Qwen3-1.7B-Instruct
Q4_K_M on :8086) via /v1/chat/completions with a hard system directive - base
models narrate instead of translating. Warm slots break determinism:
cache_prompt is always false (the chain feature rebuilds prefixes instead).

## 8. Test harness

```bash
bash bench/testrun.sh     # servers on :8081-8083 + scoreboard + escalation demo
bash bench/ab_test.sh     # the A/B: fleet vs (optionally) the 27B on tasks/*
```

**Verify**: scoreboard.tsv has per-task tok/s; the testrun demo prints
`destination hit out of scope - shall we delegate an SLM for rust?` on the
second turn. If servers fail to load, check `"$GOTATO_FLEET/testrun/server-*.log"`
and RAM headroom (`free -h` — the 27B needs ≥ 24 GB).

## 9. Known gotchas (each was hit and fixed — do not re-encounter)

1. **`gguf` f16 tensors are IEEE half, not bf16** — the `u32<<16` trick is
   bf16-only. `core/gglora/gguf.go` decodes correctly; do not "simplify" it.
2. **GGUF metadata**: bool values are 1 byte; u8/i8 1 byte; u16/i16 2 bytes.
3. **`llama-cli` one-shot** needs `-st` (single-turn); with a chat template
   the default is interactive mode.
4. **`pkill -f <pattern>` can kill your own shell** if the pattern appears in
   the command line — use `pkill -x` for exact process names.
5. **OOM**: always pass `-c 2048..4096` (default context is the model's full
   262K and its KV cache alone can OOM a 15 GB box); the DeltaNet state costs
   ~2.4 GB fixed regardless of context.
6. **`llama-quantize` refuses Q8→F16** without `--allow-requantize`.
7. **numpy ≥ 2.3 removed float8 dtypes** — the FP8 analysis tools decode
   E4M3 manually (`pipeline/../core` tooling); do not rely on numpy for FP8.
8. **LoRA parity is verified** (core/gglora vs torch, bit-exact) — if you
   change the math, re-run the dump-and-compare harness against torch.
9. **The 27B's weights contain no separable experts** (redundancy ≈ 0 across
   64 layers) — "slicing" means training small models from its *output*, not
   carving its weights. This is a design fact, not a TODO.

## 10. Machine profiles

| | potato (12-16 GB) | test box (40 GB) |
|---|---|---|
| Fleet | 2B + 4B + 0.6B-expert | same |
| 27B Q4 | ❌ (OOM) | ✅ ~2-3 tok/s, `-c 4096`, use `llama-server` |
| 2B-base LoRA | ❌ (needs 24 GB) | ✅ `--base-model unsloth/Qwen3.5-2B` |
| Distillation | ❌ | ✅ `bench/distill.py` (27B as oracle, ~2 tok/s) |
| GPU (if NVIDIA ≥ 4 GB) | torch 2.3.1+cu121 for fp16 0.6B LoRA (7× faster; pin torch ≤ 2.4 for Maxwell sm_50) | check `nvidia-smi`; MX550-class runs 2B-base |

## 11. Definition of done

```bash
bash bench/testrun.sh 2>&1 | grep -E "out of scope"   # escalation fires
cat "$GOTATO_FLEET/index.json" | grep '"ready"'       # expert built
cat "$GOTATO_FLEET/testrun/scoreboard.tsv"            # numbers > 0
```
