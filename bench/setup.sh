#!/bin/bash
# setup.sh - one-shot setup for the 40GB test bench (the ThinkPad).
# Installs llama.cpp, downloads the models, builds the Go core + the C-SIMD
# trainer. NO python anywhere. All paths are env-overridable (the path
# contract): GOTATO_FLEET, GOTATO_LLAMA, GOTATO_LLAMA_BIN.
# usage: bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

export GOTATO_FLEET="${GOTATO_FLEET:-$HOME/slm-fleet}"
export GOTATO_LLAMA="${GOTATO_LLAMA:-$HOME/llama.cpp}"
export GOTATO_LLAMA_BIN="${GOTATO_LLAMA_BIN:-$GOTATO_LLAMA/build/bin}"
REPO="$(cd .. && pwd)"

echo "==> [1/5] system deps (cmake g++ make curl go)"
sudo apt-get update -y >/dev/null 2>&1 || true
sudo apt-get install -y cmake g++ make curl >/dev/null 2>&1 || true
command -v go >/dev/null || echo "  MISSING go (https://go.dev/dl) - install before continuing"
gcc --version >/dev/null 2>&1 || echo "  MISSING gcc - needed for the AVX2 trainer kernels (cgo)"

echo "==> [2/5] llama.cpp (CPU build, incl. llama-server)"
if [ ! -d "$GOTATO_LLAMA" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$GOTATO_LLAMA"
fi
cmake -S "$GOTATO_LLAMA" -B "$GOTATO_LLAMA/build" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_NATIVE=OFF >/dev/null
cmake --build "$GOTATO_LLAMA/build" -j"$(nproc)" --target llama-server llama-cli >/dev/null

echo "==> [3/5] models (the fleet + the 27B oracle)"
mkdir -p "$GOTATO_FLEET"
command -v hf >/dev/null || pip install -q huggingface_hub[hf_transfer] 2>/dev/null || true
cd "$GOTATO_FLEET"
for m in Qwen3.5-2B-Q4_K_M.gguf Qwen3.5-4B-Q4_K_M.gguf Qwen3-0.6B-Q8_0.gguf Qwen3-1.7B-Q4_K_M.gguf; do
  if [ ! -f "$m" ]; then
    echo "  fetching $m (hf download unsloth/...)"
    hf download "unsloth/${m%-Q4_K_M.gguf}-GGUF" "$m" --local-dir . 2>/dev/null \
      || echo "  MISSING $m - copy from the dev box (or set HF_TOKEN)"
  fi
done
if [ ! -f Qwen3.8-27B-Q4_K_M.gguf ]; then
  echo "  NOTE: the 27B oracle (17GB): hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf"
  echo "        or copy from the dev box. Without it, expertd oracle falls back to --mock."
fi
# 1.7B instruct (the tool brain): the fleet file above covers it when the
# download names match; otherwise copy it from the dev box.

echo "==> [4/5] Go core (expertd) + GoTorch trainer (gglora, C kernels)"
cd "$REPO/core/expertd" && GOGC=off go build -o expertd .
cd "$REPO/core/gglora" && GOGC=off go build -o gglora .
export GOTATO_GGLORA="$REPO/core/gglora/gglora"

echo "==> [5/5] verify"
"$GOTATO_LLAMA_BIN/llama-server" --version >/dev/null 2>&1 && echo "  ok llama-server"
"$REPO/core/expertd/expertd" langs >/dev/null 2>&1 && echo "  ok expertd (langs)"
"$REPO/core/gglora/gglora" train 2>&1 | grep -q usage && echo "  ok gglora"

echo "==> done. next:"
echo "    expertd watch <your-stack>          # slice per-stack SLMs"
echo "    expertd oracle <stack> --mock       # or set GOTATO_ORACLE_URL=<27B llama-server>"
echo "    expertd serve                       # the gateway (:8090) - omp-potato connects here"
