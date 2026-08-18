#!/bin/bash
# setup.sh - one-shot setup for the 40GB laptop test bench.
# Installs llama.cpp, downloads the models, builds the Go core, creates venv.
# usage: bash setup.sh   (run from the laptop/ directory)
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/5] system deps (cmake g++ make python3)"
sudo apt-get update -y >/dev/null 2>&1 || true
sudo apt-get install -y cmake g++ make python3 python3-venv curl >/dev/null 2>&1 || true

echo "==> [2/5] llama.cpp (CPU build, incl. llama-server)"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_NATIVE=OFF >/dev/null
cmake --build llama.cpp/build -j"$(nproc)" --target llama-cli llama-server llama-perplexity llama-quantize >/dev/null

echo "==> [3/5] models (the test subjects)"
mkdir -p models
if [ ! -f models/qwen3.8-27b-q4km.gguf ]; then
  echo "  NOTE: copy qwen3.8-27b-q4km.gguf (16.8GB) from the dev box, or download:"
  echo "  hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf"
fi
for m in Qwen3.5-2B-Q4_K_M.gguf Qwen3.5-4B-Q4_K_M.gguf Qwen3-0.6B-Q8_0.gguf; do
  [ -f "models/$m" ] || echo "  missing: models/$m (copy from dev box or hf download)"
done

echo "==> [4/5] python venv (trainer)"
python3 -m venv .venv
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q transformers safetensors

echo "==> [5/5] Go core"
(cd ../slm-fleet/expertd && go build -o expertd .) 2>/dev/null \
  || (cd expertd 2>/dev/null && GOGC=off go build -o expertd . && cp expertd ../expertd 2>/dev/null) || true
[ -f expertd ] || echo "  expertd binary not built - copy from dev box"

echo "==> done. next: bash ab_test.sh"
