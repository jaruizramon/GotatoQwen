#!/bin/bash
# ab_test.sh - the architectural A/B: original 27B vs generalist fleet vs expert.
# Scores every model on the same coding-task set: wall time, tok/s, output file.
# usage: bash ab_test.sh [--quick]
set -uo pipefail
cd "$(dirname "$0")"
M=models
B=llama.cpp/build/bin
OUT=results
mkdir -p "$OUT"

MODELS=(
  "$M/qwen3.8-27b-q4km.gguf:27B-original"
  "$M/Qwen3.5-4B-Q4_K_M.gguf:fleet-4B"
  "$M/Qwen3.5-2B-Q4_K_M.gguf:fleet-2B"
  "$M/Qwen3-0.6B-Q8_0.gguf:expert-base-0.6B"
)
GEN=${GEN:-80}
SCORE="$OUT/scoreboard.tsv"
echo -e "task\tmodel\ts\tok_s\tgen_tok_s" > "$SCORE"

for task in tasks/*.txt; do
  name=$(basename "$task" .txt)
  prompt=$(head -c 2000 "$task")
  echo "=== task: $name ==="
  for entry in "${MODELS[@]}"; do
    file="${entry%%:*}"; label="${entry##*:}"
    [ -f "$file" ] || { echo "  [skip] $label (missing $file)"; continue; }
    out="$OUT/${name}__${label}.txt"
    t0=$(date +%s.%N)
    timeout 1800 $B/llama-cli -m "$file" -p "$prompt" -n $GEN -t 4 -c 4096 \
        --temp 0.3 --log-disable -st > "$out" 2>/dev/null
    t1=$(date +%s.%N)
    dt=$(echo "$t1 - $t0" | bc)
    tps=$(echo "scale=1; $GEN / $dt" | bc)
    gts=$(grep -oE "Generation: [0-9.]+ t/s" "$out" | head -1 | grep -oE "[0-9.]+")
    echo -e "$name\t$label\t$dt\t$tps\t${gts:-na}" >> "$SCORE"
    echo "  [$label] ${dt}s wall (${tps} tok/s incl-load, gen ${gts:-na} t/s)"
  done
done

echo
echo "=== scoreboard ==="
column -t -s$'\t' "$SCORE"
echo "outputs in $OUT/ for quality comparison"
