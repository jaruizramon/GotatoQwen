#!/bin/bash
# probe.sh - dump the hardware facts that decide this project's future.
# Run on the E16 (or any test machine) and paste the output back.
echo "=== CPU ==="; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Flags" | head -6
echo "=== MEMORY TOPOLOGY ==="
sudo -n dmidecode -t memory 2>/dev/null | grep -E "Size:|Speed:|Locator:|Configured" | head -12 \
  || echo "(no root - run: sudo dmidecode -t memory | grep -E 'Size:|Speed:' )"
echo "=== RAM ==="; free -h | head -2
echo "=== DISK ==="; df -h / | tail -1
echo "=== OS ==="; uname -a
echo "=== llama-bench (bandwidth probe with the actual 27B Q4) ==="
llama.cpp/build/bin/llama-bench -m models/qwen3.8-27b-q4km.gguf -p 128 -n 32 -t 4 2>/dev/null \
  | grep -E "PP|TG" || echo "(run after setup.sh)"
