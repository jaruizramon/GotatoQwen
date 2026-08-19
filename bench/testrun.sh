#!/bin/bash
# testrun.sh - test-run the GotatoQwen fleet in llama.cpp.
#
#   phase 1  status      : models, expert adapter, sub-index, ledger
#   phase 2  servers     : llama-server per SLM (8081 python-expert, 8082 2B, 8083 4B)
#   phase 3  bench       : every task against every server -> scoreboard
#   phase 4  demo        : delegation ladder + out-of-scope escalation (2 turns)
#   phase 5  interactive : keep servers up; you chat via :8081..8083
#
# usage: bash testrun.sh [--no-servers] [--no-bench] [--no-demo] [--interactive]
set -uo pipefail
cd "$(dirname "$0")"
F="${GOTATO_FLEET:-$HOME/slm-fleet}"
B="${GOTATO_LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
EXPERD="${GOTATO_EXPERD:-$(cd "$(dirname "$0")/.." && pwd)/core/expertd/expertd}"
TASKS="${GOTATO_TASKS:-$(cd "$(dirname "$0")" && pwd)/tasks}"
OUT=$F/testrun
mkdir -p "$OUT"
NO_SERVERS=0; NO_BENCH=0; NO_DEMO=0; INTERACTIVE=0
for a in "$@"; do
  case $a in
    --no-servers) NO_SERVERS=1;; --no-bench) NO_BENCH=1;;
    --no-demo) NO_DEMO=1;; --interactive) INTERACTIVE=1;;
  esac
done

echo "== [1/5] STATUS =="
for m in Qwen3.5-2B-Q4_K_M.gguf Qwen3.5-4B-Q4_K_M.gguf Qwen3-0.6B-Q8_0.gguf; do
  [ -f "$F/$m" ] && echo "  ok  $m" || echo "  MISSING $m"
done
[ -f "$F/adapters/python.gguf" ] && echo "  ok  expert adapter python.gguf" || echo "  MISSING python adapter"
[ -f "$F/subindex.json" ] && echo "  ok  sub-index ($(du -h "$F/subindex.json" | cut -f1))" || echo "  MISSING subindex (run: expertd index)"
python3 -c "import json;d=json.load(open('$F/index.json'));print('  ok  index:',{k:v.get('status') for k,v in d.items()})" 2>/dev/null || echo "  MISSING index.json"

# ---- [2/5] servers --------------------------------------------------------
if [ "$NO_SERVERS" = "0" ]; then
  echo "== [2/5] STARTING llama-servers =="
  pkill -x llama-server 2>/dev/null; sleep 1
  nohup $B/llama-server -m $F/Qwen3-0.6B-Q8_0.gguf --lora $F/adapters/python.gguf \
      -t 4 -c 4096 -np 1 --port 8081 > $OUT/server-python.log 2>&1 &
  nohup $B/llama-server -m $F/Qwen3.5-2B-Q4_K_M.gguf \
      -t 4 -c 4096 -np 1 --port 8082 > $OUT/server-2b.log 2>&1 &
  nohup $B/llama-server -m $F/Qwen3.5-4B-Q4_K_M.gguf \
      -t 4 -c 4096 -np 1 --port 8083 > $OUT/server-4b.log 2>&1 &
  sleep 3  # let the binds settle before polling
  for port in 8081 8082 8083; do
    for i in $(seq 1 90); do
      h=$(curl -s "http://localhost:$port/health")
      echo "$h" | grep -q '"ok"' && break
      sleep 2
    done
    echo "  :$port $(curl -s http://localhost:$port/health)"
  done
fi

# ---- [3/5] bench ------------------------------------------------------------
if [ "$NO_BENCH" = "0" ]; then
  echo "== [3/5] BENCH (tasks x servers, n_predict=40) =="
  echo -e "task\tpython-exp:8081\t2b:8082\t4b:8083" | tee "$OUT/scoreboard.tsv"
  for task in "$TASKS"/*.txt; do
    name=$(basename "$task" .txt)
    prompt=$(head -c 1500 "$task")
    row="$name"
    for port in 8081 8082 8083; do
      r=$(curl -s "http://localhost:$port/completion" -d "{\"prompt\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$prompt"),\"n_predict\":40,\"temperature\":0.3}" 2>/dev/null)
      tps=$(echo "$r" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); t=d.get('timings',{})
    n=t.get('predicted_n',0); ms=t.get('predicted_ms',1)
    print(f'{n/(ms/1000):.1f}')
except Exception: print('err')")
      row="$row\t$tps"
    done
    echo -e "$row" | tee -a "$OUT/scoreboard.tsv"
  done
fi

# ---- [4/5] demo -------------------------------------------------------------
if [ "$NO_DEMO" = "0" ]; then
  echo "== [4/5] DELEGATION LADDER + ESCALATION =="
  rm -f "$F/sessions.jsonl"
  cat > /tmp/test_py.py <<'EOF'
def chunked(seq, n):
    """Yield successive n-sized chunks, utils-module style with type hints."""
EOF
  cat > /tmp/test_rs.txt <<'EOF'
Implement pub fn retry<T, E, F>(f: F, attempts: usize, backoff: Duration) in rsutil with exponential backoff, std only.
EOF
  echo "  turn 1: python task -> session s1"
  timeout 120 "$EXPERD" route /tmp/test_py.py --session s1 --scope-check -n 16 2>/dev/null | grep -E "router" | head -2
  echo "  turn 2: rust task, same session -> escalation?"
  timeout 60 "$EXPERD" route /tmp/test_rs.txt --session s1 --scope-check -n 8 2>/dev/null | grep -E "router" | head -3
  echo "  ledger:"
  "$EXPERD" sessions | head -3
fi

# ---- [5/5] interactive ------------------------------------------------------
if [ "$INTERACTIVE" = "1" ] || [ "$NO_SERVERS" = "0" ]; then
  echo "== [5/5] SERVERS UP - chat with them =="
  echo "  python-expert :8081   (web UI: http://localhost:8081)"
  echo "  2B generalist :8082   (web UI: http://localhost:8082)"
  echo "  4B generalist :8083   (web UI: http://localhost:8083)"
  echo "  stop: pkill -x llama-server"
  echo "  scoreboard: $OUT/scoreboard.tsv | ledger: expertd sessions"
fi
