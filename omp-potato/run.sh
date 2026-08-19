#!/bin/bash
# omp-potato launcher: execs the omp fork with the potato system prompt.
# The gateway must be running (AGENT.md §5); omp's gateway URL comes from
# its own config (default http://localhost:8090).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OMP_BIN="${OMP_BIN:-$(command -v omp || echo /usr/local/bin/omp)}"
exec "$OMP_BIN" --system-prompt "$(cat "$DIR/potato-prompt.md")" --no-tools "$@"
