#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs/qwen3_4b}"
mkdir -p "$OUT_DIR"

echo "[monitor] log: $OUT_DIR/train.log"
echo "[monitor] progress: $OUT_DIR/progress.json"
echo "[monitor] stop with Ctrl-C (this only stops monitoring)"

while true; do
  clear || true
  date
  if [[ -f "$OUT_DIR/progress.json" ]]; then
    python - "$OUT_DIR/progress.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps(d, ensure_ascii=False, indent=2))
except Exception as e:
    print(f"progress unavailable: {e}")
PY
  else
    echo "progress.json not created yet"
  fi
  echo
  nvidia-smi || true
  sleep 10
done
