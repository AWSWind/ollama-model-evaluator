#!/usr/bin/env bash
set -euo pipefail
curl -s http://localhost:11434/api/generate -d @- <<'EOF' | head -c 800
{"model":"qwen3.6:27b","prompt":"What is 2+2? Answer with just the number.","stream":false,"options":{"num_predict":16}}
EOF
echo
