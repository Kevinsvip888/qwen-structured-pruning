#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/qwen-pruning-lab
source /root/autodl-tmp/qwen-inference-lab/.venv-serve/bin/activate
method="${1:?Usage: bash tools/serve_model.sh BF16|RandomChannels|ActivationEnergy}"
case "$method" in
  BF16) model="/root/autodl-tmp/qwen-inference-lab/models/Qwen3-4B-Instruct-2507"; alias="qwen-prune-bf16" ;;
  RandomChannels) model="./models/Qwen3-4B-Instruct-2507-MLP9216-RandomChannels"; alias="qwen-prune-random" ;;
  ActivationEnergy) model="./models/Qwen3-4B-Instruct-2507-MLP9216-ActivationEnergy"; alias="qwen-prune-energy" ;;
  *) echo "Unknown method: $method" >&2; exit 1 ;;
esac
test -f "$model/config.json"
python - <<'PY'
import socket
with socket.socket() as s:
    s.settimeout(1)
    if s.connect_ex(('127.0.0.1', 8000)) == 0:
        raise SystemExit('Port 8000 is occupied. Stop the previous server first.')
PY
export HF_HUB_OFFLINE=1
exec vllm serve "$model" --served-model-name "$alias" \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --kv-cache-memory-bytes 4294967296 \
  --attention-backend FLASH_ATTN --max-model-len 8192 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --generation-config vllm --no-enable-prefix-caching \
  --seed 42 --host 127.0.0.1 --port 8000
