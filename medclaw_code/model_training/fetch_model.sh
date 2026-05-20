#!/usr/bin/env bash
# Download the trained model from the GPU server to this machine.
# Run on your LOCAL machine (WSL), NOT on the server.
#
# Usage:
#   bash model_training/fetch_model.sh root@12.34.56.78
#   bash model_training/fetch_model.sh root@12.34.56.78 --port 2222
#
# The script pulls output/medclaw_final/ from the server into
# model_training/output/medclaw_final/ here, which is exactly what
# serve.sh expects when you run: bash start.sh --model trained

set -e

SERVER="$1"
SSH_PORT="${SSH_PORT:-22}"

# Parse optional --port flag
while [[ $# -gt 1 ]]; do
  case "$2" in
    --port) SSH_PORT="$3"; shift 2 ;;
    *) echo "Unknown option: $2"; exit 1 ;;
  esac
done

if [[ -z "$SERVER" ]]; then
  echo "Usage: bash model_training/fetch_model.sh <user>@<server_ip> [--port 22]"
  exit 1
fi

# Remote path (assumes project was set up at ~/MedClaw on the server)
REMOTE_PATH="${REMOTE_MODEL_PATH:-~/MedClaw/model_training/output/medclaw_final/}"
# Local destination
LOCAL_PATH="$(dirname "$0")/output/medclaw_final"
mkdir -p "$LOCAL_PATH"

echo "=== Fetching trained model ==="
echo "From : $SERVER:$REMOTE_PATH"
echo "To   : $LOCAL_PATH"
echo ""

# Use rsync for resumable transfer (falls back to scp if rsync absent on server)
if command -v rsync &>/dev/null; then
  rsync -avz --progress \
    -e "ssh -p $SSH_PORT" \
    "$SERVER:$REMOTE_PATH" \
    "$LOCAL_PATH/"
else
  echo "rsync not found, falling back to scp..."
  scp -P "$SSH_PORT" -r "$SERVER:$REMOTE_PATH" "$LOCAL_PATH/"
fi

echo ""
echo "=== Download complete ==="
du -sh "$LOCAL_PATH"
echo ""
echo "You can now serve the trained model:"
echo "  bash start.sh --model trained"
