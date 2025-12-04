#!/bin/bash

### ----------------------------------------------------
### Start Script for Chatterbox FastAPI Server
### ----------------------------------------------------

ENV_NAME="chatterbox"
APP_DIR="/workspace/chatterbox-api"
APP_FILE="server.py"
PORT=8000
LOG_FILE="/workspace/chatterbox-api/chatterbox.log"

echo "-------------------------------------------"
echo " Starting Chatterbox API $(date)"
echo "-------------------------------------------"

# Load conda
if [ -f "/workspace/miniconda/etc/profile.d/conda.sh" ]; then
    source /workspace/miniconda/etc/profile.d/conda.sh
else
    echo "[ERROR] conda.sh not found!"
    exit 1
fi

# Activate environment
echo "[INFO] Activating conda env: $ENV_NAME"
conda activate $ENV_NAME || {
    echo "[ERROR] Failed to activate conda environment"
    exit 1
}

# Kill old server if running
PID=$(pgrep -f "uvicorn.*$APP_FILE")
if [ ! -z "$PID" ]; then
    echo "[INFO] Killing old server process: $PID"
    kill -9 $PID
fi

# Move into app directory
cd $APP_DIR || {
    echo "[ERROR] Failed to cd into ${APP_DIR}"
    exit 1
}

# Export HuggingFace & Torch cache paths
export HF_HOME="/workspace/.cache/huggingface"
export XDG_CACHE_HOME="/workspace/.cache"
mkdir -p $HF_HOME $XDG_CACHE_HOME

echo "[INFO] Cache directories ready"

# Start FastAPI server
echo "[INFO] Launching server on port $PORT..."

nohup uvicorn server:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    >> $LOG_FILE 2>&1 &

sleep 1

NEW_PID=$(pgrep -f "uvicorn.*$APP_FILE")

if [ ! -z "$NEW_PID" ]; then
    echo "[OK] Chatterbox server running with PID: $NEW_PID"
    echo "[OK] Logs: $LOG_FILE"
else
    echo "[ERROR] Failed to start Chatterbox server!"
fi

echo "-------------------------------------------"
echo " Startup Complete"
echo "-------------------------------------------"