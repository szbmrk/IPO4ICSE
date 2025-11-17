#!/usr/bin/env bash
MODEL_DIR=~/models
LLAMA_SERVER=~/Projects/llama.cpp/build/bin/llama-server
PYTHON_SCRIPT=~/Projects/IPO4ICSE/tawos/src/main.py
VENV_PATH=~/Projects/IPO4ICSE/tawos/.venv
SERVER_PORT=8080

wait_for_server() {
    local max_attempts=60
    local attempt=0
    
    echo "Waiting for llama-server to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
            sleep 20
            echo "Server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    echo "Server failed to start within timeout"
    return 1
}

wait_for_process() {
    local pid=$1
    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
    done
}

for MODEL_PATH in "$MODEL_DIR"/*; do
    if [[ -f "$MODEL_PATH" ]]; then
        echo "Starting model: $MODEL_PATH"
        "$LLAMA_SERVER" -m "$MODEL_PATH" -ngl 99 --port "$SERVER_PORT" &
        SERVER_PID=$!
        
        if wait_for_server; then
            echo "Running Python script: $PYTHON_SCRIPT"
            source "$VENV_PATH/bin/activate"
            python "$PYTHON_SCRIPT" &
            PYTHON_PID=$!
            deactivate
            
            echo "Waiting for Python script to finish..."
            wait_for_process "$PYTHON_PID"
            
            echo "Stopping llama-server for model: $MODEL_PATH"
            kill "$SERVER_PID"
            wait "$SERVER_PID" 2>/dev/null
        else
            echo "Skipping model due to server startup failure"
            kill "$SERVER_PID" 2>/dev/null
        fi
    fi
done

echo "All models processed."
