#!/usr/bin/env bash

MODEL_DIR=~/models
LLAMA_SERVER=~/Projects/llama.cpp/build/bin/llama-server
PYTHON_SCRIPT=~/Projects/IPO4ICSE/tawos/src/main.py
VENV_PATH=~/Projects/IPO4ICSE/tawos/.venv
PROJECT_DIR=~/Projects/IPO4ICSE/tawos
SERVER_PORT=8080

# Track PIDs for cleanup
SERVER_PID=""
PYTHON_PID=""

# Cleanup function
cleanup() {
    echo ""
    echo "=========================================="
    echo "Interrupt received! Cleaning up..."
    echo "=========================================="
    
    # Kill Python process if running
    if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo "Stopping Python process (PID: $PYTHON_PID)..."
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
        wait "$PYTHON_PID" 2>/dev/null || true
    fi
    
    # Kill llama-server if running
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping llama-server (PID: $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    
    # Deactivate venv
    deactivate 2>/dev/null || true
    
    echo "Cleanup complete. Exiting."
    exit 130
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

check_mysql() {
    echo "Checking MySQL connection..."
    
    cd "$PROJECT_DIR"
    
    # Load environment variables
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    else
        echo "Error: .env file not found"
        return 1
    fi
    
    # Try to connect to MySQL using Python (more reliable than mysql client)
    if source "$VENV_PATH/bin/activate" && python3 -c "import pymysql; pymysql.connect(host='${DB_HOST:-localhost}', user='${DB_USER:-root}', password='${DB_PASSWORD}', database='${DB_NAME:-TAWOS}', connect_timeout=5).close()" 2>/dev/null; then
        echo "✓ MySQL connection successful"
        deactivate
        return 0
    else
        echo "✗ Cannot connect to MySQL database"
        deactivate 2>/dev/null
        echo ""
        echo "=========================================="
        echo "MySQL Connection Error!"
        echo "=========================================="
        echo "Please ensure MySQL is running and accessible."
        echo "To start MySQL:"
        echo "  - Linux: sudo systemctl start mysql"
        echo "  - macOS: brew services start mysql"
        echo "=========================================="
        return 1
    fi
}

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

# Check MySQL before starting
if ! check_mysql; then
    exit 1
fi

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
            PYTHON_PID=""  # Clear after process completes
            
            echo "Stopping llama-server for model: $MODEL_PATH"
            kill "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""  # Clear after stopping
        else
            echo "Skipping model due to server startup failure"
            kill "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
        fi
    fi
done

# Disable trap before normal exit
trap - SIGINT SIGTERM

if [[ "$1" == "--shutdown" ]]; then
    shutdown now
fi

echo "All models processed."
