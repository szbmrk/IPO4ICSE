#!/usr/bin/env bash

set -e  # Exit on error

# Arguments
EXPORT_DIR=$1
BENCHMARK_DIR=$2
MODEL_DIR=$3

if [ -z "$EXPORT_DIR" ] || [ -z "$BENCHMARK_DIR" ] || [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <export_dir> <benchmark_dir> <model_dir>"
    exit 1
fi

# Configuration
LLAMA_SERVER=~/Projects/llama.cpp/build/bin/llama-server
PYTHON_SCRIPT=src/main.py
VENV_PATH=.venv
SERVER_PORT=8080
TEMPS=(0.2 0.4 0.6 0.8)

# Helper functions
check_mysql() {
    echo "Checking MySQL connection..."
    
    # Load environment variables
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    else
        echo "Error: .env file not found"
        return 1
    fi
    
    # Try to connect to MySQL
    if command -v mysql &> /dev/null; then
        if mysql -h"${DB_HOST:-localhost}" -u"${DB_USER:-root}" -p"${DB_PASSWORD}" -e "USE ${DB_NAME:-TAWOS};" 2>/dev/null; then
            echo "✓ MySQL connection successful"
            return 0
        else
            echo "✗ Cannot connect to MySQL database"
            return 1
        fi
    else
        # If mysql client not available, try with Python
        if python3 -c "import pymysql; pymysql.connect(host='${DB_HOST:-localhost}', user='${DB_USER:-root}', password='${DB_PASSWORD}', database='${DB_NAME:-TAWOS}', connect_timeout=5).close()" 2>/dev/null; then
            echo "✓ MySQL connection successful"
            return 0
        else
            echo "✗ Cannot connect to MySQL database"
            return 1
        fi
    fi
}

print_mysql_error() {
    echo ""
    echo "=========================================="
    echo "MySQL Connection Error!"
    echo "=========================================="
    echo "The script requires a working MySQL connection to export data."
    echo ""
    echo "Please ensure:"
    echo "  1. MySQL server is running"
    echo "  2. Database exists and is accessible"
    echo "  3. Credentials in .env are correct"
    echo ""
    echo "To start MySQL:"
    echo "  - Linux: sudo systemctl start mysql"
    echo "  - macOS: brew services start mysql"
    echo ""
    echo "To test connection manually:"
    echo "  mysql -h \$DB_HOST -u \$DB_USER -p \$DB_NAME"
    echo "=========================================="
    echo ""
}
wait_for_server() {
    local max_attempts=60
    local attempt=0
    
    echo "Waiting for llama-server to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
            sleep 5 # Give it a bit more time to settle
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

# Activate venv
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check MySQL connection before starting
if ! check_mysql; then
    print_mysql_error
    exit 1
fi

# 1. Export Data
echo "Exporting data to $EXPORT_DIR..."
export EXPORT_FOLDER="$EXPORT_DIR"
export EXPORT_ENABLED="true"
python "$PYTHON_SCRIPT" --export --yes

# 2. Run Models
export EXPORT_ENABLED="false" # Don't export again

for MODEL_PATH in "$MODEL_DIR"/*; do
    if [[ -f "$MODEL_PATH" ]]; then
        echo "Starting model: $MODEL_PATH"
        "$LLAMA_SERVER" -m "$MODEL_PATH" -ngl 99 --port "$SERVER_PORT" > /dev/null 2>&1 &
        SERVER_PID=$!
        
        if wait_for_server; then
            for TEMP in "${TEMPS[@]}"; do
                echo "Running classification with temp $TEMP..."
                export LOCAL_MODEL_TEMP="$TEMP"
                
                python "$PYTHON_SCRIPT" &
                PYTHON_PID=$!
                
                wait_for_process "$PYTHON_PID"
            done
            
            echo "Stopping llama-server for model: $MODEL_PATH"
            kill "$SERVER_PID"
            wait_for_process "$SERVER_PID"
        else
            echo "Failed to start server for $MODEL_PATH"
            kill "$SERVER_PID" 2>/dev/null
        fi
    fi
done

# 3. Run Benchmark
echo "Running benchmark..."
export BENCHMARK_FOLDER="$EXPORT_DIR"
export BENCHMARK_OUTPUT="$BENCHMARK_DIR"
python "$PYTHON_SCRIPT" --benchmark

deactivate
echo "Done!"
