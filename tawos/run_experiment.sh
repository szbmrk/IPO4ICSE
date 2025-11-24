#!/usr/bin/env bash

RUN_EXPORT=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --export)
      RUN_EXPORT=true
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

EXPORT_DIR=$1
BENCHMARK_DIR=$2
MODEL_DIR=$3

if [ -z "$EXPORT_DIR" ] || [ -z "$BENCHMARK_DIR" ] || [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <export_dir> <benchmark_dir> <model_dir> [--export]"
    exit 1
fi

LLAMA_SERVER=~/Projects/llama.cpp/build/bin/llama-server
PYTHON_SCRIPT=src/main.py
VENV_PATH=.venv
SERVER_PORT=8080
TEMPS=(0.2 0.3 0.4 0.5 0.75 1.0)

SERVER_PID=""
PYTHON_PID=""

cleanup() {
    echo ""
    echo "=========================================="
    echo "Interrupt received! Cleaning up..."
    echo "=========================================="
    
    if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo "Stopping Python process (PID: $PYTHON_PID)..."
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
        wait "$PYTHON_PID" 2>/dev/null || true
    fi
    
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping llama-server (PID: $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    
    deactivate 2>/dev/null || true
    
    echo "Cleanup complete. Exiting."
    exit 130
}

trap cleanup SIGINT SIGTERM

check_mysql() {
    echo "Checking MySQL connection..."
    
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    else
        echo "Error: .env file not found"
        return 1
    fi
    
    if command -v mysql &> /dev/null; then
        if mysql -h"${DB_HOST:-localhost}" -u"${DB_USER:-root}" -p"${DB_PASSWORD}" -e "USE ${DB_NAME:-TAWOS};" 2>/dev/null; then
            echo "✓ MySQL connection successful"
            return 0
        else
            echo "✗ Cannot connect to MySQL database"
            return 1
        fi
    else
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
            sleep 5
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

wait_for_port_free() {
    local port=$1
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for port $port to be freed..."
    while [ $attempt -lt $max_attempts ]; do
        local port_in_use=false
        
        if command -v lsof &> /dev/null; then
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                port_in_use=true
            fi
        elif command -v ss &> /dev/null; then
            if ss -tuln | grep -q ":$port "; then
                port_in_use=true
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                port_in_use=true
            fi
        else
            if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
                port_in_use=true
            fi
        fi
        
        if [ "$port_in_use" = false ]; then
            echo "Port $port is free"
            return 0
        fi
        
        if [ $attempt -eq 10 ]; then
            echo "Port still in use, attempting to kill process..."
            if command -v lsof &> /dev/null; then
                PORT_PID=$(lsof -ti :$port 2>/dev/null)
                if [ -n "$PORT_PID" ]; then
                    echo "Killing process $PORT_PID holding port $port"
                    kill -9 $PORT_PID 2>/dev/null || true
                fi
            fi
        fi
        
        attempt=$((attempt + 1))
        sleep 1
    done
    
    echo "Warning: Port $port still in use after timeout, forcing ahead anyway"
    pkill -9 -f "llama-server.*$port" 2>/dev/null || true
    sleep 2
    return 1
}

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

export EXPORT_FOLDER="$EXPORT_DIR"

if [ "$RUN_EXPORT" = true ]; then
    if ! check_mysql; then
        print_mysql_error
        exit 1
    fi

    echo "Exporting data to $EXPORT_DIR..."
    export EXPORT_ENABLED="true"
    python "$PYTHON_SCRIPT" --export --yes
else
    echo "Skipping export..."
fi

export EXPORT_ENABLED="false"
export LOCAL_MODEL_ENABLED="false"

echo "Computing own metrics..."
python "$PYTHON_SCRIPT"
echo "Own metrics computed!"
export LOCAL_MODEL_ENABLED="true"

for MODEL_PATH in "$MODEL_DIR"/*; do
    if [[ -f "$MODEL_PATH" ]]; then
        echo "=========================================="
        echo "Starting model: $(basename "$MODEL_PATH")"
        echo "=========================================="
        "$LLAMA_SERVER" -m "$MODEL_PATH" -ngl -1 --cache-ram 0 --port "$SERVER_PORT" &> llama.log &
        SERVER_PID=$!
        echo "llama-server PID: $SERVER_PID"
        
        if wait_for_server; then
            for TEMP in "${TEMPS[@]}"; do
                echo "Running classification with temp $TEMP..."
                export LOCAL_MODEL_TEMP="$TEMP"
                
                python "$PYTHON_SCRIPT" --skip-own-metrics --skip-filtering &
                PYTHON_PID=$!
                
                wait_for_process "$PYTHON_PID"
                PYTHON_PID=""
            done
            
            echo "Stopping llama-server for model: $MODEL_PATH"
            kill -TERM "$SERVER_PID" 2>/dev/null || true
            sleep 3
            if kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "Force killing llama-server..."
                kill -9 "$SERVER_PID" 2>/dev/null || true
                sleep 2
            fi
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            
            sleep 2
            
            wait_for_port_free "$SERVER_PORT"
            
            echo "✓ Completed all temperatures for $(basename "$MODEL_PATH")"
        else
            echo "Failed to start server for $(basename "$MODEL_PATH")"
            kill -9 "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            
            sleep 2
            
            wait_for_port_free "$SERVER_PORT"
        fi
    fi
done

echo "Creating cleaned CSV files..."
python "$PYTHON_SCRIPT" --skip-own-metrics
echo "Cleaned CSV files created in ${EXPORT_DIR}-cleaned"

echo "Running benchmark..."
export BENCHMARK_FOLDER="$EXPORT_DIR"
export BENCHMARK_OUTPUT="$BENCHMARK_DIR"
python "$PYTHON_SCRIPT" --benchmark

trap - SIGINT SIGTERM

deactivate
echo "Done!"
shutdown now
