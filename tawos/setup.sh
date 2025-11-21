#!/usr/bin/env bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$HOME/llama.cpp"
MODEL_DIR="$HOME/models"
VENV_PATH="$PROJECT_ROOT/.venv"

# Helper functions
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Check system requirements
check_system_requirements() {
    print_header "Checking System Requirements"
    
    local all_good=true
    
    # Check Python
    if check_command python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_info "Python version: $PYTHON_VERSION"
        
        # Check if version is 3.9 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_success "Python version is 3.9 or higher"
        else
            print_error "Python 3.9 or higher is required"
            all_good=false
        fi
    else
        all_good=false
    fi
    
    # Check Git
    check_command git || all_good=false
    
    # Check CMake
    check_command cmake || all_good=false
    
    # Check Make
    check_command make || all_good=false
    
    # Check curl
    check_command curl || all_good=false
    
    # Check for C++ compiler
    if check_command g++ || check_command clang++; then
        print_success "C++ compiler found"
    else
        print_error "No C++ compiler found (g++ or clang++)"
        all_good=false
    fi
    
    # Check MySQL
    if check_command mysql; then
        if systemctl is-active --quiet mysql 2>/dev/null || pgrep -x mysqld > /dev/null; then
            print_success "MySQL is running"
        else
            print_warning "MySQL is installed but not running"
            print_info "You may need to start it with: sudo systemctl start mysql"
        fi
    else
        print_warning "MySQL client not found - make sure MySQL server is accessible"
    fi
    
    if [ "$all_good" = false ]; then
        print_error "Some required dependencies are missing"
        print_info "Please install missing dependencies before continuing"
        print_info "\nFor Ubuntu/Debian:"
        print_info "  sudo apt update"
        print_info "  sudo apt install python3.9 python3.9-venv python3-pip git cmake build-essential mysql-server curl"
        print_info "\nFor macOS:"
        print_info "  brew install python@3.9 git cmake mysql curl"
        exit 1
    fi
    
    print_success "All system requirements are met"
}

# Setup Python virtual environment
setup_python_env() {
    print_header "Setting Up Python Environment"
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_PATH" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    
    # Check if uv is available
    if command -v uv &> /dev/null; then
        print_info "Using uv to install dependencies..."
        uv sync
    else
        print_info "Installing dependencies with pip..."
        pip install -r requirements.txt
    fi
    
    print_success "Python environment setup complete"
}

# Clone and build llama.cpp
setup_llama_cpp() {
    print_header "Setting Up llama.cpp"
    
    print_info "Target directory: $LLAMA_DIR"
    
    # Create parent directory if it doesn't exist
    LLAMA_PARENT=$(dirname "$LLAMA_DIR")
    mkdir -p "$LLAMA_PARENT"
    
    # Clone llama.cpp if not exists
    if [ ! -d "$LLAMA_DIR" ]; then
        print_info "Cloning llama.cpp repository to $LLAMA_DIR..."
        git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
        print_success "llama.cpp cloned"
    else
        print_info "llama.cpp directory already exists at $LLAMA_DIR"
        read -p "Do you want to update llama.cpp? (y/N): " update_llama
        if [[ "$update_llama" =~ ^[Yy]$ ]]; then
            print_info "Updating llama.cpp..."
            cd "$LLAMA_DIR"
            git pull
            print_success "llama.cpp updated"
        fi
    fi
    
    cd "$LLAMA_DIR"
    
    # Determine build type
    print_info "Detecting GPU support..."
    BUILD_FLAGS=""
    
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        read -p "Build with CUDA support? (Y/n): " use_cuda
        if [[ ! "$use_cuda" =~ ^[Nn]$ ]]; then
            BUILD_FLAGS="-DLLAMA_CUDA=ON"
            print_info "Building with CUDA support..."
        fi
    elif [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        print_success "Apple Silicon detected"
        read -p "Build with Metal support? (Y/n): " use_metal
        if [[ ! "$use_metal" =~ ^[Nn]$ ]]; then
            BUILD_FLAGS="-DLLAMA_METAL=ON"
            print_info "Building with Metal support..."
        fi
    else
        print_info "No GPU detected, building with CPU only..."
    fi
    
    # Build llama.cpp
    if [ -d "build" ]; then
        print_warning "Build directory exists"
        read -p "Do you want to rebuild? (y/N): " rebuild
        if [[ "$rebuild" =~ ^[Yy]$ ]]; then
            rm -rf build
        else
            print_info "Skipping build"
            if [ -f "build/bin/llama-server" ]; then
                print_success "llama-server already built"
                return 0
            fi
        fi
    fi
    
    if [ ! -f "build/bin/llama-server" ]; then
        print_info "Building llama.cpp (this may take several minutes)..."
        mkdir -p build
        cd build
        cmake .. $BUILD_FLAGS
        cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        print_success "llama.cpp built successfully"
    fi
    
    # Verify binary exists
    if [ -f "build/bin/llama-server" ]; then
        print_success "llama-server binary found at: $LLAMA_DIR/build/bin/llama-server"
    else
        print_error "llama-server binary not found"
        exit 1
    fi
}

# Setup model directory
setup_model_directory() {
    print_header "Setting Up Model Directory"
    
    mkdir -p "$MODEL_DIR"
    print_success "Model directory created: $MODEL_DIR"
    
    # Check if models exist
    MODEL_COUNT=$(find "$MODEL_DIR" -name "*.gguf" 2>/dev/null | wc -l)
    
    if [ "$MODEL_COUNT" -eq 0 ]; then
        print_warning "No GGUF models found in $MODEL_DIR"
        print_info "\nYou need to download GGUF format models to run experiments."
        print_info "Recommended models:"
        print_info "  - Llama-3.1-8B-Instruct (Q4_K_M or Q4_1)"
        print_info "  - Mistral-7B-Instruct-v0.3 (Q5_K_M)"
        print_info "  - Phi-3.5-mini-instruct (Q8_0)"
        print_info "\nYou can download models from:"
        print_info "  - https://huggingface.co/models?library=gguf"
        print_info "  - https://huggingface.co/TheBloke"
        print_info "\nPlace .gguf files in: $MODEL_DIR"
    else
        print_success "Found $MODEL_COUNT GGUF model(s)"
        print_info "Models:"
        find "$MODEL_DIR" -name "*.gguf" -exec basename {} \;
    fi
}

# Setup configuration files
setup_configuration() {
    print_header "Setting Up Configuration Files"
    
    cd "$PROJECT_ROOT"
    
    # Setup .env file
    if [ ! -f ".env" ]; then
        print_info "Creating .env file..."
        
        read -p "Enter MySQL host [localhost]: " db_host
        db_host=${db_host:-localhost}
        
        read -p "Enter MySQL user [root]: " db_user
        db_user=${db_user:-root}
        
        read -sp "Enter MySQL password: " db_password
        echo
        
        read -p "Enter database name [TAWOS]: " db_name
        db_name=${db_name:-TAWOS}
        
        cat > .env << EOF
DB_HOST=$db_host
DB_USER=$db_user
DB_PASSWORD=$db_password
DB_NAME=$db_name
EOF
        print_success ".env file created"
    else
        print_info ".env file already exists"
    fi
    
    # Setup config.toml
    if [ ! -f "config.toml" ]; then
        print_info "Creating config.toml from example..."
        cp example.config.toml config.toml
        print_success "config.toml created"
        print_info "You can edit config.toml to customize settings"
    else
        print_info "config.toml already exists"
    fi
}

# Test database connection
test_database_connection() {
    print_header "Testing Database Connection"
    
    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"
    
    print_info "Testing MySQL connection..."
    
    if python3 -c "
import os
from dotenv import load_dotenv
import pymysql

load_dotenv()

try:
    conn = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )
    conn.close()
    print('Connection successful')
    exit(0)
except Exception as e:
    print(f'Connection failed: {e}')
    exit(1)
" 2>&1; then
        print_success "Database connection successful"
    else
        print_error "Database connection failed"
        print_info "Please check your .env file and ensure MySQL is running"
        print_info "You can test manually with: mysql -h <host> -u <user> -p <database>"
        return 1
    fi
}

# Make scripts executable
setup_scripts() {
    print_header "Making Scripts Executable"
    
    cd "$PROJECT_ROOT"
    
    if [ -f "run_experiment.sh" ]; then
        chmod +x run_experiment.sh
        print_success "run_experiment.sh is now executable"
    fi
    
    if [ -f "start_models.sh" ]; then
        chmod +x start_models.sh
        print_success "start_models.sh is now executable"
    fi
    
    chmod +x setup.sh
    print_success "setup.sh is executable"
}

# Print final instructions
print_final_instructions() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}Your TAWOS environment is ready!${NC}\n"
    
    print_info "Next steps:"
    echo "  1. Activate the virtual environment:"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  2. Download GGUF models to: $MODEL_DIR"
    echo "     (If you haven't already)"
    echo ""
    echo "  3. Review and edit configuration:"
    echo "     - .env (database settings)"
    echo "     - config.toml (pipeline settings)"
    echo ""
    echo "  4. Run a test export:"
    echo "     python src/main.py --export --yes"
    echo ""
    echo "  5. Run a full experiment:"
    echo "     ./run_experiment.sh <export_dir> <benchmark_dir> $MODEL_DIR"
    echo ""
    
    print_info "Useful commands:"
    echo "  - Test database: python src/main.py --export --yes"
    echo "  - Run benchmark: python src/main.py --benchmark"
    echo "  - Start llama-server: $LLAMA_DIR/build/bin/llama-server -m <model.gguf> -ngl 99 --port 8080"
    echo ""
    
    print_success "Setup completed successfully!"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --llama-dir)
                LLAMA_DIR="$2"
                shift 2
                ;;
            --model-dir)
                MODEL_DIR="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated setup script for TAWOS environment.

Options:
    --llama-dir DIR     Directory where llama.cpp will be installed
                        (default: $HOME/Projects/llama.cpp)
    
    --model-dir DIR     Directory where GGUF models will be stored
                        (default: $HOME/models)
    
    -h, --help          Show this help message

Examples:
    # Use default directories
    $0
    
    # Specify custom llama.cpp directory
    $0 --llama-dir /opt/llama.cpp
    
    # Specify both directories
    $0 --llama-dir ~/my-llama --model-dir ~/my-models

EOF
}

# Main setup flow
main() {
    print_header "TAWOS Setup Script"
    print_info "This script will set up your TAWOS environment from scratch"
    echo ""
    
    print_info "Configuration:"
    echo "  - llama.cpp directory: $LLAMA_DIR"
    echo "  - Model directory: $MODEL_DIR"
    echo "  - Project root: $PROJECT_ROOT"
    echo ""
    
    check_system_requirements
    setup_python_env
    setup_llama_cpp
    setup_model_directory
    setup_configuration
    setup_scripts
    test_database_connection
    print_final_instructions
}

# Parse arguments and run main setup
parse_args "$@"
main
