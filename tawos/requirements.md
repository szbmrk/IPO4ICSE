# TAWOS Data Cleaning Pipeline Requirements

This document outlines the full system requirements and specifications needed to run the TAWOS (The Agile Way of Software Engineering) data export and classification pipeline.

> **Quick Start**: If you want to set up everything automatically, run `./setup.sh` after cloning the repository. The automated setup script will install all dependencies, build llama.cpp, and configure your environment. See [Installation Steps](#installation-steps) for details.

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Python Environment](#python-environment)
- [External Dependencies](#external-dependencies)
- [Database Requirements](#database-requirements)
- [Configuration Files](#configuration-files)
- [Installation Steps](#installation-steps)
- [Running the Project](#running-the-project)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## Hardware Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 10 GB free disk space for exports and logs
- **Network**: Internet connection for package downloads and model inference

### GPU Requirements (Optional but Recommended for llama.cpp)
- **GPU Memory**: 8 GB+ VRAM for running quantized LLM models
- **CUDA Support**: NVIDIA GPU with CUDA 11.0+ (for GPU acceleration)

---

## Software Requirements

### Operating System
- **Linux** (Ubuntu 20.04+, Debian 11+, or compatible distributions)
- **macOS** 12.0+ (for Apple Silicon or Intel)
- **Windows** 10/11 (with WSL2 recommended for bash scripts)

### Required Software

#### 1. Python
- **Version**: Python 3.9 or higher

#### 2. MySQL Database
- **Version**: MySQL 5.7+ or MariaDB 10.3+
- **Status**: Must be running and accessible via TCP/IP

#### 3. Git
- **Version**: Any recent version
- **Required for**: Cloning llama.cpp repository

#### 4. C++ Build Tools
Required for building llama.cpp:
- **GCC**: 7.5+ or **Clang**: 10+
- **CMake**: 3.12 or higher
- **Make**: GNU Make 4.0+

#### 5. curl
- Required for health checks and HTTP requests
- Usually pre-installed on most systems


## Python Environment

### Package Manager
**Recommended**: Use `uv` for fast dependency management
**Alternative**: Standard `pip` package manager

### Python Dependencies

#### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| aiohttp | ≥3.13.2 | Async HTTP client for API calls |
| cryptography | ≥46.0.3 | Secure connections and encryption |
| dotenv | ≥0.9.9 | Environment variable management |
| matplotlib | ≥3.9.4 | Data visualization |
| pandas | ≥2.3.3 | Data manipulation and CSV handling |
| pymysql | ≥1.1.2 | MySQL database connectivity |
| requests | ≥2.32.5 | HTTP requests for model API |
| scipy | ≥1.13.1 | Statistical analysis |
| seaborn | ≥0.13.2 | Statistical data visualization |
| sqlalchemy | ≥2.0.44 | Database ORM and connection management |

#### Development Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| setuptools | 80.1.0 - 81.0.0 | Package building |
| pylint | 3.0.4 - 4.0.0 | Code linting |
| pytest | 8.0.2 - 9.0.0 | Testing framework |
| yapf | ≥0.43.0 | Code formatting |
| pytest-cov | 6.0.0 - 7.0.0 | Test coverage reporting |

### Installation
```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt
```

---

## External Dependencies

### llama.cpp
**Purpose**: Local LLM inference server for issue classification

#### Requirements
- **Repository**: https://github.com/ggerganov/llama.cpp
- **Build Type**: Server build with GPU support (optional)
- **Location of Install**: Not strict (configurable in scripts)

#### Installation Steps
```bash
# Clone repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CPU only
mkdir build
cd build
cmake ..
cmake --build . --config Release

# OR Build with CUDA support (NVIDIA GPU)
mkdir build
cd build
cmake .. -DLLAMA_CUDA=ON
cmake --build . --config Release

# OR Build with Metal support (Apple Silicon)
mkdir build
cd build
cmake .. -DLLAMA_METAL=ON
cmake --build . --config Release
```

#### Binary Location
After building, the server binary will be located at:
```
~/location-of-repo/llama.cpp/build/bin/llama-server
```

#### Model Files
- **Format**: GGUF quantized models
- **Location**: Configurable (default: `~/models/`)
- **Examples**: 
  - `Llama-3.1-8B-Instruct-Q4_1.gguf`
  - `Mistral-7B-Instruct-v0.3.Q5_K_M.gguf`
  - `Phi-3.5-mini-instruct.Q8_0.gguf`
  - `gemma-3-4b-it-qat-Q8_0.gguf`

#### Server Configuration
- **Port**: 8080 (default, configurable in config.toml)
- **GPU Layers**: `-ngl 99` (offload 99 layers to GPU)
- **Health Check Endpoint**: `http://localhost:8080/health`

---

## Database Requirements

### TAWOS Database
- **Database Name**: `TAWOS` (configurable)
- **Schema**: Must contain the following tables:
  - `Issue`
  - `Comment`
  - `Change_Log`
  - `Component`
  - `User`
  - `Sprint`
  - `Project`
  - `Repository`

### Database Access
- **Connection Type**: TCP/IP
- **Port**: 3306 (default MySQL port)
- **User Permissions**: SELECT access to all tables listed above
- **Character Set**: UTF-8 compatible

---

## Configuration Files

### 1. Environment Variables (`.env`)
**Required**. Create in project root:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=TAWOS
```

### 2. Application Configuration (`config.toml`)
**Required**. Create based on `example.config.toml`:

```toml
[export]
enabled = true          # Enable/disable data export
limit = 1000           # Number of issues to export (0 = no limit)
folder = "exports"     # Output directory for CSV files

[local_model]
enabled = true         # Enable/disable local model classification
url = "http://localhost:8080"  # llama.cpp server URL
batch_size = 16        # Batch size for model processing
n_predict = 20         # Max tokens to predict
temp = 0.5             # Temperature for model inference (0.0-1.0)

[benchmark]
folder = "exports"     # Input folder for benchmarking
output = "benchmark"   # Output folder for benchmark results

[logging]
level = "INFO"         # Log level: DEBUG, INFO, WARNING, ERROR
format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
file = "tawos.log"     # Log file path
```

---

## Installation Steps

### Automated Setup (Recommended)

The easiest way to set up the entire system is to use the provided setup script:

```bash
# Clone the repository
git clone https://github.com/szbmrk/IPO4ICSE.git
cd IPO4ICSE/tawos

# Run automated setup with default directories
./setup.sh

# Or specify custom directories
./setup.sh --llama-dir /opt/llama.cpp --model-dir ~/my-models
```

**Command-line Options:**
- `--llama-dir DIR`: Directory where llama.cpp will be installed (default: `~/llama.cpp`)
- `--model-dir DIR`: Directory where GGUF models will be stored (default: `~/models`)
- `--help`: Display usage information

The `setup.sh` script will:
1. ✓ Check all system requirements (Python, MySQL, Git, CMake, C++ compiler, curl)
2. ✓ Create and configure Python virtual environment
3. ✓ Install all Python dependencies (via uv or pip)
4. ✓ Clone and build llama.cpp with GPU support (if available) to the specified directory
5. ✓ Create model directory at the specified location
6. ✓ Set up configuration files (`.env` and `config.toml`) interactively
7. ✓ Test database connection
8. ✓ Make all scripts executable

After the setup completes, you just need to:
- Download GGUF models to your model directory (default: `~/models/`)
- Review/adjust `config.toml` settings
- Run `./run_experiment.sh <export_dir> <benchmark_dir> <model_dir>`

**Examples:**
```bash
# Use all defaults
./setup.sh

# Install llama.cpp to /opt
./setup.sh --llama-dir /opt/llama.cpp

# Custom locations for both
./setup.sh --llama-dir ~/ai/llama.cpp --model-dir ~/ai/models
```

### Manual Setup

If you prefer to set up components manually:

#### 1. Clone the Repository
```bash
git clone https://github.com/szbmrk/IPO4ICSE.git
cd IPO4ICSE/tawos
```

#### 2. Set Up Python Environment
```bash
# Create virtual environment
python3.9 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
# OR
pip install -r requirements.txt
```

#### 3. Install and Build llama.cpp
```bash
cd ~/Projects
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUDA=ON  # Add GPU support if available
cmake --build . --config Release
```

#### 4. Download Model Files
```bash
mkdir -p ~/models
# Download your preferred GGUF models to ~/models/
# Example models used in this project:
# - Llama-3.1-8B-Instruct-Q4_1.gguf
# - Mistral-7B-Instruct-v0.3.Q5_K_M.gguf
# - Phi-3.5-mini-instruct.Q8_0.gguf
```

#### 5. Configure Database
```bash
# Start MySQL service
sudo systemctl start mysql  # Linux
# OR
brew services start mysql   # macOS

# Import TAWOS dataset
mysql -u root -p < tawos_schema.sql  # If available
```

#### 6. Create Configuration Files
```bash
# Create .env file
cat > .env << EOF
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=TAWOS
EOF

# Copy and edit config.toml
cp example.config.toml config.toml
# Edit config.toml with your preferred settings
```

#### 7. Make Scripts Executable
```bash
chmod +x setup.sh
chmod +x run_experiment.sh
chmod +x start_models.sh
```

#### 8. Verify Installation
```bash
# Test database connection
python -c "import pymysql; pymysql.connect(host='localhost', user='root', password='your_password', database='TAWOS')"

# Test llama.cpp server
~/Projects/llama.cpp/build/bin/llama-server -m ~/models/your-model.gguf -ngl 99 --port 8080 &
curl http://localhost:8080/health
```

---

## Running the Project

### Basic Run
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main pipeline
python src/main.py
```

### Export Only
```bash
python src/main.py --export --yes
```

### Benchmark Only
```bash
python src/main.py --benchmark
```

### Run Full Experiment
```bash
./run_experiment.sh exports-5000 benchmark-5000 ~/models
```

---

## Troubleshooting

### Setup Script Issues

1. **Setup Script Fails on System Requirements**
   - **Missing Python 3.9+**: Install with your package manager
     ```bash
     # Ubuntu/Debian
     sudo apt update && sudo apt install python3.9 python3.9-venv python3-pip
     # Fedora
     sudo dnf install python3.9
     # macOS
     brew install python@3.9
     ```
   - **Missing Build Tools**: Install development tools
     ```bash
     # Ubuntu/Debian
     sudo apt install build-essential cmake git
     # Fedora
     sudo dnf groupinstall "Development Tools" && sudo dnf install cmake
     # macOS
     xcode-select --install && brew install cmake
     ```

2. **llama.cpp Build Fails**
   - **CMake version too old**: Upgrade CMake to 3.12+
   - **CUDA build fails**: Verify CUDA toolkit is installed and `nvcc` is in PATH
   - **Metal build fails**: Ensure Xcode Command Line Tools are installed
   - **Out of space**: llama.cpp build requires ~2GB free space

3. **Setup Script Prompts Are Not Interactive**
   - Run the script directly (not through another script): `bash setup.sh`
   - Ensure you're running in an interactive terminal

### Common Runtime Issues

1. **MySQL Connection Failed**
   - Verify MySQL service is running: `sudo systemctl status mysql`
   - Check credentials in `.env` file
   - Ensure database exists and user has permissions
   - Test connection: `mysql -h localhost -u root -p TAWOS`

2. **llama.cpp Server Not Starting**
   - Check if port 8080 is available: `lsof -i :8080` or `netstat -tuln | grep 8080`
   - Verify model file path and format (must be GGUF)
   - Check CUDA/GPU drivers if using GPU acceleration
   - Look at server logs for specific error messages

3. **Python Package Installation Failures**
   - Update pip: `pip install --upgrade pip`
   - Install system dependencies: 
     ```bash
     # Ubuntu/Debian
     sudo apt install python3-dev libmysqlclient-dev
     # Fedora
     sudo dnf install python3-devel mysql-devel
     ```

4. **Out of Memory Errors**
   - Reduce `batch_size` in `config.toml`
   - Use smaller quantized models (Q4 instead of Q8)
   - Reduce `-ngl` parameter for llama.cpp (offload fewer layers to GPU)

5. **"Permission Denied" When Running Scripts**
   - Make scripts executable: `chmod +x setup.sh run_experiment.sh start_models.sh`

6. **Virtual Environment Activation Issues**
   - Ensure you're in the project directory
   - Use correct activation command:
     ```bash
     # Bash/Zsh
     source .venv/bin/activate
     # Fish
     source .venv/bin/activate.fish
     # Windows
     .venv\Scripts\activate
     ```

---

## Performance Optimization

### For Faster Processing
- Use GPU acceleration with llama.cpp
- Increase `batch_size` (if memory allows)
- Use smaller quantized models (Q4_0, Q4_1)
- Enable batch processing for model inference

### For Lower Memory Usage
- Reduce `batch_size`
- Use CPU-only inference
- Limit export size with `limit` parameter
- Use smaller quantized models

---

## License & Dataset

- **TAWOS Dataset**: https://github.com/SOLAR-group/TAWOS
- **llama.cpp**: MIT License

---