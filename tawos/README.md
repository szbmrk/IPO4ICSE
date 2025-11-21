# TAWOS Data Export Script

The dataset is available here: https://github.com/SOLAR-group/TAWOS

This script connects to a MySQL database, selects the first **N issues** (defined in `config.toml`), and exports them along with all related records (comments, change logs) into CSV files.

After the export, a data cleaning step (`data_cleaning.py`) is automatically run to modify the generated files.

After the basic cleaning multiple validiation processes are run to filter out the irrelevant and troll issues.

All exported CSV files are saved into a configurable output directory (default: `exports/`).

## Quick Start

### Automated Setup

Run the automated setup script to install all dependencies and configure the environment:

```bash
# Use default directories
./setup.sh

# Or specify custom directories
./setup.sh --llama-dir /path/to/llama.cpp --model-dir /path/to/models
```

**Options:**
- `--llama-dir DIR`: Specify where to install llama.cpp (default: `~/Projects/llama.cpp`)
- `--model-dir DIR`: Specify where to store GGUF models (default: `~/models`)
- `--help`: Show usage information

This will:
- Check system requirements (Python 3.9+, MySQL, Git, CMake, etc.)
- Create and configure Python virtual environment
- Clone and build llama.cpp with GPU support (if available)
- Set up model directory for GGUF files
- Create configuration files (`.env` and `config.toml`)
- Test database connection
- Make all scripts executable

After setup completes, follow the printed instructions to download models and run your first experiment.

### What the Setup Script Does

The `setup.sh` script performs a complete automated installation:

1. **System Check**: Validates that all required tools are installed (Python 3.9+, Git, CMake, MySQL, C++ compiler, curl)
2. **Python Environment**: Creates a virtual environment and installs all dependencies
3. **llama.cpp**: Clones the repository to `~/Projects/llama.cpp` and builds with GPU support if available (CUDA/Metal)
4. **Model Directory**: Creates `~/models/` directory for storing GGUF model files
5. **Configuration**: Interactively creates `.env` and `config.toml` files
6. **Database Test**: Verifies MySQL connection with provided credentials
7. **Scripts**: Makes all shell scripts executable

After running `setup.sh`, you only need to:
- Download GGUF models to `~/models/` (e.g., from [HuggingFace](https://huggingface.co/models?library=gguf))
- Adjust `config.toml` if needed
- Run `./run_experiment.sh <export_dir> <benchmark_dir> ~/models`

## Requirements

For detailed system requirements and manual installation steps, see [requirements.md](requirements.md).

### Python Packages
Install dependencies using:

```bash
uv sync

# or if not using uv:
pip install -r ./requirements.txt
```

## Configuration

### 1. Environment variables (.env)

Create a `.env` file in the root directory.

```
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=TAWOS
```

### 2. settings

Create a `config.toml` file to control the behaviour of the pipeline.

```toml
[export]
# Is exproting needed or is it already exported
enabled = true
# Number of issues to export (set to 0 for no limit)
limit = 1000
# Output directory for CSV files
folder = "exports"

[local_model]
# Use llamacpp local model for classifying
enabled = true
# Local url for the model ran by llamacpp
url = "http://localhost:8080"
# Batch size to process the data
batch_size = 20

[logging]
level = "INFO"
# Optional: specify a file to log to
file = "tawos.log" 
format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
```

## What It Exports

| Data | Source Table | Filter Applied? |
|------|--------------|----------------|
| Issues | `Issue` | First N by ID |
| Comments | `Comment` | Only where `Issue_ID` is in exported issues |
| Change Logs | `Change_Log` | Only where `Issue_ID` is in exported issues |
| Components | `Component` | None |
| Users | `User` | None |
| Sprints | `Sprint` | None |
| Projects | `Project` | None |
| Repositories | `Repository` | None |

## Data cleaning

Data cleaning currently consists of removing unnecessery columns.

## How to Run

### Prerequisites
After running `setup.sh`, ensure you have:
1. **MySQL server is running** - The scripts will check and exit with an error if MySQL is not accessible
2. Downloaded GGUF models to `~/models/`
3. Configured `.env` with database credentials
4. Reviewed `config.toml` settings

**Important**: All scripts now automatically check MySQL connectivity before starting. If MySQL is not running, you'll see a clear error message with instructions to start it.

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main pipeline
python src/main.py

# Export only
python src/main.py --export --yes

# Benchmark only
python src/main.py --benchmark
```

## Running Experiments

To run a full experiment including data export, model classification with varying temperatures, and benchmarking, use the provided `run_experiment.sh` script.

### Usage

```bash
./run_experiment.sh <export_dir> <benchmark_dir> <model_dir>
```

### Arguments

1. `export_dir`: Directory where the exported CSV files and classification results will be saved.
2. `benchmark_dir`: Directory where the benchmark summary and plots will be saved.
3. `model_dir`: Directory containing the GGUF model files to be tested.

### What it does

1. **Exports Data**: Exports the configured number of issues from the database to `export_dir`.
2. **Runs Models**: Iterates through all `.gguf` models in `model_dir`.
    - For each model, it runs classification with temperatures: `0.2`, `0.4`, `0.6`, `0.8`.
    - Results are saved in `Issue.csv` with columns named `modelname_temp_XX_validity_point`.
3. **Benchmarks**: Generates statistical summaries and plots in `benchmark_dir`.

### Example

```bash
./run_experiment.sh exports-experiment benchmark-results ~/models
```

## Verifying Your Setup

After running `setup.sh`, verify everything is working:

1. **Check Python Environment**
   ```bash
   source .venv/bin/activate
   python --version  # Should show 3.9+
   pip list | grep pandas  # Verify packages installed
   ```

2. **Check llama.cpp**
   ```bash
   ~/llama.cpp/build/bin/llama-server --version
   ls ~/models/*.gguf  # List your models
   ```

3. **Test Database Connection**
   ```bash
   source .venv/bin/activate
   python -c "from dotenv import load_dotenv; import pymysql, os; load_dotenv(); pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'), database=os.getenv('DB_NAME')); print('✓ Database connection successful')"
   ```

4. **Test llama.cpp Server**
   ```bash
   # Start server with a model
   ~/Projects/llama.cpp/build/bin/llama-server -m ~/models/your-model.gguf -ngl 99 --port 8080 &
   sleep 5
   curl http://localhost:8080/health
   # Should return OK
   # Kill server: pkill llama-server
   ```

## Troubleshooting

### MySQL Connection Errors

All scripts (`main.py`, `run_experiment.sh`, `start_models.sh`) now automatically check MySQL connectivity before starting. If you see a MySQL connection error:

1. **Check if MySQL is running:**
   ```bash
   # Linux
   sudo systemctl status mysql
   sudo systemctl start mysql  # If not running
   
   # macOS
   brew services list
   brew services start mysql  # If not running
   ```

2. **Verify connection manually:**
   ```bash
   mysql -h localhost -u root -p TAWOS
   ```

3. **Check your .env file:**
   - Ensure `DB_HOST`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME` are correct
   - No extra spaces or quotes around values

For more common issues and solutions, see the [Troubleshooting section in requirements.md](requirements.md#troubleshooting).

## Documentation

- **[requirements.md](requirements.md)**: Detailed system requirements and installation instructions
- **[example.config.toml](example.config.toml)**: Configuration template with all options