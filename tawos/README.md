# TAWOS Data Export Script

The dataset is available here: https://github.com/SOLAR-group/TAWOS

This script connects to a MySQL database, selects the first **N issues** (defined in `config.toml`), and exports them along with all related records (comments, change logs) into CSV files.

After the export, a data cleaning step (`data_cleaning.py`) is automatically run to modify the generated files.

After the basic cleaning multiple validiation processes are run to filter out the irrelevant and troll issues.

All exported CSV files are saved into a configurable output directory (default: `exports/`).

## Requirements

### Python Packages
Install dependencies using:

```
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

```
python src/main.py
```

## Output Example

```
exports/
├── Issue.csv
├── Comment.csv
├── Change_Log.csv
├── Component.csv
├── User.csv
├── Sprint.csv
├── Project.csv
└── Repository.csv
```
