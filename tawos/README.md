# TAWOS Data Export Script

The dataset is available here: https://github.com/SOLAR-group/TAWOS

This script connects to a MySQL database, selects the first **N issues**, and exports them along with all related records (comments, change logs, etc.) into CSV files. It also exports several lookup/reference tables.

All exported CSV files are saved into the `exports/` directory.

## Requirements

### Python Packages
Install dependencies using:

```
pip install -r ./requirements.txt
```

## Configuration

In the script, two variables can be adjusted:

| Variable | Description | Default |
|---------|-------------|---------|
| `LIMIT` | Number of issues to export | `1000` |
| `OUT`   | Output directory for CSV files | `exports` |

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
