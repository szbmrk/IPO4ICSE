import argparse
from core.benchmark import run_benchmark
from core.export import export_sql_to_csv
from core.data_cleaning import (
    add_points_generated_by_local_model,
    add_points_generated_by_own_metrics,
    remove_unnecessery_columns,
    filter_by_own_metrics,
)
from core.log import get_logger
from core.db_check import ensure_mysql_connection
from config_loader import config
from core.local_model_classifier import LocalModelClassifier


logger = get_logger("main")


def export(force=False):
    if force:
        export_sql_to_csv()
        return

    confirm = input(
        "Are you sure you want to export? You could lose your current files. (y/n) "
    )
    if confirm == "y":
        export_sql_to_csv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", action="store_true", help="Run the benchmark pipeline"
    )
    parser.add_argument("--export", action="store_true", help="Only run the exporting")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        exit(0)

    # Check MySQL connection before export
    if args.export or config.EXPORT_ENABLED:
        logger.info("Checking MySQL connection...")
        ensure_mysql_connection()

    if args.export:
        export(args.yes)
        exit(0)

    if config.EXPORT_ENABLED:
        export(args.yes)

    remove_unnecessery_columns()

    add_points_generated_by_own_metrics()

    if config.LOCAL_MODEL_ENABLED:
        local_classifier = LocalModelClassifier()
        add_points_generated_by_local_model(local_classifier)

    filter_by_own_metrics()


if __name__ == "__main__":
    main()
