import argparse
import signal
import sys
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


def signal_handler(sig, frame):
    logger.warning("\nInterrupt received! Stopping gracefully...")
    logger.info("Cleaning up and exiting...")
    sys.exit(130)


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
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", action="store_true", help="Run the benchmark pipeline"
    )
    parser.add_argument("--export", action="store_true", help="Only run the exporting")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    parser.add_argument(
        "--skip-own-metrics",
        action="store_true",
        help="Skip own metrics calculation (use if already computed)",
    )
    parser.add_argument(
        "--skip-filtering",
        action="store_true",
        help="Skip filtering and cleaned CSV creation (use for intermediate runs)",
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        exit(0)

    if args.export or config.EXPORT_ENABLED:
        logger.info("Checking MySQL connection...")
        ensure_mysql_connection()

    if args.export:
        export(args.yes)
        exit(0)

    if config.EXPORT_ENABLED:
        export(args.yes)

    remove_unnecessery_columns()

    if not args.skip_own_metrics:
        add_points_generated_by_own_metrics()
    else:
        logger.info("Skipping own metrics calculation (--skip-own-metrics flag set)")

    if config.LOCAL_MODEL_ENABLED:
        local_classifier = LocalModelClassifier()
        add_points_generated_by_local_model(local_classifier)

    if not args.skip_filtering:
        filter_by_own_metrics()
    else:
        logger.info("Skipping filtering step (--skip-filtering flag set)")


if __name__ == "__main__":
    main()
