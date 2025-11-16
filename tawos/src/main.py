import argparse
from core.benchmark import run_benchmark
from core.export import export_sql_to_csv
from core.data_cleaning import (
    add_points_generated_by_local_model,
    add_points_generated_by_own_metrics,
    remove_unnecessery_columns,
)
from core.log import get_logger
from config_loader import config
from core.local_model_classifier import LocalModelClassifier


logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", action="store_true", help="Run the benchmark pipeline"
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        exit(0)

    if config.EXPORT_ENABLED:
        confirm = input(
            "Are you sure you want to export? You could lose your current files. (y/n) "
        )
        if confirm == "y":
            export_sql_to_csv()

    remove_unnecessery_columns()

    if config.LOCAL_MODEL_ENABLED:
        local_classifier = LocalModelClassifier()
        add_points_generated_by_local_model(local_classifier)

    add_points_generated_by_own_metrics()


if __name__ == "__main__":
    main()
