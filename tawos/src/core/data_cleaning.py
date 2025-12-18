import asyncio
import os

from core.local_model_classifier import LocalModelClassifier
from core.own_metrics_classifier import OwnMetricsClassifier
from core.log import get_logger
from config_loader import config
from utils.file_reading import safe_read_csv

logger = get_logger("DataCleaning")


def _remove_columns_from_csv(
    file_name: str, columns: list[str], folder_path: str = config.EXPORT_FOLDER
) -> None:
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' not found.")
        raise FileNotFoundError(f"File '{file_path}' not found.")

    df = safe_read_csv(file_path)

    for column_name in columns:
        if column_name in df.columns:
            df = df.drop(columns=[column_name])
            logger.info(f"Removed column: '{column_name}'")
        else:
            logger.warning(
                f"Column '{column_name}' not found in '{file_name}'. No changes made."
            )
            continue

    output_file = os.path.join(folder_path, f"{file_name}")
    df.to_csv(output_file, sep=";", index=False)
    logger.info(f"Saved modified file as '{output_file}'")


def remove_unnecessery_columns() -> None:
    _remove_columns_from_csv("Project.csv", ["URL", "Project_Key"])
    _remove_columns_from_csv("Repository.csv", ["URL"])
    _remove_columns_from_csv("Issue.csv", ["Jira_ID", "Issue_Key", "URL"])
    _remove_columns_from_csv("Component.csv", ["Jira_ID"])
    logger.info("Removing unnecessery columns done")


def add_points_generated_by_local_model(classifier: LocalModelClassifier) -> None:
    logger.info("Starting classifying by local model")
    issues_path = os.path.join(config.EXPORT_FOLDER, "Issue.csv")
    df = safe_read_csv(issues_path)
    points, column_name = asyncio.run(classifier.classify(df))

    if points is None:
        return

    df[column_name] = points
    df.to_csv(issues_path, sep=";", index=False)
    logger.info(f"Saved modified file as '{issues_path}'")


def add_points_generated_by_own_metrics(classifier: OwnMetricsClassifier) -> None:
    logger.info("Starting classifying by own metrics")
    issues_path = os.path.join(config.EXPORT_FOLDER, "Issue.csv")
    df = safe_read_csv(issues_path)
    points, _ = asyncio.run(classifier.classify(df))

    if "OwnMetrics_validity_point" in df.columns:
        logger.info(
            "Own metrics already computed (OwnMetrics_validity_point column exists)"
        )
        return

    df["OwnMetrics_validity_point"] = points
    df.to_csv(issues_path, sep=";", index=False)
    logger.info(f"Saved modified file as '{issues_path}'")


def filter_by_own_metrics() -> None:
    logger.info("Starting filtering by own metrics")

    source_folder = config.EXPORT_FOLDER
    target_folder = f"{source_folder}-cleaned"
    os.makedirs(target_folder, exist_ok=True)

    issues_path = os.path.join(source_folder, "Issue.csv")
    if not os.path.exists(issues_path):
        logger.error(f"File '{issues_path}' not found.")
        return

    df_issues = safe_read_csv(issues_path)

    if "OwnMetrics_validity_point" not in df_issues.columns:
        logger.error("OwnMetrics_validity_point column missing in Issue.csv")
        return

    initial_count = len(df_issues)
    df_valid_issues = df_issues[df_issues["OwnMetrics_validity_point"] >= 20].copy()
    filtered_count = len(df_valid_issues)
    logger.info(
        f"Filtered issues: {initial_count} -> {filtered_count} (Removed {initial_count - filtered_count})"
    )

    valid_ids = set(df_valid_issues["ID"])

    cols_to_drop = [c for c in df_valid_issues.columns if c.endswith("_validity_point")]
    df_valid_issues.drop(columns=cols_to_drop, inplace=True)

    target_issue_path = os.path.join(target_folder, "Issue_cleaned.csv")
    df_valid_issues.to_csv(target_issue_path, sep=";", index=False)
    logger.info(f"Saved cleaned issues to '{target_issue_path}'")

    related_files = {
        "Comment.csv": "Issue_ID",
        "Change_Log.csv": "Issue_ID",
        "Issue_Links.csv": "Issue_ID",
    }

    for filename, id_col in related_files.items():
        file_path = os.path.join(source_folder, filename)
        if os.path.exists(file_path):
            df = safe_read_csv(file_path)
            if id_col in df.columns:
                initial_rows = len(df)
                df_filtered = df[df[id_col].isin(valid_ids)]
                final_rows = len(df_filtered)

                target_path = os.path.join(
                    target_folder, filename.replace(".csv", "_cleaned.csv")
                )
                df_filtered.to_csv(target_path, sep=";", index=False)
                logger.info(
                    f"Filtered {filename}: {initial_rows} -> {final_rows} rows. Saved to {target_path}"
                )
            else:
                logger.warning(
                    f"Column {id_col} not found in {filename}, copying as is."
                )
                target_path = os.path.join(
                    target_folder, filename.replace(".csv", "_cleaned.csv")
                )
                df.to_csv(target_path, sep=";", index=False)
        else:
            logger.warning(f"{filename} not found, skipping.")

    other_files = [
        "Component.csv",
        "User.csv",
        "Sprint.csv",
        "Project.csv",
        "Repository.csv",
    ]
    for filename in other_files:
        file_path = os.path.join(source_folder, filename)
        if os.path.exists(file_path):
            df = safe_read_csv(file_path)
            target_path = os.path.join(
                target_folder, filename.replace(".csv", "_cleaned.csv")
            )
            df.to_csv(target_path, sep=";", index=False)
            logger.info(f"Copied {filename} to {target_path}")
