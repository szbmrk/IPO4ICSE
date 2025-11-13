import os
import pandas as pd

from core.log import get_logger

logger = get_logger("DataCleaning")


def __remove_columns_from_csv(
    file_name: str, columns: list[str], folder_path: str = "exports"
):
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' not found.")
        raise FileNotFoundError(f"File '{file_path}' not found.")

    df = pd.read_csv(file_path, sep=";")

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


def remove_unnecessery_columns():
    __remove_columns_from_csv("Project.csv", ["URL", "Project_Key"])
    __remove_columns_from_csv("Repository.csv", ["URL"])
    __remove_columns_from_csv("Issue.csv", ["Jira_ID", "Issue_Key", "URL"])
    __remove_columns_from_csv("Component.csv", ["Jira_ID"])
    logger.info("Removing unnecessery columns done")
