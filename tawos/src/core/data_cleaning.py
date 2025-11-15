import asyncio
import os
import pandas as pd

from core.gemini_classifier import GeminiClassifier
from core.openai_classifier import OpenAIClassifier
from core.claude_classifier import ClaudeClassifier
from core.local_model_classifier import LocalModelClassifier
from core.log import get_logger
from config_loader import config
from core.own_metrics_classifier import classify_by_own_metrics

logger = get_logger("DataCleaning")


def _remove_columns_from_csv(
    file_name: str, columns: list[str], folder_path: str = config.EXPORT_FOLDER
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
    _remove_columns_from_csv("Project.csv", ["URL", "Project_Key"])
    _remove_columns_from_csv("Repository.csv", ["URL"])
    _remove_columns_from_csv("Issue.csv", ["Jira_ID", "Issue_Key", "URL"])
    _remove_columns_from_csv("Component.csv", ["Jira_ID"])
    logger.info("Removing unnecessery columns done")


def add_points_generated_by_local_model(classifier: LocalModelClassifier):
    logger.info("Starting classifying by local model")
    issues_path = os.path.join(config.EXPORT_FOLDER, "Issue.csv")
    df = pd.read_csv(issues_path, sep=";")
    points, column_name = asyncio.run(classifier.classify(df))
    df[column_name] = points
    df.to_csv(issues_path, sep=";", index=False)
    logger.info(f"Saved modified file as '{issues_path}'")


def add_points_generated_by_gemini(classifier: GeminiClassifier):
    logger.info("Starting classifying by gemini")
    issues_path = os.path.join(config.EXPORT_FOLDER, "Issue.csv")
    df = pd.read_csv(issues_path, sep=";")
    points, column_name = asyncio.run(classifier.classify(df))
    df[column_name] = points
    df.to_csv(issues_path, sep=";", index=False)
    logger.info(f"Saved modified file as '{issues_path}'")


def add_points_generated_by_openai(classifier: OpenAIClassifier):
    logger.info("Starting classifying by openai")
    issues_path = os.path.join(config.EXPORT_FOLDER, "Issue.csv")
    df = pd.read_csv(issues_path, sep=";")
    points, column_name = asyncio.run(classifier.classify(df))
    df[column_name] = points
    df.to_csv(issues_path, sep=";", index=False)
    logger.info(f"Saved modified file as '{issues_path}'")


def add_points_generated_by_claude(classifier: ClaudeClassifier):
    logger.info("Starting classifying by claude")
    issues_path = os.path.join(config.EXPORT_FOLDER, "Issue.csv")
    df = pd.read_csv(issues_path, sep=";")
    points, column_name = asyncio.run(classifier.classify(df))
    df[column_name] = points
    df.to_csv(issues_path, sep=";", index=False)
    logger.info(f"Saved modified file as '{issues_path}'")


def add_points_generated_by_own_metrics():
    logger.info("Starting classifying by own metrics")
    issues_path = os.path.join(config.EXPORT_FOLDER, "Issue.csv")
    df = pd.read_csv(issues_path, sep=";")
    points = classify_by_own_metrics(df)
    df["own_validity_point"] = points
    df.to_csv(issues_path, sep=";", index=False)
    logger.info(f"Saved modified file as '{issues_path}'")
