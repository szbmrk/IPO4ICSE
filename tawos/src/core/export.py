import os
import pandas as pd
from sqlalchemy import create_engine, text
from config_loader import config
from core.log import get_logger

logger = get_logger("Export")


def _export(query: str, file_name: str, engine, params=None):
    df = pd.read_sql(text(query), engine, params=params or {})
    df.to_csv(f"{config.EXPORT_FOLDER}/{file_name}.csv", sep=";", index=False)
    logger.info(f"{file_name}.csv  ({len(df)} rows)")
    logger.info("Exporting related records...")


def export_sql_to_csv():
    os.makedirs(config.EXPORT_FOLDER, exist_ok=True)
    os.makedirs(f"{config.EXPORT_FOLDER}/timing", exist_ok=True)

    engine = create_engine(
        f"mysql+pymysql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}/{config.DB_NAME}"
    )

    logger.info("Loading issues...")

    if config.EXPORT_LIMIT > 1:
        issues = pd.read_sql(
            text(
                """
            SELECT *
            FROM Issue
            ORDER BY RAND()
            LIMIT :limit_val
        """
            ),
            engine,
            params={"limit_val": config.EXPORT_LIMIT},
        )
    else:
        issues = pd.read_sql(
            text(
                """
            SELECT *
            FROM Issue
            ORDER BY RAND()
        """
            ),
            engine,
        )

    issues.to_csv(f"{config.EXPORT_FOLDER}/Issue.csv", sep=";", index=False)

    issue_ids = tuple(issues["ID"].tolist())
    logger.info(f"{len(issue_ids)} issues included.")

    _export(
        """
        SELECT * FROM Comment WHERE Issue_ID IN :ids
    """,
        "Comment",
        engine,
        {"ids": issue_ids},
    )

    _export(
        """
        SELECT * FROM Change_Log WHERE Issue_ID IN :ids
    """,
        "Change_Log",
        engine,
        {"ids": issue_ids},
    )

    _export(
        """
        SELECT * FROM Issue_Link WHERE Issue_ID IN :ids
    """,
        "Issue_Links",
        engine,
        {"ids": issue_ids},
    )

    logger.info("Exporting lookup tables...")

    _export("SELECT * FROM Component", "Component", engine)
    _export("SELECT * FROM User", "User", engine)
    _export("SELECT * FROM Sprint", "Sprint", engine)
    _export("SELECT * FROM Project", "Project", engine)
    _export("SELECT * FROM Repository", "Repository", engine)

    logger.info(f"All CSVs exported into ./{config.EXPORT_FOLDER}")
