import pandas as pd
import re
from core.log import get_logger

logger = get_logger("OwnMetric")


def _calculate_row_points(title, description):
    title = title if isinstance(title, str) else ""
    description = description if isinstance(description, str) else ""

    points = 100
    spam_keywords = ["free", "win", "offer", "click", "???", "!!!", "xd", "xdd"]

    if len(title) < 10:
        points -= 20

    for keyword in spam_keywords:
        escaped = re.escape(keyword)
        pattern = rf"\b{escaped}\b"

        if re.search(pattern, title, re.IGNORECASE) or re.search(
            pattern, description, re.IGNORECASE
        ):
            points -= 30

    return max(points, 0)


def classify_by_own_metrics(df: pd.DataFrame):
    total_rows = len(df)
    logger.info(f"Local metric classification started for {total_rows} rows")

    rows = list(df[["Title", "Description"]].itertuples(index=False, name=None))

    results = [_calculate_row_points(title, description) for title, description in rows]

    logger.info(f"Local metric classification finished for {total_rows} rows")

    return results
