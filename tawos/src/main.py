import os
from core.converter import export_sql_to_csv
from core.data_cleaning import (
    add_points_generated_by_local_model,
    remove_unnecessery_columns,
)
from core.log import get_logger
from config_loader import config

logger = get_logger("main")

if __name__ == "__main__":
    if not os.path.exists(f"{config.EXPORT_FOLDER}/Issue.csv"):
        export_sql_to_csv()
    remove_unnecessery_columns()
    add_points_generated_by_local_model()
