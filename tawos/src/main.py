import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

LIMIT = 1000
OUT = "exports"


def main():
    os.makedirs(OUT, exist_ok=True)

    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

    print(f"Loading first {LIMIT} issues...")
    issues = pd.read_sql(
        text("""
        SELECT *
        FROM Issue
        ORDER BY ID
        LIMIT :limit_val
    """),
        engine,
        params={"limit_val": LIMIT},
    )
    issues.to_csv(f"{OUT}/Issue.csv", sep=";", index=False)

    issue_ids = tuple(issues["ID"].tolist())
    print(f"{len(issue_ids)} issues included.\n")

    def export(query, file_name, params=None):
        df = pd.read_sql(text(query), engine, params=params or {})
        df.to_csv(f"{OUT}/{file_name}.csv", sep=";", index=False)
        print(f"{file_name}.csv  ({len(df)} rows)")
        print("Exporting related records...\n")

    export(
        """
        SELECT * FROM Comment WHERE Issue_ID IN :ids
    """,
        "Comment",
        {"ids": issue_ids},
    )

    export(
        """
        SELECT * FROM Change_Log WHERE Issue_ID IN :ids
    """,
        "Change_Log",
        {"ids": issue_ids},
    )

    print("\nExporting lookup tables...\n")

    export("SELECT * FROM Component", "Component")
    export("SELECT * FROM User", "User")
    export("SELECT * FROM Sprint", "Sprint")
    export("SELECT * FROM Project", "Project")
    export("SELECT * FROM Repository", "Repository")

    print("\nAll CSVs exported into ./exports")


if __name__ == "__main__":
    main()
