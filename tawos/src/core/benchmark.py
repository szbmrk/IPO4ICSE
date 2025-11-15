import os
import pandas as pd
from config_loader import config
import matplotlib.pyplot as plt
from core.log import get_logger

logger = get_logger("Benchmark")

OUTPUT_DIR = "benchmark"


def __save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def __failure_rate_analysis(df, point_columns):
    logger.info("Calculating failure rates")
    failure_stats = (df[point_columns] == -1).mean().sort_values()
    plt.figure(figsize=(10, 5))
    failure_stats.plot(kind="bar")
    plt.title("Failure Rate per Model (Couldn't classify)")
    plt.ylabel("Failure Rate")
    plt.xticks(rotation=45)
    __save_plot("failure_rate_per_model.png")


def run_benchmark():
    logger.info("Running benchmarks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(config.BENCHMARK_FOLDER, "Issue.csv")
    df = pd.read_csv(file_path, sep=";")

    point_columns = [c for c in df.columns if c.endswith("_validity_point")]

    df_clean = df[point_columns].replace(-1, pd.NA)
    df_clean = df.dropna()

    __failure_rate_analysis(df, point_columns)

    logger.info("Benchmarking complete")
    logger.info(f"All plots saved to: {OUTPUT_DIR}/")
