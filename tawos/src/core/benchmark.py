import os
import pandas as pd
from config_loader import config
import matplotlib.pyplot as plt
import seaborn as sns
from core.log import get_logger

logger = get_logger("Benchmark")

OUTPUT_DIR = "benchmark"


def __save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def __failure_rate_analysis(df, point_columns):
    logger.info("Calculating failure rates")
    failure_stats = (df[point_columns] == -1).mean().sort_values()
    plt.figure(figsize=(10, 5))
    failure_stats.plot(kind="bar")
    plt.title("Failure Rate (-1) per Model")
    plt.ylabel("Failure Rate")
    plt.xticks(rotation=45)
    __save_plot("failure_rate_per_model.png")


def __agreement_matrix(df):
    logger.info("Creating agreement matrix")
    agreement = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Model Agreement (Correlation Matrix)")
    __save_plot("agreement_matrix.png")


def __consensus_per_task(df):
    logger.info("Calculating consensus std")
    df["consensus_std"] = df.std(axis=1)
    df["consensus_range"] = df.max(axis=1) - df.min(axis=1)
    df["consensus_mean"] = df.mean(axis=1)

    plt.figure(figsize=(10, 5))
    df["consensus_std"].hist(bins=40)
    plt.title("Consensus Standard Deviation Across Models")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Number of Tasks")
    __save_plot("consensus_std_distribution.png")


def run_benchmark():
    logger.info("Running benchmarks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(config.BENCHMARK_FOLDER, "Issue.csv")
    df = pd.read_csv(file_path, sep=";")

    point_columns = [c for c in df.columns if c.endswith("_validity_points")]

    df_clean = df[point_columns].replace(-1, pd.NA)

    __failure_rate_analysis(df, point_columns)
    __agreement_matrix(df_clean)
    __consensus_per_task(df_clean)

    logger.info("Benchmarking complete")
    logger.info(f"All plots saved to: {OUTPUT_DIR}/")
