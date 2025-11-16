import os
import pandas as pd
import numpy as np
from config_loader import config
import matplotlib.pyplot as plt
import seaborn as sns
from core.log import get_logger

logger = get_logger("Benchmark")
OUTPUT_DIR = config.BENCHMARK_OUTPUT

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")


def _save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def _get_model_names(point_columns):
    return [c.replace("_validity_point", "") for c in point_columns]


def _statistical_summary(df, point_columns):
    logger.info("Generating statistical summary")

    df_numeric = df[point_columns].apply(pd.to_numeric, errors="coerce")
    df_valid = df_numeric.replace(-1, np.nan)
    model_names = _get_model_names(point_columns)

    summary = pd.DataFrame(
        {
            "Model": model_names,
            "Mean": df_valid.mean().values,
            "Median": df_valid.median().values,
            "Std Dev": df_valid.std().values,
            "Min": df_valid.min().values,
            "Max": df_valid.max().values,
            "Failure Rate (%)": (df_numeric == -1).mean().values * 100,
            "Valid Samples": df_valid.count().values,
        }
    )

    summary_path = os.path.join(OUTPUT_DIR, "statistical_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved statistical summary to {summary_path}")

    return summary


def _model_comparison_boxplot(df, point_columns):
    logger.info("Creating model comparison boxplot")

    df_valid = (
        df[point_columns].apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    )
    model_names = _get_model_names(point_columns)

    fig, ax = plt.subplots(figsize=(12, 6))

    data_to_plot = [df_valid[col].dropna() for col in point_columns]

    bp = ax.boxplot(
        data_to_plot,
        labels=model_names,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )

    colors = sns.color_palette("husl", len(point_columns))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validity Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribution of Validity Scores by Model", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    _save_plot("model_comparison_boxplot.png")


def _agreement_heatmap(df, point_columns):
    logger.info("Creating agreement heatmap")

    df_valid = (
        df[point_columns].apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    )
    model_names = _get_model_names(point_columns)

    corr_matrix = df_valid.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticklabels(model_names)

    for i in range(len(model_names)):
        for j in range(len(model_names)):
            text = ax.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title("Inter-Model Agreement (Correlation)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Correlation Coefficient")

    _save_plot("model_agreement_heatmap.png")


def _failure_analysis_detailed(df, point_columns):
    """Detailed failure analysis"""
    logger.info("Creating detailed failure analysis")

    failure_data = []
    for col in point_columns:
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        failures = df[col_numeric == -1]
        failure_data.append(
            {
                "Model": col.replace("_validity_point", ""),
                "Total Failures": len(failures),
                "Failure Rate (%)": (len(failures) / len(df)) * 100,
                "Bug Failures": len(failures[failures["Type"] == "Bug"]),
                "Feature Failures": len(failures[failures["Type"] == "Feature"]),
                "TechDebt Failures": len(failures[failures["Type"] == "Tech Debt"]),
            }
        )

    failure_df = pd.DataFrame(failure_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Failure rate by model
    ax1.barh(failure_df["Model"], failure_df["Failure Rate (%)"], color="coral")
    ax1.set_xlabel("Failure Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Classification Failure Rate by Model", fontsize=13, fontweight="bold"
    )
    ax1.grid(axis="x", alpha=0.3)

    # Failure by task type
    failure_by_type = failure_df[
        ["Model", "Bug Failures", "Feature Failures", "TechDebt Failures"]
    ].set_index("Model")
    failure_by_type.plot(kind="bar", stacked=True, ax=ax2)
    ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of Failures", fontsize=12, fontweight="bold")
    ax2.set_title("Failure Distribution by Task Type", fontsize=13, fontweight="bold")
    ax2.legend(title="Task Type", bbox_to_anchor=(1.05, 1))
    ax2.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    _save_plot("failure_analysis_detailed.png")

    failure_path = os.path.join(OUTPUT_DIR, "failure_summary.csv")
    failure_df.to_csv(failure_path, index=False)
    logger.info(f"Saved failure summary to {failure_path}")


def _pairwise_difference_analysis(df, point_columns):
    logger.info("Analyzing pairwise model differences")

    df_valid = (
        df[point_columns].apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    )
    model_names = _get_model_names(point_columns)

    n_models = len(point_columns)
    diff_matrix = np.zeros((n_models, n_models))

    for i, col1 in enumerate(point_columns):
        for j, col2 in enumerate(point_columns):
            if i != j:
                valid_both = df_valid[[col1, col2]].dropna()
                if len(valid_both) > 0:
                    diff_matrix[i, j] = np.mean(
                        np.abs(valid_both[col1] - valid_both[col2])
                    )

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(diff_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticklabels(model_names)

    for i in range(len(model_names)):
        for j in range(len(model_names)):
            if i != j:
                text = ax.text(
                    j,
                    i,
                    f"{diff_matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

    ax.set_title(
        "Mean Absolute Difference Between Models", fontsize=14, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="Mean Absolute Difference")

    _save_plot("pairwise_difference_heatmap.png")


def run_benchmark():
    logger.info("Running benchmarks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_path = os.path.join(config.BENCHMARK_FOLDER, "Issue.csv")
    df = pd.read_csv(file_path, sep=";")
    point_columns = [c for c in df.columns if c.endswith("_validity_point")]

    logger.info(f"Found {len(point_columns)} models to analyze")
    logger.info(f"Analyzing {len(df)} tasks")

    _statistical_summary(df, point_columns)
    _model_comparison_boxplot(df, point_columns)
    _agreement_heatmap(df, point_columns)
    _failure_analysis_detailed(df, point_columns)
    _pairwise_difference_analysis(df, point_columns)

    logger.info("Benchmarking complete")
    logger.info(f"All plots and summaries saved to: {OUTPUT_DIR}/")
