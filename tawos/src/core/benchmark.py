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

    _, ax = plt.subplots(figsize=(12, 6))

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
            ax.text(
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
    logger.info("Creating detailed failure analysis")

    all_types = df["Type"].dropna().unique()
    logger.info(f"Found task types: {', '.join(all_types)}")

    failure_data = []
    for col in point_columns:
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        failures = df[col_numeric == -1]

        model_dict = {
            "Model": col.replace("_validity_point", ""),
            "Total Failures": len(failures),
            "Failure Rate (%)": (len(failures) / len(df)) * 100,
        }

        for task_type in all_types:
            model_dict[f"{task_type} Failures"] = len(
                failures[failures["Type"] == task_type]
            )

        failure_data.append(model_dict)

    failure_df = pd.DataFrame(failure_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(failure_df["Model"], failure_df["Failure Rate (%)"], color="coral")
    ax.set_xlabel("Failure Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Classification Failure Rate by Model", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 100)
    _save_plot("failure_rate_by_model.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    type_columns = [col for col in failure_df.columns if col.endswith(" Failures")]
    failure_by_type = failure_df[["Model"] + type_columns].set_index("Model")

    failure_by_type.columns = [
        col.replace(" Failures", "") for col in failure_by_type.columns
    ]

    failure_by_type.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Failures", fontsize=12, fontweight="bold")
    ax.set_title("Failure Distribution by Task Type", fontsize=13, fontweight="bold")
    ax.legend(title="Task Type", bbox_to_anchor=(1.05, 1))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    _save_plot("failure_distribution_by_type.png")

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
                ax.text(
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


def _spam_detected_analysis(df, point_columns):
    logger.info("Analyzing spam detection rates")

    spam_data = []
    for col in point_columns:
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        spam_detected = df[(col_numeric < 20) & (col_numeric != -1)]

        model_dict = {
            "Model": col.replace("_validity_point", ""),
            "Total Spam Detected": len(spam_detected),
            "Spam Detection Rate (%)": (len(spam_detected) / len(df)) * 100,
        }

        spam_data.append(model_dict)

    spam_df = pd.DataFrame(spam_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(spam_df["Model"], spam_df["Spam Detection Rate (%)"], color="skyblue")
    ax.set_xlabel("Spam Detection Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Spam Detected by Model (validity_point < 20)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 100)
    _save_plot("spam_detection_rate_by_model.png")

    spam_path = os.path.join(OUTPUT_DIR, "spam_detection_summary.csv")
    spam_df.to_csv(spam_path, index=False)
    logger.info(f"Saved spam detection summary to {spam_path}")


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
    _spam_detected_analysis(df, point_columns)

    logger.info("Benchmarking complete")
    logger.info(f"All plots and summaries saved to: {OUTPUT_DIR}/")
