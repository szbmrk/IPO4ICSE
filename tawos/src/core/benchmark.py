import os
import pandas as pd
import numpy as np
from config_loader import config
import matplotlib.pyplot as plt
import seaborn as sns
from core.log import get_logger
from matplotlib.colors import LinearSegmentedColormap

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

    _, ax = plt.subplots(figsize=(10, 6))
    ax.barh(failure_df["Model"], failure_df["Failure Rate (%)"], color="coral")
    ax.set_xlabel("Failure Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Classification Failure Rate by Model", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 100)
    _save_plot("failure_rate_by_model.png")

    failure_path = os.path.join(OUTPUT_DIR, "failure_summary.csv")
    failure_df.to_csv(failure_path, index=False)
    logger.info(f"Saved failure summary to {failure_path}")


def _spam_agreement_heatmap(df, point_columns, spam_threshold=20):
    logger.info("Creating spam agreement heatmap")

    df_valid = (
        df[point_columns].apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    )
    model_names = _get_model_names(point_columns)

    spam_matrix = (df_valid < spam_threshold).astype(float)

    n_models = len(point_columns)
    agreement = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            col_i = spam_matrix.iloc[:, i]
            col_j = spam_matrix.iloc[:, j]
            both_valid = ~(col_i.isna() | col_j.isna())

            if both_valid.sum() == 0:
                agreement[i, j] = np.nan
            else:
                agreement[i, j] = (
                    col_i.loc[both_valid] == col_j.loc[both_valid]
                ).mean()

    colors = ["#ff4d4d", "#ffec99", "#4caf50"]
    cmap = LinearSegmentedColormap.from_list("agreement", colors)

    _, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(agreement, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticklabels(model_names)

    for i in range(n_models):
        for j in range(n_models):
            ax.text(
                j, i, f"{agreement[i, j]:.2f}", ha="center", va="center", color="black"
            )

    ax.set_title("Inter-Model Spam Agreement", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Spam Agreement Rate")

    _save_plot("model_spam_agreement_heatmap.png")


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

    _, ax = plt.subplots(figsize=(10, 6))
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


def _model_execution_time_analysis():
    logger.info("Creating model execution time analysis")

    timing_data = []
    export_folder = config.BENCHMARK_FOLDER + "/timing"

    own_metrics_timing_path = os.path.join(export_folder, "own_metrics_timing.csv")
    if os.path.exists(own_metrics_timing_path):
        df_timing = pd.read_csv(own_metrics_timing_path)
        total_time = df_timing["Timing (s)"].sum()
        avg_time = df_timing["Timing (s)"].mean()
        timing_data.append(
            {
                "Model": "OwnMetrics",
                "Total Time (s)": total_time,
                "Average Time per Item (ms)": avg_time * 1000,
                "Items Processed": len(df_timing),
            }
        )
        logger.info(f"Found timing data for OwnMetrics: {len(df_timing)} items")

    for filename in os.listdir(export_folder):
        if "timing" in filename.lower() and filename.endswith(".csv"):
            if filename == "own_metrics_timing.csv":
                continue

            filepath = os.path.join(export_folder, filename)
            try:
                df_timing = pd.read_csv(filepath)
                if "Timing (s)" in df_timing.columns:
                    total_time = df_timing["Timing (s)"].sum()
                    avg_time = df_timing["Timing (s)"].mean()
                    model_name = (
                        filename.replace("_timing.csv", "").replace("_", " ").title()
                    )
                    timing_data.append(
                        {
                            "Model": model_name,
                            "Total Time (s)": total_time,
                            "Average Time per Item (ms)": avg_time * 1000,
                            "Items Processed": len(df_timing),
                        }
                    )
                    logger.info(
                        f"Found timing data for {model_name}: {len(df_timing)} items"
                    )
            except Exception as e:
                logger.warning(f"Could not read timing file {filename}: {e}")

    if not timing_data:
        logger.warning("No timing data found. Skipping execution time analysis.")
        return

    timing_df = pd.DataFrame(timing_data)

    _, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(timing_df))
    ax.barh(timing_df["Model"], timing_df["Total Time (s)"], color=colors)
    ax.set_xlabel("Total Execution Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Total Execution Time by Model", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(timing_df["Total Time (s)"]):
        ax.text(v, i, f" {v:.2f}s", va="center", fontsize=10)

    _save_plot("model_execution_time_total.png")

    _, ax = plt.subplots(figsize=(10, 6))
    ax.barh(timing_df["Model"], timing_df["Average Time per Item (ms)"], color=colors)
    ax.set_xlabel(
        "Average Time per Item (milliseconds)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Average Execution Time per Item by Model", fontsize=13, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(timing_df["Average Time per Item (ms)"]):
        ax.text(v, i, f" {v:.2f}ms", va="center", fontsize=10)

    _save_plot("model_execution_time_average.png")

    timing_summary_path = os.path.join(OUTPUT_DIR, "execution_time_summary.csv")
    timing_df.to_csv(timing_summary_path, index=False)
    logger.info(f"Saved execution time summary to {timing_summary_path}")


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
    _failure_analysis_detailed(df, point_columns)
    _spam_agreement_heatmap(df, point_columns, 20)
    _spam_detected_analysis(df, point_columns)
    _model_execution_time_analysis()

    logger.info("Benchmarking complete")
    logger.info(f"All plots and summaries saved to: {OUTPUT_DIR}/")
