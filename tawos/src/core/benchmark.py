import os
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
from config_loader import config
import matplotlib.pyplot as plt
import seaborn as sns
from core.log import get_logger
from matplotlib.colors import LinearSegmentedColormap

from utils.file_reading import safe_read_csv

logger = get_logger("Benchmark")
OUTPUT_DIR = config.BENCHMARK_OUTPUT

plt.style.use("dark_background")
sns.set_palette("husl")

DARK_BG = "#181825"
DARK_GRID = "#1e1e2e"
DARK_TEXT = "#cdd6f4"
ACCENT_COLOR = "#89b4fa"


@dataclass
class ModelInfo:
    column_name: str
    base_name: str
    temperature: str | None
    display_name: str

    @classmethod
    def from_column(cls, column: str) -> "ModelInfo":
        name = column.replace("_validity_point", "")

        if "_temp_" in name:
            base_name, temperature = name.split("_temp_", 1)
        else:
            base_name = name
            temperature = None

        display_name = name.replace("_", " ").title()

        if display_name == "Ownmetrics":
            display_name = "Own Metrics"

        return cls(
            column_name=column,
            base_name=base_name,
            temperature=temperature,
            display_name=display_name,
        )


class ModelAnalyzer:
    def __init__(self, point_columns: List[str]):
        self.models = [ModelInfo.from_column(col) for col in point_columns]
        self._color_map = self._generate_color_map()

    def _generate_color_map(self) -> dict:
        unique_base_names = sorted(set(m.base_name for m in self.models))
        palette = sns.color_palette("husl", len(unique_base_names))
        return dict(zip(unique_base_names, palette))

    def get_colors(self) -> List:
        return [self._color_map[m.base_name] for m in self.models]

    def get_display_names(self) -> List[str]:
        display_names = []
        previous_base_name = None
        counter = 0
        for m in self.models:
            if m.base_name == previous_base_name:
                counter += 1
                invisible_suffix = "\u200b" * counter
                display_names.append(f"Temp {m.temperature}{invisible_suffix}")
            else:
                display_names.append(m.display_name)
            previous_base_name = m.base_name
        return display_names

    def get_column_names(self) -> List[str]:
        return [m.column_name for m in self.models]


def _configure_dark_plot(ax):
    ax.set_facecolor(DARK_BG)
    ax.figure.patch.set_facecolor(DARK_BG)
    ax.tick_params(colors=DARK_TEXT)
    ax.spines["bottom"].set_color(DARK_TEXT)
    ax.spines["top"].set_color(DARK_TEXT)
    ax.spines["left"].set_color(DARK_TEXT)
    ax.spines["right"].set_color(DARK_TEXT)
    ax.xaxis.label.set_color(DARK_TEXT)
    ax.yaxis.label.set_color(DARK_TEXT)
    ax.title.set_color(DARK_TEXT)
    ax.grid(alpha=0.2, color=DARK_GRID)


def _save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"Saved: {path}")


def _statistical_summary(df, point_columns, analyzer: ModelAnalyzer):
    logger.info("Generating statistical summary")

    df_numeric = df[point_columns].apply(pd.to_numeric, errors="coerce")
    df_valid = df_numeric.replace(-1, np.nan)

    summary = pd.DataFrame(
        {
            "Model": analyzer.get_display_names(),
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


def _format_time(value):
    if value >= 60:
        return f"{value / 60:.2f}m"
    elif value >= 1:
        return f"{value:.2f}s"
    else:
        return f"{value * 1000:.2f}ms"


def _model_comparison_boxplot(df, point_columns, analyzer: ModelAnalyzer):
    logger.info("Creating model comparison boxplot")

    df_valid = (
        df[point_columns].apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    )
    plot_labels = analyzer.get_display_names()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    data_to_plot = [df_valid[col].dropna() for col in point_columns]

    bp = ax.boxplot(
        data_to_plot,
        labels=plot_labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(color=ACCENT_COLOR, linewidth=2),
        meanprops=dict(color="#ff6b6b", linewidth=2),
        whiskerprops=dict(color=DARK_TEXT),
        capprops=dict(color=DARK_TEXT),
        flierprops=dict(markeredgecolor=DARK_TEXT, marker="o", markersize=4, alpha=0.5),
    )

    colors = analyzer.get_colors()
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(DARK_TEXT)

    # for i, data in enumerate(data_to_plot, 1):
    #     median = data.median()
    #     mean = data.mean()
    #     q1 = data.quantile(0.25)
    #     q3 = data.quantile(0.75)
    #     iqr = q3 - q1
    #     lower_bound = q1 - 1.5 * iqr
    #     upper_bound = q3 + 1.5 * iqr
    #     outliers = data[(data < lower_bound) | (data > upper_bound)]

    #     label_text = f"Med: {median:.2f}\nMean: {mean:.2f}\nOutliers: {len(outliers)}"
    #     ax.text(
    #         i + 0.1,
    #         (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 - 20,
    #         label_text,
    #         ha="left",
    #         va="center",
    #         fontsize=8,
    #         fontweight="bold",
    #         color=DARK_TEXT,
    #         bbox=dict(
    #             boxstyle="round,pad=0.3",
    #             facecolor=colors[i - 1],
    #             alpha=0.5,
    #             edgecolor=DARK_TEXT,
    #         ),
    #     )

    ax.set_xticklabels(plot_labels, fontweight="bold", color=DARK_TEXT)
    ax.set_ylabel("Validity Score", fontsize=12, fontweight="bold", color=DARK_TEXT)
    ax.set_title(
        f"Distribution of Validity Scores (samples: {len(df)})",
        fontsize=14,
        fontweight="bold",
        color=DARK_TEXT,
    )
    plt.xticks(rotation=20, ha="right", fontsize=10, color=DARK_TEXT)
    _configure_dark_plot(ax)

    _save_plot("model_comparison_boxplot.png")


def _failure_analysis_detailed(df, point_columns, analyzer: ModelAnalyzer):
    logger.info("Creating detailed failure analysis")

    failure_data = []
    plot_labels = analyzer.get_display_names()

    for i, col in enumerate(point_columns):
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        failures = df[col_numeric == -1]

        model_dict = {
            "Model": plot_labels[i],
            "Total Failures": len(failures),
            "Failure Rate (%)": (len(failures) / len(df)) * 100,
        }

        failure_data.append(model_dict)

    failure_df = pd.DataFrame(failure_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    colors = analyzer.get_colors()

    y_pos = np.arange(len(failure_df))
    bars = ax.barh(y_pos, failure_df["Failure Rate (%)"], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(failure_df["Model"], fontweight="bold", color=DARK_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Failure Rate (%)", fontsize=12, fontweight="bold", color=DARK_TEXT)
    ax.set_title(
        f"Classification Failure Rate (samples: {len(df)})",
        fontsize=13,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_xlim(0, 100)

    for i, (bar, count, rate) in enumerate(
        zip(bars, failure_df["Total Failures"], failure_df["Failure Rate (%)"])
    ):
        ax.text(
            rate + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({rate:.1f}%)",
            va="center",
            fontsize=10,
            color=DARK_TEXT,
        )

    _configure_dark_plot(ax)
    _save_plot("failure_rate_by_model.png")


def _spam_agreement_heatmap(
    df, point_columns, analyzer: ModelAnalyzer, spam_threshold=20
):
    logger.info("Creating spam agreement heatmap")

    df_valid = (
        df[point_columns].apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    )
    plot_labels = analyzer.get_display_names()

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

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    im = ax.imshow(agreement, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_models + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_models + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=DARK_TEXT, linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(
        plot_labels, rotation=45, ha="right", fontweight="bold", color=DARK_TEXT
    )
    ax.set_yticklabels(plot_labels, fontweight="bold", color=DARK_TEXT)

    ax.set_title(
        f"Spam Agreement (samples: {len(df)})",
        fontsize=14,
        fontweight="bold",
        color=DARK_TEXT,
    )
    cbar = plt.colorbar(im, ax=ax, label="Spam Agreement Rate")
    cbar.ax.yaxis.label.set_color(DARK_TEXT)
    cbar.ax.tick_params(colors=DARK_TEXT)
    cbar.outline.set_edgecolor(DARK_TEXT)

    _save_plot("model_spam_agreement_heatmap.png")


def _spam_detected_analysis(df, point_columns, analyzer: ModelAnalyzer):
    logger.info("Analyzing spam detection rates")

    spam_data = []
    plot_labels = analyzer.get_display_names()

    for i, col in enumerate(point_columns):
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        spam_detected = df[(col_numeric < 20) & (col_numeric != -1)]

        model_dict = {
            "Model": plot_labels[i],
            "Total Spam Detected": len(spam_detected),
            "Spam Detection Rate (%)": (len(spam_detected) / len(df)) * 100,
        }

        spam_data.append(model_dict)

    spam_df = pd.DataFrame(spam_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    colors = analyzer.get_colors()

    y_pos = np.arange(len(spam_df))
    bars = ax.barh(
        y_pos,
        spam_df["Spam Detection Rate (%)"],
        color=colors,
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(spam_df["Model"], fontweight="bold", color=DARK_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Spam Detection Rate (%)", fontsize=12, fontweight="bold", color=DARK_TEXT
    )
    ax.set_title(
        f"Spam Detected (validity_point < 20) (samples: {len(df)})",
        fontsize=13,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_xlim(0, 100)

    for i, (bar, count, rate) in enumerate(
        zip(bars, spam_df["Total Spam Detected"], spam_df["Spam Detection Rate (%)"])
    ):
        ax.text(
            rate + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({rate:.1f}%)",
            va="center",
            fontsize=10,
            color=DARK_TEXT,
        )

    _configure_dark_plot(ax)
    _save_plot("spam_detection_rate_by_model.png")


def _model_execution_time_analysis(analyzer: ModelAnalyzer):
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
                "Model": "Own Metrics",
                "RawName": "own_metrics",
                "Total Time (s)": total_time,
                "Average Time per Item (s)": avg_time,
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
                    raw_name = filename.replace("_timing.csv", "")
                    model_name = raw_name.replace("_", " ").title()
                    timing_data.append(
                        {
                            "Model": model_name,
                            "RawName": raw_name,
                            "Total Time (s)": total_time,
                            "Average Time per Item (s)": avg_time,
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

    model_order = {
        m.base_name
        if m.temperature is None
        else m.column_name.replace("_validity_point", ""): i
        for i, m in enumerate(analyzer.models)
    }

    timing_df["sort_order"] = timing_df["RawName"].map(
        lambda x: model_order.get(x, 999)
    )
    timing_df = timing_df.sort_values("sort_order").reset_index(drop=True)

    base_models = timing_df["RawName"].apply(
        lambda x: ModelInfo.from_column(x + "_validity_point").base_name
    )
    model_color_map = analyzer._color_map
    colors = [model_color_map[m] for m in base_models]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    y_pos = np.arange(len(timing_df))
    ax.barh(y_pos, timing_df["Total Time (s)"], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(timing_df["Model"], fontweight="bold", color=DARK_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Total Execution Time (s)", fontsize=12, fontweight="bold", color=DARK_TEXT
    )
    ax.set_title(
        f"Total Execution Time by Model (samples: {len(df_timing)})",
        fontsize=13,
        fontweight="bold",
        color=DARK_TEXT,
    )

    for i, v in enumerate(timing_df["Total Time (s)"]):
        ax.text(v, i, f" {_format_time(v)}", va="center", fontsize=10, color=DARK_TEXT)

    _configure_dark_plot(ax)
    _save_plot("model_execution_time_total.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    y_pos = np.arange(len(timing_df))
    ax.barh(
        y_pos,
        timing_df["Average Time per Item (s)"],
        color=colors,
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(timing_df["Model"], fontweight="bold", color=DARK_TEXT)
    ax.invert_yaxis()  # Invert so first item appears at top
    ax.set_xlabel(
        "Average Time per Item (s)", fontsize=12, fontweight="bold", color=DARK_TEXT
    )
    ax.set_title(
        f"Average Execution Time per Item by Model (samples: {len(df_timing)})",
        fontsize=13,
        fontweight="bold",
        color=DARK_TEXT,
    )

    for i, v in enumerate(timing_df["Average Time per Item (s)"]):
        ax.text(v, i, f" {_format_time(v)}", va="center", fontsize=10, color=DARK_TEXT)

    _configure_dark_plot(ax)
    _save_plot("model_execution_time_average.png")


def run_benchmark():
    logger.info("Running benchmarks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_path = os.path.join(config.BENCHMARK_FOLDER, "Issue.csv")
    df = safe_read_csv(file_path)
    point_columns = [c for c in df.columns if c.endswith("_validity_point")]

    logger.info(f"Found {len(point_columns)} models to analyze")
    logger.info(f"Analyzing {len(df)} tasks")

    analyzer = ModelAnalyzer(point_columns)

    _statistical_summary(df, point_columns, analyzer)
    _model_comparison_boxplot(df, point_columns, analyzer)
    _failure_analysis_detailed(df, point_columns, analyzer)
    _spam_agreement_heatmap(df, point_columns, analyzer, spam_threshold=20)
    _spam_detected_analysis(df, point_columns, analyzer)
    _model_execution_time_analysis(analyzer)

    logger.info("Benchmarking complete")
    logger.info(f"All plots and summaries saved to: {OUTPUT_DIR}/")
