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
        if temperature is not None:
            temperature = float(temperature)
            display_name = f"{display_name.split('-')[0]} t={temperature:.2f}"
        else:
            display_name = f"{display_name.split('-')[0]}"

        if display_name == "Ownmetrics":
            display_name = "Own Metrics"
            base_name = "own_metrics"
        temperature = str(temperature)
        return cls(
            column_name=column,
            base_name=base_name,
            temperature=temperature,
            display_name=display_name,
        )


class ModelAnalyzer:
    def __init__(self, point_columns: List[str]) -> None:
        self.models = [ModelInfo.from_column(col) for col in point_columns]

        def sort_key(m: ModelInfo):
            is_not_own_metrics = m.base_name != "own_metrics"
            try:
                temp_val = float(m.temperature) if m.temperature else -1.0
            except ValueError:
                temp_val = -1.0
            return (is_not_own_metrics, m.base_name.lower(), -temp_val)

        self.models.sort(key=sort_key)
        self._color_map = self._generate_color_map()

    def _generate_color_map(self) -> dict:
        unique_base_names = sorted(set(m.base_name for m in self.models))
        palette = sns.color_palette("husl", len(unique_base_names))
        return dict(zip(unique_base_names, palette))

    def get_colors(self) -> List:
        return [self._color_map[m.base_name] for m in self.models]

    def get_display_names(self) -> List[str]:
        return [m.display_name for m in self.models]

    def get_column_names(self) -> List[str]:
        return [m.column_name for m in self.models]


def _configure_dark_plot(ax: object) -> None:
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


def _save_plot(filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"Saved: {path}")


def _statistical_summary(
    df: pd.DataFrame, point_columns: list[str], analyzer: ModelAnalyzer
) -> pd.DataFrame:
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


def _format_time(value: float) -> str:
    if value >= 60:
        return f"{value / 60:.2f}m"
    elif value >= 1:
        return f"{value:.2f}s"
    else:
        return f"{value * 1000:.2f}ms"


def _create_single_boxplot(
    df: pd.DataFrame,
    models: List[ModelInfo],
    filename: str,
    title: str,
    color_map: dict,
) -> None:
    point_columns = [m.column_name for m in models]
    df_valid = (
        df[point_columns].apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    )
    plot_labels = [m.display_name for m in models]

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

    colors = [color_map.get(m.base_name, "#89b4fa") for m in models]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(DARK_TEXT)

    ax.set_xticklabels(plot_labels, fontweight="bold", color=DARK_TEXT)
    ax.set_ylabel("Validity Score", fontsize=12, fontweight="bold", color=DARK_TEXT)
    ax.set_title(
        f"{title} (samples: {len(df)})",
        fontsize=14,
        fontweight="bold",
        color=DARK_TEXT,
    )
    plt.xticks(rotation=20, ha="right", fontsize=10, color=DARK_TEXT)
    _configure_dark_plot(ax)

    _save_plot(filename)


def _model_comparison_boxplot(
    df: pd.DataFrame, point_columns: list[str], analyzer: ModelAnalyzer
) -> None:
    logger.info("Creating model comparison boxplots")

    models_by_base = {}
    own_metrics_model = None

    for m in analyzer.models:
        if m.base_name == "own_metrics":
            own_metrics_model = m
            continue

        if m.base_name not in models_by_base:
            models_by_base[m.base_name] = []
        models_by_base[m.base_name].append(m)

    for base_name, models in models_by_base.items():
        models.sort(key=lambda x: float(x.temperature) if x.temperature else -1)

        _create_single_boxplot(
            df,
            models,
            f"model_comparison_boxplot_{base_name}.png",
            f"Validity Scores - {base_name}",
            analyzer._color_map,
        )

    best_models = []
    if own_metrics_model:
        best_models.append(own_metrics_model)

    for base_name, models in models_by_base.items():
        best_model = None
        min_failure_rate = float("inf")

        for m in models:
            col_numeric = pd.to_numeric(df[m.column_name], errors="coerce")
            failure_rate = (col_numeric == -1).mean()

            if failure_rate < min_failure_rate:
                min_failure_rate = failure_rate
                best_model = m

        if best_model:
            best_models.append(best_model)

    _create_single_boxplot(
        df,
        best_models,
        "model_comparison_boxplot_best.png",
        "Validity Scores - Lowest Failure Rate",
        analyzer._color_map,
    )


def _failure_analysis_detailed(
    df: pd.DataFrame, point_columns: list[str], analyzer: ModelAnalyzer
) -> None:
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
        f"Classification Failure Rate with Different Temperatures (samples: {len(df)})",
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
    df: pd.DataFrame,
    point_columns: list[str],
    analyzer: ModelAnalyzer,
    spam_threshold: int = 20,
) -> None:
    logger.info("Creating spam agreement heatmap for best models")

    models_by_base = {}
    own_metrics_model = None

    for m in analyzer.models:
        if m.base_name == "own_metrics":
            own_metrics_model = m
            continue

        if m.base_name not in models_by_base:
            models_by_base[m.base_name] = []
        models_by_base[m.base_name].append(m)

    best_models = []
    if own_metrics_model:
        best_models.append(own_metrics_model)

    for base_name, models in models_by_base.items():
        best_model = None
        min_failure_rate = float("inf")

        for m in models:
            col_numeric = pd.to_numeric(df[m.column_name], errors="coerce")
            failure_rate = (col_numeric == -1).mean()

            if failure_rate < min_failure_rate:
                min_failure_rate = failure_rate
                best_model = m

        if best_model:
            best_models.append(best_model)

    selected_point_columns = [m.column_name for m in best_models]
    plot_labels = [m.display_name for m in best_models]

    df_valid = (
        df[selected_point_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace(-1, np.nan)
    )

    spam_matrix = (df_valid < spam_threshold).astype(float)

    n_models = len(selected_point_columns)
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

    for i in range(n_models):
        for j in range(n_models):
            if not np.isnan(agreement[i, j]):
                ax.text(
                    j,
                    i,
                    f"{agreement[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=18,
                )

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
        f"Spam Agreement - Lowest failure rate Models (samples: {len(df)})",
        fontsize=14,
        fontweight="bold",
        color=DARK_TEXT,
    )
    cbar = plt.colorbar(im, ax=ax, label="Spam Agreement Rate")
    cbar.ax.yaxis.label.set_color(DARK_TEXT)
    cbar.ax.tick_params(colors=DARK_TEXT)
    cbar.outline.set_edgecolor(DARK_TEXT)

    _save_plot("model_spam_agreement_heatmap.png")


def _spam_detected_analysis(
    df: pd.DataFrame, point_columns: list[str], analyzer: ModelAnalyzer
) -> None:
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


def _model_execution_time_analysis(analyzer: ModelAnalyzer) -> None:
    logger.info("Creating model execution time analysis")

    timing_data = []
    export_folder = config.BENCHMARK_FOLDER + "/timing"

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
                    model_name = ModelInfo.from_column(
                        raw_name + "_validity_point"
                    ).display_name
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

    if not timing_data:
        logger.warning("No timing data found. Skipping execution time analysis.")
        return

    timing_df = pd.DataFrame(timing_data)

    model_order = {
        (
            m.base_name
            if m.temperature is None
            else m.column_name.replace("_validity_point", "")
        ): i
        for i, m in enumerate(analyzer.models)
    }

    timing_df["sort_order"] = timing_df["RawName"].map(
        lambda x: -1 if x == "own_metrics" else model_order.get(x, 999)
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
        f"Total Execution Time (samples: {len(df_timing)})",
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
    ax.invert_yaxis()
    ax.set_xlabel(
        "Average Time per Item (s)", fontsize=12, fontweight="bold", color=DARK_TEXT
    )
    ax.set_title(
        "Average Execution Time per Item (samples: 10000)",
        fontsize=13,
        fontweight="bold",
        color=DARK_TEXT,
    )

    for i, v in enumerate(timing_df["Average Time per Item (s)"]):
        ax.text(v, i, f" {_format_time(v)}", va="center", fontsize=10, color=DARK_TEXT)

    _configure_dark_plot(ax)
    _save_plot("model_execution_time_average.png")


def run_benchmark() -> None:
    logger.info("Running benchmarks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_path = os.path.join(config.BENCHMARK_FOLDER, "Issue.csv")
    df = safe_read_csv(file_path)
    point_columns = [c for c in df.columns if c.endswith("_validity_point")]

    logger.info(f"Found {len(point_columns)} models to analyze")
    logger.info(f"Analyzing {len(df)} tasks")

    analyzer = ModelAnalyzer(point_columns)
    sorted_point_columns = analyzer.get_column_names()

    _statistical_summary(df, sorted_point_columns, analyzer)
    _model_comparison_boxplot(df, sorted_point_columns, analyzer)
    _failure_analysis_detailed(df, sorted_point_columns, analyzer)
    _spam_agreement_heatmap(df, sorted_point_columns, analyzer, spam_threshold=20)
    _spam_detected_analysis(df, sorted_point_columns, analyzer)
    _model_execution_time_analysis(analyzer)

    logger.info("Benchmarking complete")
    logger.info(f"All plots and summaries saved to: {OUTPUT_DIR}/")
