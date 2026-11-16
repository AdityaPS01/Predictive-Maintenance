"""Generate EDA charts from the official AI4I CSV (never from fabricated data)."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ml.ml_model import DATA_PATH, load_and_clean_data

OUT = ROOT / "ui" / "static" / "charts"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="darkgrid", context="notebook")
BG = "#0c1219"
FG = "#dce9ee"
CYAN = "#4be7d2"
ORANGE = "#ff9c65"


def finish(fig, filename, output_dir=OUT):
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        for spine in ax.spines.values():
            spine.set_color("#2a3a45")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main(csv_path=DATA_PATH, output_dir=OUT):
    try:
        df, cleaning = load_and_clean_data(csv_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot generate charts: {error}", file=sys.stderr)
        print("Add the official CSV at data/ai4i2020.csv, then run this script again.", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(7, 3.5))
    counts = df["Machine failure"].value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar(["No failure (0)", "Failure (1)"], counts.values, color=[CYAN, ORANGE], width=.55)
    ax.set_title("Machine Failure Distribution", loc="left", pad=12, fontsize=12)
    ax.set_ylabel("Number of records")
    ax.bar_label(bars, padding=3, color=FG, fontsize=9)
    finish(fig, "failure_distribution.png", output_dir)

    distributions = [
        ("Air temperature [K]", "Air temperature distribution", "air_temperature.png"),
        ("Process temperature [K]", "Process temperature distribution", "process_temperature.png"),
        ("Rotational speed [rpm]", "Rotational speed distribution", "rotational_speed.png"),
        ("Torque [Nm]", "Torque distribution", "torque.png"),
        ("Tool wear [min]", "Tool wear distribution", "tool_wear.png"),
    ]
    for column, title, filename in distributions:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.histplot(data=df, x=column, bins=35, color=CYAN, edgecolor=BG, ax=ax)
        ax.set_title(title, loc="left", pad=12, fontsize=12)
        ax.set_ylabel("Count")
        finish(fig, filename)

    numeric = df.select_dtypes(include="number").drop(columns=["UDI"], errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(numeric.corr(), cmap="mako", center=0, annot=True, fmt=".2f",
                linewidths=.4, linecolor=BG, annot_kws={"size": 7}, ax=ax,
                cbar_kws={"shrink": .8})
    ax.set_title("Correlation Heatmap", loc="left", pad=12, fontsize=12)
    ax.tick_params(axis="x", labelrotation=35)
    ax.tick_params(axis="y", labelrotation=0)
    finish(fig, "correlation_heatmap.png", output_dir)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x="Torque [Nm]", y="Tool wear [min]", hue="Machine failure",
                    palette={0: CYAN, 1: ORANGE}, alpha=.65, s=18, ax=ax)
    ax.set_title("Torque vs Tool Wear", loc="left", pad=12, fontsize=12)
    ax.legend(title="Failure", labels=["No failure (0)", "Failure (1)"], frameon=False)
    finish(fig, "torque_tool_wear.png", output_dir)
    print(f"Generated 8 charts in {output_dir}")
    print(f"Rows used: {len(df):,}; duplicates removed: {cleaning['duplicates_removed']}; missing values before cleaning: {cleaning['missing_values_before_cleaning']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
