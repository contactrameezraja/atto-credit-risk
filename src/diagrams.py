"""Generate professional diagrams for the README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

DARK = "#1E293B"
BLUE = "#2563EB"
LIGHT_BLUE = "#DBEAFE"
GREEN = "#059669"
LIGHT_GREEN = "#D1FAE5"
ORANGE = "#D97706"
LIGHT_ORANGE = "#FEF3C7"
PURPLE = "#7C3AED"
LIGHT_PURPLE = "#EDE9FE"
GREY = "#64748B"
LIGHT_GREY = "#F1F5F9"
WHITE = "#FFFFFF"


def draw_box(ax, x, y, w, h, label, sublabel=None, color=BLUE, bg=LIGHT_BLUE):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=bg, edgecolor=color, linewidth=2.5
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.22, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=DARK)
        ax.text(x, y - 0.18, sublabel, ha="center", va="center",
                fontsize=8.5, color=GREY, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=DARK)


def arrow_down(ax, x, y_from, y_to, color=GREY):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))


def arrow_right(ax, x_from, x_to, y, color=GREY):
    ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 13)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # Title
    ax.text(8, 12.5, "Credit Risk Prediction Pipeline", ha="center",
            fontsize=22, fontweight="bold", color=DARK)
    ax.text(8, 12.0, "End-to-end flow from raw data to live predictions",
            ha="center", fontsize=12, color=GREY)

    # Divider line between Part 1 and Part 2
    ax.plot([8.2, 8.2], [1.0, 11.5], color="#E2E8F0", linewidth=1.5, linestyle="--")

    # === LEFT COLUMN: Part 1 ===
    cx = 4.5

    ax.text(cx, 11.2, "PART 1: DATA ENGINEERING", ha="center", fontsize=10,
            fontweight="bold", color=GREEN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_GREEN, edgecolor=GREEN))

    # Inputs side by side
    draw_box(ax, 3, 10, 2.2, 0.8, "transactions.csv", "10 rows, 6 cols", BLUE, LIGHT_BLUE)
    draw_box(ax, 6, 10, 2.2, 0.8, "labels.csv", "5 customers", BLUE, LIGHT_BLUE)

    # Arrows from both inputs merge into Load
    arrow_down(ax, 3, 9.6, 9.15, BLUE)
    arrow_down(ax, 6, 9.6, 9.15, BLUE)

    # Pipeline steps - clean vertical chain
    draw_box(ax, cx, 8.7, 3.5, 0.8, "Load & Explore", "Data Quality Report", GREEN, LIGHT_GREEN)
    arrow_down(ax, cx, 8.3, 7.75, GREEN)

    draw_box(ax, cx, 7.3, 3.5, 0.8, "Clean", "Nulls, Duplicates, Outliers", GREEN, LIGHT_GREEN)
    arrow_down(ax, cx, 6.9, 6.35, GREEN)

    draw_box(ax, cx, 5.9, 3.5, 0.8, "Feature Engineering", "Core (4) + Custom (3)", GREEN, LIGHT_GREEN)
    arrow_down(ax, cx, 5.5, 4.95, GREEN)

    draw_box(ax, cx, 4.5, 3.5, 0.8, "Text Processing", "kw_ keywords + has_ categories", GREEN, LIGHT_GREEN)
    arrow_down(ax, cx, 4.1, 3.55, GREEN)

    # Output
    draw_box(ax, cx, 3.1, 3.5, 0.8, "training_set.csv", "5 rows x 21 columns", ORANGE, LIGHT_ORANGE)

    # EDA branch going down-left from training_set
    draw_box(ax, 1.2, 2.1, 2.0, 0.8, "EDA Charts", "3 visualisations", ORANGE, LIGHT_ORANGE)
    ax.annotate("", xy=(1.8, 2.5), xytext=(3.0, 2.7),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))

    # === RIGHT COLUMN: Part 2 ===
    rx = 11.5

    ax.text(rx, 11.2, "PART 2: API SERVICE", ha="center", fontsize=10,
            fontweight="bold", color=PURPLE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_PURPLE, edgecolor=PURPLE))

    draw_box(ax, rx, 10, 2.8, 0.8, "model.joblib", "LogisticRegression", PURPLE, LIGHT_PURPLE)
    arrow_down(ax, rx, 9.6, 9.15, PURPLE)

    draw_box(ax, rx, 8.7, 2.8, 0.8, "FastAPI Service", "app.py + uvicorn", PURPLE, LIGHT_PURPLE)

    # Two branches from FastAPI
    arrow_down(ax, 10.5, 8.3, 7.75, GREY)
    arrow_down(ax, 12.5, 8.3, 7.75, PURPLE)

    draw_box(ax, 10.5, 7.3, 2.2, 0.8, "GET /health", "Readiness probe", GREY, LIGHT_GREY)
    draw_box(ax, 12.5, 7.3, 2.2, 0.8, "POST /predict", "JSON request", PURPLE, LIGHT_PURPLE)

    arrow_down(ax, 12.5, 6.9, 6.35, PURPLE)

    draw_box(ax, 12.5, 5.9, 2.8, 0.8, "Validate Input", "Pydantic constraints", PURPLE, LIGHT_PURPLE)
    arrow_down(ax, 12.5, 5.5, 4.95, PURPLE)

    draw_box(ax, 12.5, 4.5, 2.8, 0.8, "Model Inference", "predict_proba()", PURPLE, LIGHT_PURPLE)
    arrow_down(ax, 12.5, 4.1, 3.55, PURPLE)

    draw_box(ax, 12.5, 3.1, 2.8, 0.8, "JSON Response", "probability + prediction", ORANGE, LIGHT_ORANGE)

    # Dashed arrow connecting Part 1 to Part 2
    ax.annotate("", xy=(10.1, 10), xytext=(7.1, 10),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8, linestyle="--"))
    ax.text(8.6, 10.35, "model trained on", ha="center", fontsize=8.5, color=GREY, style="italic")

    # Bottom note
    ax.text(8, 0.6, "Part 1 produces the training data. Part 2 serves predictions from the trained model.",
            ha="center", fontsize=10, color=GREY, style="italic")

    fig.tight_layout()
    out = str(Path(__file__).resolve().parents[1] / "artifacts" / "diagram_pipeline_flow.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved: {out}")


def project_structure_diagram():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(7, 8.5, "Project Structure", ha="center", va="center",
            fontsize=20, fontweight="bold", color=DARK)

    # Data section
    box = FancyBboxPatch((0.3, 5.5), 3.2, 2.4, boxstyle="round,pad=0.2",
                          facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=2.5)
    ax.add_patch(box)
    ax.text(1.9, 7.5, "data/", fontsize=14, fontweight="bold", color=BLUE, ha="center")
    ax.text(1.9, 6.9, "transactions.csv", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")
    ax.text(1.9, 6.5, "10 rows, 6 columns", fontsize=8, color=GREY, ha="center", style="italic")
    ax.text(1.9, 6.0, "labels.csv", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")

    # Artifacts section
    box = FancyBboxPatch((3.9, 5.5), 3.8, 2.4, boxstyle="round,pad=0.2",
                          facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=2.5)
    ax.add_patch(box)
    ax.text(5.8, 7.5, "artifacts/", fontsize=14, fontweight="bold", color=ORANGE, ha="center")
    ax.text(5.8, 6.9, "model.joblib", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")
    ax.text(5.8, 6.5, "Pre-trained model (provided)", fontsize=8, color=GREY, ha="center", style="italic")
    ax.text(5.8, 6.0, "training_set.csv + EDA charts", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")
    ax.text(5.8, 5.6, "Generated by pipeline", fontsize=8, color=GREY, ha="center", style="italic")

    # Src section
    box = FancyBboxPatch((8.1, 5.5), 5.5, 2.4, boxstyle="round,pad=0.2",
                          facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=2.5)
    ax.add_patch(box)
    ax.text(10.85, 7.5, "src/", fontsize=14, fontweight="bold", color=GREEN, ha="center")
    ax.text(9.5, 6.85, "prepare_data.py", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")
    ax.text(9.5, 6.45, "Part 1", fontsize=8, color=GREY, ha="center", style="italic")
    ax.text(12.2, 6.85, "app.py", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")
    ax.text(12.2, 6.45, "Part 2", fontsize=8, color=GREY, ha="center", style="italic")
    ax.text(10.85, 5.85, "eda.py", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")

    # Root files section
    box = FancyBboxPatch((0.3, 3.5), 3.2, 1.5, boxstyle="round,pad=0.2",
                          facecolor=LIGHT_PURPLE, edgecolor=PURPLE, linewidth=2.5)
    ax.add_patch(box)
    ax.text(1.9, 4.6, "Root Files", fontsize=14, fontweight="bold", color=PURPLE, ha="center")
    ax.text(1.9, 4.05, "requirements.txt", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")
    ax.text(1.9, 3.6, "README.md", fontsize=10, color=DARK, ha="center",
            fontfamily="monospace")

    # Legend
    ax.text(7, 2.7, "Provided:  data/*.csv  +  artifacts/model.joblib",
            ha="center", fontsize=10, color=BLUE, fontweight="bold")
    ax.text(7, 2.2, "Generated:  artifacts/training_set.csv  +  artifacts/eda_*.png",
            ha="center", fontsize=10, color=ORANGE, fontweight="bold")
    ax.text(7, 1.7, "Your code:  src/prepare_data.py  +  src/app.py  +  src/eda.py",
            ha="center", fontsize=10, color=GREEN, fontweight="bold")

    fig.tight_layout()
    out = str(Path(__file__).resolve().parents[1] / "artifacts" / "diagram_project_structure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    pipeline_diagram()
    project_structure_diagram()
    print("Diagrams complete.")
