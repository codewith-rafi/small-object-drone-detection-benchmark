#!/usr/bin/env python3
"""
Generate simplified architecture figures (short labels, no overflow).

Outputs mirrored to:
  report/static/images/
  01_benchmark/results/architecture/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    REPO / "report" / "static" / "images",
    REPO / "01_benchmark" / "results" / "architecture",
]

C_BASE = "#bfdbfe"
C_SAHI = "#a5b4fc"
C_P2 = "#c4b5fd"
C_INT = "#86efac"
C_AN = "#fde68a"
C_EDGE = "#1e293b"
C_MUTED = "#64748b"
C_DARK = "#0f172a"
C_NECK = "#e0e7ff"
C_GRAY = "#e2e8f0"


def _ensure_dirs() -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    for d in OUT_DIRS:
        fig.savefig(d / name, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.25)
        print(f"wrote {d / name}")
    if name == "architecture_phases.png":
        root = REPO / "01_benchmark" / "results" / name
        fig.savefig(root, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.25)
        print(f"wrote {root}")
    plt.close(fig)


def _box(ax, x, y, w, h, title, subtitle="", facecolor=C_GRAY, title_size=11, sub_size=9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=facecolor,
            edgecolor=C_EDGE,
            linewidth=1.3,
        )
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=C_DARK,
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color="#334155",
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=C_DARK,
        )


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=C_EDGE, lw=1.6),
    )


def fig_phases_overview() -> None:
    phases = [
        ("Phase 1", "Baseline", C_BASE),
        ("Phase 2", "SAHI", C_SAHI),
        ("Phase 3", "P2 Head", C_P2),
        ("Phase 4", "Interaction", C_INT),
        ("Phase 5", "Analysis", C_AN),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 2.6))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.set_title("Research Pipeline", fontsize=14, fontweight="bold", pad=10)

    w, h, gap, x0, y = 2.0, 1.35, 0.28, 0.4, 0.55
    for i, (phase, name, color) in enumerate(phases):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, phase, name, color, title_size=10, sub_size=12)
        if i < len(phases) - 1:
            _arrow(ax, x + w + 0.04, y + h / 2, x + w + gap - 0.04, y + h / 2)
    _save(fig, "architecture_phases.png")


def fig01_system_pipeline() -> None:
    """Clear vertical flow — no crossing arrows, short labels."""
    fig, ax = plt.subplots(figsize=(10.0, 9.0))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 9.0)
    ax.axis("off")
    ax.set_title("Figure 1 — System Pipeline", fontsize=14, fontweight="bold", pad=12)

    # Row 0: data
    _box(ax, 0.5, 7.9, 2.0, 0.75, "VisDrone", "raw data", C_GRAY)
    _arrow(ax, 2.55, 8.25, 3.0, 8.25)
    _box(ax, 3.05, 7.9, 2.0, 0.75, "Convert", "9 classes", C_BASE)
    _arrow(ax, 5.1, 8.25, 5.55, 8.25)
    _box(ax, 5.6, 7.9, 2.0, 0.75, "Dataset", "train / val", C_GRAY)
    _arrow(ax, 7.65, 8.25, 8.1, 8.25)
    _box(ax, 8.15, 7.9, 1.5, 0.75, "EDA", "", C_AN, title_size=11)

    # Phase 1
    _arrow(ax, 6.6, 7.9, 6.6, 7.35)
    _box(ax, 3.3, 6.55, 6.6, 0.75, "Phase 1 — Train YOLOv8-XL", "", C_BASE, title_size=12)
    _arrow(ax, 6.6, 6.55, 6.6, 6.05)
    _box(ax, 3.6, 5.25, 6.0, 0.75, "Checkpoint", "0.5375 mAP@0.5", "#93c5fd", title_size=12, sub_size=10)

    # Branch to SAHI / P2 (no direct arrow to Phase 4 — cleaner)
    _arrow(ax, 5.0, 5.25, 2.5, 4.55)
    _arrow(ax, 8.2, 5.25, 7.5, 4.55)
    _box(ax, 0.6, 3.7, 3.6, 0.8, "Phase 2 — SAHI", "sliced inference", C_SAHI, title_size=11, sub_size=9)
    _box(ax, 5.8, 3.7, 3.6, 0.8, "Phase 3 — P2", "high-res head", C_P2, title_size=11, sub_size=9)

    # Merge into interaction
    _arrow(ax, 2.4, 3.7, 4.4, 3.15)
    _arrow(ax, 7.6, 3.7, 5.6, 3.15)
    _box(ax, 2.8, 2.3, 4.4, 0.8, "Phase 4 — Interaction", "A / B / C / D", C_INT, title_size=11, sub_size=9)

    # Analysis then artifacts
    _arrow(ax, 5.0, 2.3, 5.0, 1.8)
    _box(ax, 2.8, 0.95, 4.4, 0.75, "Phase 5 — Analysis", "", C_AN, title_size=12)
    _arrow(ax, 5.0, 0.95, 5.0, 0.55)
    _box(ax, 2.8, 0.15, 4.4, 0.35, "Public artifacts", "", "#d1fae5", title_size=11)

    _save(fig, "fig01_system_pipeline.png")


def fig02_yolov8_vs_p2() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    fig.suptitle("Figure 2 — YOLOv8-XL vs P2 Head", fontsize=14, fontweight="bold", y=0.98)

    def draw(ax, title, use_p2: bool):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        specs = [
            (7.6, "P2  1/4", use_p2),
            (5.7, "P3  1/8", True),
            (3.8, "P4  1/16", True),
            (1.9, "P5  1/32", True),
        ]
        ax.text(2.0, 9.3, "Backbone", ha="center", fontsize=10, fontweight="bold")
        for y, label, active in specs:
            if active:
                color = C_P2 if label.startswith("P2") else C_BASE
                _box(ax, 0.6, y, 2.8, 1.15, label, "", color, title_size=11)
            else:
                ax.add_patch(
                    FancyBboxPatch(
                        (0.6, y),
                        2.8,
                        1.15,
                        boxstyle="round,pad=0.02,rounding_size=0.1",
                        facecolor="#f1f5f9",
                        edgecolor="#94a3b8",
                        linewidth=1.2,
                        linestyle="--",
                    )
                )
                ax.text(2.0, y + 0.58, label + " (unused)", ha="center", va="center", fontsize=10, color=C_MUTED)

        _box(ax, 4.0, 3.2, 2.0, 4.4, "Neck", "FPN + PAN", C_NECK, title_size=11, sub_size=9)
        for y, _, active in specs:
            if active:
                _arrow(ax, 3.45, y + 0.55, 3.95, 5.4)

        ax.text(8.0, 9.3, "Detect", ha="center", fontsize=10, fontweight="bold")
        for y, label, active in specs:
            short = label.split()[0]
            if active and (use_p2 or not short.startswith("P2")):
                color = C_P2 if short == "P2" else "#93c5fd"
                _box(ax, 6.6, y, 2.8, 1.15, short, "", color, title_size=11)
                _arrow(ax, 6.05, y + 0.55, 6.55, y + 0.55)
            elif short == "P2" and not use_p2:
                ax.add_patch(
                    FancyBboxPatch(
                        (6.6, y),
                        2.8,
                        1.15,
                        boxstyle="round,pad=0.02,rounding_size=0.1",
                        facecolor="#f8fafc",
                        edgecolor="#94a3b8",
                        linewidth=1.2,
                        linestyle="--",
                    )
                )
                ax.text(8.0, y + 0.58, "P2 off", ha="center", va="center", fontsize=10, color=C_MUTED)

        foot = "P2–P5 heads" if use_p2 else "P3–P5 heads"
        ax.text(5.0, 0.55, foot, ha="center", fontsize=10, color="#334155")

    draw(axes[0], "Baseline YOLOv8-XL", use_p2=False)
    draw(axes[1], "P2-YOLOv8-XL", use_p2=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    _save(fig, "fig02_yolov8_vs_p2.png")


def fig03_sahi_inference() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 6.0))
    fig.suptitle("Figure 3 — SAHI Inference", fontsize=14, fontweight="bold")

    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 8.5)
    ax1.axis("off")
    ax1.set_title("How SAHI works", fontsize=12, fontweight="bold", pad=8)

    # Outer frame large enough for 2x2 tiles + padding
    frame_x, frame_y, frame_w, frame_h = 0.55, 5.35, 4.9, 2.75
    ax1.add_patch(Rectangle((frame_x, frame_y), frame_w, frame_h, fill=False, edgecolor=C_EDGE, lw=1.6))
    pad = 0.35
    tile_w = (frame_w - 3 * pad) / 2
    tile_h = (frame_h - 3 * pad) / 2
    ox, oy = frame_x + pad, frame_y + pad
    for i, (col, row) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        tx = ox + col * (tile_w + pad)
        ty = oy + row * (tile_h + pad)
        ax1.add_patch(
            Rectangle((tx, ty), tile_w, tile_h, facecolor=C_SAHI, alpha=0.55, edgecolor="#4338ca", lw=1.1)
        )
        ax1.text(tx + tile_w / 2, ty + tile_h / 2, f"t{i + 1}", ha="center", va="center", fontsize=10)
    ax1.text(3.0, 5.05, "Overlapping tiles", ha="center", fontsize=9, color=C_MUTED)

    _arrow(ax1, 3.0, 5.35, 3.0, 4.7)
    _box(ax1, 1.3, 3.7, 3.4, 0.9, "Per-tile YOLO", "", C_BASE, title_size=11)
    _arrow(ax1, 3.0, 3.7, 3.0, 3.1)
    _box(ax1, 1.3, 2.1, 3.4, 0.9, "Merge + NMS", "", C_INT, title_size=11)
    _arrow(ax1, 3.0, 2.1, 3.0, 1.5)
    _box(ax1, 1.3, 0.5, 3.4, 0.9, "Full-image boxes", "", "#93c5fd", title_size=11)

    ax2.set_title("Parameter grid (16 configs)", fontsize=12, fontweight="bold", pad=8)
    slices = [320, 640, 1024, 1280]
    overlaps = [0.1, 0.2, 0.3, 0.4]
    ax2.set_xlim(-0.6, 4.6)
    ax2.set_ylim(-1.85, 4.6)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([str(o) for o in overlaps], fontsize=10)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels([str(s) for s in slices], fontsize=10)
    ax2.set_xlabel("Overlap", fontsize=11)
    ax2.set_ylabel("Slice size", fontsize=11)
    for i in range(4):
        for j in range(4):
            ax2.add_patch(
                Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, facecolor=C_SAHI, edgecolor=C_EDGE, linewidth=1.1, alpha=0.65)
            )
    ax2.add_patch(Rectangle((-0.4, -1.6), 4.0, 0.42, facecolor=C_BASE, edgecolor=C_EDGE, lw=1.1))
    ax2.text(1.6, -1.39, "Baseline (no SAHI)", ha="center", va="center", fontsize=10, fontweight="bold")
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig03_sahi_inference.png")


def fig04_interaction_matrix() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Figure 4 — Interaction Study", fontsize=14, fontweight="bold", pad=12)

    ax.text(3.35, 5.55, "No SAHI", ha="center", fontsize=12, fontweight="bold")
    ax.text(7.0, 5.55, "SAHI", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.55, 3.9, "YOLOv8-XL", ha="center", va="center", fontsize=11, fontweight="bold", rotation=90)
    ax.text(0.55, 1.7, "P2 model", ha="center", va="center", fontsize=11, fontweight="bold", rotation=90)

    cells = [
        (1.4, 2.95, "A", "Baseline", C_BASE),
        (5.05, 2.95, "B", "+ SAHI", C_SAHI),
        (1.4, 0.75, "C", "P2", C_P2),
        (5.05, 0.75, "D", "P2 + SAHI", C_INT),
    ]
    for x, y, code, name, color in cells:
        _box(ax, x, y, 3.2, 1.85, code, name, color, title_size=16, sub_size=12)

    _save(fig, "fig04_interaction_matrix.png")


def main() -> None:
    _ensure_dirs()
    fig_phases_overview()
    fig01_system_pipeline()
    fig02_yolov8_vs_p2()
    fig03_sahi_inference()
    fig04_interaction_matrix()
    print("Done.")


if __name__ == "__main__":
    main()
