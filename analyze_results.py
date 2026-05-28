#!/usr/bin/env python3
"""
analyze_results.py — Analyze GRNITE pipeline results.

Usage:
    python analyze_results.py --embedding_source genePT
    python analyze_results.py --embedding_source Qwen
    python analyze_results.py --embedding_source genePT --datasets hESC hHep
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LaTeX-style (Computer Modern) fonts without requiring a LaTeX installation
plt.rcParams.update({
    "font.family":           "serif",
    "mathtext.fontset":      "cm",
    "axes.titlesize":        13,
    "axes.labelsize":        11,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       9,
    "figure.titlesize":      14,
})

import numpy as np
import pandas as pd
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_DATASETS = ["hESC", "hHep", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"] # "mDC",
# DEFAULT_DATASETS = ["PBMC-ALL-Human", "PBMC-CTL-Human", "Tumor-ALL", "Tumor-malignant", "Dahlin", ] # "BoneMarrow"
GRN_METRICS = ["AUROC", "AUPRC", "JC"]
LORA_METRIC = "AUROC_vs_Atrue"

COLORS = {
    "teacher":  "#5B8DB8",
    "grnite":   "#E07B39",
    "ablation": "#8AB49A",
    "initial":  "#5B8DB8",
    "lora":     "#E07B39",
    "random":   "#8AB49A",
}

# Teacher methods to drop entirely from the analysis
TEACHER_EXCLUDE = {"celloracle-base"} # , "dazzle-full_filtered"

# Display-name overrides for teacher methods (raw GRN pattern → clean label)
TEACHER_NAME_MAP = {
    "celloracle-whole":    "celloracle",
    "correlation_thresh":  "correlation",
    "dazzle-full_filtered": "dazzle",
    "scenic-network":      "scenic",
}

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_dataset_results(
    dataset: str, embedding_source: str, data_root: Path
) -> tuple:
    path = data_root / dataset / f"{dataset}_pipeline_results_{embedding_source}.xlsx"
    if not path.exists():
        print(f"  [skip] not found: {path}")
        return None, None
    xl = pd.ExcelFile(path)
    grn  = xl.parse("GRN_Eval")
    prep = xl.parse("Preprocessing")
    return grn, prep


def classify_row(method: str) -> str:
    if method.endswith("_grnite_ablation"):
        return "ablation"
    if "_grnite" in method:
        return "grnite"
    return "teacher"


def parse_grn_eval(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Convert GRN_Eval sheet to long-format DataFrame; all metrics scaled to %."""
    df = df.copy()
    df["role"] = df["Method"].apply(classify_row)

    records = []
    for i in range(0, len(df), 3):
        group = df.iloc[i : i + 3]
        if len(group) < 3:
            continue

        by_role = {role: group[group["role"] == role] for role in ("teacher", "grnite", "ablation")}
        if any(len(v) == 0 for v in by_role.values()):
            print(f"  [warn] incomplete group at rows {i}–{i+2} in {dataset}, skipping")
            continue

        teacher_name = by_role["teacher"].iloc[0]["Method"]
        if teacher_name in TEACHER_EXCLUDE:
            continue
        display_name = TEACHER_NAME_MAP.get(teacher_name, teacher_name)
        for role, rows in by_role.items():
            row = rows.iloc[0]
            records.append(
                {
                    "dataset": dataset,
                    "teacher": display_name,
                    "role": role,
                    **{m: row[m] * 100.0 for m in GRN_METRICS},   # convert to %
                }
            )

    return pd.DataFrame(records)


def parse_preprocessing(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    label_map = {"A_emb": "initial", "A_emb_lora": "lora", "A_emb_rnd": "random"}
    df = df.copy()
    df["embedding"] = df["Embedding"].map(label_map).fillna(df["Embedding"])
    df["dataset"] = dataset
    df = df.rename(columns={LORA_METRIC: "AUROC"})
    df["AUROC"] = df["AUROC"] * 100.0   # convert to %
    return df[["dataset", "embedding", "AUROC"]]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _darken(hex_color: str, factor: float = 0.65) -> str:
    """Return a darkened version of a hex color by scaling RGB channels."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(int(r * factor), int(g * factor), int(b * factor))


def fmt(df: pd.DataFrame) -> str:
    return df.round(2).to_string()


def save_csv(df: pd.DataFrame, path: Path, **kwargs):
    df.to_csv(path, **kwargs)
    print(f"  Saved: {path.name}")


def save_fig(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Goal 1: LoRA Analysis ──────────────────────────────────────────────────────

def analyze_lora(prep_frames: list, output_dir: Path, embedding_source: str) -> pd.DataFrame:
    df = pd.concat(prep_frames, ignore_index=True)

    summary = df.pivot(index="dataset", columns="embedding", values="AUROC")
    for col in ("initial", "lora", "random"):
        if col not in summary.columns:
            summary[col] = np.nan
    summary = summary[["initial", "lora", "random"]]
    summary["lora_vs_initial"] = summary["lora"] - summary["initial"]
    summary["lora_vs_random"]  = summary["lora"] - summary["random"]

    print("\n" + "=" * 60)
    print("Goal 1: LoRA Embedding Quality (AUROC vs True GRN, %)")
    print("=" * 60)
    print(fmt(summary))
    print(f"\n  Mean LoRA AUROC        : {summary['lora'].mean():.2f}%")
    print(f"  Mean Initial AUROC     : {summary['initial'].mean():.2f}%")
    print(f"  Mean gain vs initial   : +{summary['lora_vs_initial'].mean():.2f}%")
    print(f"  Mean gain vs random    : +{summary['lora_vs_random'].mean():.2f}%")

    save_csv(summary, output_dir / "lora_auroc_summary.csv")
    _plot_lora_bars(summary, output_dir, embedding_source)

    return summary


def _plot_lora_bars(summary: pd.DataFrame, output_dir: Path, embedding_source: str):
    datasets = summary.index.tolist()
    x = np.arange(len(datasets))
    w = 0.3

    initial_label = f"Initial {embedding_source} (pre-LoRA)"
    lora_label    = f"LoRA-tuned {embedding_source}"
    random_label  = "Random initialization (LoRA-tuned)"

    fig, ax = plt.subplots(figsize=(max(9, len(datasets) * 1.5), 5))
    ax.bar(x - w, summary["initial"], w, label=initial_label, color=COLORS["initial"], edgecolor="white")
    ax.bar(x,     summary["lora"],    w, label=lora_label,    color=COLORS["lora"],    edgecolor="white")
    b_rnd = ax.bar(x + w, summary["random"], w, label=random_label, color=COLORS["random"], edgecolor="white")

    # Annotate above LoRA bar: lora − initial (gain from LoRA training)
    # Annotate above random bar: random − lora (ablation − main; negative = ablation is worse)
    for xi, ds in enumerate(datasets):
        lora_val    = summary.loc[ds, "lora"]
        initial_val = summary.loc[ds, "initial"]
        random_val  = summary.loc[ds, "random"]

        diff_lora = lora_val - initial_val
        sign_lora = "+" if diff_lora >= 0 else ""
        ax.text(
            xi, max(lora_val, 0) + 1.0,
            f"{sign_lora}{diff_lora:.1f}%",
            ha="center", va="bottom", fontsize=8,
            color=_darken(COLORS["lora"]), fontweight="bold",
        )

        diff_rnd = random_val - lora_val
        sign_rnd = "+" if diff_rnd >= 0 else ""
        ax.text(
            xi + w, max(random_val, 0) + 1.0,
            f"{sign_rnd}{diff_rnd:.1f}%",
            ha="center", va="bottom", fontsize=8,
            color=_darken(COLORS["random"]), fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("AUROC vs.\ True GRN (\%)")
    ax.set_title(f"LoRA Embedding Quality --- AUROC per Dataset ({embedding_source})")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)

    # Build legend: bar entries + annotation explanation entries
    import matplotlib.patches as mpatches
    bar_handles, bar_labels = ax.get_legend_handles_labels()
    annot_handles = [
        mpatches.Patch(color=_darken(COLORS["lora"]),
                       label=f"Annotation above LoRA bar: LoRA $-$ initial ({embedding_source})"),
        mpatches.Patch(color=_darken(COLORS["random"]),
                       label="Annotation above random bar: random $-$ LoRA (ablation gap)"),
    ]
    ax.legend(handles=bar_handles + annot_handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    save_fig(fig, output_dir / "lora_auroc_bar.png")


# ── Goal 2: GRNITE vs Teacher ──────────────────────────────────────────────────

def compute_improvements(grn_frames: list) -> pd.DataFrame:
    df = pd.concat(grn_frames, ignore_index=True)

    pivot = df.pivot_table(
        index=["dataset", "teacher"], columns="role", values=GRN_METRICS, aggfunc="first"
    )
    pivot.columns = [f"{metric}_{role}" for metric, role in pivot.columns]
    pivot = pivot.reset_index()

    for m in GRN_METRICS:
        pivot[f"{m}_vs_teacher"]  = pivot[f"{m}_grnite"] - pivot[f"{m}_teacher"]
        pivot[f"{m}_vs_ablation"] = pivot[f"{m}_grnite"] - pivot[f"{m}_ablation"]

    return pivot


def report_and_save_goal2(pivot: pd.DataFrame, grn_frames: list, output_dir: Path):
    df_all = pd.concat(grn_frames, ignore_index=True)

    # ── Baseline teacher table ──
    teachers = df_all[df_all["role"] == "teacher"].copy()
    baseline_auroc = teachers.pivot(index="dataset", columns="teacher", values="AUROC")
    print("\n" + "=" * 60)
    print("Teacher Baselines — AUROC (%)")
    print("=" * 60)
    print(fmt(baseline_auroc))
    save_csv(teachers,       output_dir / "baseline_all.csv",   index=False)
    save_csv(baseline_auroc, output_dir / "baseline_AUROC.csv")

    # ── Improvement tables ──
    vs_teacher_cols  = ["dataset", "teacher"] + [f"{m}_vs_teacher"  for m in GRN_METRICS]
    vs_ablation_cols = ["dataset", "teacher"] + [f"{m}_vs_ablation" for m in GRN_METRICS]

    vs_teacher  = pivot[vs_teacher_cols].copy()
    vs_ablation = pivot[vs_ablation_cols].copy()

    print("\n" + "=" * 60)
    print("GRNITE improvement over teacher — AUROC (pp)")
    print("=" * 60)
    print(fmt(vs_teacher.pivot(index="dataset", columns="teacher", values="AUROC_vs_teacher")))

    print("\n" + "=" * 60)
    print("GRNITE improvement over ablation — AUROC (pp)")
    print("=" * 60)
    print(fmt(vs_ablation.pivot(index="dataset", columns="teacher", values="AUROC_vs_ablation")))

    save_csv(pivot,       output_dir / "improvements_full.csv",        index=False)
    save_csv(vs_teacher,  output_dir / "improvements_vs_teacher.csv",  index=False)
    save_csv(vs_ablation, output_dir / "improvements_vs_ablation.csv", index=False)

    # ── Aggregates ──
    delta_cols = [f"{m}_{c}" for m in GRN_METRICS for c in ("vs_teacher", "vs_ablation")]

    by_dataset = pivot.groupby("dataset")[delta_cols].mean()
    by_teacher = pivot.groupby("teacher")[delta_cols].mean()

    print("\n" + "=" * 60)
    print("Aggregate by Dataset — mean over teachers (pp)")
    print("=" * 60)
    print(fmt(by_dataset))

    print("\n" + "=" * 60)
    print("Aggregate by Teacher — mean over datasets (pp)")
    print("=" * 60)
    print(fmt(by_teacher))

    save_csv(by_dataset, output_dir / "aggregate_by_dataset.csv")
    save_csv(by_teacher, output_dir / "aggregate_by_teacher.csv")

    return by_dataset, by_teacher


# ── Visualizations — Goal 2 ────────────────────────────────────────────────────

def plot_heatmaps(pivot: pd.DataFrame, output_dir: Path):
    for comp, label in [("vs_teacher", "vs.\ Teacher"), ("vs_ablation", "vs.\ Ablation")]:
        fig, axes = plt.subplots(
            1, len(GRN_METRICS),
            figsize=(6 * len(GRN_METRICS), max(4, len(pivot["dataset"].unique()) * 0.7 + 2)),
        )
        if len(GRN_METRICS) == 1:
            axes = [axes]

        for ax, m in zip(axes, GRN_METRICS):
            col  = f"{m}_{comp}"
            heat = pivot.pivot(index="dataset", columns="teacher", values=col)
            vals = heat.values[~np.isnan(heat.values)]
            vmax = max(abs(vals).max(), 0.01) if len(vals) else 1.0
            sns.heatmap(
                heat, ax=ax, annot=True, fmt=".1f",
                center=0, vmin=-vmax, vmax=vmax,
                cmap="coolwarm", linewidths=0.5,
                cbar_kws={"label": f"$\\Delta$ {m} (pp)"},
            )
            ax.set_title(f"$\\Delta$ {m}  (GRNITE {label})")
            ax.set_xlabel("Teacher Method")
            ax.set_ylabel("Dataset" if m == GRN_METRICS[0] else "")
            ax.tick_params(axis="x", rotation=35)

        comp_str = comp.replace("_", " ")
        fig.suptitle(f"GRNITE improvement {comp_str} (percentage points)", y=1.01)
        fig.tight_layout()
        save_fig(fig, output_dir / f"heatmap_{comp}.png")


def _aggregate_bar_panel(ax, df: pd.DataFrame, metric: str, ylabel: bool):
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df[f"{metric}_vs_teacher"],  w,
           label="vs.\ Teacher",  color=COLORS["teacher"],  edgecolor="white")
    ax.bar(x + w / 2, df[f"{metric}_vs_ablation"], w,
           label="vs.\ Ablation", color=COLORS["ablation"], edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    if ylabel:
        ax.set_ylabel(f"$\\Delta$ {metric} (pp)")
    ax.legend()


def plot_aggregate_bars_by_dataset(by_dataset: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(
        1, len(GRN_METRICS),
        figsize=(5 * len(GRN_METRICS), 4.5),
        sharey=False,
    )
    if len(GRN_METRICS) == 1:
        axes = [axes]
    for ax, m in zip(axes, GRN_METRICS):
        _aggregate_bar_panel(ax, by_dataset, m, ylabel=True)
        ax.set_title(f"Avg $\\Delta$ {m} by Dataset")
    fig.suptitle("GRNITE Average Improvement --- by Dataset (mean over teachers)")
    fig.tight_layout()
    save_fig(fig, output_dir / "aggregate_improvements_by_dataset.png")


def plot_aggregate_bars_by_teacher(by_teacher: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(
        1, len(GRN_METRICS),
        figsize=(5 * len(GRN_METRICS), 4.5),
        sharey=False,
    )
    if len(GRN_METRICS) == 1:
        axes = [axes]
    for ax, m in zip(axes, GRN_METRICS):
        _aggregate_bar_panel(ax, by_teacher, m, ylabel=True)
        ax.set_title(f"Avg $\\Delta$ {m} by Teacher Method")
    fig.suptitle("GRNITE Average Improvement --- by Teacher Method (mean over datasets)")
    fig.tight_layout()
    save_fig(fig, output_dir / "aggregate_improvements_by_teacher.png")


def plot_per_metric_bars(pivot: pd.DataFrame, output_dir: Path):
    """One figure per metric × comparison: datasets on x-axis, teachers as bar groups."""
    for comp, label in [("vs_teacher", "vs.\ Teacher"), ("vs_ablation", "vs.\ Ablation")]:
        for m in GRN_METRICS:
            col  = f"{m}_{comp}"
            heat = pivot.pivot(index="dataset", columns="teacher", values=col)
            teachers = heat.columns.tolist()
            datasets = heat.index.tolist()

            x = np.arange(len(datasets))
            n_t = len(teachers)
            w = 0.7 / n_t
            offsets = np.linspace(-(n_t - 1) / 2 * w, (n_t - 1) / 2 * w, n_t)

            cmap = plt.colormaps.get_cmap("tab10")
            fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.4), 5))
            for t_i, teacher in enumerate(teachers):
                vals = heat[teacher].values
                ax.bar(x + offsets[t_i], vals, w, label=teacher,
                       color=cmap(t_i / max(n_t - 1, 1)), edgecolor="white")

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=30, ha="right")
            ax.set_title(f"GRNITE $\\Delta$ {m} {label} --- per Dataset $\\times$ Teacher")
            ax.set_ylabel(f"$\\Delta$ {m} (pp)")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            save_fig(fig, output_dir / f"bar_{m}_{comp}.png")


# ── Markdown Summary ──────────────────────────────────────────────────────────

def write_markdown_summary(
    lora_summary:     pd.DataFrame,
    by_dataset:       pd.DataFrame,
    by_teacher:       pd.DataFrame,
    embedding_source: str,
    output_dir:       Path,
):
    def _table(df: pd.DataFrame) -> str:
        try:
            return df.round(2).to_markdown()
        except Exception:
            return df.round(2).to_string()

    lines = [
        f"# GRNITE Results Summary --- {embedding_source}\n",
        "---\n",
        "## Goal 1: LoRA Embedding Quality (AUROC vs True GRN, %)\n",
        _table(lora_summary),
        "",
        f"- Mean LoRA AUROC: **{lora_summary['lora'].mean():.2f}%**",
        f"- Mean Initial AUROC: {lora_summary['initial'].mean():.2f}%",
        f"- Mean LoRA gain over initial: **+{lora_summary['lora_vs_initial'].mean():.2f} pp**",
        f"- Mean LoRA gain over random: **+{lora_summary['lora_vs_random'].mean():.2f} pp**",
        "",
        "---\n",
        "## Goal 2: GRNITE Improvement over Teacher / Ablation (percentage points)\n",
        "### Average by Dataset (mean over all teacher methods)\n",
        _table(by_dataset),
        "",
        "### Average by Teacher Method (mean over all datasets)\n",
        _table(by_teacher),
        "",
        "---",
        f"*Generated by analyze_results.py --- embedding source: {embedding_source}*",
    ]

    path = output_dir / "summary.md"
    path.write_text("\n".join(str(l) for l in lines))
    print(f"  Saved: summary.md")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze GRNITE pipeline results.")
    parser.add_argument(
        "--embedding_source", default="Qwen", choices=["genePT", "Qwen"],
        help="Embedding source used during the pipeline run.",
    )
    parser.add_argument("--data_root", default="Data_plot/TF500", help="Root directory for dataset folders. TF500, TF1000, GG")
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="Dataset names to include (folder names under data_root).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save outputs (default: results_{data_root_leaf}_{embedding_source}/).",
    )
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    data_root_leaf = data_root.name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"results_{data_root_leaf}_{args.embedding_source}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {output_dir}")
    print(f"Embedding source : {args.embedding_source}")
    print(f"Datasets         : {args.datasets}")

    grn_frames, prep_frames = [], []
    for ds in args.datasets:
        print(f"\nLoading {ds} ...")
        grn, prep = load_dataset_results(ds, args.embedding_source, data_root)
        if grn is None:
            continue
        grn_frames.append(parse_grn_eval(grn, ds))
        prep_frames.append(parse_preprocessing(prep, ds))

    if not grn_frames:
        print("\nNo results files found. Check --data_root and --embedding_source.")
        sys.exit(1)

    print(f"\nLoaded {len(grn_frames)} datasets.\n")

    # Goal 1 — LoRA embeddings
    lora_summary = analyze_lora(prep_frames, output_dir, args.embedding_source)

    # Goal 2 — GRNITE vs teacher
    pivot = compute_improvements(grn_frames)
    by_dataset, by_teacher = report_and_save_goal2(pivot, grn_frames, output_dir)

    plot_heatmaps(pivot, output_dir)
    plot_aggregate_bars_by_dataset(by_dataset, output_dir)
    plot_aggregate_bars_by_teacher(by_teacher, output_dir)
    plot_per_metric_bars(pivot, output_dir)

    write_markdown_summary(lora_summary, by_dataset, by_teacher, args.embedding_source, output_dir)

    print(f"\nDone. All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
