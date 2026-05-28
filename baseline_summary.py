#!/usr/bin/env python3
"""
baseline_summary.py — Average baseline method performance across dataset groups.
Produces printed tables and CSV files; no plots.

Usage:
    python baseline_summary.py
    python baseline_summary.py --embedding_source Qwen
    python baseline_summary.py --output_dir baseline_tables/
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_ROOT = Path("Data")
GRN_METRICS = ["AUROC", "AUPRC", "JC"]

# Embedding sources tried in order; first file found for each dataset is used.
# Teacher rows are identical across embedding sources, so one file suffices.
PREFERRED_SOURCES = ["Qwen", "genePT"]

DATASET_GROUPS = {
    "TF500":  ["hESC", "hHep", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"],
    "TF1000": ["hESC", "hHep", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"],
    "GG":     ["PBMC-ALL-Human", "PBMC-CTL-Human", "Tumor-ALL",
               "Tumor-malignant", "Dahlin", "BoneMarrow"],
}

# Which groups each method is averaged over
METHOD_SCOPE = {
    "CellOracle":  ["TF500", "TF1000", "GG"],
    "GRNBoost":    ["TF500", "TF1000", "GG"],
    "SCENIC":      ["TF500", "TF1000", "GG"],
    "PORTIA":      ["TF500", "TF1000", "GG"],
    "Correlation": ["TF500", "TF1000", "GG"],
    "DAZZLE":      ["TF500", "TF1000"],
}

# Raw method name in GRN_Eval sheet → display name
METHOD_NAME_MAP = {
    "celloracle-whole":      "CellOracle",
    "grnboost":              "GRNBoost",
    "scenic-network":        "SCENIC",
    "portia":                "PORTIA",
    "dazzle-full_filtered":  "DAZZLE",
    "correlation_thresh":    "Correlation",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def find_results_file(group: str, dataset: str) -> Path | None:
    for src in PREFERRED_SOURCES:
        p = DATA_ROOT / group / dataset / f"{dataset}_pipeline_results_{src}.xlsx"
        if p.exists():
            return p
    return None


def classify_row(method: str) -> str:
    if method.endswith("_grnite_ablation"):
        return "ablation"
    if "_grnite" in method:
        return "grnite"
    return "teacher"


def load_teacher_rows(path: Path, group: str, dataset: str) -> list[dict]:
    """Return one record per teacher method found in the file."""
    df = pd.ExcelFile(path).parse("GRN_Eval")
    df["role"] = df["Method"].apply(classify_row)
    teachers = df[df["role"] == "teacher"].copy()

    records = []
    for _, row in teachers.iterrows():
        raw_name = row["Method"]
        display  = METHOD_NAME_MAP.get(raw_name)
        if display is None:
            continue  # skip unmapped methods (e.g. celloracle-base)
        records.append({
            "method":  display,
            "group":   group,
            "dataset": dataset,
            **{m: row[m] * 100.0 for m in GRN_METRICS},  # convert to %
        })
    return records


# ── Aggregation ────────────────────────────────────────────────────────────────

def collect_all_records() -> pd.DataFrame:
    records = []
    for group, datasets in DATASET_GROUPS.items():
        for ds in datasets:
            path = find_results_file(group, ds)
            if path is None:
                print(f"  [skip] not found: {group}/{ds}")
                continue
            records.extend(load_teacher_rows(path, group, ds))
    return pd.DataFrame(records)


def build_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns a dict of metric → summary DataFrame.
    Rows = methods, columns = groups + "Overall".
    """
    summaries = {}
    for metric in GRN_METRICS:
        rows = {}
        for method, scope in METHOD_SCOPE.items():
            row = {}
            all_vals = []
            for group in ["TF500", "TF1000", "GG"]:
                subset = df[(df["method"] == method) & (df["group"] == group)][metric]
                if group in scope and len(subset) > 0:
                    row[group] = f"{subset.mean():.2f} ± {subset.std():.2f}"
                    all_vals.extend(subset.tolist())
                else:
                    row[group] = "—"
            row["Overall"] = f"{np.mean(all_vals):.2f} ± {np.std(all_vals):.2f}" if all_vals else "—"
            rows[method] = row
        summaries[metric] = pd.DataFrame(rows).T
        summaries[metric].index.name = "Method"
    return summaries


# ── Output ─────────────────────────────────────────────────────────────────────

def print_and_save(summaries: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print each metric table
    for metric, tbl in summaries.items():
        print(f"\n{'=' * 60}")
        print(f"  {metric}  (mean ± std across datasets, %)")
        print("=" * 60)
        print(tbl.to_string())

    # Save individual CSVs
    for metric, tbl in summaries.items():
        path = output_dir / f"baseline_{metric}.csv"
        tbl.to_csv(path)
        print(f"\n  Saved: {path}")

    # Save combined Excel with one sheet per metric
    excel_path = output_dir / "baseline_summary.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        for metric, tbl in summaries.items():
            tbl.to_excel(writer, sheet_name=metric)
    print(f"  Saved: {excel_path}")

    # Also print a compact combined view (Overall column only)
    print(f"\n{'=' * 60}")
    print("  Combined — Overall average (mean ± std, %)")
    print("=" * 60)
    combined = pd.concat(
        {m: summaries[m]["Overall"] for m in GRN_METRICS}, axis=1
    )
    combined.columns = GRN_METRICS
    combined.index.name = "Method"
    print(combined.to_string())
    combined.to_csv(output_dir / "baseline_overall.csv")
    print(f"  Saved: baseline_overall.csv")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline method summary across dataset groups.")
    parser.add_argument("--data_root", default="Data", help="Root data directory.")
    parser.add_argument("--output_dir", default="baseline_tables", help="Where to save CSV/Excel output.")
    args = parser.parse_args()

    global DATA_ROOT
    DATA_ROOT = Path(args.data_root)

    print("Collecting teacher rows from results files...")
    df = collect_all_records()
    if df.empty:
        print("No data found. Check DATA_ROOT and that results files exist.")
        sys.exit(1)

    print(f"Loaded {len(df)} dataset × method records from "
          f"{df[['group','dataset']].drop_duplicates().shape[0]} files.\n")

    summaries = build_summary(df)
    print_and_save(summaries, Path(args.output_dir))
    print("\nDone.")


if __name__ == "__main__":
    main()
