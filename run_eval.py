"""
Scriptable evaluation wrapper around Eval.py PRROC.

Usage:
    python run_eval.py \
        --dataset lora/hESC \
        --methods celloracle-whole,scenic-network,grnboost,\
                  genePT-celloracle-whole_grnite,genePT-scenic-network_grnite,genePT-grnboost_grnite,\
                  random-celloracle-whole_grnite_ablation,random-scenic-network_grnite_ablation,random-grnboost_grnite_ablation \
        --output results_hESC.xlsx \
        --preprocessing_metrics_text Data/lora/hESC/preprocessing_metrics_genePT.json \
        --preprocessing_metrics_random Data/lora/hESC/preprocessing_metrics_random.json
"""

import argparse
import json
import os

import pandas as pd

from Eval import PRROC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch PRROC evaluation.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset path under Data/ (e.g. lora/hESC)")
    parser.add_argument("--embedding_source", type=str, default=None,
                        help="Embedding source used (genePT, Qwen, random). Selects the matching ref_present file.")
    parser.add_argument("--methods", type=str, required=True,
                        help="Comma-separated method suffixes to evaluate.")
    parser.add_argument("--output", type=str, default=None,
                        help="Excel output path. Defaults to Data/<dataset>/eval_results.xlsx")
    parser.add_argument("--preprocessing_metrics_text", type=str, default=None,
                        help="Path to preprocessing_metrics_{genePT|Qwen}.json for the primary embedding source.")
    parser.add_argument("--preprocessing_metrics_random", type=str, default=None,
                        help="Path to preprocessing_metrics_random.json.")
    return parser.parse_args()


def load_preprocessing_table(text_path: str | None, random_path: str | None) -> pd.DataFrame | None:
    rows = []

    def _add(path, label):
        if path is None or not os.path.exists(path):
            return
        with open(path) as f:
            m = json.load(f)
        emb_before = m.get("A_emb", {})
        emb_after = m.get("A_emb_lora") or m.get("A_emb_rnd") or {}
        after_label = "A_emb_lora" if "A_emb_lora" in m else "A_emb_rnd"
        for name, d in [("A_emb", emb_before), (after_label, emb_after)]:
            rows.append({
                "Embedding": name,
                "AUROC_vs_Atrue": round(d.get("auroc", float("nan")), 4),
                "AUPRC_vs_Atrue": round(d.get("ap", float("nan")), 4),
                "F1": round(d.get("f1", float("nan")), 4),
                "Precision": round(d.get("precision", float("nan")), 4),
                "Recall": round(d.get("recall", float("nan")), 4),
                "BalAcc": round(d.get("balanced_acc", float("nan")), 4),
            })

    _add(text_path, "text")
    _add(random_path, "random")

    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["Embedding"])
    return df


def main() -> None:
    args = parse_args()

    data_root = f"Data/{args.dataset}"
    print(data_root)
    data_name = args.dataset.rstrip("/").split("/")[-1]

    # Resolve ref_present: prefer source-specific file, fall back to untagged legacy file.
    if args.embedding_source:
        ref_path = os.path.join(data_root, f"{data_name}-ref_present_{args.embedding_source}.csv")
    else:
        ref_path = os.path.join(data_root, f"{data_name}-ref_present.csv")

    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference network not found: {ref_path}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    output_path = args.output or os.path.join(data_root, "eval_results.xlsx")

    rows = []
    for method in methods:
        pred_path = os.path.join(data_root, f"{data_name}-{method}.csv")
        print(f"\n--- {method} ---")
        if not os.path.exists(pred_path):
            print(f"  MISSING: {pred_path}")
            rows.append({"Method": method, "AUROC": None, "AUPRC": None,
                         "Precision": None, "Recall": None, "F1": None, "JC": None,
                         "Pred_Edges": None})
            continue
        try:
            cm, precision, recall, f1, AUPRC, AUROC, ball_acc, jc, possible_edges, true_edges, pred_edges = PRROC(
                ref_path=ref_path,
                input_path=pred_path,
                output_path=data_root + "/",
                method=method,
                directed=True,
                selfEdges=False,
                plotFlag=False,
                restrict=False,
            )
            print(f"  AUROC={AUROC:.4f}  AUPRC={AUPRC:.4f}  F1={f1:.4f}  edges={pred_edges}")
            rows.append({
                "Method": method,
                "AUROC": round(AUROC, 4),
                "AUPRC": round(AUPRC, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
                "JC": round(jc, 4),
                "Pred_Edges": int(pred_edges),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"Method": method, "AUROC": None, "AUPRC": None,
                         "Precision": None, "Recall": None, "F1": None, "JC": None,
                         "Pred_Edges": None})

    df_results = pd.DataFrame(rows)
    print("\n===== GRN Evaluation Results =====")
    print(df_results.to_string(index=False))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Try loading preprocessing metrics
    text_json = args.preprocessing_metrics_text
    if text_json is None:
        for suffix in ["genePT", "Qwen"]:
            candidate = os.path.join(data_root, f"preprocessing_metrics_{suffix}.json")
            if os.path.exists(candidate):
                text_json = candidate
                break
    random_json = args.preprocessing_metrics_random
    if random_json is None:
        candidate = os.path.join(data_root, "preprocessing_metrics_random.json")
        if os.path.exists(candidate):
            random_json = candidate

    df_preprocess = load_preprocessing_table(text_json, random_json)
    if df_preprocess is not None:
        print("\n===== Preprocessing Ablation =====")
        print(df_preprocess.to_string(index=False))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="GRN_Eval", index=False)
        if df_preprocess is not None:
            df_preprocess.to_excel(writer, sheet_name="Preprocessing", index=False)

    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
