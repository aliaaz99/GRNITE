import argparse
import os
import pickle
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, roc_auc_score

from utils import csv2A, get_present2, load_edge_set


HUMAN_DATASETS = {
    "PBMC-ALL-Human",
    "PBMC_CTL-Human",
    "Tumor-ALL",
    "Tumor-malignant",
    "hESC",
    "hHep",
    "PBMC",
    "HFD",
    "classicalmonocytes",
    "myeloidDC",
    "naiveBcells",
    "naiveCD4Tcells",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA contrastive refinement for gene embeddings.")

    parser.add_argument("--data-path", type=str, default="lora/hESC", help="Relative dataset path under Data/.")
    parser.add_argument("--embedding-source", type=str, default="genePT", choices=["genePT", "Qwen", "random"])
    parser.add_argument(
        "--gene-source",
        type=str,
        default=None,
        choices=["genePT", "Qwen"],
        help="Which embedding vocabulary to use for determining present genes. "
             "Required when --embedding-source random; ignored otherwise.",
    )
    parser.add_argument(
        "--input-feature-case",
        type=str,
        default="text_expr",
        choices=["random_only", "text_only", "expr_only", "text_expr"],
        help="How to build X_sample_emb fed into train_lora.",
    )
    parser.add_argument("--random-embedding-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0, help="Global random seed for reproducibility.")

    parser.add_argument("--n-low", type=int, default=128)
    parser.add_argument("--neg-multiplier", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=float, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--base-negs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)

    parser.add_argument("--expr-svd-components", type=int, default=64)
    parser.add_argument("--expr-fusion-weight", type=float, default=1.0)
    parser.add_argument("--expr-knn-k", type=int, default=10)
    parser.add_argument("--expr-prior-weight", type=float, default=0.5)

    parser.add_argument("--subsample", type=int, default=None, help="Optional 2000-cell block index.")
    parser.add_argument("--save-embedding", action="store_true", help="Save learned low-dim embedding as .npy.")
    parser.add_argument(
        "--embedding-out-path",
        type=str,
        default=None,
        help="Optional explicit path for saving X_sample_lora_low (.npy). If set, embedding is saved regardless of --save-embedding.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to Data/<data-path>/plots.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting.")

    return parser.parse_args()


def infer_species(data_name: str) -> str:
    return "human" if data_name in HUMAN_DATASETS else "mouse"


def build_edge_index_from_adj(A):
    if isinstance(A, torch.Tensor):
        A_np = A.detach().cpu().numpy()
    else:
        A_np = np.asarray(A)

    src, dst = np.where(A_np == 1)
    keep = src != dst
    return torch.as_tensor(src[keep], dtype=torch.long), torch.as_tensor(dst[keep], dtype=torch.long)


def sample_negatives_for_anchors(anchors, A_train_np: np.ndarray, num_negs: int, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    anchors_np = anchors.detach().cpu().numpy()
    n = A_train_np.shape[0]
    negs = np.empty((len(anchors_np), num_negs), dtype=np.int64)

    all_idx = np.arange(n)
    for row, i in enumerate(anchors_np):
        candidates = np.where((A_train_np[i] == 0) & (all_idx != i))[0]
        if len(candidates) == 0:
            candidates = np.setdiff1d(all_idx, np.array([i]))
        replace = len(candidates) < num_negs
        negs[row] = rng.choice(candidates, size=num_negs, replace=replace)

    return torch.as_tensor(negs, dtype=torch.long, device=anchors.device)


class LoRAProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False)
        nn.init.kaiming_uniform_(self.base.weight, a=np.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Linear(d_in, self.r, bias=False)
        self.B = nn.Linear(self.r, d_out, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.dropout(self.A(x))) * self.scaling


def row_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def cosine_sim_matrix(Z: np.ndarray) -> np.ndarray:
    return Z @ Z.T


def choose_threshold_by_target_edgecount(S: np.ndarray, target_edges: int):
    n = S.shape[0]
    S2 = np.asarray(S).copy()
    np.fill_diagonal(S2, -np.inf)

    flat = S2.reshape(-1)
    flat = flat[np.isfinite(flat)]

    target_edges = int(np.clip(target_edges, 0, flat.size))
    if target_edges == 0:
        return np.inf, np.zeros((n, n), dtype=np.int8)

    kth = target_edges - 1
    thr = -np.partition(-flat, kth)[kth]

    A_pred = (S2 >= float(thr)).astype(np.int8)
    np.fill_diagonal(A_pred, 0)
    return float(thr), A_pred


def choose_threshold_by_mean_kstd(S: np.ndarray, k_std: float = 0.0):
    S2 = np.asarray(S).copy()
    np.fill_diagonal(S2, np.nan)

    vals = S2[~np.isnan(S2)]
    thr = float(vals.mean() + k_std * vals.std())

    A_pred = (np.asarray(S) >= thr).astype(np.int8)
    np.fill_diagonal(A_pred, 0)
    return thr, A_pred


def metrics_against_true(A_pred: np.ndarray, A_true: np.ndarray):
    A_p = np.asarray(A_pred).astype(np.int8)
    A_t = np.asarray(A_true).astype(np.int8)

    if A_p.shape != A_t.shape:
        raise ValueError(f"Shape mismatch: pred={A_p.shape}, true={A_t.shape}")

    n = A_t.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y_pred = A_p[mask].reshape(-1)
    y_true = A_t[mask].reshape(-1)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    tnr = tn / (tn + fp + 1e-12)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": 100.0 * float(precision),
        "recall": 100.0 * float(recall),
        "f1": 100.0 * float(f1),
        "balanced_acc": 50.0 * float(recall + tnr),
    }


def print_metrics(name: str, m: dict):
    print(
        f"{name:12s} | F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
        f"BalAcc={m['balanced_acc']:.4f} | TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}"
    )


# Maps gene_source to the emb_name_column used for bio_adj cache and present_genes files.
_GENE_SOURCE_EMB_NAMES = {
    "genePT": "GenePT-Embedding-8B",
    "Qwen":   "Qwen3-Embedding-8B",
}


def load_embeddings(species: str, embedding_source: str, gene_source: str):
    """Load gene embeddings and vocabulary.

    gene_source controls WHICH vocabulary (gene set) is loaded.
    embedding_source == "random" means the loaded embeddings will be replaced by
    random noise inside build_fused_embedding — the vocab still comes from gene_source.
    Output emb_name_column is "random" when embedding_source=="random" so that all
    saved files (present_genes, lora npy, metrics json) get a "_random" suffix and
    never overwrite the primary-source files.
    """
    if gene_source == "genePT":
        emb_name_column_vocab = "GenePT-Embedding-8B"
        emb_path = "Gene_embeddings/GenePT_gene_embedding_ada_text.pickle"
        with open(emb_path, "rb") as f:
            data_emb = pickle.load(f)
        gene_names_all = list(data_emb.keys())
    elif gene_source == "Qwen":
        emb_name_column_vocab = "Qwen3-Embedding-8B"
        emb_path = f"Gene_embeddings/{species}_embeds-{emb_name_column_vocab}.h5"
        data_h5 = pd.read_hdf(emb_path)
        gene_names_all = data_h5["Symbol"].values
        data_emb = {
            row["Symbol"]: np.array(row[emb_name_column_vocab])
            for _, row in data_h5.iterrows()
        }
    else:
        raise ValueError(f"Unknown gene_source: {gene_source!r}. Choose from genePT, Qwen.")

    # Files for the random run get "_random" suffix; primary-source files keep their own suffix.
    emb_name_column = "random" if embedding_source == "random" else emb_name_column_vocab

    print("Number of all genes for embeddings:", len(gene_names_all))
    return emb_name_column, data_emb, gene_names_all


def load_expression(data_path: str, subsample: Optional[int]):
    expr_path = f"Data/{data_path}/ExpressionData.csv"
    if subsample is not None:
        cell_range = [0] + list(range(2000 * (subsample - 1), 2000 * subsample))
        data_sample = pd.read_csv(expr_path, header=0, index_col=0, usecols=cell_range).T
    else:
        data_sample = pd.read_csv(expr_path, header=0, index_col=0).T

    data_sample = data_sample.transform(lambda x: np.log(x + 1))
    gene_names_sample = [g.upper() for g in data_sample.columns]
    data_sample.columns = gene_names_sample

    print("Reading data completed!")
    print(f"Number of cells: {data_sample.shape[0]}, Number of genes: {data_sample.shape[1]}")
    return data_sample, gene_names_sample


def build_fused_embedding(
    data_sample: pd.DataFrame,
    gene_names_sample,
    gene_names_all,
    data_emb,
    embedding_source: str,
    emb_name_column: str,
    random_embedding_seed: int,
    expr_svd_components: int,
    expr_fusion_weight: float,
    data_path: str,
    input_feature_case: str,
):
    present_genes, X_sample_emb_text = get_present2(gene_names_sample, gene_names_all, data_emb)
    n = len(present_genes)
    print(f"Number of present genes: {n}")

    X_text = X_sample_emb_text.astype(np.float32)
    rng = np.random.default_rng(random_embedding_seed)
    X_random = rng.standard_normal(size=X_text.shape).astype(np.float32)

    # Keep backwards-compatible behavior: embedding_source=random means random replaces the text-like branch.
    if embedding_source == "random":
        X_base = X_random
        print(f"Using random base embedding for source='{embedding_source}' with shape: {X_base.shape}")
    else:
        X_base = X_text
        print(f"Using loaded base embedding for source='{embedding_source}' with shape: {X_base.shape}")

    non_present_genes = set(gene_names_sample) - set(present_genes)
    with open(f"Data/{data_path}/non_present_genes_{emb_name_column}.txt", "w") as f:
        f.write("Genes present in expression data but not in text embeddings:\n")
        for gene in non_present_genes:
            f.write(gene + "\n")

    with open(f"Data/{data_path}/present_genes_{emb_name_column}.txt", "w") as f:
        f.write("Genes present in both expression data and text embeddings:\n")
        for gene in present_genes:
            f.write(gene + "\n")
    print(f"Saved present_genes to: Data/{data_path}/present_genes_{emb_name_column}.txt ({len(present_genes)} genes)")

    X_sample = data_sample[present_genes].T.values.astype(np.float32)
    print("X_sample shape:", X_sample.shape)

    expr_gene = X_sample.copy()
    expr_gene = (expr_gene - expr_gene.mean(axis=1, keepdims=True)) / (expr_gene.std(axis=1, keepdims=True) + 1e-6)
    expr_gene = np.nan_to_num(expr_gene, nan=0.0, posinf=0.0, neginf=0.0)

    max_components = max(1, min(expr_gene.shape[0] - 1, expr_gene.shape[1] - 1))
    n_components = min(int(expr_svd_components), max_components)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    X_expr_low = svd.fit_transform(expr_gene).astype(np.float32)
    X_expr_low = row_normalize(X_expr_low)
    print(f"Expression SVD feature shape: {X_expr_low.shape}")

    if input_feature_case == "random_only":
        X_input = X_random
        case_desc = "random only"
    elif input_feature_case == "text_only":
        X_input = X_base
        case_desc = "base/text only"
    elif input_feature_case == "expr_only":
        X_input = X_expr_low.astype(np.float32)
        case_desc = "expression only"
    elif input_feature_case == "text_expr":
        X_input = np.concatenate([X_base.astype(np.float32), expr_fusion_weight * X_expr_low], axis=1)
        case_desc = "base/text + expression"
    else:
        raise ValueError(f"Unknown input_feature_case: {input_feature_case}")

    print(f"Training input case: {input_feature_case} ({case_desc})")
    print(f"X_sample_emb shape used for train_lora: {X_input.shape}")

    return present_genes, X_input, X_expr_low


def load_or_build_bio_prior(data_path: str, emb_name_column: str, species: str, present_genes):
    t0 = time.time()
    data_path_bio_adj = f"Data/{data_path}/bio_adj-{emb_name_column}.pickle"
    data_path_bio_edges = f"Data/{data_path}/bio_grn-{emb_name_column}.csv"

    if os.path.exists(data_path_bio_adj):
        print("Loading celloracle adjacency matrix as bio prior from pickle file...")
        with open(data_path_bio_adj, "rb") as f:
            A_bio = pickle.load(f)
            A_bio = A_bio[: len(present_genes), : len(present_genes)]
        _ = load_edge_set(data_path_bio_edges)
        print("Celloracle adjacency matrix loaded successfully!")
    else:
        print("Celloracle adjacency matrix not found. Computing from edge list...")
        base_path = (
            "Gene_embeddings/celloracle_baseGRN.csv"
            if species == "human"
            else "Gene_embeddings/celloracle_mouse_baseGRN.csv"
        )
        bio_grn = load_edge_set(base_path)
        print("Number of edges in base celloracle:", len(bio_grn))

        present_set = set(present_genes)
        bio_grn = {(g1, g2) for g1, g2 in bio_grn if g1 in present_set and g2 in present_set}
        print("Number of edges in base celloracle GRN with present genes:", len(bio_grn))

        pd.DataFrame(list(bio_grn), columns=["Gene1", "Gene2"]).to_csv(data_path_bio_edges, index=False)
        A_bio = csv2A(bio_grn, present_genes)

        with open(data_path_bio_adj, "wb") as f:
            pickle.dump(A_bio, f)

    elapsed = time.time() - t0
    print(f"A_bio built/loaded in {elapsed:.2f}s")

    A_bio = np.asarray(A_bio).astype(np.int8)
    np.fill_diagonal(A_bio, 0)
    print("A_bio shape:", A_bio.shape)
    return A_bio


def build_training_adjacency(A_bio: np.ndarray, X_expr_low: np.ndarray, expr_knn_k: int, expr_prior_weight: float):
    n = A_bio.shape[0]

    S_expr = X_expr_low @ X_expr_low.T
    np.fill_diagonal(S_expr, -np.inf)

    k_expr = max(1, min(int(expr_knn_k), n - 1))
    idx = np.argpartition(-S_expr, kth=k_expr - 1, axis=1)[:, :k_expr]

    A_expr = np.zeros_like(A_bio, dtype=np.int8)
    rows = np.arange(n)[:, None]
    A_expr[rows, idx] = 1
    np.fill_diagonal(A_expr, 0)

    mix = (1.0 - float(expr_prior_weight)) * A_bio + float(expr_prior_weight) * A_expr
    A_train = (mix >= 0.5).astype(np.int8)
    np.fill_diagonal(A_train, 0)

    print(
        f"A_train shape: {A_train.shape} | bio edges={int(A_bio.sum())} | "
        f"expr edges={int(A_expr.sum())} | train edges={int(A_train.sum())}"
    )
    return A_train


def train_lora(
    X_sample_emb: np.ndarray,
    A_train: np.ndarray,
    n_low: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lr: float,
    tau: float,
    num_epoch: int,
    base_negs: int,
    neg_multiplier: float,
    batch_size: int,
    device: torch.device,
):
    X_emb_t = torch.as_tensor(X_sample_emb, dtype=torch.float32, device=device)
    d_emb = X_emb_t.shape[1]

    src_pos, dst_pos = build_edge_index_from_adj(A_train)
    if src_pos.numel() == 0:
        raise ValueError("A_train has no positive edges. Cannot train contrastive LoRA.")

    pairs = torch.stack([src_pos, dst_pos], dim=1).to(device)
    num_negs = max(1, int(base_negs * float(neg_multiplier)))

    proj = LoRAProjector(d_in=d_emb, d_out=n_low, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout).to(device)
    opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    A_train_np = np.asarray(A_train)
    rng = np.random.default_rng(0)

    proj.train()
    loss_over_epochs = []

    for epoch in range(1, num_epoch + 1):
        perm = torch.randperm(pairs.shape[0], device=device)
        pairs_shuf = pairs[perm]

        total_loss = 0.0
        total_batches = 0

        for start in range(0, pairs_shuf.shape[0], batch_size):
            batch = pairs_shuf[start : start + batch_size]
            anchors = batch[:, 0]
            pos = batch[:, 1]

            neg = sample_negatives_for_anchors(anchors, A_train_np, num_negs=num_negs, rng=rng)

            z = F.normalize(proj(X_emb_t), dim=1)
            z_a = z[anchors]
            z_p = z[pos]
            z_n = z[neg]

            pos_logits = (z_a * z_p).sum(dim=1, keepdim=True) / tau
            neg_logits = (z_a.unsqueeze(1) * z_n).sum(dim=2) / tau
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

            loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item()
            total_batches += 1

        epoch_loss = total_loss / max(1, total_batches)
        loss_over_epochs.append(epoch_loss)
        if epoch % 20 == 0 or epoch == 1:
            print(f"[LoRA-Contrastive] epoch {epoch:4d}/{num_epoch} | loss={epoch_loss:.4f}")

    proj.eval()
    with torch.no_grad():
        X_lora = F.normalize(proj(X_emb_t), dim=1).detach().cpu().numpy()

    return X_lora, loss_over_epochs


def evaluate_balanced_pos_neg(A_ref: np.ndarray, X_before: np.ndarray, X_after: np.ndarray, title: str):
    A_np = np.asarray(A_ref).astype(np.int8)
    n = A_np.shape[0]

    pos_i, pos_j = np.where(A_np == 1)
    keep = pos_i != pos_j
    pos_i, pos_j = pos_i[keep], pos_j[keep]

    if len(pos_i) == 0:
        raise ValueError(f"No positive edges in {title}; cannot evaluate.")

    rng = np.random.default_rng(123)
    all_idx = np.arange(n)
    neg_i = pos_i.copy()
    neg_j = np.empty_like(pos_j)

    for idx, i in enumerate(neg_i):
        candidates = np.where((A_np[i] == 0) & (all_idx != i))[0]
        if len(candidates) == 0:
            candidates = np.setdiff1d(all_idx, np.array([i]))
        neg_j[idx] = rng.choice(candidates)

    Z_after = row_normalize(X_after)
    pos_scores = np.sum(Z_after[pos_i] * Z_after[pos_j], axis=1)
    neg_scores = np.sum(Z_after[neg_i] * Z_after[neg_j], axis=1)

    y_true = np.concatenate([np.ones_like(pos_scores, dtype=np.int8), np.zeros_like(neg_scores, dtype=np.int8)])
    y_score = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    print(f"\n=== Balanced Pos/Neg Evaluation vs {title} ===")
    print(f"Mean cosine sim (pos): {pos_scores.mean():.4f} +/- {pos_scores.std():.4f}")
    print(f"Mean cosine sim (neg): {neg_scores.mean():.4f} +/- {neg_scores.std():.4f}")
    print(f"ROC-AUC: {auc:.4f} | AP: {ap:.4f}")

    Z_before = row_normalize(X_before)
    pos_scores0 = np.sum(Z_before[pos_i] * Z_before[pos_j], axis=1)
    neg_scores0 = np.sum(Z_before[neg_i] * Z_before[neg_j], axis=1)
    y_score0 = np.concatenate([pos_scores0, neg_scores0])

    auc0 = roc_auc_score(y_true, y_score0)
    ap0 = average_precision_score(y_true, y_score0)

    print("[Baseline: original embedding]")
    print(f"Mean cosine sim (pos): {pos_scores0.mean():.4f} +/- {pos_scores0.std():.4f}")
    print(f"Mean cosine sim (neg): {neg_scores0.mean():.4f} +/- {neg_scores0.std():.4f}")
    print(f"ROC-AUC: {auc0:.4f} | AP: {ap0:.4f}")

    metrics_before = {
        "auroc": float(auc0), "ap": float(ap0),
        "cos_sim_pos": float(pos_scores0.mean()), "cos_sim_neg": float(neg_scores0.mean()),
    }
    metrics_after = {
        "auroc": float(auc), "ap": float(ap),
        "cos_sim_pos": float(pos_scores.mean()), "cos_sim_neg": float(neg_scores.mean()),
    }
    return pos_i, pos_j, neg_i, neg_j, metrics_before, metrics_after


def plot_similarity_violin(X_before, X_after, pos_i, pos_j, neg_i, neg_j, save_path: Optional[str] = None):
    before = row_normalize(X_before)
    after = row_normalize(X_after)

    pos_before = np.sum(before[pos_i] * before[pos_j], axis=1)
    neg_before = np.sum(before[neg_i] * before[neg_j], axis=1)
    pos_after = np.sum(after[pos_i] * after[pos_j], axis=1)
    neg_after = np.sum(after[neg_i] * after[neg_j], axis=1)

    rng = np.random.default_rng(0)

    def maybe_downsample(arr, max_points=200000):
        if len(arr) <= max_points:
            return arr
        return arr[rng.choice(len(arr), size=max_points, replace=False)]

    data = [
        maybe_downsample(pos_before),
        maybe_downsample(neg_before),
        maybe_downsample(pos_after),
        maybe_downsample(neg_after),
    ]

    plt.figure(figsize=(8, 3))
    plt.violinplot(data, positions=[1, 2, 4, 5], showmeans=True, showmedians=True, showextrema=False)
    plt.axvline(3, linestyle="--", linewidth=1)
    plt.xticks([1, 2, 4, 5], ["Before +", "Before -", "After +", "After -"])
    plt.ylabel("Cosine similarity")
    plt.title("Cosine similarity distributions before vs after contrastive LoRA")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.ylim(-1.05, 1.05)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("Saved plot:", save_path)
    plt.show()


def load_reference_adjacency(data_path: str, data_name: str, present_genes, embedding_source: str = ""):
    ref_path = f"Data/{data_path}/refNetwork.csv"
    if not os.path.exists(ref_path):
        print(f"Reference network not found: {ref_path}")
        return None

    ref_grn_un = load_edge_set(ref_path, undirected=False)
    present_genes_upper = set(g.upper() for g in present_genes)
    ref_grn_present = [(g1, g2) for g1, g2 in ref_grn_un if g1 in present_genes_upper and g2 in present_genes_upper]

    suffix = f"_{embedding_source}" if embedding_source else ""
    ref_out_path = f"Data/{data_path}/{data_name}-ref_present{suffix}.csv"
    pd.DataFrame(ref_grn_present, columns=["Gene1", "Gene2"]).to_csv(ref_out_path, index=False)
    print("Reference GRN with present genes saved!")

    A_true = csv2A(ref_grn_present, present_genes)
    A_true = np.asarray(A_true).astype(np.int8)
    np.fill_diagonal(A_true, 0)
    print("A_true from reference GRN built successfully!", A_true.shape)
    return A_true


def save_binary_adjacency_as_edge_csv(A_bin: np.ndarray, present_genes, out_path: str) -> None:
    A = np.asarray(A_bin).astype(np.int8)
    n = A.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    keep = A[rows, cols] == 1
    rows, cols = rows[keep], cols[keep]

    edges_df = pd.DataFrame({
        "Gene1": [present_genes[i] for i in rows],
        "Gene2": [present_genes[j] for j in cols],
    })

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    edges_df.to_csv(out_path, index=False)
    print("Saved A_emb_lora_bin edges:", out_path)


def evaluate_binary_and_threshold_free(A_true, A_bio, A_train, X_before, X_after, present_genes, data_path: str, data_name: str, emb_name_column: str = ""):
    Z0 = row_normalize(X_before)
    Z1 = row_normalize(X_after)
    S0 = cosine_sim_matrix(Z0)
    S1 = cosine_sim_matrix(Z1)

    A_bio_np = np.asarray(A_bio).astype(np.int8)
    np.fill_diagonal(A_bio_np, 0)

    target_edges = int(np.asarray(A_train).sum())
    thr0, A_emb_bin = choose_threshold_by_target_edgecount(S0, target_edges)
    thr1, A_emb_lora_bin = choose_threshold_by_target_edgecount(S1, target_edges)
    suffix = f"_{emb_name_column}" if emb_name_column else ""
    a_emb_lora_out = f"Data/{data_path}/{data_name}-A_emb_lora_bin{suffix}.csv"
    save_binary_adjacency_as_edge_csv(A_emb_lora_bin, present_genes, a_emb_lora_out)

    print(f"Threshold picked (initial emb): {thr0:.6f} | edges={int(A_emb_bin.sum())}")
    print(f"Threshold picked (lora emb): {thr1:.6f} | edges={int(A_emb_lora_bin.sum())}")

    m_bio = metrics_against_true(A_bio_np, A_true)
    m_emb = metrics_against_true(A_emb_bin, A_true)
    m_lora = metrics_against_true(A_emb_lora_bin, A_true)

    print("\n=== Compare against A_true (directed, excluding diagonal) ===")
    print_metrics("A_bio", m_bio)
    print_metrics("A_emb", m_emb)
    print_metrics("A_emb_lora", m_lora)

    n = A_true.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y_true = A_true[mask].reshape(-1)

    auc_emb = roc_auc_score(y_true, S0[mask].reshape(-1))
    ap_emb = average_precision_score(y_true, S0[mask].reshape(-1))
    auc_lora = roc_auc_score(y_true, S1[mask].reshape(-1))
    ap_lora = average_precision_score(y_true, S1[mask].reshape(-1))

    print("\n=== AUROC / PRAUC vs A_true (off-diagonal directed pairs) ===")
    print(f"A_emb      | AUROC={auc_emb:.6f}  PRAUC={ap_emb:.6f}")
    print(f"A_emb_lora | AUROC={auc_lora:.6f}  PRAUC={ap_lora:.6f}")

    print("\n=== Mean + k*std threshold sweep ===")
    for k in [0.0, 0.5, 1.0, 1.5, 2.0]:
        thr0, A0 = choose_threshold_by_mean_kstd(S0, k_std=k)
        thr1, A1 = choose_threshold_by_mean_kstd(S1, k_std=k)

        m0 = metrics_against_true(A0, A_true)
        m1 = metrics_against_true(A1, A_true)

        print(f"\nk={k:.2f}")
        print(f"A_emb      thr={thr0:.6f} edges={int(A0.sum())}")
        print_metrics("A_emb", m0)
        print(f"A_emb_lora thr={thr1:.6f} edges={int(A1.sum())}")
        print_metrics("A_emb_lora", m1)

    print_metrics("A_bio", m_bio)

    metrics_emb = {
        "auroc": float(auc_emb),
        "ap": float(ap_emb),
        "f1": float(m_emb["f1"]),
        "precision": float(m_emb["precision"]),
        "recall": float(m_emb["recall"]),
        "balanced_acc": float(m_emb["balanced_acc"]),
    }
    metrics_lora = {
        "auroc": float(auc_lora),
        "ap": float(ap_lora),
        "f1": float(m_lora["f1"]),
        "precision": float(m_lora["precision"]),
        "recall": float(m_lora["recall"]),
        "balanced_acc": float(m_lora["balanced_acc"]),
    }
    return metrics_emb, metrics_lora


def main():
    _main_start = time.time()
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_name = args.data_path.split("/")[-1]
    species = infer_species(data_name)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Data path:", args.data_path)
    print("Data name:", data_name)
    print("Species:", species)
    print("Using device:", device)
    print("Input feature case:", args.input_feature_case)
    print("Global seed:", args.seed)

    plot_dir = args.plot_dir or f"Data/{args.data_path}/plots"

    if args.embedding_source == "random":
        if args.gene_source is None:
            raise ValueError(
                "--gene-source must be specified when --embedding-source random. "
                "Pass the primary embedding source (genePT or Qwen) so the gene vocabulary matches."
            )
        gene_source = args.gene_source
    else:
        gene_source = args.embedding_source

    emb_name_column, data_emb, gene_names_all = load_embeddings(species, args.embedding_source, gene_source)
    data_sample, gene_names_sample = load_expression(args.data_path, args.subsample)

    present_genes, X_sample_emb, X_expr_low = build_fused_embedding(
        data_sample=data_sample,
        gene_names_sample=gene_names_sample,
        gene_names_all=gene_names_all,
        data_emb=data_emb,
        embedding_source=args.embedding_source,
        emb_name_column=emb_name_column,
        random_embedding_seed=args.random_embedding_seed,
        expr_svd_components=args.expr_svd_components,
        expr_fusion_weight=args.expr_fusion_weight,
        data_path=args.data_path,
        input_feature_case=args.input_feature_case,
    )

    # bio_adj depends only on the gene set (present_genes), not on whether features are random.
    # Always use the vocab-specific cache key so the random run reuses the primary source's
    # bio_adj pickle rather than creating a stale gene-set-agnostic "bio_adj-random.pickle".
    bio_cache_key = _GENE_SOURCE_EMB_NAMES[gene_source]
    A_bio = load_or_build_bio_prior(args.data_path, bio_cache_key, species, present_genes)
    A_train = build_training_adjacency(A_bio, X_expr_low, args.expr_knn_k, args.expr_prior_weight)

    X_sample_lora_low, loss_over_epochs = train_lora(
        X_sample_emb=X_sample_emb,
        A_train=A_train,
        n_low=args.n_low,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        tau=args.tau,
        num_epoch=args.num_epoch,
        base_negs=args.base_negs,
        neg_multiplier=args.neg_multiplier,
        batch_size=args.batch_size,
        device=device,
    )
    print("X_sample_lora_low shape:", X_sample_lora_low.shape)

    if args.save_embedding or args.embedding_out_path:
        out_path = args.embedding_out_path or f"Data/{args.data_path}/X_sample_lora_low_{emb_name_column}.npy"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(out_path, X_sample_lora_low)
        print("Saved:", out_path)

    if not args.no_plots:
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(loss_over_epochs, label="Contrastive Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LoRA Contrastive Training Loss")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        train_loss_plot_path = os.path.join(plot_dir, "training_loss.png")
        plt.savefig(train_loss_plot_path, dpi=200, bbox_inches="tight")
        print("Saved plot:", train_loss_plot_path)
        plt.show()

    pos_i, pos_j, neg_i, neg_j, _mbio_before, _mbio_after = evaluate_balanced_pos_neg(
        A_ref=A_bio,
        X_before=X_sample_emb,
        X_after=X_sample_lora_low,
        title="A_bio",
    )

    if not args.no_plots:
        plot_similarity_violin(
            X_sample_emb,
            X_sample_lora_low,
            pos_i,
            pos_j,
            neg_i,
            neg_j,
            save_path=os.path.join(plot_dir, "violin_similarity_vs_abio.png"),
        )

    A_true = load_reference_adjacency(args.data_path, data_name, present_genes, args.embedding_source)
    if A_true is None:
        print("Skipping A_true-based evaluations (reference network unavailable).")
        return

    pos_i_t, pos_j_t, neg_i_t, neg_j_t, _mtrue_before, _mtrue_after = evaluate_balanced_pos_neg(
        A_ref=A_true,
        X_before=X_sample_emb,
        X_after=X_sample_lora_low,
        title="A_true",
    )
    if not args.no_plots:
        plot_similarity_violin(
            X_sample_emb,
            X_sample_lora_low,
            pos_i_t,
            pos_j_t,
            neg_i_t,
            neg_j_t,
            save_path=os.path.join(plot_dir, "violin_similarity_vs_atrue.png"),
        )

    metrics_emb, metrics_lora = evaluate_binary_and_threshold_free(
        A_true=A_true,
        A_bio=A_bio,
        A_train=A_train,
        X_before=X_sample_emb,
        X_after=X_sample_lora_low,
        present_genes=present_genes,
        data_path=args.data_path,
        data_name=data_name,
        emb_name_column=emb_name_column,
    )

    import json
    after_key = "A_emb_lora" if args.embedding_source != "random" else "A_emb_rnd"
    preprocessing_metrics = {
        "embedding_source": args.embedding_source,
        "A_emb": metrics_emb,
        after_key: metrics_lora,
    }
    metrics_out_path = os.path.join(
        "Data", args.data_path, f"preprocessing_metrics_{args.embedding_source}.json"
    )
    os.makedirs(os.path.dirname(metrics_out_path), exist_ok=True)
    with open(metrics_out_path, "w") as f:
        json.dump(preprocessing_metrics, f, indent=2)
    print(f"Saved preprocessing metrics: {metrics_out_path}")
    print(f"Total stage1_lora.py time in seconds: {time.time() - _main_start:.2f}")


if __name__ == "__main__":
    main()
