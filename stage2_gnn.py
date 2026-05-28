import argparse
import copy
import os
import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from utils import GNNEncoder, GraphAutoEncoder, adj_bce_loss, create_mask_from_adj, csv2A, load_edge_set

warnings.filterwarnings("ignore")


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

METHOD_PATTERNS = {
    "bio": "celloracle-base",
    "celloracle": "celloracle-whole",
    "scenic": "scenic-network",
    "grnboost": "grnboost",
    "portia": "portia",
    "deeprig": "celloracle-deeprig_filtered",
    "correlation": "correlation_thresh",
    "dazzle": "dazzle-full_filtered",
    "knn": "expr-knn",
}


class MultiBilinearDecoder(nn.Module):
    """
    More expressive decoder than a single bilinear form:
      score = sum_h alpha_h * tanh(z W_h z^T) + gamma * (g(z) g(z)^T)
    """

    def __init__(self, latent_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = int(num_heads)
        self.W = nn.Parameter(torch.empty(self.num_heads, latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W)

        self.head_alpha = nn.Parameter(torch.ones(self.num_heads))
        self.gamma = nn.Parameter(torch.tensor(1.0))

        hidden = max(32, latent_dim // 2)
        self.residual_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((z.size(0), z.size(0)), device=z.device, dtype=z.dtype)
        for h in range(self.num_heads):
            bil = z @ self.W[h] @ z.T
            out = out + self.head_alpha[h] * torch.tanh(bil)

        z_res = F.normalize(self.residual_mlp(z), dim=1)
        out = out + self.gamma * (z_res @ z_res.T)
        return out


class GraphAutoEncoderComplex(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels, decoder_heads: int = 4):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels_list, out_channels)
        self.decoder = MultiBilinearDecoder(out_channels, num_heads=decoder_heads)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z)


def select_train_nodes(n_nodes: int, node_fraction: float, seed: int) -> torch.Tensor:
    """Return a sorted 1-D tensor of node indices to use as the training node set."""
    n_train = max(1, int(round(n_nodes * node_fraction)))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_nodes, generator=g)
    return perm[:n_train].sort().values


def create_incident_mask(true_adj: torch.Tensor, train_nodes: torch.Tensor,
                         neg_multiplier: float, seed: int) -> torch.Tensor:
    """
    Build a training mask restricted to edges incident to at least one node in train_nodes.

    Positives: upper-triangle edges in true_adj that are incident to train_nodes.
    Negatives: upper-triangle non-edges that are incident to train_nodes,
               randomly sampled at neg_multiplier * num_pos.
    Returns a bool mask over the full N×N matrix (upper triangle only).
    """
    device = true_adj.device
    n = true_adj.shape[0]
    triu = torch.triu_indices(n, n, offset=1, device=device)

    # Determine which upper-triangle pairs have at least one endpoint in train_nodes
    train_set = torch.zeros(n, dtype=torch.bool, device=device)
    train_set[train_nodes] = True
    incident = train_set[triu[0]] | train_set[triu[1]]

    labels = true_adj[triu[0], triu[1]]
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

    # Restrict to incident pairs
    pos_idx = pos_idx[incident[pos_idx]]
    neg_idx = neg_idx[incident[neg_idx]]

    num_neg = int(max(0, neg_multiplier) * pos_idx.numel())
    num_neg = min(num_neg, neg_idx.numel())

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if num_neg > 0:
        neg_perm = torch.randperm(neg_idx.numel(), generator=g, device=device)[:num_neg]
        selected = torch.cat([pos_idx, neg_idx[neg_perm]])
    else:
        selected = pos_idx

    mask = torch.zeros_like(true_adj, dtype=torch.bool)
    mask[triu[0][selected], triu[1][selected]] = True
    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-step GRNITE with LoRA node features.")

    parser.add_argument("--dataPath", type=str, default="TF500/hESC", help="Path under Data/.")
    parser.add_argument(
        "--targetMethod",
        type=str,
        default="celloracle",
        choices=list(METHOD_PATTERNS.keys()),
        help="Target GRN method to mimic.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="single",
        choices=["single", "consensus"],
        help="Use a single teacher graph or a consensus target built from multiple methods.",
    )
    parser.add_argument(
        "--consensus_methods",
        type=str,
        default="celloracle,knn",
        help="Comma-separated methods used when --target_mode consensus.",
    )
    parser.add_argument(
        "--consensus_min_votes",
        type=int,
        default=0,
        help="Minimum votes for consensus edge (0 = strict majority).",
    )
    parser.add_argument("--gnn_dim_hidden", type=str, default="32,32", help="Hidden units per layer, comma separated.")
    parser.add_argument("--expr_svd_components", type=int, default=32, help="SVD components for expression-derived kNN target.")
    parser.add_argument("--expr_knn_k", type=int, default=10, help="k for expression-derived kNN target.")
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="multi_bilinear",
        choices=["bilinear", "multi_bilinear"],
        help="Decoder type for reconstructing adjacency.",
    )
    parser.add_argument("--decoder_heads", type=int, default=2, help="Number of heads for multi_bilinear decoder.")
    parser.add_argument("--neg_multiplier", type=float, default=1.0, help="Ratio of negatives to positives in the mask.")
    parser.add_argument(
        "--train_node_fraction",
        type=float,
        default=1.0,
        help="Fraction of nodes to use when building the training mask. "
             "Training edges are those incident to (at least one endpoint in) the selected nodes. "
             "Default 1.0 = use all nodes (same as original uniform-edge sampling).",
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for target loss vs prior loss.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for GNN model.")
    parser.add_argument("--num_epoch", type=int, default=500, help="Number of training epochs.")
    parser.add_argument(
        "--edge_count_scale",
        type=float,
        default=1.0,
        help="Scale factor k for edge-count matching at inference (predict k * target_edges).",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id.")
    parser.add_argument("--sample", type=int, default=None, help="Subsample id used in target-method file naming.")
    parser.add_argument("--name", type=str, default="grnite_lora", help="Suffix for saved graph file.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument(
        "--embedding_source",
        type=str,
        default="Qwen",
        choices=["genePT", "Qwen", "random"],
        help="Node feature source: 'genePT' or 'Qwen' load the corresponding LoRA-refined embedding; 'random' uses random Gaussian features.",
    )
    parser.add_argument(
        "--gene_source",
        type=str,
        default=None,
        choices=["genePT", "Qwen"],
        help="Which present_genes file to use when --embedding_source random. "
             "Should match the primary EMBEDDING_SOURCE from run_pipeline.sh.",
    )
    parser.add_argument(
        "--random_feature_dim",
        type=int,
        default=128,
        help="Dimensionality of random node features when --embedding_source random.",
    )
    parser.add_argument("--use_confidence_weighting", action="store_true", help="Use confidence-weighted BCE for target/prior losses.")
    parser.add_argument("--target_confidence_alpha", type=float, default=2.0, help="Scale for target confidence weighting.")
    parser.add_argument("--prior_confidence_alpha", type=float, default=1.0, help="Scale for prior confidence weighting.")

    parser.add_argument(
        "--lora_embedding_path",
        type=str,
        default=None,
        help="Explicit path to a saved LoRA embedding .npy (overrides auto-detection from --embedding_source).",
    )
    parser.add_argument(
        "--present_genes_path",
        type=str,
        default=None,
        help="Path to present_genes.txt from stage1_lora.py. Defaults to Data/<dataPath>/present_genes.txt.",
    )
    parser.add_argument("--lambda_emb", type=float, default=5.0,
                        help="Weight for embedding cosine similarity auxiliary loss (0 = disabled).")
    parser.add_argument("--label_smoothing", type=float, default=0.2,
                        help="Label smoothing ε: teacher 0/1 labels become eps and 1−eps. "
                             "Prevents overconfident fitting to noisy teacher edges.")
    parser.add_argument("--lambda_sparse", type=float, default=1.0,
                        help="Weight for sparsity regularization: penalises mean predicted edge probability. "
                             "Encourages sparse GRN predictions consistent with true GRN topology.")

    return parser.parse_args()


def infer_species(data_name: str) -> str:
    return "human" if data_name in HUMAN_DATASETS else "mouse"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_EMBEDDING_FILENAMES = {
    "genePT": "X_sample_lora_low_GenePT-Embedding-8B.npy",
    "Qwen":   "X_sample_lora_low_Qwen3-Embedding-8B.npy",
}

_PRESENT_GENES_FILENAMES = {
    "genePT": "present_genes_GenePT-Embedding-8B.txt",
    "Qwen":   "present_genes_Qwen3-Embedding-8B.txt",
    "random": "present_genes_random.txt",
}

_GENE_SOURCE_EMB_NAMES = {
    "genePT": "GenePT-Embedding-8B",
    "Qwen":   "Qwen3-Embedding-8B",
}


def resolve_lora_embedding_path(data_path: str, embedding_source: str, requested_path: Optional[str]) -> Optional[str]:
    """Return the path to the LoRA embedding .npy, or None for 'random'."""
    if embedding_source == "random":
        return None

    if requested_path is not None:
        if not os.path.exists(requested_path):
            raise FileNotFoundError(f"LoRA embedding file not found: {requested_path}")
        return requested_path

    filename = _EMBEDDING_FILENAMES[embedding_source]
    path = os.path.join(f"Data/{data_path}", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LoRA embedding file not found: {path}\n"
            f"Run stage1_lora.py with --embedding-source {'genePT' if embedding_source == 'genePT' else 'text'} --save-embedding, "
            "or pass --lora_embedding_path explicitly."
        )
    return path


def resolve_present_genes_path(
    data_path: str,
    embedding_source: str,
    gene_source: Optional[str],
    requested_path: Optional[str],
) -> str:
    """Return the path to the source-specific present_genes .txt file.

    For random ablation, gene_source overrides embedding_source so the random run
    uses the same gene vocabulary as the primary embedding source.
    """
    if requested_path is not None:
        if not os.path.exists(requested_path):
            raise FileNotFoundError(f"present_genes file not found: {requested_path}")
        return requested_path

    # Determine which vocabulary key to look up:
    # - random + gene_source provided → use the primary source's present_genes directly
    # - random + no gene_source → fall back to present_genes_random.txt (written by stage1_lora.py)
    # - any other source → use its own file
    if embedding_source == "random" and gene_source is not None:
        lookup_key = gene_source
    else:
        lookup_key = embedding_source

    default_dir = f"Data/{data_path}"
    path = os.path.join(default_dir, _PRESENT_GENES_FILENAMES[lookup_key])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"present_genes file not found: {path}\n"
            f"Run stage1_lora.py with --embedding-source {lookup_key} --save-embedding first."
        )
    return path


def load_present_genes(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"present_genes file not found: {path}")

    genes: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("genes present"):
                continue
            genes.append(line)

    if not genes:
        raise ValueError(f"No genes parsed from {path}")
    return genes


def load_or_build_A_bio(data_path: str, species: str, present_genes: List[str], source_label: str = ""):
    suffix = f"-{source_label}" if source_label else ""
    data_path_bio_edges = f"Data/{data_path}/bio_grn{suffix}.csv"

    if os.path.exists(data_path_bio_edges):
        bio_grn = load_edge_set(data_path_bio_edges)
        print("Loaded bio GRN edge list from:", data_path_bio_edges)
    else:
        base_path = "Gene_embeddings/celloracle_baseGRN.csv" if species == "human" else "Gene_embeddings/celloracle_mouse_baseGRN.csv"
        bio_grn = load_edge_set(base_path)
        present_set = set(present_genes)
        bio_grn = {(g1, g2) for g1, g2 in bio_grn if g1 in present_set and g2 in present_set}
        pd.DataFrame(list(bio_grn), columns=["Gene1", "Gene2"]).to_csv(data_path_bio_edges, index=False)
        print("Saved filtered bio GRN edge list to:", data_path_bio_edges)

    A_bio = csv2A(bio_grn, present_genes)
    A_bio = np.asarray(A_bio, dtype=np.float32)
    np.fill_diagonal(A_bio, 0.0)
    return A_bio, bio_grn


def load_target_graph(
    data_path: str,
    data_name: str,
    target_method: str,
    subsample: Optional[int],
    present_genes: List[str],
    expr_svd_components: int = 64,
    expr_knn_k: int = 10,
):
    if target_method == "bio":
        raise ValueError("target_method='bio' should be resolved from loaded A_bio in main().")

    if target_method == "knn":
        A_knn, _ = build_expression_knn_target(
            data_path=data_path,
            present_genes=present_genes,
            subsample=subsample,
            expr_svd_components=expr_svd_components,
            expr_knn_k=expr_knn_k,
        )
        return A_knn, set()

    base = METHOD_PATTERNS[target_method]
    if subsample is not None:
        if target_method in ["deeprig", "dazzle"]:
            data_path_method = f"Data/{data_path}/{data_name}-{base.replace('_filtered', '')}-sample-{subsample}_filtered.csv"
        else:
            data_path_method = f"Data/{data_path}/{data_name}-{base}-sample-{subsample}.csv"
    else:
        data_path_method = f"Data/{data_path}/{data_name}-{base}.csv"

    target_grn = load_edge_set(data_path_method)
    target_grn = set((g1, g2) for g1, g2 in target_grn if g1 in present_genes and g2 in present_genes)
    print("Target method file:", data_path_method)
    print("Number of edges in target GRN with present genes:", len(target_grn))

    A_target = csv2A(target_grn, present_genes)
    A_target = np.asarray(A_target, dtype=np.float32)
    np.fill_diagonal(A_target, 0.0)
    return A_target, target_grn


def build_expression_knn_target(
    data_path: str,
    present_genes: List[str],
    subsample: Optional[int],
    expr_svd_components: int,
    expr_knn_k: int,
):
    expr_path = f"Data/{data_path}/ExpressionData.csv"
    if subsample is not None:
        # Keep behavior aligned with existing project conventions.
        cell_range = [0] + list(range(2000 * (subsample - 1), 2000 * subsample))
        data_sample = pd.read_csv(expr_path, header=0, index_col=0, usecols=cell_range).T
    else:
        data_sample = pd.read_csv(expr_path, header=0, index_col=0).T

    data_sample = data_sample.transform(lambda x: np.log(x + 1))
    data_sample.columns = [g.upper() for g in data_sample.columns]

    missing = [g for g in present_genes if g not in data_sample.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} present genes are missing in expression matrix while building knn target."
        )

    X_sample = data_sample[present_genes].T.values.astype(np.float32)  # gene x cell
    expr_gene = X_sample.copy()
    expr_gene = (expr_gene - expr_gene.mean(axis=1, keepdims=True)) / (expr_gene.std(axis=1, keepdims=True) + 1e-6)
    expr_gene = np.nan_to_num(expr_gene, nan=0.0, posinf=0.0, neginf=0.0)

    max_components = max(1, min(expr_gene.shape[0] - 1, expr_gene.shape[1] - 1))
    n_components = min(int(expr_svd_components), max_components)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    X_expr_low = svd.fit_transform(expr_gene).astype(np.float32)
    X_expr_low = X_expr_low / (np.linalg.norm(X_expr_low, axis=1, keepdims=True) + 1e-12)

    S_expr = X_expr_low @ X_expr_low.T
    np.fill_diagonal(S_expr, -np.inf)
    n = S_expr.shape[0]
    k_expr = max(1, min(int(expr_knn_k), n - 1))
    idx = np.argpartition(-S_expr, kth=k_expr - 1, axis=1)[:, :k_expr]

    A_expr = np.zeros((n, n), dtype=np.float32)
    rows = np.arange(n)[:, None]
    A_expr[rows, idx] = 1.0
    np.fill_diagonal(A_expr, 0.0)

    # Make undirected to stay consistent with csv2A-built targets.
    A_expr = np.maximum(A_expr, A_expr.T)
    np.fill_diagonal(A_expr, 0.0)
    print(f"Expression kNN target built: k={k_expr}, edges={int(A_expr.sum())}")
    return A_expr, X_expr_low


def save_knn_target_csv(
    A_knn: np.ndarray,
    present_genes: List[str],
    data_path: str,
    data_name: str,
    subsample: Optional[int],
):
    n = A_knn.shape[0]
    i, j = np.triu_indices(n, k=1)
    keep = A_knn[i, j] > 0.5
    i, j = i[keep], j[keep]

    df = pd.DataFrame({
        "Gene1": [present_genes[src] for src in i],
        "Gene2": [present_genes[tgt] for tgt in j],
    })

    if subsample is not None:
        out_path = f"Data/{data_path}/{data_name}-expr-knn-sample-{subsample}.csv"
    else:
        out_path = f"Data/{data_path}/{data_name}-expr-knn.csv"
    df.to_csv(out_path, index=False)
    print("Saved kNN target graph CSV:", out_path)


def load_consensus_target_graph(
    data_path: str,
    data_name: str,
    methods: List[str],
    subsample: Optional[int],
    present_genes: List[str],
    min_votes: int = 0,
    expr_svd_components: int = 64,
    expr_knn_k: int = 10,
    A_bio: Optional[np.ndarray] = None,
):
    if not methods:
        raise ValueError("consensus methods list is empty.")

    A_list = []
    used_methods = []
    for method in methods:
        if method not in METHOD_PATTERNS:
            raise ValueError(f"Unknown method in consensus list: {method}")
        if method == "bio":
            if A_bio is None:
                raise ValueError("A_bio is required when 'bio' is included in consensus methods.")
            A_m = np.asarray(A_bio).astype(np.float32)
            np.fill_diagonal(A_m, 0.0)
        elif method == "knn":
            A_m, _ = build_expression_knn_target(
                data_path=data_path,
                present_genes=present_genes,
                subsample=subsample,
                expr_svd_components=expr_svd_components,
                expr_knn_k=expr_knn_k,
            )
            save_knn_target_csv(
                A_knn=A_m,
                present_genes=present_genes,
                data_path=data_path,
                data_name=data_name,
                subsample=subsample,
            )
        else:
            A_m, _ = load_target_graph(
                data_path=data_path,
                data_name=data_name,
                target_method=method,
                subsample=subsample,
                present_genes=present_genes,
                expr_svd_components=expr_svd_components,
                expr_knn_k=expr_knn_k,
            )
        A_list.append(A_m.astype(np.float32))
        used_methods.append(method)

    vote_count = np.sum(np.stack(A_list, axis=0), axis=0)
    k = len(A_list)
    if min_votes <= 0:
        # strict majority
        min_votes = (k // 2) + 1
    min_votes = max(1, min(min_votes, k))

    A_consensus = (vote_count >= float(min_votes)).astype(np.float32)
    np.fill_diagonal(A_consensus, 0.0)
    vote_frac = (vote_count / float(k)).astype(np.float32)
    np.fill_diagonal(vote_frac, 0.0)

    print(f"Consensus target built from methods={used_methods}, min_votes={min_votes}, edges={int(A_consensus.sum())}")
    return A_consensus, vote_frac, used_methods


def masked_weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: Optional[torch.Tensor] = None):
    pred_flat = logits[mask]
    target_flat = target[mask]
    if weight is None:
        return F.binary_cross_entropy_with_logits(pred_flat, target_flat)
    weight_flat = weight[mask]
    return F.binary_cross_entropy_with_logits(pred_flat, target_flat, weight=weight_flat)


def binarize_by_edgecount_matching(adj_score: torch.Tensor, adj_target: torch.Tensor, edge_count_scale: float = 1.0):
    """
    Binarize symmetric score matrix by matching the number of undirected edges in target.
    Returns (adj_bin, threshold, target_edges, pred_edges).
    """
    S = adj_score.detach().clone()
    T = adj_target.detach()
    n = S.shape[0]

    triu = torch.triu_indices(n, n, offset=1, device=S.device)
    score_upper = S[triu[0], triu[1]]
    target_upper = T[triu[0], triu[1]]

    target_edges_base = int((target_upper > 0.5).sum().item())
    target_edges = int(round(float(edge_count_scale) * float(target_edges_base)))
    total_candidates = score_upper.numel()
    target_edges = max(0, min(target_edges, total_candidates))

    adj_bin = torch.zeros_like(S)
    if target_edges == 0:
        thr = float("inf")
        return adj_bin, thr, target_edges, 0

    topk_vals, topk_idx = torch.topk(score_upper, k=target_edges, largest=True, sorted=False)
    adj_bin[triu[0][topk_idx], triu[1][topk_idx]] = 1.0
    adj_bin[triu[1][topk_idx], triu[0][topk_idx]] = 1.0
    thr = float(topk_vals.min().item())
    return adj_bin, thr, target_edges_base, target_edges


def save_predicted_graph(adj_pred_eval_binary, adj_pred_eval_score, present_genes, data_path, data_name, target_method, sample, name, node_feature_source="lora"):
    infered_grn = adj_pred_eval_binary.cpu().numpy()

    i, j = np.triu_indices_from(infered_grn, k=1)
    nonzero_mask = infered_grn[i, j] != 0
    i, j = i[nonzero_mask], j[nonzero_mask]

    edges = [(present_genes[src], present_genes[tgt]) for src, tgt in zip(i, j)]
    weights = adj_pred_eval_score[i, j].cpu().numpy()

    edges_df = pd.DataFrame({
        "Gene1": [e[0] for e in edges],
        "Gene2": [e[1] for e in edges],
        "weight": weights,
    })

    grn_mapping = {
        "bio": "-celloracle-base",
        "celloracle": "-celloracle-whole",
        "scenic": "-scenic-network",
        "portia": "-portia",
        "grnboost": "-grnboost",
        "deeprig": "-celloracle-deeprig",
        "dazzle": "-dazzle-full_filtered",
        "correlation": "-correlation_thresh",
        "consensus": "-consensus",
    }
    case_name = f"{data_name}-{node_feature_source}{grn_mapping.get(target_method, '-unknown')}"

    if sample is not None:
        out_path = f"Data/{data_path}/{case_name}-sample-{sample}_{name}.csv"
    else:
        out_path = f"Data/{data_path}/{case_name}_{name}.csv"

    edges_df.to_csv(out_path, index=False)
    print("Saved inferred GRN:", out_path)


def save_target_graph_csv(
    A_target: np.ndarray,
    present_genes: List[str],
    data_path: str,
    data_name: str,
    target_label: str,
    sample: Optional[int],
    vote_frac: Optional[np.ndarray] = None,
):
    n = A_target.shape[0]
    i, j = np.triu_indices(n, k=1)
    keep = A_target[i, j] > 0.5
    i, j = i[keep], j[keep]

    out = {
        "Gene1": [present_genes[src] for src in i],
        "Gene2": [present_genes[tgt] for tgt in j],
    }
    if vote_frac is not None:
        out["vote_fraction"] = vote_frac[i, j]

    df = pd.DataFrame(out)
    if sample is not None:
        out_path = f"Data/{data_path}/{data_name}-target_{target_label}-sample-{sample}.csv"
    else:
        out_path = f"Data/{data_path}/{data_name}-target_{target_label}.csv"
    df.to_csv(out_path, index=False)
    print("Saved target graph CSV:", out_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_path = args.dataPath
    data_name = data_path.split("/")[-1]
    species = infer_species(data_name)
    gnn_dim_hidden = [int(x) for x in args.gnn_dim_hidden.split(",")]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Data path:", data_path)
    print("Data name:", data_name)
    print("Species:", species)
    print("Using device:", device)
    print("Decoder type:", args.decoder_type)
    print("Embedding source:", args.embedding_source)
    print("Edge-count scale:", args.edge_count_scale)
    print("Target mode:", args.target_mode)
    if args.target_mode == "single":
        print("Target method:", args.targetMethod)
    else:
        print("Consensus methods:", args.consensus_methods)
    print("Confidence weighting:", args.use_confidence_weighting)

    lora_path = resolve_lora_embedding_path(data_path, args.embedding_source, args.lora_embedding_path)
    present_genes_path = resolve_present_genes_path(data_path, args.embedding_source, args.gene_source, args.present_genes_path)
    present_genes = load_present_genes(present_genes_path)
    print("Loaded present genes:", present_genes_path, f"({len(present_genes)} genes)")

    if lora_path is not None:
        X_lora = np.load(lora_path).astype(np.float32)
        if X_lora.shape[0] != len(present_genes):
            raise ValueError(
                f"Row mismatch: X_lora has {X_lora.shape[0]} rows but present_genes has {len(present_genes)} genes. "
                "Ensure both files come from the same stage1_lora.py run."
            )
        print("Loaded LoRA embedding:", lora_path)
        print("X_lora shape:", X_lora.shape)
    else:
        X_lora = None

    # Derive bio_grn source label from the gene vocabulary, never "random".
    _bio_gene_source = args.gene_source if args.embedding_source == "random" and args.gene_source else args.embedding_source
    bio_source_label = _GENE_SOURCE_EMB_NAMES.get(_bio_gene_source, _bio_gene_source)

    start_time1 = time.time()
    A_bio, _ = load_or_build_A_bio(data_path, species, present_genes, source_label=bio_source_label)
    if args.target_mode == "single":
        if args.targetMethod == "bio":
            A_target = np.asarray(A_bio).astype(np.float32)
            np.fill_diagonal(A_target, 0.0)
        else:
            A_target, _ = load_target_graph(
                data_path=data_path,
                data_name=data_name,
                target_method=args.targetMethod,
                subsample=args.sample,
                present_genes=present_genes,
                expr_svd_components=args.expr_svd_components,
                expr_knn_k=args.expr_knn_k,
            )
        if args.targetMethod == "knn":
            save_knn_target_csv(
                A_knn=A_target,
                present_genes=present_genes,
                data_path=data_path,
                data_name=data_name,
                subsample=args.sample,
            )
        target_method_label = args.targetMethod
        target_vote_frac = None
    else:
        consensus_methods = [m.strip() for m in args.consensus_methods.split(",") if m.strip()]
        A_target, target_vote_frac, _ = load_consensus_target_graph(
            data_path=data_path,
            data_name=data_name,
            methods=consensus_methods,
            subsample=args.sample,
            present_genes=present_genes,
            min_votes=args.consensus_min_votes,
            expr_svd_components=args.expr_svd_components,
            expr_knn_k=args.expr_knn_k,
            A_bio=A_bio,
        )
        target_method_label = "consensus"
        save_target_graph_csv(
            A_target=A_target,
            present_genes=present_genes,
            data_path=data_path,
            data_name=data_name,
            target_label="consensus",
            sample=args.sample,
            vote_frac=target_vote_frac,
        )
    all_time1 = time.time() - start_time1

    A_bio_sparse = sp.csr_matrix(A_bio)
    edge_index_bio, _ = from_scipy_sparse_matrix(A_bio_sparse)

    if args.embedding_source == "random":
        rng = np.random.default_rng(args.seed)
        x_np = rng.standard_normal(size=(len(present_genes), args.random_feature_dim)).astype(np.float32)
        print("Using random node features with shape:", x_np.shape)
    else:
        x_np = X_lora
        print(f"Using {args.embedding_source} LoRA node features with shape:", x_np.shape)

    x = torch.tensor(x_np, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index_bio).to(device)

    adj_target = torch.tensor(A_target, dtype=torch.float32, device=device)
    A_prior = torch.tensor(A_bio, dtype=torch.float32, device=device)

    # Load ground truth for edge-count matching at inference time (if available)
    try:
        ref_grn_un = load_edge_set(f"Data/{data_path}/refNetwork.csv", undirected=False)
        present_genes_upper = set(g.upper() for g in present_genes)
        ref_grn_present = [(g1, g2) for g1, g2 in ref_grn_un if g1 in present_genes_upper and g2 in present_genes_upper]
        A_true_np = csv2A(ref_grn_present, present_genes)
        adj_true_for_edgecount = torch.tensor(np.asarray(A_true_np, dtype=np.float32), device=device)
        print(f"Ground truth loaded for edge-count matching: {int(adj_true_for_edgecount.triu(1).sum().item())} edges")
    except Exception as e:
        print(f"Ground truth not found, falling back to adj_target for edge-count matching: {e}")
        adj_true_for_edgecount = adj_target
    if args.use_confidence_weighting:
        if target_vote_frac is not None:
            vote_frac_t = torch.tensor(target_vote_frac, dtype=torch.float32, device=device)
            target_conf = torch.abs(vote_frac_t - 0.5) * 2.0
            target_weight = 1.0 + float(args.target_confidence_alpha) * target_conf
        else:
            target_weight = torch.ones_like(adj_target)

        x_norm = F.normalize(data.x, dim=1)
        sim01 = (x_norm @ x_norm.T + 1.0) * 0.5
        prior_conf = A_prior * sim01 + (1.0 - A_prior) * (1.0 - sim01)
        prior_weight = 1.0 + float(args.prior_confidence_alpha) * prior_conf
    else:
        target_weight = None
        prior_weight = None

    start_time2 = time.time()
    if args.decoder_type == "bilinear":
        model = GraphAutoEncoder(
            in_channels=data.x.shape[1],
            hidden_channels_list=gnn_dim_hidden[:-1],
            out_channels=gnn_dim_hidden[-1],
        ).to(device)
    else:
        model = GraphAutoEncoderComplex(
            in_channels=data.x.shape[1],
            hidden_channels_list=gnn_dim_hidden[:-1],
            out_channels=gnn_dim_hidden[-1],
            decoder_heads=args.decoder_heads,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    effective_lambda_emb = args.lambda_emb if args.embedding_source != "random" else 0.0
    if effective_lambda_emb != args.lambda_emb:
        print(f"lambda_emb forced to 0 (embedding_source=random; cosine similarity of random features is uninformative)")
    if effective_lambda_emb > 0:
        x_norm = F.normalize(torch.tensor(x_np, dtype=torch.float32, device=device), dim=1)
        S_emb = (x_norm @ x_norm.T)          # cosine similarities in [-1, 1]
        S_emb = (S_emb + 1.0) / 2.0          # rescale to [0, 1]
        S_emb.fill_diagonal_(0.0)
        triu_all = torch.triu_indices(len(present_genes), len(present_genes), offset=1, device=device)
        S_emb_flat = S_emb[triu_all[0], triu_all[1]]  # precompute flat upper-tri once
        print(f"Embedding similarity matrix computed | lambda_emb={args.lambda_emb}")
    else:
        S_emb_flat = None
        triu_all   = None

    # Select training nodes once before the loop (fixed for the entire run).
    n_nodes = len(present_genes)
    use_node_masking = args.train_node_fraction < 1.0
    if use_node_masking:
        train_nodes = select_train_nodes(n_nodes, args.train_node_fraction, seed=args.seed)
        print(
            f"Node-based masking: {train_nodes.numel()}/{n_nodes} nodes selected "
            f"(train_node_fraction={args.train_node_fraction})"
        )
    else:
        train_nodes = None

    model.train()
    best_state = None
    loss_min = float("inf")
    epoch_min = 0

    for epoch in range(0, args.num_epoch + 1):
        optimizer.zero_grad()

        if use_node_masking:
            mask_posneg_1 = create_incident_mask(adj_target, train_nodes,
                                                  neg_multiplier=args.neg_multiplier, seed=epoch)
            mask_posneg_2 = create_incident_mask(A_prior, train_nodes,
                                                  neg_multiplier=args.neg_multiplier, seed=epoch)
        else:
            mask_posneg_1 = create_mask_from_adj(adj_target, neg_multiplier=args.neg_multiplier, seed=epoch)
            mask_posneg_2 = create_mask_from_adj(A_prior, neg_multiplier=args.neg_multiplier, seed=epoch)

        adj_pred = model(data.x, data.edge_index)
        # Label smoothing: replace hard 0/1 teacher labels with (ε, 1−ε) to avoid
        # overconfident fitting to noisy teacher edges. Uncomment args block above too.
        eps = args.label_smoothing
        adj_target_smooth = adj_target * (1.0 - eps) + (1.0 - adj_target) * eps
        A_prior_smooth    = A_prior    * (1.0 - eps) + (1.0 - A_prior)    * eps
        if args.use_confidence_weighting:
            loss_target = masked_weighted_bce_with_logits(adj_pred, adj_target, mask_posneg_1, target_weight)
            loss_prior = masked_weighted_bce_with_logits(adj_pred, A_prior, mask_posneg_2, prior_weight)
        else:
            # Replace adj_target / A_prior with adj_target_smooth / A_prior_smooth to enable label smoothing
            loss_target = adj_bce_loss(adj_pred, adj_target, mask=mask_posneg_1, pos_weight=None)
            loss_prior = adj_bce_loss(adj_pred, A_prior, mask=mask_posneg_2, pos_weight=None)
            # loss_target = adj_bce_loss(adj_pred, adj_target_smooth, mask=mask_posneg_1, pos_weight=None)
            # loss_prior = adj_bce_loss(adj_pred, A_prior_smooth, mask=mask_posneg_2, pos_weight=None)

        loss = args.beta * loss_target + (1.0 - args.beta) * loss_prior

        if S_emb_flat is not None:
            pred_flat = torch.sigmoid(adj_pred[triu_all[0], triu_all[1]])
            loss_emb = F.mse_loss(pred_flat, S_emb_flat)
            loss = loss + effective_lambda_emb * loss_emb
        else:
            loss_emb = torch.tensor(0.0)

        # Sparsity regularization: penalise mean predicted edge probability.
        # Real GRNs are sparse; this discourages memorising dense teacher false positives.
        # Uncomment args block above too.
        # loss_sparse = torch.sigmoid(adj_pred).mean()
        # loss = loss + args.lambda_sparse * loss_sparse

        if loss.item() < loss_min:
            loss_min = loss.item()
            epoch_min = epoch
            best_state = copy.deepcopy(model.state_dict())

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch}, Loss={loss.item():.4f}, "
                f"Loss_target={loss_target.item():.4f}, Loss_prior={loss_prior.item():.4f}, "
                f"Loss_emb={loss_emb.item():.4f}, "
                f"Best Loss={loss_min:.4f}, Best Epoch={epoch_min}"
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        adj_pred_eval = torch.sigmoid(model(data.x, data.edge_index))

    adj_pred_eval = adj_pred_eval * (1 - torch.eye(adj_pred_eval.size(0), device=adj_pred_eval.device))
    N = adj_pred_eval.shape[0]
    triu_indices = torch.triu_indices(N, N, offset=1)
    adj_pred_eval[triu_indices[1], triu_indices[0]] = adj_pred_eval[triu_indices[0], triu_indices[1]]

    adj_pred_eval_binary, matched_thr, target_edges_base, pred_edges = binarize_by_edgecount_matching(
        adj_pred_eval, adj_target, edge_count_scale=args.edge_count_scale # adj_true_for_edgecount
    )
    print(
        f"Edge-count matching threshold={matched_thr:.6f} | "
        f"target_edges(base)={target_edges_base} | "
        f"edge_count_scale={args.edge_count_scale:.4f} | "
        f"predicted_edges={pred_edges}"
    )

    all_time2 = time.time() - start_time2

    print("Step 1 time (preprocessing) in seconds: {:.2f}".format(all_time1))
    print("Step 2 (training and evaluation) in seconds: {:.2f}".format(all_time2))
    print("Total training and evaluation time in seconds: {:.2f}".format(all_time1 + all_time2))

    # For random ablation, encode the gene vocabulary in the filename so genePT and
    # Qwen ablation outputs don't collide (e.g. random_genePT vs random_Qwen).
    if args.embedding_source == "random" and args.gene_source:
        effective_source = f"random_{args.gene_source}"
    else:
        effective_source = args.embedding_source

    save_predicted_graph(
        adj_pred_eval_binary=adj_pred_eval_binary,
        adj_pred_eval_score=adj_pred_eval,
        present_genes=present_genes,
        data_path=data_path,
        data_name=data_name,
        target_method=target_method_label,
        sample=args.sample,
        name=args.name,
        node_feature_source=effective_source,
    )

    # Optional: save the subset of reference edges restricted to present genes.
    try:
        data_path_true = f"Data/{data_path}/refNetwork.csv"
        ref_grn_un = load_edge_set(data_path_true, undirected=False)
        present_genes_upper = set(g.upper() for g in present_genes)
        ref_grn_present = [(g1, g2) for g1, g2 in ref_grn_un if g1 in present_genes_upper and g2 in present_genes_upper]
        ref_grn_present_path = f"Data/{data_path}/{data_name}-ref_present_{args.embedding_source}.csv"
        pd.DataFrame(list(ref_grn_present), columns=["Gene1", "Gene2"]).to_csv(ref_grn_present_path, index=False)
        print("Reference GRN with present genes saved!")
    except Exception as e:
        print("Could not save reference GRN with present genes:", e)


if __name__ == "__main__":
    main()
