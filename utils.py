import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import pandas as pd
import torch
import copy
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv


 
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super(GNNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TAGConv(in_channels, hidden_channels_list[0]))
        for i in range(1, len(hidden_channels_list)):
            self.layers.append(TAGConv(hidden_channels_list[i-1], hidden_channels_list[i]))
        self.layers.append(TAGConv(hidden_channels_list[-1], out_channels))
        self.final_layer = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        z = self.final_layer(x)
        return z

class BilinearDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(BilinearDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        # adj_reconstructed = torch.sigmoid(z @ self.weight @ z.T)
        adj_reconstructed = z @ self.weight @ z.T
        return adj_reconstructed
    

class GraphAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels_list, out_channels)
        self.decoder = BilinearDecoder(out_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_reconstructed = self.decoder(z)
        return adj_reconstructed



# Construc an adjacency matrix from the distance matrix
def get_adjacency_matrix(distance_matrix, k=None, threshold=None):
    """
    Given a distance matrix, return an adjacency matrix.
    If k is provided, threshold = mean + k * std of upper-triangle distances.
    Otherwise, use threshold if provided, else fallback to mean.
    """
    distance_upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    if k is not None:
        mu = np.mean(distance_upper_triangle)
        sigma = np.std(distance_upper_triangle)
        threshold = mu - k * sigma
    elif threshold is None:
        threshold = np.mean(distance_upper_triangle)

    adjacency_matrix = (distance_matrix < threshold).astype(int)

    # Set diagonal to 0
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix


# Given gene name, get the embedding and consrtuct a distance matrix between the embeddings of those genes
def get_present(gene_names, gene_names_all, data):
    """
    Given a list of gene names and a dictionary of gene embeddings, 
    return the distance matrix between the embeddings of those genes.
    """
    N = len(gene_names)
    # Get the embeddings for the given gene names
    gene_embeddings_sample= []
    present_genes = []
    for gene_name in gene_names:
        if gene_name in gene_names_all:
            gene_embeddings_sample.append(data[gene_name])
            present_genes.append(gene_name)
        else:
            continue
    # Convert the list of embeddings to a numpy array
    embeddings = np.array(gene_embeddings_sample)
    print("Present genes found and embeddings shape:", embeddings.shape)

    return present_genes, embeddings



    
def wgcna_grn(X, beta=1):
    """
    Vectorized WGCNA GRN inference using correlation and soft-thresholding.

    Args:
        X (np.ndarray): Gene expression matrix [num_genes x num_cells]
        beta (float): Soft-thresholding power

    Returns:
        A (np.ndarray): Adjacency matrix [num_genes x num_genes]
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X)

    # Apply soft thresholding (absolute value raised to power beta)
    A = np.abs(corr_matrix) ** beta

    # Zero out diagonal
    np.fill_diagonal(A, 0.0)

    # fill the nan values with 0
    A = np.nan_to_num(A)
    

    return A




def adj_bce_loss(pred_adj, true_adj, mask=None, pos_weight=None):
    if mask is None:
        N = pred_adj.shape[0]
        triu_indices = torch.triu_indices(N, N, offset=1)
        pred_flat = pred_adj[triu_indices[0], triu_indices[1]]
        true_flat = true_adj[triu_indices[0], triu_indices[1]]
    else:
        pred_flat = pred_adj[mask]
        true_flat = true_adj[mask]

    if pos_weight is not None:
        weight = torch.ones_like(true_flat)
        weight[true_flat == 1] = pos_weight  # Increase weight for positive samples
    else:
        weight = None

    loss_fn = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')
    return loss_fn(pred_flat, true_flat)



def load_edge_set(csv_path, undirected=True):
    df = pd.read_csv(csv_path)
    
    # Detect column names automatically
    if {'Gene1', 'Gene2'}.issubset(df.columns):
        source_col, target_col = 'Gene1', 'Gene2'
    else:
        raise ValueError("CSV must contain either ['Gene1', 'Gene2'] or ['TF', 'Target'] columns.")
    
    # Build edge set
    if undirected:
        edge_set = set(tuple(sorted([row[source_col].upper(), row[target_col].upper()])) for _, row in df.iterrows())
    else:
        edge_set = [(row[source_col].upper(), row[target_col].upper()) for _, row in df.iterrows()]

    return edge_set



def create_mask_from_adj(true_adj, neg_multiplier=1, seed=42):
    """
    Create a mask that includes all positive edges (1s) and
    a random subset of negative edges (0s) from the upper triangle of the adjacency matrix.
    """
    neg_multiplier = int(neg_multiplier)
    device = true_adj.device
    torch.manual_seed(seed)
    N = true_adj.shape[0]
    
    # Make sure triu_indices are on the correct device
    triu_indices = torch.triu_indices(N, N, offset=1, device=device)

    # Get the edge values in the upper triangle
    true_flat = true_adj[triu_indices[0], triu_indices[1]]

    # Positive and negative edge indices
    pos_idx = (true_flat == 1).nonzero(as_tuple=True)[0]
    neg_idx = (true_flat == 0).nonzero(as_tuple=True)[0]

    # Sample negative edges
    num_pos = pos_idx.size(0)
    num_neg_to_sample = min(neg_multiplier * num_pos, neg_idx.size(0))
    perm = torch.randperm(neg_idx.size(0), device=device)[:num_neg_to_sample]
    neg_idx_sampled = neg_idx[perm]

    # Combine
    selected_idx = torch.cat([pos_idx, neg_idx_sampled], dim=0)

    # Create mask on full matrix
    mask = torch.zeros_like(true_adj, dtype=torch.bool)
    i_selected = triu_indices[0][selected_idx]
    j_selected = triu_indices[1][selected_idx]
    mask[i_selected, j_selected] = True

    return mask



def csv2A(grn, present_genes):
    N = len(present_genes)
    A = np.zeros((N, N))
    
    # Convert grn edges to uppercase for consistent comparison
    grn_upper = set((g1.upper(), g2.upper()) for g1, g2 in grn)
    
    for i in range(N):
        for j in range(i, N):
            if i != j:
                # Compare using uppercase versions of the genes
                gene_i = present_genes[i].upper()
                gene_j = present_genes[j].upper()
                A[i, j] = 1 if ((gene_i, gene_j) in grn_upper or (gene_j, gene_i) in grn_upper) else 0
                A[j, i] = A[i, j]  # make symmetric

    return A
