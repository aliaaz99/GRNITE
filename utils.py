import pickle
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import pearsonr

from sklearn.preprocessing import normalize
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super(GCNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels_list[0]))
        for i in range(1, len(hidden_channels_list)):
            self.layers.append(GCNConv(hidden_channels_list[i-1], hidden_channels_list[i]))
        self.layers.append(GCNConv(hidden_channels_list[-1], out_channels))

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
        z = self.layers[-1](x, edge_index)
        return z


class BilinearDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(BilinearDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        # z @ W @ z.T
        adj_reconstructed = torch.sigmoid(z @ self.weight @ z.T)
        return adj_reconstructed

class GraphAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels_list, out_channels)
        self.decoder = BilinearDecoder(out_channels)
        # self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_reconstructed = self.decoder(z)
        return adj_reconstructed

# Construc an adjacency matrix from the distance matrix
def get_adjacency_matrix(distance_matrix, threshold=None):
    """
    Given a distance matrix, return an adjacency matrix.
    """
    # Create an adjacency matrix based on the distance matrix
    if threshold is not None:
        adjacency_matrix = (distance_matrix < threshold).astype(int)
    else:
        distance_upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        threshold = np.mean(distance_upper_triangle)
        adjacency_matrix = (distance_matrix < threshold).astype(int)
    # Set the diagonal to 0
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix


# Given gene name, get the embedding and consrtuct a distance matrix between the embeddings of those genes
def get_distance_matrix(gene_names, gene_names_all, data):
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
    print(embeddings.shape)
    
    # Compute the distance matrix
    embeddings = normalize(embeddings, norm='l2')
    distance_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2) 
    # # diagonal to zero
    # np.fill_diagonal(distance_matrix, 0)


    return distance_matrix, present_genes, embeddings

# plot the graph of the adjacency matrix
def plot_graph(adjacency_matrix, gene_names):
    """
    Given an adjacency matrix and gene names, plot the graph.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_matrix(adjacency_matrix)
    
    # Set the node labels to the gene names
    mapping = {i: gene_names[i] for i in range(len(gene_names))}
    G = nx.relabel_nodes(G, mapping)
    
    # Draw the graph
    plt.figure(figsize=(4, 4), facecolor='white')
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=8)
    plt.title('Graph of Gene Embeddings')



def plot_diff_graph(adj_original, adj_reconstructed, node_labels=None):
    """
    Plots a graph comparing original and reconstructed adjacency matrices.
    Green: added edges, Red: removed edges, Gray: retained edges.
    """
    # Binarize reconstructed adjacency
    adj_reconstructed_bin = adj_reconstructed

    # Get edge sets
    original_edges = set(zip(*np.where(np.triu(adj_original, 1))))
    reconstructed_edges = set(zip(*np.where(np.triu(adj_reconstructed_bin, 1))))

    # Compute differences
    added_edges = reconstructed_edges - original_edges
    removed_edges = original_edges - reconstructed_edges
    unchanged_edges = original_edges & reconstructed_edges

    # Create graph and add all edges
    G = nx.Graph()
    G.add_nodes_from(range(adj_original.shape[0]))
    G.add_edges_from(unchanged_edges)
    G.add_edges_from(added_edges)
    G.add_edges_from(removed_edges)

    # Positioning (layout)
    plt.figure(figsize=(4, 4), facecolor='white')
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)

    # Draw edges by category
    nx.draw_networkx_edges(G, pos, edgelist=list(unchanged_edges), edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=list(added_edges), edge_color='green', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=list(removed_edges), edge_color='red', width=2)

    # Draw labels
    if node_labels is not None:
        mapping = {i: label for i, label in enumerate(node_labels)}
        nx.draw_networkx_labels(G, pos, labels=mapping, font_size=10)
    else:
        nx.draw_networkx_labels(G, pos, font_size=10)

    # Title and legend-like explanation
    plt.title("Graph Reconstruction Differences")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("Plots/graph_reconstruction_diff.jpg", dpi=300)
    
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




def adj_bce_loss(pred_adj, true_adj, mask=None):
    if mask is None:
        N = pred_adj.shape[0]
        triu_indices = torch.triu_indices(N, N, offset=1)
        pred_flat = pred_adj[triu_indices[0], triu_indices[1]]
        true_flat = true_adj[triu_indices[0], triu_indices[1]]
    else:
        pred_flat = pred_adj[mask]
        true_flat = true_adj[mask]

    bce_loss = nn.BCELoss()
    return bce_loss(pred_flat, true_flat)



def compute_confusion_matrix(pred_grn, ref_grn, all_genes):
    # Generate all possible edges (upper triangle only)
    from itertools import combinations
    all_possible_edges = set(tuple(sorted(edge)) for edge in combinations(all_genes, 2))

    tp = len(pred_grn & ref_grn)
    fp = len(pred_grn - ref_grn)
    fn = len(ref_grn - pred_grn)
    tn = len(all_possible_edges - (pred_grn | ref_grn))

    # Calculate F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 100 * 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return np.array([[tn, fp], [fn, tp]], dtype=int), f1_score


# Compute pairwise Jaccard Index
def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def load_edge_set(csv_path):
    df = pd.read_csv(csv_path)
    edge_set = set(tuple(sorted([row['Gene1'], row['Gene2']])) for _, row in df.iterrows())
    return edge_set