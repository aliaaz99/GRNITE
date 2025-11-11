from utils import *
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', type=str, default='TF500/hESC', help='Path to the refNetwork and Expression matrix')
parser.add_argument('--step', type=int, default=1, help='Step 1 or 2 of GRNITE, Step1 for getting prior graph, Step 2 for improving existing GRN method')
parser.add_argument('--targetMethod', type=str, default='celloracle', help='Target GRN method name to improve (if running Step 2)')
parser.add_argument('--n_low', type=int, default=100, help='Number of dimensions to reduce to in X_sample_reduced, default is 100')
parser.add_argument('--neg_multiplier', type=float, default=1.0, help='Ratio of negative samples to positive samples, default is 1')
parser.add_argument('--gnn_dim_hidden', type=str, default="32,32", help='hidden units per layer for GNN encoder, comma separated')
parser.add_argument('--name', type=str, default='grnite', help='name added to the end of saved graphs')
parser.add_argument('--sample', type=int, default=None, help='used for small subsamples of GroundGAN datasets')
parser.add_argument('--beta', type=float, default=0.5, help='weight for target loss vs prior loss in step 2, default is 0.5')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for GNN model, default is 0.01')
parser.add_argument('--num_epoch', type=int, default=5000, help='number of training epochs, default is 5000')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use, default is 0')


# Names and paths
args = parser.parse_args()
dataPath = args.dataPath
dataName = dataPath.split('/')[-1]
print("Data path: ", dataPath)
print("Data name: ", dataName)
species = 'human' if dataName in ['PBMC-ALL-Human', 'PBMC_CTL-Human', 'Tumor-ALL', 'Tumor-malignant', 'hESC', 'hHep'] else 'mouse'
step = args.step
n_low = args.n_low
neg_multiplier = args.neg_multiplier
targetMethod = args.targetMethod
subsample = args.sample
gnn_dim_hidden = [int(x) for x in args.gnn_dim_hidden.split(',')]
beta = args.beta
lr = args.lr
num_epoch = args.num_epoch
gpu = args.gpu
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load the prior text embeddings
emb_name_column = 'Qwen3-Embedding-8B'
genePTPath = f"Gene_emebddings/{species}_embeds-{emb_name_column}.h5"
# Load HDF file and construct dict
data_h5 = pd.read_hdf(genePTPath)
gene_names_all = data_h5['Symbol'].values
data_emb = {
    row['Symbol']: np.array(row[emb_name_column])
    for _, row in data_h5.iterrows()
}
print("Number of all genes for text-emb: ", len(gene_names_all))



# Load gene experssion data
data_path_sample = 'Data/' + dataPath + '/ExpressionData.csv'
if subsample is not None:
    # range is from 2000*(sample-1) to 2000*sample, [0] is for gene names
    cell_range = cell_range = [0] + list(range(2000 * (subsample - 1), 2000 * subsample))
    data_sample = pd.read_csv(data_path_sample, header=0, index_col=0, usecols=cell_range).T 
else:
    data_sample = pd.read_csv(data_path_sample, header=0, index_col=0).T
data_sample = data_sample.transform(lambda x: np.log(x + 1)) 
print("Reading data completed!")
num_cells_sample, num_genes_sample = data_sample.shape
print(f"Number of cells: {num_cells_sample}, Number of genes: {num_genes_sample}")
gene_names_sample = list(data_sample.columns)

# Function to get the present genes and their embeddings
# X_sample_emb: the gene embeddings from PT for the genes present in the sample
present_genes, X_sample_emb = get_present(gene_names_sample, gene_names_all, data_emb)
N = len(present_genes)
print(f"Number of present genes: {N}")
non_present_genes = set(gene_names_sample) - set(present_genes)
# save the non-present genes to a file
non_present_genes_path = 'Data/' + dataPath + '/non_present_genes.txt'
with open(non_present_genes_path, 'w') as f:
    f.write("Genes present in expression data but not in text embeddings:\n")
    for gene in non_present_genes:
        f.write(gene + '\n')

# get the expression of the present genes
present_expression = data_sample[present_genes].T.values
X_sample = np.array(present_expression)
print("X_sample shape: ", X_sample.shape)


# load the bio base GRN or compute it if not exists:
start_time1 = time.time()
data_path_bio_adj = 'Data/' + dataPath + '/bio_adj-' + emb_name_column + '.pickle'
data_path_bio_edges = 'Data/' + dataPath + '/bio_grn.csv'
bioExists = os.path.exists(data_path_bio_adj)
if bioExists:
    print("Loading celloracle adjacency matrix as bio prior from pickle file...")
    with open(data_path_bio_adj, 'rb') as f:
        A_bio = pickle.load(f)
    bio_grn = load_edge_set(data_path_bio_edges)
    print("Celloracle adjacency matrix loaded successfully!")
else:
    data_path_bio = 'Gene_emebddings/celloracle_baseGRN.csv' if species=='human' else 'Gene_emebddings/celloracle_mouse_baseGRN.csv'
    bio_grn = load_edge_set(data_path_bio)
    print("Number of edges in base celloracle: ", len(bio_grn))
    bio_grn = set((g1, g2) for g1, g2 in bio_grn if g1 in present_genes and g2 in present_genes)
    print("Number of edges in the base celloracle GRN with present genes: ", len(bio_grn))
    pd.DataFrame(list(bio_grn), columns=['Gene1', 'Gene2']).to_csv(data_path_bio_edges, index=False)
    A_bio = csv2A(bio_grn, present_genes)
    # save the celloracle adjacency matrix using pickle:
    with open(data_path_bio_adj, 'wb') as f:
        pickle.dump(A_bio, f)
end_time1 = time.time()
all_time1 = end_time1 - start_time1

# text distance, correlation:
D_emb = cdist(X_sample_emb, X_sample_emb, metric='cosine')
np.fill_diagonal(D_emb, 0)
D_emb[D_emb < 0] = 0
A_emb = get_adjacency_matrix(D_emb, k=None)


# save correlation graph with weights if needed as target:
if targetMethod == 'correlation':
    # correlation matrix and adjacency
    Corr_matrix = wgcna_grn(X_sample, beta=1)
    Corr_dist = 1 - Corr_matrix
    A_corr = get_adjacency_matrix(Corr_dist, threshold=None)
    corr_values = Corr_matrix[np.triu_indices(N, k=1)]
    i_corr, j_corr = np.triu_indices_from(A_corr, k=1)
    nonzero_mask_corr = A_corr[i_corr, j_corr] != 0
    i_corr = i_corr[nonzero_mask_corr]
    j_corr = j_corr[nonzero_mask_corr]
    edges_corr = [(present_genes[src], present_genes[tgt]) for src, tgt in zip(i_corr, j_corr)]
    weights_corr = Corr_matrix[i_corr, j_corr]  # extract the correlation weights
    edges_corr_df = pd.DataFrame(edges_corr, columns=['Gene1', 'Gene2'])
    edges_corr_df['weight'] = weights_corr
    if subsample is not None:
        edges_corr_df.to_csv('Data/target/' + dataName + f'-corr_edges-{subsample}.csv', index=False)
    else:
        edges_corr_df.to_csv('Data/target/' + dataName + '-corr_edges.csv', index=False)
    print("Corr edges saved!")


# Prepare the GNN data:
# Choose the initial embeddings for nodes based on the step
# Use gene expression features reduced by TruncatedSVD for step 1
# Use random Gaussian features for step 2
if step==1:
    svd = TruncatedSVD(n_components=n_low, random_state=42)
    X_sample_low = svd.fit_transform(X_sample)
    print("TruncatedSVD reduced feature shape: ", X_sample_low.shape)
    scaler = StandardScaler()
    X_sample_reduced = scaler.fit_transform(X_sample_low)
    x = torch.Tensor(X_sample_reduced) # gene expression features
elif step==2:
    num_nodes = A_emb.shape[0]
    x = torch.randn((num_nodes, n_low)) # random guassian features

    
# Get the target graph for decoder:
if step ==1:
    adj_target = torch.FloatTensor(A_bio).to(device)
    target_grn = bio_grn
elif step==2:
    # Define method-specific filename patterns
    patterns = {
        'celloracle': 'celloracle-whole',
        'scenic': 'scenic-network',
        'grnboost': 'grnboost',
        'portia': 'portia',
        'deeprig': 'celloracle-deeprig_filtered',
        'correlation': 'corr_edges',
        'dazzle': 'dazzle-full_filtered'
    }

    if targetMethod not in patterns:
        raise ValueError(f"Unknown target method for step 2: {targetMethod}")

    base = patterns[targetMethod]

    # Handle subsample logic
    if subsample is not None:
        if targetMethod in ['deeprig', 'dazzle']: # these methods have '_filtered' suffix because edges are filtered
            data_path_method = f"Data/{dataPath}/{dataName}-{base.replace('_filtered', '')}-sample-{subsample}_filtered.csv"
        else:
            data_path_method = f"Data/{dataPath}/{dataName}-{base}-sample-{subsample}.csv"
    else:
        data_path_method = f"Data/{dataPath}/{dataName}-{base}.csv"
    target_grn = load_edge_set(data_path_method)
    target_grn = set((g1, g2) for g1, g2 in target_grn if g1 in present_genes and g2 in present_genes)
    print("Number of edges in the target GRN with present genes: ", len(target_grn))
    A_method = csv2A(target_grn, present_genes)
    adj_target = torch.FloatTensor(A_method).to(device)


# Get the underlying (source) graph:
if step==1:
    A_emb_sparse = sp.csr_matrix(A_emb)
    edge_index_emb, _ = from_scipy_sparse_matrix(A_emb_sparse)
    edge_index_gnn = edge_index_emb
    A_gnn = torch.FloatTensor(A_emb).to(device)
elif step==2:
    # load the step1_grnite graph if exists which is the output of step 1:
    data_path_prior = 'Data/' + dataPath + '/' + dataName + '-step1_grnite.csv'
    cell_plus_exists = os.path.exists(data_path_prior)
    if cell_plus_exists:
        print("Loading step 1 prio graph from file...")
        bio_grn_plus = load_edge_set(data_path_prior)
        A_cell_plus = csv2A(bio_grn_plus, present_genes)
        A_cell_plus_sparse = sp.csr_matrix(A_cell_plus)
        edge_index_celloracle_plus, _ = from_scipy_sparse_matrix(A_cell_plus_sparse)
        edge_index_gnn = edge_index_celloracle_plus
        A_gnn = torch.FloatTensor(A_cell_plus).to(device)
    else:
        print("step 1 prior graph not found, using text dist graph instead.")
        A_emb_sparse = sp.csr_matrix(A_emb)
        edge_index_emb, _ = from_scipy_sparse_matrix(A_emb_sparse)
        edge_index_gnn = edge_index_emb
        A_gnn = torch.FloatTensor(A_emb).to(device)



start_time2 = time.time()
data = Data(x=x, edge_index=edge_index_gnn).to(device)
model = GraphAutoEncoder(in_channels=data.x.shape[1], hidden_channels_list=gnn_dim_hidden[:-1], out_channels=gnn_dim_hidden[-1]).to(device)
print("Model structure:")
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.train()
loss_min, epoch_min = float('inf'), 0
best_state = None
for epoch in range(0, num_epoch+1):
    optimizer.zero_grad()
    mask_posneg = create_mask_from_adj(adj_target, neg_multiplier=neg_multiplier, seed = epoch) # negative sampling
    adj_pred = model(data.x, data.edge_index)
    loss1 = adj_bce_loss(adj_pred, adj_target, mask=mask_posneg, pos_weight=None) # loss for target graph
    if step==1:
        loss = loss1
    elif step==2:
        loss2 = adj_bce_loss(adj_pred, A_gnn, mask=mask_posneg, pos_weight=None) # loss for prior graph
        loss = beta * loss1 + (1-beta) * loss2
    if loss.item() < loss_min:
        loss_min = loss.item()
        epoch_min = epoch
        best_state = copy.deepcopy(model.state_dict())
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Best Loss: {loss_min:.4f}, Best Epoch: {epoch_min}")


# Evaluate the model compared to celloracle graph:.
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    adj_pred_eval = torch.sigmoid(model(data.x, data.edge_index))
adj_pred_eval = adj_pred_eval * (1 - torch.eye(adj_pred_eval.size(0)).to(adj_pred_eval.device))
N = adj_pred_eval.shape[0]
triu_indices = torch.triu_indices(N, N, offset=1)
adj_pred_eval[triu_indices[1], triu_indices[0]] = adj_pred_eval[triu_indices[0], triu_indices[1]]
adj_true_flat = adj_target[triu_indices[0], triu_indices[1]]
adj_pred_flat = adj_pred_eval[triu_indices[0], triu_indices[1]]
threshold = 0.50
adj_pred_eval_binary = (adj_pred_eval > threshold).float()
adj_pred_binary_flat = adj_pred_eval_binary[triu_indices[0], triu_indices[1]]

end_time2 = time.time()
all_time2 = end_time2 - start_time2

all_time = all_time1 + all_time2
print("Step 1 time (preprocessing) in seconds: {:.2f}".format(all_time1))
print("Step 2 (training and evaluation) in seconds: {:.2f}".format(all_time2))
print(f"Total training and evaluation time in seconds: {all_time:.2f}")


# Convert predicted adjacency to numpy and save csv
infered_grn = adj_pred_eval_binary.cpu().numpy()
i, j = np.triu_indices_from(infered_grn, k=1)
nonzero_mask = infered_grn[i, j] != 0
i, j = i[nonzero_mask], j[nonzero_mask]
edges = [(present_genes[src], present_genes[tgt]) for src, tgt in zip(i, j)]
weights = adj_pred_eval[i, j].cpu().numpy()
edges_df = pd.DataFrame({
    'Gene1': [e[0] for e in edges],
    'Gene2': [e[1] for e in edges],
    'weight': weights
})
edges_df_rev = edges_df.rename(columns={'Gene1': 'Gene2', 'Gene2': 'Gene1'})
edges_df_final = pd.concat([edges_df, edges_df_rev], ignore_index=True)


if step==1:
    caseName = f"{dataName}-step1"
elif step==2:
    grn_mapping = {
        'celloracle': '-celloracle-whole',
        'scenic': '-scenic-network',
        'portia': '-portia',
        'grnboost': '-grnboost',
        'deeprig': '-celloracle-deeprig',
        'dazzle': '-dazzle-full',
        'correlation': '-corr_edges'
    }
    caseName = f"{dataName}{grn_mapping.get(targetMethod, '-unknown')}"
else:
    caseName = 'unknown'

if subsample is not None:
    edges_df_final.to_csv(f'Data/{dataPath}/{caseName}-sample-{subsample}_{args.name}.csv', index=False)
else:
    edges_df_final.to_csv(f'Data/{dataPath}/{caseName}_{args.name}.csv', index=False)




# (Optional) Get the subset of the reference GRN with only present genes:
data_path_true = 'Data/' + dataPath + '/refNetwork.csv'
ref_grn_un = load_edge_set(data_path_true, undirected=False)
present_genes_upper = set(g.upper() for g in present_genes)
ref_grn_present = [(g1, g2) for g1, g2 in ref_grn_un if g1 in present_genes_upper and g2 in present_genes_upper]
ref_grn_present_path = 'Data/' + dataPath + '/' + dataName + '-ref_present.csv'
ref_present_df = pd.DataFrame(list(ref_grn_present), columns=['Gene1', 'Gene2'])
ref_present_df.to_csv(ref_grn_present_path, index=False)
