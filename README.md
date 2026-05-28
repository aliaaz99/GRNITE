# GRNITE

This repository contains the code and commands for **"GRNITE: Gene Regulatory Network Inference with Text Embeddings."**

## Overview

GRNITE improves gene regulatory network (GRN) inference from single-cell RNA-seq data. Rather than being another standalone inference algorithm, it is a lightweight **meta-method** that takes the output of *any* existing method and enhances it by combining LLM-based text descriptions of genes, curated biological prior knowledge, and the chosen method's data-driven GRN. This yields consistent gains across methods with minimal extra compute.

The pipeline runs in three stages: **Stage 1** (`stage1_lora.py`) refines text-based gene embeddings, **Stage 2** (`stage2_gnn.py`) uses a graph neural network to combine those embeddings with a "teacher" GRN, and **Stage 3** (`run_eval.py`) evaluates the results against a ground-truth network.

### Overview figure

<!-- TODO: replace the placeholder below with the updated GRNITE overview figure. -->
<img width="7386" height="2796" alt="GRNITE overview (placeholder — updated figure to be added)" src="https://github.com/user-attachments/assets/64b4e60c-56e5-4eb6-820d-f7d3a9ade033" />

> **Note:** The figure above is a placeholder showing the previous overview. The updated three-stage overview figure will be added here.

## Installation & Dependencies

The code runs on Unix-like operating systems (macOS, Linux).

You can build a Conda environment with all dependencies from the provided `environment.yml`. The environment must be activated for each usage.

```sh
conda env create -f environment.yml
conda activate GRNITE
```

Alternatively (or additionally), install the Python dependencies with `pip`:

```sh
pip install -r requirements.txt
```

> For a CUDA build of PyTorch, install the wheel matching your CUDA version from https://pytorch.org (the paper environment used `torch==2.5.1+cu121`).

## Repository structure

```
GRNITE/
├── run_pipeline.sh        # main entry point — runs the full pipeline end-to-end
├── stage1_lora.py         # Stage 1: contrastive LoRA refinement of gene embeddings
├── stage2_gnn.py          # Stage 2: GNN edge prediction against a teacher GRN
├── run_eval.py            # Stage 3: batch evaluation wrapper
├── Eval.py                # core AUROC/AUPRC/Jaccard evaluation routines
├── utils.py               # shared GNN modules and helpers
├── analyze_results.py     # summarize/visualize results (teacher- or method-based averages)
├── baseline_summary.py    # report average teacher-method performance
├── environment.yml        # Conda environment
├── requirements.txt       # pip dependencies
├── Gene_embeddings/       # downloaded text embeddings + base GRNs (see its README)
├── Data/                  # per-dataset inputs and generated outputs (see its README)
└── logs/                  # pipeline run logs
```

## Inputs

For each dataset you create a folder under `Data/`, e.g. `Data/TF500/hESC/`, containing:

- `ExpressionData.csv` — **required.** Genes × cells expression matrix.
- `refNetwork.csv` — *optional*, needed only for evaluation. The ground-truth GRN.
- `<dataset>-<teacher>.csv` — the teacher GRN(s) you want to enhance, one file per baseline method (e.g. `hESC-scenic-network.csv`, `hESC-grnboost.csv`).

All GRN and reference CSVs must contain `Gene1` and `Gene2` columns. See `Data/README.md` for the full teacher-method → filename mapping.

You also need the shared resources in `Gene_embeddings/` (text embeddings and the CellOracle base GRNs). Download the four files from [this link](https://zenodo.org/records/17705020) and place them in `Gene_embeddings/` as described in `Gene_embeddings/README.md`.

## Outputs

Running the pipeline produces, inside each dataset folder:

- The refined gene embeddings (`X_sample_lora_low_*.npy`) and the list of genes present in both the expression data and the embedding vocabulary (`present_genes_*.txt`).
- The GRNITE-enhanced GRN for each teacher method, as an edge list with weights (`<dataset>-<source>-<teacher>_grnite.csv`).
- Stage 1 preprocessing metrics (`preprocessing_metrics_<source>.json`) and diagnostic plots under `plots/`.
- An evaluation workbook (`<dataset>_pipeline_results_<source>.xlsx`) with a `GRN_Eval` sheet (per-method AUROC/AUPRC/Jaccard) and a `Preprocessing` sheet, when a reference network is available.

## Usage

The main entry point is **`run_pipeline.sh`**, which runs Stage 1 → Stage 2 → Stage 3 for the dataset and teacher methods you configure at the top of the file.

### Run the main pipeline

Edit the `CONFIGURATION` block in `run_pipeline.sh` (set `DATASET`, `METHODS`, `EMBEDDING_SOURCE`, etc.), then:

```sh
bash run_pipeline.sh > logs/example_run.log
```

You can also run the stages individually, for example:

```sh
# Stage 1 — refine embeddings for one dataset
python stage1_lora.py --data-path TF500/hESC --embedding-source Qwen --save-embedding

# Stage 2 — enhance a single teacher GRN (e.g. SCENIC)
python stage2_gnn.py --dataPath TF500/hESC --targetMethod scenic --embedding_source Qwen --name grnite

# Stage 3 — evaluate baselines and their GRNITE versions
python run_eval.py --dataset TF500/hESC --embedding_source Qwen \
    --methods scenic-network,Qwen-scenic-network_grnite \
    --output Data/TF500/hESC/hESC_pipeline_results_Qwen.xlsx
```

### (Optional) Run the ablation experiments

The two ablations from the paper — a **Stage 1** variant initialized from random Gaussian features instead of text embeddings, and a **Stage 2** variant that uses random node features — are included in `run_pipeline.sh` but **commented out**. To run them, open `run_pipeline.sh` and uncomment the blocks marked `[ABLATION]` (the random Stage 1 baseline, the random Stage 2 training loop, and the ablation lines in the Stage 3 evaluation). They can also be invoked directly:

```sh
# Stage 1 ablation — random-initialized embeddings (same gene vocabulary as Qwen)
python stage1_lora.py --data-path TF500/hESC --embedding-source random --gene-source Qwen --save-embedding

# Stage 2 ablation — random node features
python stage2_gnn.py --dataPath TF500/hESC --targetMethod scenic \
    --embedding_source random --gene_source Qwen --name grnite_ablation
```

## Analyzing and summarizing results

**`analyze_results.py`** visualizes the pipeline results. It compares each GRNITE-enhanced network against its teacher baseline (and, if present, the ablation), and reports the improvements both as **teacher-based averages** (mean improvement per teacher method, across datasets) and as **method/dataset-based averages** (mean improvement per dataset, across teachers). It writes summary tables (CSV/Markdown) and figures (heatmaps and bar charts) to an output directory.

```sh
python analyze_results.py --embedding_source Qwen --data_root Data/TF500
```

**`baseline_summary.py`** reports the average performance of the teacher methods themselves (CellOracle, GRNBoost, SCENIC, PORTIA, etc.), aggregated across datasets and dataset groups, as printed tables plus CSV/Excel files.

```sh
python baseline_summary.py --data_root Data
```

## Example data

An `ExpressionData.csv` for `TF500/hESC` can be used as a starting example. For the GRouNdGAN datasets, download the expression matrix (e.g. `PBMC-ALL-Human`) from the [GRouNdGAN benchmarking page](https://emad-combine-lab.github.io/GRouNdGAN/benchmarking) and place it at `Data/GG/PBMC-ALL-Human/ExpressionData.csv`.

## Citation

If you use GRNITE in your research, please cite:

```
GRNITE: Gene regulatory network inference with text embeddings.
Ali Azizpour, Narein Rao, Santiago Segarra, Luay Nakhleh, Nicolae Sapoval.
```
