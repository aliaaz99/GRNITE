#!/usr/bin/env bash
# =============================================================================
# GRNITE end-to-end pipeline
#
#   Stage 1: stage1_lora.py  — contrastive LoRA refinement of gene embeddings
#   Stage 2: stage2_gnn.py    — GNN edge prediction against a teacher GRN
#   Stage 3: run_eval.py      — evaluation against the ground-truth GRN
#
# Configure the variables in the CONFIGURATION block below, then run:
#
#   bash run_pipeline.sh
#
# Ablation experiments (Stage 1 / Stage 2 random-feature variants) are included
# below but COMMENTED OUT. Uncomment the marked blocks if you want to run them.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------
# CONFIGURATION — edit here
# -----------------------------------------------------------------------
DATASET="TF500/hESC"                         # path under Data/
METHODS="celloracle grnboost scenic portia"  # teacher GRN methods to enhance
GPU=0
SEED=0

# Stage 1 (stage1_lora.py) settings
LORA_RANK=128
LORA_EPOCHS=100              # 100 for main results; reduce for quick testing
EMBEDDING_SOURCE="Qwen"     # "genePT" | "Qwen" | "random"

# Stage 2 (stage2_gnn.py) settings
GNN_EPOCHS=500
LAMBDA_EMB=5.0
NEG_MULT=1.0

# -----------------------------------------------------------------------
# Derived variables
# -----------------------------------------------------------------------
DATA_NAME=$(basename "$DATASET")
DATA_ROOT="Data/${DATASET}"
LORA_EMBEDDING_SOURCE="$EMBEDDING_SOURCE"

# -----------------------------------------------------------------------
# Stage 1: LoRA contrastive refinement of gene embeddings
# -----------------------------------------------------------------------
echo "============================================================"
echo "Stage 1: LoRA refinement — embedding_source=${EMBEDDING_SOURCE}"
echo "============================================================"
python -u stage1_lora.py \
    --data-path "$DATASET" \
    --embedding-source "$LORA_EMBEDDING_SOURCE" \
    --lora-rank "$LORA_RANK" \
    --num-epoch "$LORA_EPOCHS" \
    --gpu "$GPU" \
    --seed "$SEED" \
    --save-embedding

# # ---------------------------------------------------------------------
# # [ABLATION] Stage 1 — random-initialized embedding baseline
# # Uncomment to train a LoRA projector from random Gaussian features
# # (same vocabulary as EMBEDDING_SOURCE) instead of text embeddings.
# # ---------------------------------------------------------------------
# echo "============================================================"
# echo "[ABLATION] Stage 1: random embedding baseline"
# echo "============================================================"
# python -u stage1_lora.py \
#     --data-path "$DATASET" \
#     --embedding-source "random" \
#     --gene-source "$EMBEDDING_SOURCE" \
#     --lora-rank "$LORA_RANK" \
#     --num-epoch "$LORA_EPOCHS" \
#     --gpu "$GPU" \
#     --seed "$SEED" \
#     --save-embedding

# -----------------------------------------------------------------------
# Stage 2: GNN training for each teacher method
#   text/LoRA features  → suffix: grnite           (main result)
#   random features     → suffix: grnite_ablation  (ablation)
# -----------------------------------------------------------------------
echo "============================================================"
echo "Stage 2: GNN training"
echo "============================================================"

for METHOD in $METHODS; do
    echo "--- Teacher: ${METHOD} | embedding: ${EMBEDDING_SOURCE} ---"
    python -u stage2_gnn.py \
        --dataPath "$DATASET" \
        --targetMethod "$METHOD" \
        --embedding_source "$EMBEDDING_SOURCE" \
        --lambda_emb "$LAMBDA_EMB" \
        --neg_multiplier "$NEG_MULT" \
        --num_epoch "$GNN_EPOCHS" \
        --gpu "$GPU" \
        --seed "$SEED" \
        --name grnite

    # # [ABLATION] Stage 2 — random node features (uncomment to run)
    # echo "--- Teacher: ${METHOD} | embedding: random (ablation) ---"
    # python -u stage2_gnn.py \
    #     --dataPath "$DATASET" \
    #     --targetMethod "$METHOD" \
    #     --embedding_source random \
    #     --gene_source "$EMBEDDING_SOURCE" \
    #     --lambda_emb "$LAMBDA_EMB" \
    #     --neg_multiplier "$NEG_MULT" \
    #     --num_epoch "$GNN_EPOCHS" \
    #     --gpu "$GPU" \
    #     --seed "$SEED" \
    #     --name grnite_ablation
done

# -----------------------------------------------------------------------
# Stage 3: Evaluation against the ground-truth GRN
#   Requires refNetwork.csv in the dataset folder.
#   Evaluates each teacher baseline together with its GRNITE-enhanced graph.
# -----------------------------------------------------------------------
echo "============================================================"
echo "Stage 3: Evaluation"
echo "============================================================"

# Map short method names to the GRN filename patterns produced above.
METHOD_GRN_MAP=(
    "celloracle:celloracle-whole"
    "scenic:scenic-network"
    "grnboost:grnboost"
    "portia:portia"
    "dazzle:dazzle-full_filtered"
    "correlation:correlation_thresh"
    "bio:celloracle-base"
)

EVAL_METHODS=""
for METHOD in $METHODS; do
    GRN_PATTERN=""
    for MAPPING in "${METHOD_GRN_MAP[@]}"; do
        KEY="${MAPPING%%:*}"
        VAL="${MAPPING##*:}"
        if [ "$KEY" = "$METHOD" ]; then
            GRN_PATTERN="$VAL"
            break
        fi
    done
    [ -z "$GRN_PATTERN" ] && GRN_PATTERN="$METHOD"

    # Teacher baseline
    [ -n "$EVAL_METHODS" ] && EVAL_METHODS="${EVAL_METHODS},"
    EVAL_METHODS="${EVAL_METHODS}${GRN_PATTERN}"

    # GRNITE-enhanced graph (main result)
    EVAL_METHODS="${EVAL_METHODS},${EMBEDDING_SOURCE}-${GRN_PATTERN}_grnite"

    # # [ABLATION] GRNITE with random features — uncomment if Stage 2 ablation was run
    # EVAL_METHODS="${EVAL_METHODS},random_${EMBEDDING_SOURCE}-${GRN_PATTERN}_grnite_ablation"
done

python -u run_eval.py \
    --dataset "$DATASET" \
    --embedding_source "$EMBEDDING_SOURCE" \
    --methods "$EVAL_METHODS" \
    --preprocessing_metrics_text "${DATA_ROOT}/preprocessing_metrics_${LORA_EMBEDDING_SOURCE}.json" \
    --output "${DATA_ROOT}/${DATA_NAME}_pipeline_results_${EMBEDDING_SOURCE}.xlsx"
    # # [ABLATION] also fold random preprocessing metrics into the eval workbook:
    # --preprocessing_metrics_random "${DATA_ROOT}/preprocessing_metrics_random.json"

echo "============================================================"
echo "Done. Results: ${DATA_ROOT}/${DATA_NAME}_pipeline_results_${EMBEDDING_SOURCE}.xlsx"
echo "============================================================"
