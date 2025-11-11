# Example for running GRNITE on TF500 dataset
echo "Running GRNITE on TF500 hESC dataset..."
python -u Main.py --dataPath TF500/hESC --step 1 --name grnite > logs/TF500_hESC_step1.log
echo "Step 1 completed."

# Example for running GRNITE on PBMC-ALL-Human dataset from GroundGAN
echo "Running GRNITE on PBMC-ALL-Human dataset..."
python -u Main.py --dataPath GroundGAN/PBMC-ALL-Human --step 1 --name grnite > logs/PBMC-ALL-Human_step1.log
echo "Step 1 completed."