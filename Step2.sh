# Example for running GRNITE on TF500 dataset
echo "Running GRNITE on TF500 hESC dataset..."
python -u Main.py --dataPath TF500/hESC --step 2 --name grnite --targetMethod scenic > logs/TF500_hESC_step2_scenic.log
echo "Step 2 with scenic completed."
python -u Main.py --dataPath TF500/hESC --step 2 --name grnite --targetMethod grnboost > logs/TF500_hESC_step2_grnboost.log
echo "Step 2 with GRNBoost completed."


# Example for running GRNITE on PBMC-ALL-Human dataset from GroundGAN
echo "Running GRNITE on PBMC-ALL-Human dataset..."
python -u Main.py --dataPath GroundGAN/PBMC-ALL-Human --step 2 --name grnite --targetMethod grnboost > logs/PBMC-ALL-Human_step2_grnboost.log
echo "Step 2 with GRNBoost completed."
python -u Main.py --dataPath GroundGAN/PBMC-ALL-Human --step 2 --name grnite --targetMethod scenic > logs/PBMC-ALL-Human_step2_scenic.log
echo "Step 2 with scenic completed."
