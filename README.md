# GRNITE

This repository contains all the necessary code and commands for "GRNITE: Gene Regulatory Network Inference with Text Embeddings."

In order to improve the quality of GRN inference and enable more comprehensive exploratory analyses of GRNs across various phenotypes of interest we developed a two-stage meta-method called GRNITE. In the first step, GRNITE leverages LLM-based embeddings of plain text gene descriptions to create a prior gene interaction graph which is then optimized with a graph neural network (GNN) to achieve a **universal** biological prior for GRN inference. In the second step, GRNITE uses a GNN to incorporate information from a GRN inferred from scRNA-seq data with any baseline inference method into our prior. The result of this two-step approach is a near-universal improvement in AUROC and recall of all evaluated methods, with minor trade-offs in precision. Furthermore, GRNITE is a lightweight meta-method, which adds minimal amount of extra compute time on top of the original GRN inference performed.
<img width="7386" height="2796" alt="grn-inference-1-long" src="https://github.com/user-attachments/assets/14cb56d9-25aa-4436-bb1e-1c745ade174c" />


## Installation & Dependencies

The code is based on Python 3.7 and should run on Unix-like operating systems (MacOS, Linux).

To install the dependencies for GRNITE, you can use the `environment.yml` file provided to build a Conda environment with all necessary dependencies. 
The GRNITE environment will need to be activated for each usage. 

```sh
conda env create -f environment.yml
conda activate GRNITE
```

Additionally, you will need to install other packages using `pip` after creating and activating the environment.

```sh
pip install -r requirements.txt
```

  
## Running the Codes

1. **Folder Structure:**

You should have the following directory structure in the project folder:

```
├── Main.py
├── utils.py
├── Eval.py
├── Gene_emebddings
│ ├── human_embeds-Qwen3-Embedding-8B.h5
│ ├── mouse_embeds-Qwen3-Embedding-8B.h5
│ ├── celloracle_baseGRN.csv
│ ├── celloracle_mouse_baseGRN.csv
├── Data
│ ├── TF500/hESC
│ │ ├── ExpressionData.csv
│ │ ├── refNetwork.csv (Optional for evaluation)
│ │ ├── hESC-{MethodXName}.csv (Target graph obtained from method X)
│ ├── GroundGAN/PBMC-ALL-Human
│ │ ├── ExpressionData.csv
│ │ ├── refNetwork.csv (Optional for evaluation)
│ │ ├── PBMC-ALL-Human-{MethodXName}.csv (Target graph obtained from method X)
│ ├── (other folders for different cases)
```

You need to place your data files in the respective folders inside the `Data` directory. An example is provided above for `hESC` dataset.

2. **Generating the results:**


3. **Evaluation:**


## Output files:

GRNITE generates the following outputs:



## Example Usage

1. **Download the data**

Download the expression matrix for the `PBMC-ALL-Human` dataset from the [GroundGAN Benchmarking page](https://emad-combine-lab.github.io/GRouNdGAN/benchmarking).  
Place the file at: ```Data/GroundGAN/PBMC-ALL-Human/ExpressionData.csv``` as described in the setup section above.

2. **Run Step 1 – Prior graph generation**

Execute the following bash script to run the first step of GRNITE, which constructs the prior graph for both datasets: `TF500/hESC` and `Data/GroundGAN/PBMC-ALL-Human`.

```bash
bash Step1.sh > logs/example_1.log
```

3. **Run Step 2 – Enhanced GRN generation**

The output GRNs from SCENIC and GRNBoost are already included in the corresponding folders.
You can now run the next script to generate the enhanced GRNs using GRNITE:

  ```sh
  bash Step1.sh > logs/example_2.log
  ```

4. **Evaluate the results**

Finally, evaluate the base GRNs (SCENIC and GRNBoost) along with their GRNITE-enhanced versions using:

```sh
python Eval.py > logs/example_eval.log
```

This will generate `Example_Eval.xlsx`, which contains different evaluation metrics for each method and dataset. Each sheet in the Excel file corresponds to a separate dataset.

# Citation
If you use GRNITE in your research, please cite the following paper:

