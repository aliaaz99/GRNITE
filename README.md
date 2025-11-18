# GRNITE

This repository contains all the necessary code and commands for "GRNITE: Gene Regulatory Network Inference with Text Embeddings."

In order to improve the quality of GRN inference and enable more comprehensive exploratory analyses of GRNs across various phenotypes of interest we developed a two-stage meta-method called GRNITE. In the first step, GRNITE leverages LLM-based embeddings of plain text gene descriptions to create a prior gene interaction graph which is then optimized with a graph neural network (GNN) to achieve a **universal** biological prior for GRN inference. In the second step, GRNITE uses a GNN to incorporate information from a GRN inferred from scRNA-seq data with any baseline inference method into our prior. The result of this two-step approach is a near-universal improvement in AUROC and recall of all evaluated methods, with minor trade-offs in precision. Furthermore, GRNITE is a lightweight meta-method, which adds minimal amount of extra compute time on top of the original GRN inference performed.

<img width="7386" height="2796" alt="grn-inference-1-long" src="https://github.com/user-attachments/assets/64b4e60c-56e5-4eb6-820d-f7d3a9ade033" />


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

You need to place your data files in the respective folders inside the `Data` directory. An example is provided above for `TF500/hESC` and `GroundGAN/PBMC-ALL-Human` dataset.

You can download the four files in the `Gene_embeddings` folder, which include the human and mouse text embeddings as well as the base CellOracle graph, from this [link](https://rice.box.com/s/qhpgkp0813zhfhjy2hmi04x2asdmaalv).

2. **Generating the results:**

   You can run `Main.py` to execute GRNITE.  
   To do so:

   - If you are running the **first step** to generate the prior graph, specify:
     ```bash
     python Main.py --step 1
     ```
   - If you have already completed the first step and want to **enhance the GRN** of another method, run:
     ```bash
     python Main.py --step 2 --targetMethod $NAME
     ```
     where `$NAME` can be one of
     ```bash
     {scenic, grnboost, celloracle, portia, correlation, deeprig, dazzle}.
     ```
    **Note:** You must include the target GRN file of the method you wish to enhance in the corresponding data folder.
    Below is the mapping between each method name and the expected GRN file name:
   
    | Method       | Expected GRN File Name                       |
    | ------------ | -------------------------------------------- |
    | `scenic`     | `{dataName}-scenic-network.csv`              |
    | `grnboost`   | `{dataName}-grnboost.csv`                    |
    | `celloracle` | `{dataName}-celloracle-whole.csv`            |
    | `portia`     | `{dataName}-portia.csv`                      |
    | `deeprig`    | `{dataName}-celloracle-deeprig_filtered.csv` |
    | `dazzle`     | `{dataName}-dazzle-full_filtered.csv`        |


   You can also run `Eval.py` to evaluate the GRNs you have generated against a reference network:
   ```bash
   python Eval.py
   ```
  Ensure that a file named `refNetwork.csv` is present in the data folder of each dataset to serve as the reference GRN if you want to perform evaluation.

  **Note:** All GRN files should contain columns named `Gene1` and `Gene2`.

  
3. **Parameters**

   You can modify the following parameters when running `Main.py` to adjust GRNITE’s behavior and training configuration.

| Parameter | Default Value | Description |
|------------|----------------|--------------|
| `--dataPath` | `'TF500/hESC'` | Path to the expression matrix and reference network. |
| `--step` | `1` | Specifies which stage of GRNITE to run: Step 1 for generating the prior graph, Step 2 for enhancing an existing GRN method. |
| `--targetMethod` | `'celloracle'` | Name of the GRN method to improve (used only in Step 2). |
| `--n_low` | `100` | Number of dimensions for dimensionality reduction of the Expression matrix. |
| `--neg_multiplier` | `1` | Ratio of negative samples to positive samples during training. |
| `--gnn_dim_hidden` | `"32,32"` | Hidden layer sizes for the GNN encoder (comma-separated). |
| `--name` | `'grnite'` | Suffix added to the names of saved graph files. |
| `--sample` | `None` | Index for subsampling GroundGAN datasets (uses cell indices from `2000 × (sample – 1)` to `2000 × sample`). |
| `--beta` | `0.5` | Weight balancing target loss vs. prior loss in Step 2. |
| `--lr` | `0.01` | Learning rate for training the GNN model. |
| `--num_epoch` | `5000` | Number of training epochs. |
| `--gpu` | `0` | GPU ID to use (default: 0). |

## Example Usage

1. **Download the data**

Download the expression matrix for the `PBMC-ALL-Human` dataset from the [GroundGAN Benchmarking page](https://emad-combine-lab.github.io/GRouNdGAN/benchmarking).  
Place the file at: ```Data/GroundGAN/PBMC-ALL-Human/ExpressionData.csv``` as described in the setup section above.
The `ExpressionData.csv` for `TF500/hESC` is included here.

2. **Run Step 1 – Prior graph generation**

Execute the following bash script to run the first step of GRNITE, which constructs the prior graph for both datasets: `TF500/hESC` and `Data/GroundGAN/PBMC-ALL-Human`.

```bash
bash Step1.sh > logs/example_step_1.log
```

3. **Run Step 2 – Enhanced GRN generation**

The output GRNs from **SCENIC** and **GRNBoost** are already included in the corresponding folders with `{dataname}-scenic-network.csv` and `{dataname}-grnboost.csv` names.
You can now run the next script to generate the enhanced GRNs using GRNITE:

  ```sh
  bash Step2.sh > logs/example_step_2.log
  ```

4. **Evaluate the results**

Finally, evaluate the base GRNs (SCENIC and GRNBoost) along with their GRNITE-enhanced versions using:

```sh
python Eval.py > logs/example_eval.log
```

This will generate `Example_Eval.xlsx`, which contains different evaluation metrics for each method and dataset. Each sheet in the Excel file corresponds to a separate dataset.

# Citation
If you use GRNITE in your research, please cite the following paper:

