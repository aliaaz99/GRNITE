# Data/

One subfolder per dataset, grouped by benchmark family. The folder name after
the last `/` is the dataset name used throughout the pipeline (e.g. `hESC`).

```
Data/
├── TF500/hESC/                 # BEELINE TF+500 example
│   ├── ExpressionData.csv      # REQUIRED — genes × cells expression matrix
│   ├── refNetwork.csv          # OPTIONAL — ground-truth GRN (needed for evaluation)
│   └── hESC-<teacher>.csv      # teacher GRN(s) to enhance (one per method)
├── TF1000/hESC/
├── GG/PBMC-ALL-Human/          # GRouNdGAN simulated example
└── real_data/Cd81_h/           # biological single-cell example
```

Teacher GRN filenames follow the pattern `<dataset>-<pattern>.csv`:

| Method      | Expected teacher GRN file        |
| ----------- | -------------------------------- |
| celloracle  | `<dataset>-celloracle-whole.csv` |
| scenic      | `<dataset>-scenic-network.csv`   |
| grnboost    | `<dataset>-grnboost.csv`         |
| portia      | `<dataset>-portia.csv`           |
| dazzle      | `<dataset>-dazzle-full_filtered.csv` |
| correlation | `<dataset>-correlation_thresh.csv`   |

All GRN/reference CSV files must contain `Gene1` and `Gene2` columns.
