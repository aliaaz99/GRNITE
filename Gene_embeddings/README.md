# Gene_embeddings/

Place the downloaded shared resource files here. These are **not** tracked in
git because of their size.

Download the four files from Zenodo:
https://zenodo.org/records/17705020

Expected contents:

```
Gene_embeddings/
├── human_embeds-Qwen3-Embedding-8B.h5    # human text embeddings (Qwen3-Embedding-8B)
├── mouse_embeds-Qwen3-Embedding-8B.h5    # mouse text embeddings (Qwen3-Embedding-8B)
├── celloracle_baseGRN.csv                # CellOracle base GRN (human), used as biological prior
└── celloracle_mouse_baseGRN.csv          # CellOracle base GRN (mouse)
```

If you use GenePT embeddings instead of Qwen, also place
`GenePT_gene_embedding_ada_text.pickle` here.
