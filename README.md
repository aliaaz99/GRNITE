# GRN

### Inputs

- **V** ∈ ℝ<sup>G×F</sup>  
  - *G*: Number of available genes from **genePT**  
  - *F*: Dimension of embeddings  

- **X** ∈ ℝ<sup>L×N</sup>  
  - *L*: Number of cells in the sample  
  - *N*: Number of genes in the sample  

### Objective

- Leverage the prior knowledge encoded in **V** to infer a more accurate gene regulatory network (GRN) from the observed gene expression data **X**.
- **Output**: 𝒢 = (**𝒱**, **A**), where  
  - |𝒱| = **N**  
  - **A** ∈ {0,1}<sup>N×N</sup>
