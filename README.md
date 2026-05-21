# FCM-UMAP
# Flow Cytometry UMAP Analysis Pipeline

## Overview

This notebook contains a workflow for loading, preprocessing, pooling, and visualizing flow cytometry datasets using UMAP dimensionality reduction. The project focuses on comparing healthy and diseased leukemia samples through pooled embeddings, sample-level visualization, and clustering behavior.

The pipeline:

* Loads `.fcs.txt` flow cytometry datasets
* Associates metadata labels with each sample
* Pools healthy and diseased samples separately
* Performs preprocessing and numeric feature selection
* Runs UMAP dimensionality reduction
* Generates 2D visualizations of cellular distributions
* Supports large-scale datasets through subsampling

The notebook was designed for exploratory analysis of leukemia-related flow cytometry data and downstream visualization.

---

# Features

## Data Loading

* Reads flow cytometry `.fcs.txt` files from local directories
* Maps metadata from CSV label tables
* Tracks sample identity using a `sample_id` column
* Separates healthy vs diseased cohorts using labels

## Preprocessing

* Numeric feature extraction using pandas
* Optional subsampling to reduce computational load
* Type coercion for mixed-format datasets
* Pooling across samples for cohort-level analysis

## Dimensionality Reduction

Uses UMAP with configurable parameters:

* `n_neighbors=15`
* `min_dist=0.1`
* `random_state=42`

Outputs:

* `UMAP1`
* `UMAP2`

coordinates for downstream visualization.

## Visualization

Visualizations include:

* UMAP scatter plots
* Sample-colored embeddings
* Healthy vs diseased pooled projections
* Cluster exploration and comparison

Libraries used:

* matplotlib
* seaborn

---

# Project Structure

```text
UMAP.ipynb
├── Data loading
├── Metadata integration
├── Healthy/diseased separation
├── Pooling and subsampling
├── Numeric feature extraction
├── UMAP embedding generation
└── Visualization and plotting
```

---

# Requirements

## Python Packages

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn umap-learn scikit-learn
```

## Main Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
```

---

# Expected Data Format

## Input Files

The notebook expects:

* Flow cytometry text exports:

```text
*.fcs.txt
```

* Metadata CSV containing labels and file names

Example columns:

| Column      | Description                 |
| ----------- | --------------------------- |
| `File`      | Sample filename             |
| `Label`     | Class label                 |
| `sample_id` | Generated sample identifier |

### Label Definitions

| Label | Meaning  |
| ----- | -------- |
| `0`   | Healthy  |
| `1`   | Diseased |

---

# Example Workflow

## 1. Load Metadata

```python
noprop = pd.read_csv("CLL_test_noprop.csv")
```

## 2. Separate Healthy and Diseased Samples

```python
healthy_df = noprop[noprop["Label"] == 0]
diseased_df = noprop[noprop["Label"] == 1]
```

## 3. Pool Samples

```python
pooled_df = pd.concat(dfs, ignore_index=True)
```

## 4. Select Numeric Features

```python
X = pooled_df.select_dtypes(include=["number"])
```

## 5. Run UMAP

```python
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

embedding = reducer.fit_transform(X)
```

## 6. Visualize Embedding

```python
sns.scatterplot(
    data=pooled_df,
    x="UMAP1",
    y="UMAP2",
    hue="sample_id"
)
```

---

# Computational Considerations

Large pooled flow cytometry datasets can contain millions of cells. To reduce runtime and memory usage, the notebook includes:

* Per-sample cell caps
* Random subsampling
* Numeric-only feature selection
* Lightweight preprocessing steps

Example:

```python
per_sample_cap = 5000
```

This keeps computations tractable while preserving representative distributions.

---

# Outputs

The notebook generates:

* UMAP embeddings
* Healthy vs diseased visual comparisons
* Pooled cohort projections
* Sample-level clustering plots
* Exported metadata CSVs

Generated columns:

| Column  | Description                |
| ------- | -------------------------- |
| `UMAP1` | First embedding dimension  |
| `UMAP2` | Second embedding dimension |

---

# Notes

* Some paths in the notebook are hardcoded to local directories.
* Update file paths before running on a new system.
* Large datasets may require substantial RAM.
* Results can vary slightly across runs depending on random initialization.

---

# Potential Extensions

Possible future improvements:

* PCA preprocessing before UMAP
* Leiden clustering integration
* Batch normalization comparisons
* Stability analysis across repeated UMAP runs
* Statistical comparison of cluster distributions
* Interactive visualization dashboards

---

# Author

Developed for flow cytometry and leukemia visualization research involving dimensionality reduction and clustering analysis.
