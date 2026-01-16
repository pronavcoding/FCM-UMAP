import numpy as np
import pandas as pd
import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

sns.set(style="white", context="poster", rc={"figure.figsize": (14, 10)})

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser()

parser.add_argument("--files", nargs="+", required=True,
                    help="List of .fcs.txt files")
parser.add_argument("-n", "--n_neighbors", type=int, default=30)
parser.add_argument("-m", "--min_dist", type=float, default=0.25)
parser.add_argument("--outdir", default="umap_results")

# safety caps (THIS FIXES YOUR CRASH)
parser.add_argument("--per_sample_cap", type=int, default=5000,
                    help="Max cells per sample in pooled embedding")
parser.add_argument("--max_cells", type=int, default=200000,
                    help="Max total pooled cells")

args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# -----------------------
# Load + downsample per sample
# -----------------------
dfs = []

for f in args.files:
    df = pd.read_csv(f, sep="\t")
    sid = os.path.basename(f).replace(".fcs.txt", "")
    df["sample_id"] = sid

    if len(df) > args.per_sample_cap:
        df = df.sample(args.per_sample_cap, random_state=42)

    dfs.append(df)

pooled_df = pd.concat(dfs, ignore_index=True)

# cap total cells
if len(pooled_df) > args.max_cells:
    pooled_df = pooled_df.sample(args.max_cells, random_state=42)

print("FINAL pooled shape:", pooled_df.shape)

# -----------------------
# Feature selection
# -----------------------
feature_cols = pooled_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in feature_cols if c not in ["sample_id"]]

X = pooled_df[feature_cols].values
X = StandardScaler().fit_transform(X)

# -----------------------
# UMAP (SAFE)
# -----------------------
umap_model = umap.UMAP(
    n_neighbors=args.n_neighbors,
    min_dist=args.min_dist,
    n_components=2,
    random_state=None,      # allow parallelism
    n_jobs=1                # prevent fork crashes on macOS
)

U = umap_model.fit_transform(X)

# -----------------------
# Clustering
# -----------------------
db = DBSCAN(eps=0.6, min_samples=20).fit(U)
pooled_df["dbscan"] = db.labels_.astype(str)

# -----------------------
# Save coordinates
# -----------------------
out = pooled_df[["sample_id"]].copy()
out["UMAP1"] = U[:, 0]
out["UMAP2"] = U[:, 1]
out["cluster"] = pooled_df["dbscan"]

out.to_csv(os.path.join(args.outdir, "pooled_cell_coordinates.csv"), index=False)

# -----------------------
# Plot pooled UMAP
# -----------------------
plt.figure()
sns.scatterplot(
    x=U[:, 0],
    y=U[:, 1],
    hue=pooled_df["sample_id"],
    s=1,
    linewidth=0,
    alpha=0.7,
    legend=False
)
plt.title("Pooled UMAP (downsampled)")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "pooled_umap.png"), dpi=300)
plt.close()

print("✅ Finished safely. Results in:", args.outdir)


