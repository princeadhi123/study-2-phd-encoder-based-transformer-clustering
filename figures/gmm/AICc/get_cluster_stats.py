"""
Quick script to get k=5 GMM-AICc cluster means for proper analysis.
"""
import pandas as pd
from pathlib import Path

BASE       = Path(r"c:\Users\pdaadh\Desktop\Study-2")
FEATS_PATH = BASE / "diagnostics" / "cluster input features" / "derived_features.csv"
CLUST_PATH = BASE / "diagnostics" / "student cluster labels" / "student_clusters.csv"

FEATURE_COLS = [
    "accuracy",
    "avg_rt",
    "var_rt",
    "rt_cv",
    "longest_correct_streak",
    "longest_incorrect_streak",
    "consecutive_correct_rate",
]

feats    = pd.read_csv(FEATS_PATH)
clusters = pd.read_csv(CLUST_PATH)[["IDCode", "gmm_aicc_best_label"]]
df       = feats.merge(clusters, on="IDCode", how="inner").dropna(subset=FEATURE_COLS)

print(f"Total students: {len(df)}")
print(f"Clusters: {sorted(df['gmm_aicc_best_label'].unique())}\n")

summary = df.groupby("gmm_aicc_best_label")[FEATURE_COLS].mean()
sizes   = df.groupby("gmm_aicc_best_label").size().rename("n")
summary = summary.join(sizes)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(summary.to_string())

summary.to_csv(Path(__file__).parent / "gmm_cluster_means_k5.csv")
print("\nSaved: gmm_cluster_means_k5.csv")
