import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
np.random.seed(0)


# === Load your data ===
path = r"data/filtered_change_seq.csv" #Relative path
df = pd.read_csv(path)

# === Treat CHANGEseq_reads as 'experimental activity' ===
df['experimental_activity'] = df['CHANGEseq_reads']

# === Simulate predicted activity (for now we can add a bit of random noise) ===
df['predicted_activity'] = df['CHANGEseq_reads'] + np.random.normal(0, 50, size=len(df))

# === Rank both experimental and predicted ===
df['exp_rank'] = df['experimental_activity'].rank(ascending=False)
df['pred_rank'] = df['predicted_activity'].rank(ascending=False)

# === Compare rankings ===
rho, pval = spearmanr(df['experimental_activity'], df['predicted_activity'])
print(f"Spearman correlation between experimental and predicted activity: {rho:.3f} (p={pval:.3e})")

# === Sort by predicted activity and display ===
ranked = df.sort_values('predicted_activity', ascending=False)
print("\nTop off-targets by predicted activity:\n")
print(ranked[['name', 'offtarget_sequence', 'predicted_activity', 'experimental_activity']])




