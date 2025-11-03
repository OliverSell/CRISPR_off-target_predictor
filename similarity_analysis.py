#/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt
import sys

max_mismatches = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# Load processed dataset
infile = "data/filtered_change_seq.csv"
try:
    changeseq = pd.read_csv(infile)[["CHANGEseq_reads", "offtarget_sequence", "target"]]
except FileNotFoundError:
    raise SystemExit("NO! Filtered CSV not found. Run preprocess_data.py first.")

ontargets = list(changeseq["target"].unique())

chars = sorted(set("".join(ontargets)))
char_to_int = {c: i for i, c in enumerate(chars)}
X = np.array([[char_to_int[c] for c in seq] for seq in ontargets])


# Hamming distances
dist_matrix = pairwise_distances(X, metric="hamming")


# Similarity graph
seq_len = len(ontargets[0])
threshold = max_mismatches / seq_len  # fraction

G = nx.Graph()
G.add_nodes_from(range(len(ontargets)))

for i in range(len(ontargets)):
    for j in range(i + 1, len(ontargets)):
        if dist_matrix[i, j] <= threshold:
            G.add_edge(i, j)


components = [list(c) for c in nx.connected_components(G)]
print(f"Found {len(components)} target clusters/groups")

# Plot
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42, k=0.5)
colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
for color, comp in zip(colors, components):
    nx.draw_networkx_nodes(G, pos, nodelist=comp, node_color=[color], node_size=800)
nx.draw_networkx_edges(G, pos, alpha=0.4)
plt.title(f"Sequence Similarity Graph (≤ {max_mismatches} mismatches)")
plt.axis("off")
plt.show()