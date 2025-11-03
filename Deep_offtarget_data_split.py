# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 10:15:49 2025

@author: zbs768
"""

#/usr/bin/env python3
import pandas as pd
import numpy as np
from Bio.Seq import Seq
import pickle
from sklearn.metrics import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt
import random



def EnergyCalcLocal(offtarget,ontarget,Gammas):
    mismatch_count = sum(c1!=c2 for c1,c2 in zip(offtarget[:-3],ontarget[:-3]))
    Delta_G_pos = [0] * 19
    loop_list = [0] * 20
    offtarget = Seq(offtarget).complement()
    #Anotate loops
    for i in range(len(loop_list)):
        if RI_MATCH_noGU[ontarget[i]][offtarget[i]] == True:
            loop_list[i] = 0
        else:
            loop_list[i] = 1
    i=0
    loop_start = None
    while i < 20:
        if loop_list[i] == 1 and loop_start == None:
            loop_start = i 
            loop_size = 0
            #SUM WHILE
        if loop_start != None:
            loop_size += 1
        if loop_start != None and (loop_list[i] == 0 or i == 19):
            if i == 19 and loop_list[i] > 0:
                for j in range(loop_size):
                    loop_list[i-j] = loop_size
            else:
                for j in range(loop_size-1):
                    loop_list[i-j-1] = loop_size-1
            loop_start = None
        i += 1 
    
    for i in range(len(Delta_G_pos)):

        #Normal stacking
        if loop_list[i] == 0:
            Delta_G_pos[i] = energy_dics[0][ontarget[i:i+2]][offtarget[i:i+2]]
        #Short loops
        if loop_list[i] == 1 and i != 0:
                Delta_G_pos[i] = energy_dics[1][ontarget[i-1:i+2]][offtarget[i-1:i+2]]/2
                Delta_G_pos[i-1] = energy_dics[1][ontarget[i-1:i+2]][offtarget[i-1:i+2]]/2
        if loop_list[i] == 2 and loop_list[i-1] == 2:
            if i not in (0,1) and i < 18:
                Delta_G_pos[i] = energy_dics[2][ontarget[i-2:i+2]][offtarget[i-2:i+2]]/3
                Delta_G_pos[i-1] = energy_dics[2][ontarget[i-2:i+2]][offtarget[i-2:i+2]]/3
                Delta_G_pos[i-2] = energy_dics[2][ontarget[i-2:i+2]][offtarget[i-2:i+2]]/3
            else:
                Delta_G_pos[i] = 0
                Delta_G_pos[i-1] = 0 
        #Long loops 3+
        if loop_list[i] > 2 and loop_list[i-(loop_list[i]-1)] == loop_list[i]:
            j = loop_list[i]
            if j == i+1:
                pass
            elif offtarget[i:i+2] in energy_dics[0][ontarget[i:i+2]] and offtarget[i-j:i-j+2] in energy_dics[0][ontarget[i-j:i-j+2]]:
                for loop_pos in range(j+1):
                    Delta_G_pos[i-loop_pos] = (energy_dics[0][ontarget[i:i+2]][offtarget[i:i+2]] +  energy_dics[0][ontarget[i-j:i-j+2]][offtarget[i-j:i-j+2]] + RNA_DNA_loop_table[j-1])/(j+1)
            
    
    #External loops
    if Delta_G_pos[0] == 0:
        i = 0
        while Delta_G_pos[i] == 0:
            i+= 1
        if ontarget[i] == 'T' or offtarget[i] == 'T':
            Delta_G_pos[i] += 0.25

    if Delta_G_pos[-1] >= 0:
        i = -1
        while Delta_G_pos[i] >= 0:
            Delta_G_pos[i] = 0
            i-= 1
        if ontarget[i-3] == 'T' or offtarget[i-3] == 'T':
            Delta_G_pos[i] += 0.25
    
    
    Delta_G = sum(Delta_G_pos)
    Weight_DG_pos = np.multiply(Delta_G_pos,Gammas)
    Weight_DG = sum(Weight_DG_pos)
        
    return Delta_G,Delta_G_pos,Weight_DG,Weight_DG_pos,mismatch_count, loop_list


#Initialization

infile = "data/41587_2020_555_MOESM3_ESM.xlsx"

xls = pd.ExcelFile(infile)

with open('data/energy_dics.pkl', 'rb') as f:
    energy_dics = pickle.load(f)

#Ferhardts Gammas
POS_WGH=[1.80067099242007,
         1.95666668400006, 
         1.90472004401173,
         2.13047270152512,
         1.37853848098249,
         1.46460783730408,
         1.0,
         1.387220146823,
         1.51401000729362,
         1.98058344620751,
         1.87939168587699,
         1.7222593588838,
         2.02228445489326,
         1.92692086621503,
         2.08041972716723,
         1.94496755678903,
         2.14539112893591,
         2.04277109036766,
         2.24911493451185,
         2.25]

RNA_DNA_loop_table = ["*","*",3.2,3.555,3.725,3.975,4.16,4.33,4.495,4.6,4.7]
DNA_DNA_loop_table = [0,3.6,4.4,4.8,4.9,5.2,5.4,5.6,5.8]

RI_MATCH_noGU = {'A':{'A':False, 'C':False, 'G':False, 'T':True},
         'C':{'A':False, 'C':False, 'G':True, 'T':False},
         'G':{'A':False, 'C':True, 'G':False, 'T':False},
         'T':{'A':True, 'C':False, 'G':False, 'T':False}}

RI_REV_NT_MAP = {'-':'', 'a':'T', 'A':'T', 'c':'G', 'C':'G', 'g':'C', 'G':'C',
              't':'A', 'T':'A', 'u':'A', 'U':'A', 'n':'N', 'N':'N'}

#Data
Raw = pd.read_excel(xls, 'CHANGE-seq_Supp_Table_3')
Specificities = pd.read_excel(xls, 'CHANGE-seq_Supp_Table_4')

#Filtering
#Filters out all offtargets below detection limit of 100
#Filters out outlier with 70k offtargets
#Filters out NA's
#Filters out bulges as they cannot be correctly scored using this scema

Filtered = Raw.query('CHANGEseq_reads > 100')
Filtered = Filtered.query("target != 'GGTGGTGTGGGCCCAGGAGGNGG'")
Filtered = Filtered.drop(columns='Unnamed: 7')
Filtered = Filtered.dropna()
Filtered = Filtered[~Filtered.offtarget_sequence.str.contains('-')]
Filtered = Filtered[Filtered.offtarget_sequence.apply(lambda x: len(str(x)))==23]


#Calculate Hamming distance
changeseq = Filtered[["CHANGEseq_reads","offtarget_sequence","target"]]

ontargets = list(changeseq["target"].unique())

# --- Step 1: Convert sequences to numeric array ---
# Map each unique character to an integer
chars = sorted(set("".join(ontargets)))
char_to_int = {c: i for i, c in enumerate(chars)}

# Encode sequences numerically
X = np.array([[char_to_int[c] for c in seq] for seq in ontargets])

# --- Step 2: Compute Hamming distances ---
dist_matrix = pairwise_distances(X, metric='hamming')

# Step 3: Define similarity threshold (e.g. ≤ 3 mismatches)
seq_len = len(ontargets[0])
max_mismatches = 3
threshold = max_mismatches / seq_len  # fraction

# Step 4: Build similarity graph
G = nx.Graph()
G.add_nodes_from(range(len(ontargets)))

for i in range(len(ontargets)):
    for j in range(i+1, len(ontargets)):
        if dist_matrix[i, j] <= threshold:
            G.add_edge(i, j)
            

# Step 5: Find connected components (groups of similar sequences)
components = [list(comp) for comp in nx.connected_components(G)]
print(f"Found {len(components)} clusters/groups")

# Step 6: Plot the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42, k=0.5)  # layout based on connectivity
colors = plt.cm.tab10(np.linspace(0, 1, len(components)))  # color by component

for color, comp in zip(colors, components):
    nx.draw_networkx_nodes(G, pos, nodelist=comp, node_color=[color], node_size=800)
    #nx.draw_networkx_labels(G, pos, labels={i: ontargets[i] for i in comp}, font_size=9)
nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.title(f"Sequence Similarity Graph (≤ {max_mismatches} mismatches)")
plt.axis("off")
plt.show()