import pandas as pd
import numpy as np
from Bio.Seq import Seq
import pickle
from sklearn.metrics import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt
import random
import os

# Initialization

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


# Paths
infile = "data/41587_2020_555_MOESM3_ESM.xlsx"
infile2 = "data/41587_2015_BFnbt3117_MOESM22_ESM.xlsx"
outfile = "data/filtered_change_seq.csv"
trainfile = "data/train_set.csv"
valfile = "data/validation_set.csv"
testfile = "data/test_set.csv"
ext_testfile = "data/ext_test_set.csv"


# Load Excel
xls = pd.ExcelFile(infile)
Raw = pd.read_excel(xls, "CHANGE-seq_Supp_Table_3")
Specificities = pd.read_excel(xls, "CHANGE-seq_Supp_Table_4")

xls = pd.ExcelFile(infile2)
guide_seq = pd.read_excel(xls, "Sheet1")



# Filtering
Filtered = Raw.query('CHANGEseq_reads > 100')
Filtered = Filtered.query("target != 'GGTGGTGTGGGCCCAGGAGGNGG'")
Filtered = Filtered.drop(columns='Unnamed: 7')
Filtered = Filtered.dropna()
Filtered = Filtered[~Filtered.offtarget_sequence.str.contains('-')]
Filtered = Filtered[Filtered.offtarget_sequence.apply(lambda x: len(str(x)))==23]
changeseq = Filtered[["CHANGEseq_reads","offtarget_sequence","target"]]


guide_seq.rename(columns={'GUIDE-Seq Reads' : 'GUIDEseq_Reads'}, inplace=True)
guide_seq = guide_seq.query('GUIDEseq_Reads > 100')
guide_seq = guide_seq[guide_seq.Offtarget_Sequence.apply(lambda x: len(str(x)))==23]
guide_seq = guide_seq[["GUIDEseq_Reads","Offtarget_Sequence","Target_Sequence"]]

# Score off-targets

score_list = list()
mismatch_list = list()

for offtarget in changeseq.iloc:
    off = offtarget["offtarget_sequence"]
    on = offtarget["target"]
    score = EnergyCalcLocal(off,on,POS_WGH[:-1])
    score_list.append(score[2])
    mismatch_list.append(score[4])
    
changeseq["DeltaGH"] = score_list
changeseq["mismatches"] = mismatch_list

score_list = list()
mismatch_list = list()

for offtarget in guide_seq.iloc:
    off = offtarget["Offtarget_Sequence"]
    on = offtarget["Target_Sequence"]
    score = EnergyCalcLocal(off,on,POS_WGH[:-1])
    score_list.append(score[2])
    mismatch_list.append(score[4])
    
guide_seq["DeltaGH"] = score_list
guide_seq["mismatches"] = mismatch_list



os.makedirs(os.path.dirname(outfile), exist_ok=True)

changeseq.to_csv(outfile, index=False)
guide_seq.to_csv(ext_testfile, index=False)

print(f"Filtered data saved to {outfile}")

ontargets = changeseq.groupby("target")

ontarget_dict = dict()

for ontarget in ontargets:
    ont = list(ontarget[1]["target"].unique())[0]
    ontarget_dict[ont] = len(ontarget[1])
    
def split_dict_by_sum(d, proportions=(0.6, 0.2, 0.2)):
    # Sort items by value (descending) so large contributors are assigned first
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in items)
    targets = [p * total for p in proportions]

    groups = [[], [], []]  # lists of keys
    sums = [0.0, 0.0, 0.0]

    for key, value in items:
        # Choose group currently furthest below its target proportion
        ratios = [s / t if t > 0 else 0 for s, t in zip(sums, targets)]
        idx = np.argmin(ratios)
        groups[idx].append(key)
        sums[idx] += value

    return groups, sums, targets

groups, sums, targets = split_dict_by_sum(ontarget_dict)

for i, (g, s, t) in enumerate(zip(groups, sums, targets), start=1):
    print(f"Group {i}: {g}, sum={s:.2f}, target≈{t:.2f}")
    
training_list = groups[0]
validation_list = groups[1]
test_list = groups[2]

training = changeseq[changeseq['target'].isin(training_list)]
validation = changeseq[changeseq['target'].isin(validation_list)]
test = changeseq[changeseq['target'].isin(test_list)]

training.to_csv(trainfile, index=False)
validation.to_csv(valfile, index=False)
test.to_csv(testfile, index=False)
