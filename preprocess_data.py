#/usr/bin/env python3
import pandas as pd
import os

# Paths
infile = "data/41587_2020_555_MOESM3_ESM.xlsx"
outfile = "data/filtered_change_seq.csv"

# Load Excel
xls = pd.ExcelFile(infile)
Raw = pd.read_excel(xls, "CHANGE-seq_Supp_Table_3")
Specificities = pd.read_excel(xls, "CHANGE-seq_Supp_Table_4")

# Filtering
Filtered = Raw.query("CHANGEseq_reads > 100")
Filtered = Filtered.query("target != 'GGTGGTGTGGGCCCAGGAGGNGG'")
Filtered = Filtered.drop(columns="Unnamed: 7", errors="ignore")
Filtered = Filtered.dropna()
Filtered = Filtered[~Filtered.offtarget_sequence.str.contains("-")]
Filtered = Filtered[Filtered.offtarget_sequence.apply(lambda x: len(str(x))) == 23]

os.makedirs(os.path.dirname(outfile), exist_ok=True)

Filtered.to_csv(outfile, index=False)
print(f"Filtered data saved to {outfile}")