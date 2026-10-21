"""Full-dataset reference-tier summary (all 50k molecules).
For the OR-of-unanimities tasks the intrinsic interaction order equals the number
of ground-truth atoms (the largest Moebius term is the union of all marked atoms),
so the summary is computed directly from the SDF property counts -- no 2^n build.
Outputs intrinsic_order_summary.csv and prints per-task label-match rate.
rings-count is circuit-rank/global and is reported separately (see run_bxaic_reference.py)."""
import csv
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

PROP = {"B": "B", "P": "P", "X": "X", "indole": "indole", "PAINS": "pains",
        "rings-max": "largest_rings"}
labels = pd.read_csv("data/data.csv")

supp = Chem.ForwardSDMolSupplier("data/explanations.sdf", sanitize=False)
orders = {t: [] for t in PROP}          # intrinsic order per positive molecule
match = {t: [0, 0] for t in PROP}       # [matched, total]
i = 0
for mol in supp:
    if mol is not None:
        for t, p in PROP.items():
            s = mol.GetProp(p).strip() if mol.HasProp(p) else ""
            k = 0 if s == "" else len(s.split(","))
            lab = int(labels[t].iloc[i])
            match[t][0] += int((k > 0) == (lab == 1))
            match[t][1] += 1
            if k > 0:
                orders[t].append(k)
    i += 1

rows = [("task", "n_molecules", "n_positive", "label_match_rate",
         "intrinsic_order_min", "intrinsic_order_median",
         "intrinsic_order_mean", "intrinsic_order_max")]
print(f"full dataset: {i} molecules")
for t in PROP:
    o = np.array(orders[t]); m = match[t]
    rows.append((t, m[1], len(o), round(m[0]/m[1], 4),
                 int(o.min()), int(np.median(o)), round(float(o.mean()), 2), int(o.max())))
    print(f"  {t:10s} pos={len(o):5d} label_match={m[0]/m[1]:.4f} "
          f"io[min/med/mean/max]={o.min()}/{int(np.median(o))}/{o.mean():.1f}/{o.max()}")
csv.writer(open("intrinsic_order_summary.csv", "w")).writerows(rows)
print("wrote intrinsic_order_summary.csv")
