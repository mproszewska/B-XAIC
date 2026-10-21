"""Continuous null-explanation (NE) metric: node-attribution Gini impurity on
negative-class molecules. High normalized impurity = flat attributions = good NE."""
import numpy as np

def gini_impurity(attr, use_abs=True, center=None):
    a = np.asarray(attr, dtype=float)
    if center == "median":
        a = a - np.median(a)
    w = np.abs(a) if use_abs else np.clip(a, 0, None)
    s = w.sum()
    n = len(w)
    if n <= 1:
        return np.nan
    if s <= 1e-12:               # all-zero attributions = perfectly flat
        return 1.0
    p = w / s
    G = 1.0 - np.sum(p ** 2)
    return G / (1.0 - 1.0 / n)   # normalized to [0,1]

if __name__ == "__main__":
    import argparse, json, torch
    import torch.nn.functional as F
    from rdkit import Chem
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    import pandas as pd
    import run_onmodel as O, compare as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n_mols", type=int, default=30)
    ap.add_argument("--max_atoms", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv("data/data.csv")
    nA = [ (Chem.MolFromSmiles(s).GetNumAtoms() if Chem.MolFromSmiles(s) else -1) for s in df["smiles"] ]
    df["nA"] = nA
    neg_idx = df.index[(df[args.task]==0)&(df.nA>1)&(df.nA<=args.max_atoms)].tolist()[:args.n_mols]

    model, ck = O.load_model(args.task)
    supp = Chem.SDMolSupplier("data/explanations.sdf"); _=len(supp)
    ginis, preds = [], []
    for i in neg_idx:
        mol = supp[i]
        if mol is None: continue
        n = mol.GetNumAtoms(); x, ei = O.mol_graph(mol)
        with torch.no_grad():
            preds.append(F.softmax(model(x, ei, torch.zeros(n, dtype=torch.long)), -1)[0,1].item())
        mi = O.exact_moebius_from_value(O.make_value_fn(model, x, ei), n)
        sv = C.sv_from_mi(mi, n)
        ginis.append(gini_impurity(sv))
    ginis = np.array(ginis)
    print(f"task={args.task}  negatives={len(ginis)} (n<= {args.max_atoms})  "
          f"mean P(y=1)={np.mean(preds):.3f}")
    print(f"  NE Gini impurity (model Shapley): mean={np.nanmean(ginis):.3f} "
          f"median={np.nanmedian(ginis):.3f} min={np.nanmin(ginis):.3f}  "
          f"(1.0 = perfectly flat = ideal NE)")
