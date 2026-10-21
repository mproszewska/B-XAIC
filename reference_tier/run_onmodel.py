"""
On-model experiment (reviewer h14X, Q2 on-model half), scaled + resumable.

For each task's trained GIN we compute the EXACT Shapley values and Moebius
interactions of the model under NODE-DELETION masking (matching the rule game),
via shapiq, and compare to the exact rule reference:
  * atom-recovery AUROC : model Shapley vs ground-truth atom set
  * interaction MSE     : model Moebius vs reference Moebius
  * gap decomposition   : deletion masking on both sides => masking-mismatch = 0,
                          so gap = model-error (+ lambda-truncation at cap lambda).

Molecules are taken from a precomputed index list (idx_<task>.json, small
positives) via random SDF access. Results are appended per molecule to
onmodel_<task>.csv and molecules already present are skipped, so the job is
resumable across invocations. Each call runs until --time_budget seconds.

The 2^n induced-subgraph forward is vectorized: all coalitions in one batched
GIN call (no Python per-coalition loop), so a whole task fits in one call.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GIN, global_add_pool
from torch.nn import Linear
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import shapiq

import compare as C
import run_bxaic_reference as R

MODEL_DIR = "/sessions/festive-focused-darwin/mnt/bxaic_rebuttal"


def build_gin():
    return Sequential("x, edge_index, batch", [
        (GIN(11, 32, num_layers=3, out_channels=32, norm="batch_norm"), "x, edge_index -> x"),
        (global_add_pool, "x, batch -> x"),
        Linear(32, 2)])


def load_model(task):
    ck = torch.load(f"{MODEL_DIR}/gin_{task}_split0.pt", map_location="cpu", weights_only=False)
    m = build_gin(); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def mol_graph(mol):
    types = torch.tensor([R.atom_label(a) for a in mol.GetAtoms()])
    x = F.one_hot(types, 11).float()
    row, col = [], []
    for b in mol.GetBonds():
        a, bb = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        row += [a, bb]; col += [bb, a]
    ei = torch.tensor([row, col], dtype=torch.long) if row else torch.zeros(2, 0, dtype=torch.long)
    return x, ei


def make_value_fn(model, x, ei, target=1):
    """Vectorized deletion-masking oracle: coalitions (m x n bool) -> P(target).
    Builds all induced subgraphs as one batched graph with tensor ops; empty
    coalitions pool to zero and hence to the readout baseline automatically."""
    n = x.shape[0]
    src, dst = ei[0], ei[1]

    def value(coalitions):
        M = torch.as_tensor(np.asarray(coalitions, dtype=bool))          # C x n
        Cn = M.shape[0]
        counts = M.sum(1)                                                 # C
        kept_coal, kept_node = M.nonzero(as_tuple=True)                   # row-major
        total = kept_coal.shape[0]
        node_gid = torch.full((Cn, n), -1, dtype=torch.long)
        node_gid[kept_coal, kept_node] = torch.arange(total)
        x_batch = x[kept_node]
        batch_index = kept_coal
        # edges present where both endpoints kept
        Se = M[:, src] & M[:, dst]                                        # C x |E|
        ce, ee = Se.nonzero(as_tuple=True)
        new_src = node_gid[ce, src[ee]]
        new_dst = node_gid[ce, dst[ee]]
        ei_batch = torch.stack([new_src, new_dst]) if ee.numel() else torch.zeros(2, 0, dtype=torch.long)
        with torch.no_grad():
            h = model.module_0(x_batch, ei_batch) if total else torch.zeros(0, 32)
            pooled = global_add_pool(h, batch_index, size=Cn) if total else torch.zeros(Cn, 32)
            logits = model.module_2(pooled)
            p = F.softmax(logits, dim=-1)[:, target].numpy()
        return p
    return value


def exact_moebius_from_value(value, n):
    """Model Moebius under deletion masking: evaluate nu on all 2^n coalitions
    (one batched forward) then invert with the fast subset transform. Independent
    of shapiq and ~10x faster in this loop; cross-checked against shapiq."""
    masks = np.arange(1 << n)
    bits = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)   # 2^n x n
    nu = np.asarray(value(bits), dtype=float)
    m = nu.copy()
    for i in range(n):
        bit = 1 << i
        has = (masks & bit) > 0
        m[has] -= m[masks[has] ^ bit]
    mi = {}
    nz = np.nonzero(np.abs(m) > 1e-9)[0]
    for mask in nz:
        mi[frozenset(int(b) for b in range(n) if (mask >> b) & 1)] = float(m[mask])
    return mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--lam", type=int, default=2)
    ap.add_argument("--time_budget", type=float, default=38.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"onmodel_{args.task}.csv"
    done = set()
    if os.path.exists(out):
        import csv
        with open(out) as f:
            for r in csv.DictReader(f):
                done.add(int(r["mol"]))
    idx = json.load(open(f"idx_{args.task}.json"))
    todo = [i for i in idx if i not in done]

    import pandas as pd
    labels = pd.read_csv(f"{args.data_dir}/data.csv")[args.task].tolist()
    model, ck = load_model(args.task)
    supp = Chem.SDMolSupplier(f"{args.data_dir}/explanations.sdf")
    _ = len(supp)
    ref_args = argparse.Namespace(max_support=20, ring_thresh=4, ring_size=6)

    header = not os.path.exists(out)
    fout = open(out, "a")
    if header:
        fout.write("mol,n,pred_p,intrinsic_order,auroc,int_mse,model_err,lam_trunc\n")

    t0 = time.time(); processed = 0
    for i in todo:
        if time.time() - t0 > args.time_budget:
            break
        mol = supp[i]
        if mol is None:
            continue
        entry, _ = R.build_entry(args.task, mol, labels[i], ref_args)
        if entry.label != 1 or len(entry.gt_atoms) == 0:
            continue
        n = mol.GetNumAtoms()
        x, ei = mol_graph(mol)
        with torch.no_grad():
            pred_p = F.softmax(model(x, ei, torch.zeros(n, dtype=torch.long)), -1)[0, 1].item()
        mi_model = exact_moebius_from_value(make_value_fn(model, x, ei), n)
        sv_model = C.sv_from_mi(mi_model, n)
        gt = entry.gt_mask()
        au = C.auroc(gt, sv_model) if gt.min() != gt.max() else ""
        imse = C.interaction_mse(mi_model, entry.mi)
        d = C.decompose_gap(entry.mi, C.truncate_mi(mi_model, args.lam), entry.mi, mi_model)
        fout.write(f"{i},{n},{pred_p:.4f},{entry.intrinsic_order},"
                   f"{au if au=='' else round(au,4)},{imse:.6f},"
                   f"{d.model_error_norm:.4f},{d.lambda_truncation_norm:.4f}\n")
        fout.flush()
        processed += 1
    fout.close()

    # aggregate
    import csv
    rows = list(csv.DictReader(open(out)))
    aur = [float(r["auroc"]) for r in rows if r["auroc"] != ""]
    print(f"task={args.task}: processed {processed} this call, total {len(rows)}/{len(idx)} "
          f"(remaining {len(idx)-len(rows)}), GIN f1={ck['f1']:.3f}")
    if rows:
        print(f"  atom-recovery AUROC: mean={np.mean(aur):.3f} median={np.median(aur):.3f} "
              f"min={np.min(aur):.3f}  (n={len(aur)})")
        print(f"  model-error norm:    mean={np.mean([float(r['model_err']) for r in rows]):.3f}")
        print(f"  lambda{args.lam}-trunc norm:   mean={np.mean([float(r['lam_trunc']) for r in rows]):.3f}")


if __name__ == "__main__":
    main()
