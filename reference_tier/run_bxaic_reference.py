"""
run_bxaic_reference.py -- build the exact rule-game reference tier from the real
B-XAIC files (data/data.csv + data/explanations.sdf) and emit reference_<task>.csv
consumed by compare.py.

The cooperative game nu(T) is the labelling rule itself under node deletion:
keep atom subset T, re-run the rule, take the 0/1 label. Its exact Moebius
interactions (MI) and Shapley values (SV) ARE the ground-truth explanation.

Locality: every task's game depends only on a small support, so the exact
Moebius is cheap for any molecule size --
  * B / P / X            : nu(T)=1 iff some ground-truth element atom survives
                           -> OR-of-unanimities over connected components of the
                           stored gt atoms (singletons for isolated atoms).
  * indole / PAINS       : nu(T)=1 iff some full-molecule pattern match <= T
                           -> OR-of-unanimities over connected components of the
                           stored gt atoms (approx when occurrences share atoms).
  * rings-max            : nu(T)=1 iff some SSSR ring with size>ring_size is intact
                           -> OR-of-unanimities over the qualifying rings (exact
                           even for fused/overlapping rings).
  * rings-count          : nu(T)=1 iff circuit-rank(subgraph on kept ring atoms)
                           > ring_thresh -> exact 2^r Moebius over the r ring atoms.

MI for an OR-of-unanimities is closed form: nu = 1 - prod_i (1-u_{Ci}), so
  MI[union of any nonempty subset J of the sets] += (-1)^{|J|+1}
(collisions summed), which is exact for arbitrary (even overlapping) supports.

Usage
  python run_bxaic_reference.py --data_dir data --task indole
  python run_bxaic_reference.py --data_dir data --task rings-count --ring_thresh 4
  python run_bxaic_reference.py --data_dir data --task rings-max  --ring_size 6
  # --limit N to cap molecules, --max_support to cap exact-game support size
"""
from __future__ import annotations

import argparse
import os
from itertools import combinations
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
from rdkit import Chem

import compare as C

# atom typing identical to dataset.py so graph_key matches explanations.pt
SYMBOLS = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]
PROP = {"P": "P", "X": "X", "B": "B", "indole": "indole", "PAINS": "pains",
        "rings-count": "rings", "rings-max": "largest_rings"}


def atom_label(atom) -> int:
    s = atom.GetSymbol()
    return SYMBOLS.index(s) if s in SYMBOLS else len(SYMBOLS) - 1


def mol_graph_key(mol) -> str:
    atom_types = np.array([atom_label(a) for a in mol.GetAtoms()])
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei.append((i, j)); ei.append((j, i))
    edge_index = np.array(ei).T if ei else np.zeros((2, 0), dtype=int)
    return C.graph_key(atom_types, edge_index)


def parse_atom_prop(mol, prop_name) -> List[int]:
    if not mol.HasProp(prop_name):
        return []
    p = mol.GetProp(prop_name).strip()
    if p == "":
        return []
    return [int(x) for x in p.split(",")]


def bond_adjacency(mol) -> Dict[int, set]:
    adj: Dict[int, set] = {a.GetIdx(): set() for a in mol.GetAtoms()}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].add(j); adj[j].add(i)
    return adj


def connected_components(atoms: List[int], adj: Dict[int, set]) -> List[FrozenSet[int]]:
    atoms = set(atoms)
    seen, comps = set(), []
    for a in atoms:
        if a in seen:
            continue
        stack, comp = [a], set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.add(u)
            for v in adj[u]:
                if v in atoms and v not in seen:
                    stack.append(v)
        comps.append(frozenset(comp))
    return comps


def mi_or_of_unanimities(sets: List[FrozenSet[int]], max_sets: int) -> Dict[FrozenSet[int], float]:
    """Exact Moebius of nu = OR_i unanimity(sets[i]); collisions on equal unions
    are summed. Returns {} if empty. Raises if too many sets for 2^m expansion."""
    sets = [s for s in sets if s]
    if not sets:
        return {}
    if len(sets) > max_sets:
        raise ValueError(f"{len(sets)} occurrences > max_support={max_sets}")
    mi: Dict[FrozenSet[int], float] = {}
    m = len(sets)
    for r in range(1, m + 1):
        sign = (-1) ** (r + 1)
        for J in combinations(range(m), r):
            union = frozenset().union(*(sets[i] for i in J))
            mi[union] = mi.get(union, 0.0) + sign
    return {k: v for k, v in mi.items() if abs(v) > 1e-12}


# ---------------- rings-count: exact 2^r Moebius over ring atoms ------------- #
def ring_bonds(mol, ring_atoms: set) -> List[Tuple[int, int]]:
    out = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in ring_atoms and j in ring_atoms:
            out.append((i, j))
    return out


def circuit_rank(kept: frozenset, atoms: List[int], bonds: List[Tuple[int, int]]) -> int:
    """E - V + C for the subgraph induced on `kept` (via union-find)."""
    idx = {a: k for k, a in enumerate(atoms)}
    parent = list(range(len(atoms)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    V = 0
    keptset = kept
    present = [a for a in atoms if a in keptset]
    V = len(present)
    E = 0
    for i, j in bonds:
        if i in keptset and j in keptset:
            E += 1
            ri, rj = find(idx[i]), find(idx[j])
            if ri != rj:
                parent[ri] = rj
    roots = {find(idx[a]) for a in present}
    Cc = len(roots)
    return E - V + Cc


def mi_rings_count(mol, ring_atoms: List[int], thresh: int, max_support: int
                   ) -> Dict[FrozenSet[int], float]:
    atoms = sorted(ring_atoms)
    r = len(atoms)
    if r == 0:
        return {}
    if r > max_support:
        raise ValueError(f"ring support {r} > max_support={max_support}")
    bonds = ring_bonds(mol, set(atoms))
    nu = np.zeros(1 << r)
    for mask in range(1 << r):
        kept = frozenset(atoms[b] for b in range(r) if (mask >> b) & 1)
        nu[mask] = 1.0 if circuit_rank(kept, atoms, bonds) > thresh else 0.0
    # fast subset Moebius inversion
    m = nu.copy()
    for i in range(r):
        bit = 1 << i
        for mask in range(1 << r):
            if mask & bit:
                m[mask] -= m[mask ^ bit]
    mi: Dict[FrozenSet[int], float] = {}
    for mask in range(1 << r):
        if abs(m[mask]) > 1e-9:
            mi[frozenset(atoms[b] for b in range(r) if (mask >> b) & 1)] = float(m[mask])
    return mi


def sssr_rings(mol) -> List[FrozenSet[int]]:
    return [frozenset(r) for r in Chem.GetSymmSSSR(mol)]


# --------------------------------- driver ----------------------------------- #
def build_entry(task, mol, label, args) -> Tuple[C.RefEntry, bool]:
    n = mol.GetNumAtoms()
    adj = bond_adjacency(mol)
    if task in ("B", "P", "X"):
        # element-presence rule: nu=1 iff ANY atom of the element survives.
        # Each atom is its own occurrence (OR of singletons), regardless of
        # whether two same-element atoms happen to be bonded.
        gt = parse_atom_prop(mol, PROP[task])
        comps = [frozenset({a}) for a in gt]
        mi = mi_or_of_unanimities(comps, args.max_support)
        gt_atoms = frozenset(gt)
    elif task in ("indole", "PAINS"):
        # connected-pattern rule: OR-of-unanimities over connected components of
        # the stored gt atoms. Exact when occurrences are disjoint; approximate
        # when two occurrences share/bond atoms (they merge into one component).
        gt = parse_atom_prop(mol, PROP[task])
        comps = connected_components(gt, adj)
        mi = mi_or_of_unanimities(comps, args.max_support)
        gt_atoms = frozenset(gt)
    elif task == "rings-max":
        rings = [r for r in sssr_rings(mol) if len(r) > args.ring_size]
        mi = mi_or_of_unanimities(rings, args.max_support)
        gt_atoms = frozenset().union(*rings) if rings else frozenset()
    elif task == "rings-count":
        ring_atoms = parse_atom_prop(mol, "rings")
        mi = mi_rings_count(mol, ring_atoms, args.ring_thresh, args.max_support)
        gt_atoms = frozenset(ring_atoms) if mi else frozenset()
    else:
        raise ValueError(task)

    sv = C.sv_from_mi(mi, n)
    rule_label = 1 if len(mi) > 0 else 0
    entry = C.RefEntry(
        n_atoms=n, mi=mi, sv=sv,
        gt_atoms=frozenset(int(i) for i in np.nonzero(np.abs(sv) > 1e-9)[0]),
        intrinsic_order=C.intrinsic_order(mi),
        label=int(label), graph_key=mol_graph_key(mol),
    )
    return entry, (rule_label == int(label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--task", required=True, choices=list(PROP))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_support", type=int, default=20,
                    help="cap on exact-game support (2^support or 2^m expansion)")
    ap.add_argument("--ring_thresh", type=int, default=4, help="rings-count: circuit rank > thresh")
    ap.add_argument("--ring_size", type=int, default=6, help="rings-max: SSSR ring size > ring_size")
    args = ap.parse_args()

    df_path = os.path.join(args.data_dir, "data.csv")
    sdf_path = os.path.join(args.data_dir, "explanations.sdf")
    import pandas as pd
    labels = pd.read_csv(df_path)[args.task].tolist()

    out_path = args.out or f"reference_{args.task}.csv"
    supp = Chem.SDMolSupplier(sdf_path)

    entries, matched, total, skipped = [], 0, 0, 0
    io_hist: Dict[int, int] = {}
    for i, mol in enumerate(supp):
        if args.limit is not None and i >= args.limit:
            break
        if mol is None:
            skipped += 1
            continue
        try:
            entry, ok = build_entry(args.task, mol, labels[i], args)
        except ValueError:
            skipped += 1
            continue
        entry.mol_id = i
        entries.append(entry)
        matched += int(ok)
        total += 1
        io_hist[entry.intrinsic_order] = io_hist.get(entry.intrinsic_order, 0) + 1

    C.write_reference_csv(entries, out_path)
    pos = [e for e in entries if e.label == 1]
    print(f"task={args.task}  molecules={total}  skipped(support>cap/none)={skipped}")
    print(f"label_matches_rule = {matched}/{total} = {matched/max(total,1):.4f}")
    print(f"positives={len(pos)}  mean intrinsic order (pos) = "
          f"{np.mean([e.intrinsic_order for e in pos]) if pos else 0:.2f}")
    print(f"intrinsic-order histogram (all): "
          f"{dict(sorted(io_hist.items()))}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
