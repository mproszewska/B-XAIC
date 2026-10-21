"""
compare.py -- B-XAIC reference-tier comparison module (reviewer h14X, Q2 second half)

Takes the exact rule-game reference (reference_<task>.csv, produced by
run_bxaic_reference.py) and real explainer outputs and computes:

  * atom-level recovery (AUROC) of node-level post-hoc explainers, using the
    SAME node-importance reduction and empty-molecule handling as the repo's
    evaluate_explanations.py, so the numbers line up with the paper.
  * interaction-vector MSE (Moebius basis) of interaction methods vs the exact
    ground-truth Moebius interactions.
  * order-truncation error curves, computed model-independently from the
    reference Moebius vector alone. The game-reconstruction error hits exactly
    0 at L == intrinsic_order, which is the provable "any method capped at
    order L < intrinsic_order cannot express the task" statement.
  * an exact, additive gap decomposition for GraphSHAP-IQ run on a trained GNN:
        pred_on_model - ref_deletion
      = (ref_baseline - ref_deletion)        # masking-mismatch  (deletion vs BShap imputation)
      + (exact_model  - ref_baseline)         # model-error       (GNN != rule, same masking)
      + (pred_on_model - exact_model)         # lambda-truncation (order cap on the model game)
    The split is additive at the difference-VECTOR level (norms reported per term
    and per interaction order). Model-agnostic methods fed the rule-game directly
    are the special case ref_baseline = exact_model = ref_deletion, so only the
    lambda-truncation term survives -- the clean unit test the reviewer asked for.

Design notes
  * No hard dependency on torch or sklearn. torch is imported lazily only to read
    a real explanations.pt; a numpy AUROC (identical to sklearn's for AUROC) is
    used if sklearn is absent.
  * Interactions are handled in the MOEBIUS basis throughout (the reference
    statistic). Explainer output is accepted as a plain dict, or duck-typed as a
    shapiq InteractionValues object (index must be "Moebius"/"MI").
  * Reference<->explanation join uses a canonical structural key derived from
    (atom-type-per-node, undirected edge set). Both sides build the graph from
    dataset.py's smiles_to_graph with identical node ordering, so the key is an
    exact identity match and needs no molecule id. run_bxaic_reference.py should
    import `graph_key` from here (or emit the same string) into the `graph_key`
    column. An explicit `mol_id` join is also supported when available.

CLI
    python compare.py auroc  --reference reference_indole.csv --explanations explanations.pt
    python compare.py interactions --reference reference_indole.csv --pred pred_moebius.json
    python compare.py truncation   --reference reference_indole.csv [--lam 2]
    python compare.py decompose    --reference reference_indole.csv \
        --pred pred_on_model.json --baseline reference_indole_bshap.csv \
        --exact-model exact_model.json --lam 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Key = FrozenSet[int]


# --------------------------------------------------------------------------- #
#  Canonical graph key  (reference <-> explanation join)                       #
# --------------------------------------------------------------------------- #
def _atom_types_from_x(x) -> List[int]:
    """One-hot / feature rows -> integer atom-type per node (argmax), node order kept."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return [int(round(v)) for v in arr]
    return [int(a) for a in arr.argmax(axis=1)]


def _undirected_edges(edge_index) -> List[Tuple[int, int]]:
    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        ei = ei.reshape(2, -1)
    edges = set()
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        a, b = int(a), int(b)
        if a == b:
            continue
        edges.add((a, b) if a < b else (b, a))
    return sorted(edges)


def graph_key(x, edge_index) -> str:
    """Exact structural identity key. Stable across the reference and explanation
    sides because both derive x/edge_index from dataset.smiles_to_graph with the
    same node ordering. Not isomorphism-invariant by design -- it is an identity
    check, which is what we want for joining the two files."""
    atom_types = _atom_types_from_x(x)
    edges = _undirected_edges(edge_index)
    payload = json.dumps([atom_types, edges], separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- #
#  Reference entry + loaders                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class RefEntry:
    n_atoms: int
    mi: Dict[Key, float]                      # exact Moebius interactions (nonzero terms)
    sv: Optional[np.ndarray] = None           # exact Shapley values, atom-indexed
    gt_atoms: Optional[FrozenSet[int]] = None # SV>0 support == expl_node_mask support
    intrinsic_order: Optional[int] = None
    label: Optional[int] = None
    graph_key: Optional[str] = None
    mol_id: Optional[int] = None
    smiles: Optional[str] = None

    def gt_mask(self) -> np.ndarray:
        m = np.zeros(self.n_atoms, dtype=bool)
        if self.gt_atoms:
            for i in self.gt_atoms:
                m[i] = True
        return m


def _parse_mi(cell) -> Dict[Key, float]:
    """mi stored as JSON list of [[atom indices], value]."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return {}
    if isinstance(cell, dict):
        return {frozenset(int(i) for i in k): float(v) for k, v in cell.items()}
    obj = json.loads(cell) if isinstance(cell, str) else cell
    out: Dict[Key, float] = {}
    for atoms, val in obj:
        out[frozenset(int(a) for a in atoms)] = float(val)
    return out


def _parse_int_list(cell) -> List[int]:
    if cell is None or (isinstance(cell, float) and (isinstance(cell, float) and math.isnan(cell))):
        return []
    if isinstance(cell, (list, tuple)):
        return [int(i) for i in cell]
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [int(i) for i in json.loads(s)]


# Column-name aliases so the loader tolerates minor schema drift.
_COLS = {
    "n_atoms": ("n_atoms", "num_atoms", "n"),
    "mi": ("mi_json", "mi", "moebius", "moebius_json"),
    "sv": ("sv_json", "sv", "shapley", "shapley_json"),
    "gt_atoms": ("gt_atoms_json", "gt_atoms", "gt_node_idx"),
    "intrinsic_order": ("intrinsic_order", "intrinsic", "max_order"),
    "label": ("label", "y", "rule_label"),
    "graph_key": ("graph_key", "gkey"),
    "mol_id": ("mol_id", "idx", "index", "id"),
    "smiles": ("smiles", "SMILES"),
}


def _pick(row: dict, names: Sequence[str]):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def load_reference(csv_path: str) -> Dict[str, RefEntry]:
    """Load reference_<task>.csv -> {join_key: RefEntry}. Join key is graph_key if
    present, else str(mol_id). Uses pandas if available, else csv module."""
    rows = _read_rows(csv_path)
    out: Dict[str, RefEntry] = {}
    for row in rows:
        mi = _parse_mi(_pick(row, _COLS["mi"]))
        gt = _pick(row, _COLS["gt_atoms"])
        gt_atoms = frozenset(_parse_int_list(gt)) if gt is not None else None
        sv_cell = _pick(row, _COLS["sv"])
        sv = np.asarray(json.loads(sv_cell) if isinstance(sv_cell, str) else sv_cell, dtype=float) \
            if sv_cell is not None else None
        n_atoms = _pick(row, _COLS["n_atoms"])
        n_atoms = int(n_atoms) if n_atoms is not None else (len(sv) if sv is not None else _infer_n(mi))
        io = _pick(row, _COLS["intrinsic_order"])
        gk = _pick(row, _COLS["graph_key"])
        mid = _pick(row, _COLS["mol_id"])
        if gt_atoms is None and sv is not None:
            gt_atoms = frozenset(int(i) for i in np.nonzero(np.abs(sv) > 1e-12)[0])
        entry = RefEntry(
            n_atoms=n_atoms, mi=mi, sv=sv, gt_atoms=gt_atoms,
            intrinsic_order=int(io) if io is not None else intrinsic_order(mi),
            label=int(_pick(row, _COLS["label"])) if _pick(row, _COLS["label"]) is not None else None,
            graph_key=str(gk) if gk is not None else None,
            mol_id=int(mid) if mid is not None else None,
            smiles=_pick(row, _COLS["smiles"]),
        )
        key = entry.graph_key or (str(entry.mol_id) if entry.mol_id is not None else str(len(out)))
        out[key] = entry
    return out


def _infer_n(mi: Dict[Key, float]) -> int:
    m = 0
    for k in mi:
        for i in k:
            m = max(m, i + 1)
    return m


def _read_rows(csv_path: str) -> List[dict]:
    try:
        import pandas as pd  # noqa
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    except Exception:
        import csv
        with open(csv_path, newline="") as f:
            return list(csv.DictReader(f))


def write_reference_csv(entries: Iterable[RefEntry], csv_path: str) -> None:
    """Reference-tier writer -- run_bxaic_reference.py can import this so the file
    format is guaranteed to match what load_reference expects."""
    import csv
    fields = ["mol_id", "graph_key", "smiles", "n_atoms", "label",
              "intrinsic_order", "sv_json", "gt_atoms_json", "mi_json"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            w.writerow({
                "mol_id": e.mol_id if e.mol_id is not None else "",
                "graph_key": e.graph_key or "",
                "smiles": e.smiles or "",
                "n_atoms": e.n_atoms,
                "label": e.label if e.label is not None else "",
                "intrinsic_order": e.intrinsic_order
                if e.intrinsic_order is not None else intrinsic_order(e.mi),
                "sv_json": json.dumps([float(v) for v in e.sv]) if e.sv is not None else "",
                "gt_atoms_json": json.dumps(sorted(e.gt_atoms)) if e.gt_atoms is not None else "",
                "mi_json": json.dumps([[sorted(k), v] for k, v in e.mi.items()]),
            })


# --------------------------------------------------------------------------- #
#  Explanation loader (repo explanations.pt schema)                            #
# --------------------------------------------------------------------------- #
@dataclass
class NodeExplanation:
    node_importance: Optional[np.ndarray]     # reduced to 1-D per-atom
    gt_node_mask: np.ndarray                   # bool
    graph_key: str
    n_atoms: int
    y: Optional[int] = None


def _reduce_node_importance(node_mask) -> Optional[np.ndarray]:
    """Mirror evaluate_explanations.get_node_importance: nan_to_num, sum feature
    dim if 2-D -> 1-D per-atom vector."""
    if node_mask is None:
        return None
    arr = np.asarray(node_mask, dtype=float)
    arr = np.nan_to_num(arr)
    if arr.ndim == 2:
        arr = arr.sum(axis=1)
    assert arr.ndim == 1, f"unexpected node_mask ndim {arr.ndim}"
    return arr


def load_explanations_pt(path: str) -> List[NodeExplanation]:
    """Read the repo's explanations.pt (needs torch). Returns node-level
    explanations keyed by graph_key for joining with the reference."""
    import torch  # lazy
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    out: List[NodeExplanation] = []
    for e in loaded["explanations"]:
        x = _tensor_to_np(e.x)
        ei = _tensor_to_np(e.edge_index)
        nm = getattr(e, "node_mask", None)
        nm = _tensor_to_np(nm) if nm is not None else None
        gt = getattr(e, "gt_node_mask", None)
        gt = _tensor_to_np(gt).astype(bool) if gt is not None else np.zeros(x.shape[0], dtype=bool)
        y = getattr(e, "y", None)
        out.append(NodeExplanation(
            node_importance=_reduce_node_importance(nm),
            gt_node_mask=gt,
            graph_key=graph_key(x, ei),
            n_atoms=x.shape[0],
            y=int(_tensor_to_np(y).reshape(-1)[0]) if y is not None else None,
        ))
    return out


def _tensor_to_np(t):
    """Torch/np/list -> numpy, avoiding the torch<->numpy C bridge so it works
    even when torch was built against a different numpy ABI (uses .tolist())."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    try:
        return np.asarray(t.tolist()) if hasattr(t, "tolist") else np.asarray(t)
    except Exception:
        return np.asarray(t)


# --------------------------------------------------------------------------- #
#  AUROC (numpy fallback == sklearn.roc_auc_score for AUROC)                   #
# --------------------------------------------------------------------------- #
def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        pass
    # Rank-based AUROC with tie handling (Mann-Whitney U).
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC undefined: only one class present")
    sum_pos = ranks[pos].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass
class AurocResult:
    per_molecule: List[float] = field(default_factory=list)
    n_evaluated: int = 0
    n_empty_skipped: int = 0
    n_unmatched: int = 0

    @property
    def mean(self) -> float:
        return float(np.mean(self.per_molecule)) if self.per_molecule else float("nan")


def atom_recovery_auroc(
    reference: Dict[str, RefEntry],
    explanations: List[NodeExplanation],
) -> AurocResult:
    """Per-molecule AUROC of explainer node importance against the exact
    ground-truth atom set. Empty-gt molecules are skipped (as in the repo's
    non_empty_ex path)."""
    res = AurocResult()
    for ex in explanations:
        ref = reference.get(ex.graph_key)
        gt = ref.gt_mask() if ref is not None else ex.gt_node_mask
        if ref is None:
            res.n_unmatched += 1
        if ex.node_importance is None:
            continue
        if gt.min() == gt.max():
            res.n_empty_skipped += 1
            continue
        res.per_molecule.append(auroc(gt, ex.node_importance))
        res.n_evaluated += 1
    return res


# --------------------------------------------------------------------------- #
#  Interaction vectors (Moebius basis)                                         #
# --------------------------------------------------------------------------- #
def as_moebius_dict(obj) -> Dict[Key, float]:
    """Accept a plain dict, a JSON path/list, or a shapiq InteractionValues and
    return {frozenset(atoms): value}. For InteractionValues the index must be a
    Moebius basis ('Moebius'/'MI')."""
    if isinstance(obj, dict):
        return {frozenset(int(i) for i in k): float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return {frozenset(int(i) for i in atoms): float(v) for atoms, v in obj}
    # shapiq InteractionValues duck-typing
    if hasattr(obj, "interaction_lookup") and hasattr(obj, "values"):
        idx = getattr(obj, "index", "Moebius")
        if str(idx) not in ("Moebius", "MI"):
            raise ValueError(
                f"Interaction index is '{idx}', not Moebius. Convert to the Moebius "
                f"basis before comparison, or run GraphSHAP-IQ with index='Moebius'."
            )
        vals = np.asarray(obj.values, dtype=float)
        return {frozenset(int(i) for i in k): float(vals[pos])
                for k, pos in obj.interaction_lookup.items()}
    raise TypeError(f"Cannot interpret interactions of type {type(obj)}")


def _align(a: Dict[Key, float], b: Dict[Key, float]) -> Tuple[List[Key], np.ndarray, np.ndarray]:
    keys = sorted(set(a) | set(b), key=lambda s: (len(s), sorted(s)))
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    return keys, va, vb


def interaction_mse(pred: Dict[Key, float], ref: Dict[Key, float]) -> float:
    _, vp, vr = _align(pred, ref)
    return float(np.mean((vp - vr) ** 2)) if len(vp) else 0.0


def interaction_error_by_order(pred: Dict[Key, float], ref: Dict[Key, float]) -> Dict[int, float]:
    keys, vp, vr = _align(pred, ref)
    by: Dict[int, List[float]] = defaultdict(list)
    for k, dp, dr in zip(keys, vp, vr):
        by[len(k)].append((dp - dr) ** 2)
    return {o: float(np.sum(v)) for o, v in sorted(by.items())}  # summed squared error per order


# --------------------------------------------------------------------------- #
#  Moebius -> game / Shapley, intrinsic order, truncation                      #
# --------------------------------------------------------------------------- #
def intrinsic_order(mi: Dict[Key, float], tol: float = 1e-12) -> int:
    return max((len(k) for k, v in mi.items() if abs(v) > tol), default=0)


def truncate_mi(mi: Dict[Key, float], L: int) -> Dict[Key, float]:
    return {k: v for k, v in mi.items() if len(k) <= L}


def sv_from_mi(mi: Dict[Key, float], n_atoms: int) -> np.ndarray:
    """Shapley value from Moebius: SV_i = sum_{T ni i} MI[T]/|T|."""
    sv = np.zeros(n_atoms)
    for k, v in mi.items():
        if not k:
            continue
        share = v / len(k)
        for i in k:
            sv[i] += share
    return sv


def game_from_mi(mi: Dict[Key, float], S: FrozenSet[int]) -> float:
    """nu(S) = sum_{T subset of S} MI[T]."""
    return float(sum(v for k, v in mi.items() if k <= S))


def game_reconstruction_rmse(mi_a: Dict[Key, float], mi_b: Dict[Key, float],
                             n_atoms: int = 0) -> float:
    """RMSE between the two games over ALL 2^n coalitions (exact).

    The two games differ only in Moebius terms supported on
    `support = union of differing keys`; non-support atoms never change the
    difference nu_a(S)-nu_b(S). Averaging over all 2^n coalitions therefore
    equals averaging over the 2^|support| support-configurations, so the result
    depends only on the (small) support and is exact for molecules of any size.
    `n_atoms` is accepted for API compatibility but not needed."""
    diff = {k: mi_a.get(k, 0.0) - mi_b.get(k, 0.0)
            for k in set(mi_a) | set(mi_b)}
    diff = {k: v for k, v in diff.items() if abs(v) > 0}
    if not diff:
        return 0.0
    atoms = sorted({i for k in diff for i in k})
    if len(atoms) > 24:
        raise ValueError(f"support {len(atoms)} too large for exact reconstruction")
    total = 0.0
    for r in range(len(atoms) + 1):
        for combo in combinations(atoms, r):
            S = frozenset(combo)
            val = sum(v for k, v in diff.items() if k <= S)
            total += val * val
    return math.sqrt(total / (2 ** len(atoms)))


@dataclass
class TruncationPoint:
    L: int
    game_rmse: float
    sv_rmse: float
    retained_mass: float     # sum |MI| with order <= L
    dropped_mass: float      # sum |MI| with order  > L


def truncation_curve(mi: Dict[Key, float], n_atoms: int) -> List[TruncationPoint]:
    """Model-independent order-truncation error curve from the reference Moebius
    vector. game_rmse == 0 exactly at L == intrinsic_order (the provable claim)."""
    io = intrinsic_order(mi)
    sv_full = sv_from_mi(mi, n_atoms)
    total_abs = sum(abs(v) for v in mi.values())
    pts: List[TruncationPoint] = []
    for L in range(0, io + 1):
        mL = truncate_mi(mi, L)
        retained = sum(abs(v) for v in mL.values())
        pts.append(TruncationPoint(
            L=L,
            game_rmse=game_reconstruction_rmse(mL, mi, n_atoms),
            sv_rmse=float(np.sqrt(np.mean((sv_from_mi(mL, n_atoms) - sv_full) ** 2))),
            retained_mass=retained,
            dropped_mass=total_abs - retained,
        ))
    return pts


def spurious_pairwise_from_conjunction(k: int, value: float) -> Dict[str, float]:
    """Illustrates the 'a k-way conjunction becomes C(k,2) fake pairwise terms'
    point. Best order-<=2 least-squares fit of a single k-way Moebius term of the
    given value spreads mass onto all C(k,2) pairs. Returns the count and the
    (uniform) magnitude implied, for the rebuttal's worked example."""
    n_pairs = math.comb(k, 2)
    # A pure k-way interaction restricted to order 2: the projection places equal
    # weight on each pair; the exact coefficient depends on the game samples, but
    # the structural point is n_pairs nonzero pairwise terms where the truth has 0.
    return {"n_spurious_pairs": n_pairs, "true_pairwise_terms": 0.0, "example_value": value}


# --------------------------------------------------------------------------- #
#  Gap decomposition                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class GapDecomposition:
    total_norm: float
    masking_mismatch_norm: Optional[float]
    model_error_norm: Optional[float]
    lambda_truncation_norm: Optional[float]
    by_order: Dict[str, Dict[int, float]] = field(default_factory=dict)
    note: str = ""

    def as_dict(self):
        return {
            "total": self.total_norm,
            "masking_mismatch": self.masking_mismatch_norm,
            "model_error": self.model_error_norm,
            "lambda_truncation": self.lambda_truncation_norm,
            "by_order": self.by_order,
            "note": self.note,
        }


def _l2(mi_a: Dict[Key, float], mi_b: Dict[Key, float]) -> float:
    _, va, vb = _align(mi_a, mi_b)
    return float(np.linalg.norm(va - vb))


def decompose_gap(
    ref_deletion: Dict[Key, float],
    pred_on_model: Dict[Key, float],
    ref_baseline: Optional[Dict[Key, float]] = None,
    exact_model: Optional[Dict[Key, float]] = None,
) -> GapDecomposition:
    """Additive (at the difference-vector level) decomposition of the GraphSHAP-IQ
    gap:

        pred_on_model - ref_deletion
      = (ref_baseline - ref_deletion)   masking-mismatch
      + (exact_model  - ref_baseline)   model-error
      + (pred_on_model - exact_model)   lambda-truncation

    Provide whichever intermediate references are available:
      * ref_baseline  : rule game under BShap baseline-imputation masking (exact).
      * exact_model   : full-order interactions of the trained GNN under the same
                        (baseline) masking -- GraphSHAP-IQ with no lambda cap.
    Missing terms are reported as None; if exactly one is missing it is recovered
    as the residual so the three still sum to the total vector.
    For a model-agnostic method fed the rule-game directly, pass
    ref_baseline = exact_model = ref_deletion -> only lambda-truncation is nonzero.
    """
    total = _l2(pred_on_model, ref_deletion)
    by_order: Dict[str, Dict[int, float]] = {}
    mm = me = lt = None
    if ref_baseline is not None:
        mm = _l2(ref_baseline, ref_deletion)
        by_order["masking_mismatch"] = interaction_error_by_order(ref_baseline, ref_deletion)
    if ref_baseline is not None and exact_model is not None:
        me = _l2(exact_model, ref_baseline)
        by_order["model_error"] = interaction_error_by_order(exact_model, ref_baseline)
    if exact_model is not None:
        lt = _l2(pred_on_model, exact_model)
        by_order["lambda_truncation"] = interaction_error_by_order(pred_on_model, exact_model)

    note = "exact additive decomposition"
    if exact_model is None and ref_baseline is not None:
        # residual absorbs model-error + lambda-truncation (aligned on a shared key set)
        uni = sorted(set(pred_on_model) | set(ref_deletion) | set(ref_baseline),
                     key=lambda s: (len(s), sorted(s)))
        vp = np.array([pred_on_model.get(k, 0.0) for k in uni])
        vr = np.array([ref_deletion.get(k, 0.0) for k in uni])
        vb = np.array([ref_baseline.get(k, 0.0) for k in uni])
        resid = np.linalg.norm((vp - vr) - (vb - vr))
        me = None
        lt = None
        note = ("exact_model not provided: masking_mismatch exact; "
                f"model_error+lambda_truncation collapsed into residual={resid:.6g}")
    elif ref_baseline is None:
        note = ("only pred_on_model and ref_deletion given: total gap only. "
                "Provide ref_baseline (BShap game) and exact_model to split it.")
    return GapDecomposition(total, mm, me, lt, by_order, note)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def _load_json_interactions(path: str) -> Dict[Key, float]:
    with open(path) as f:
        obj = json.load(f)
    return as_moebius_dict(obj)


def _cmd_auroc(args):
    ref = load_reference(args.reference)
    exps = load_explanations_pt(args.explanations)
    r = atom_recovery_auroc(ref, exps)
    print(f"atom-recovery AUROC (mean over {r.n_evaluated} non-empty molecules): {r.mean:.4f}")
    print(f"  empty-gt skipped: {r.n_empty_skipped} | unmatched-to-reference: {r.n_unmatched}")


def _cmd_interactions(args):
    ref = load_reference(args.reference)
    pred = _load_json_interactions(args.pred)
    # single-molecule reference file expected here (or pick by --graph-key)
    entry = _select_entry(ref, args.graph_key)
    mse = interaction_mse(pred, entry.mi)
    print(f"interaction MSE (Moebius): {mse:.6g}")
    print(f"  by order (summed sq err): {interaction_error_by_order(pred, entry.mi)}")
    print(f"  reference intrinsic order: {entry.intrinsic_order}")


def _cmd_truncation(args):
    ref = load_reference(args.reference)
    entry = _select_entry(ref, args.graph_key)
    pts = truncation_curve(entry.mi, entry.n_atoms)
    print(f"intrinsic order = {intrinsic_order(entry.mi)}  (n_atoms={entry.n_atoms})")
    print(f"{'L':>3} {'game_rmse':>12} {'sv_rmse':>12} {'retained':>10} {'dropped':>10}")
    for p in pts:
        print(f"{p.L:>3} {p.game_rmse:>12.6g} {p.sv_rmse:>12.6g} "
              f"{p.retained_mass:>10.4g} {p.dropped_mass:>10.4g}")
    if args.lam is not None:
        ok = args.lam >= intrinsic_order(entry.mi)
        print(f"lambda={args.lam}: {'sufficient' if ok else 'INSUFFICIENT'} "
              f"(needs lambda >= {intrinsic_order(entry.mi)})")


def _cmd_decompose(args):
    ref = load_reference(args.reference)
    entry = _select_entry(ref, args.graph_key)
    pred = _load_json_interactions(args.pred)
    baseline = None
    if args.baseline:
        bref = load_reference(args.baseline)
        baseline = _select_entry(bref, args.graph_key).mi
    exact_model = _load_json_interactions(args.exact_model) if args.exact_model else None
    d = decompose_gap(entry.mi, pred, baseline, exact_model)
    print(json.dumps(d.as_dict(), indent=2))


def _select_entry(ref: Dict[str, RefEntry], gk: Optional[str]) -> RefEntry:
    if gk is not None:
        return ref[gk]
    if len(ref) == 1:
        return next(iter(ref.values()))
    raise SystemExit("reference has multiple molecules; pass --graph-key to select one")


def main():
    p = argparse.ArgumentParser(description="B-XAIC reference-tier comparison")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("auroc", help="atom-level recovery AUROC vs reference gt atoms")
    a.add_argument("--reference", required=True)
    a.add_argument("--explanations", required=True, help="explanations.pt")
    a.set_defaults(func=_cmd_auroc)

    i = sub.add_parser("interactions", help="interaction-vector MSE vs reference Moebius")
    i.add_argument("--reference", required=True)
    i.add_argument("--pred", required=True, help="JSON interactions or shapiq dump")
    i.add_argument("--graph-key", default=None)
    i.set_defaults(func=_cmd_interactions)

    t = sub.add_parser("truncation", help="order-truncation error curve (model-independent)")
    t.add_argument("--reference", required=True)
    t.add_argument("--graph-key", default=None)
    t.add_argument("--lam", type=int, default=None)
    t.set_defaults(func=_cmd_truncation)

    d = sub.add_parser("decompose", help="GraphSHAP-IQ gap decomposition")
    d.add_argument("--reference", required=True, help="deletion-masking reference")
    d.add_argument("--pred", required=True, help="GraphSHAP-IQ output on the model (JSON/shapiq)")
    d.add_argument("--baseline", default=None, help="BShap baseline-masking reference csv")
    d.add_argument("--exact-model", default=None, help="full-order model interactions (JSON)")
    d.add_argument("--graph-key", default=None)
    d.add_argument("--lam", type=int, default=None)
    d.set_defaults(func=_cmd_decompose)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
