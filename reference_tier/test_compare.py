"""
Self-contained validation for compare.py. Uses an INDEPENDENT brute-force
Moebius transform (mi[T] = sum_{S subset T} (-1)^{|T-S|} nu(S)) to build exact
ground-truth games for the canonical B-XAIC task shapes, then checks every
comparison metric against hand-derived expectations. No torch / HF / real data.

Run:  python test_compare.py
"""
import json
import math
import os
import tempfile
from itertools import combinations

import numpy as np

import compare as C


# ----- independent exact game -> Moebius (do NOT reuse compare's helpers) ----
def moebius_from_game(nu, n):
    atoms = list(range(n))
    mi = {}
    for r in range(n + 1):
        for T in combinations(atoms, r):
            Tset = frozenset(T)
            s = 0.0
            for k in range(len(T) + 1):
                for S in combinations(T, k):
                    s += ((-1) ** (len(T) - len(S))) * nu(frozenset(S))
            if abs(s) > 1e-12:
                mi[Tset] = s
    return mi


def shapley_from_game(nu, n):
    sv = np.zeros(n)
    for i in range(n):
        others = [a for a in range(n) if a != i]
        for r in range(len(others) + 1):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in combinations(others, r):
                Sset = frozenset(S)
                sv[i] += w * (nu(Sset | {i}) - nu(Sset))
    return sv


# ----- task game definitions ------------------------------------------------
def dictator_game(n, atom=0):
    return lambda S: 1.0 if atom in S else 0.0


def or_game(n, atoms):
    atoms = set(atoms)
    return lambda S: 1.0 if (set(S) & atoms) else 0.0


def conjunction_game(n, atoms):
    atoms = frozenset(atoms)
    return lambda S: 1.0 if atoms <= set(S) else 0.0


PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ===========================================================================
print("== 1. Moebius / intrinsic order match hand derivation ==")
n = 8
mi_dict = C.as_moebius_dict  # noqa (referenced for clarity)

mi_B = moebius_from_game(dictator_game(n, 0), n)
check("dictator MI == {{0}:1}", mi_B == {frozenset({0}): 1.0}, str(mi_B))
check("dictator intrinsic order 1", C.intrinsic_order(mi_B) == 1)

mi_X = moebius_from_game(or_game(n, [0, 1]), n)
check("halogen OR MI structure",
      mi_X == {frozenset({0}): 1.0, frozenset({1}): 1.0, frozenset({0, 1}): -1.0}, str(mi_X))
check("halogen intrinsic order 2", C.intrinsic_order(mi_X) == 2)

mi_ind = moebius_from_game(conjunction_game(n, [0, 1, 2, 3]), n)
check("indole conjunction single 4-way term",
      mi_ind == {frozenset({0, 1, 2, 3}): 1.0}, str(mi_ind))
check("indole intrinsic order 4", C.intrinsic_order(mi_ind) == 4)

mi_ring = moebius_from_game(conjunction_game(n, [0, 1, 2, 3, 4, 5]), n)
check("ring intrinsic order 6", C.intrinsic_order(mi_ring) == 6)

# ===========================================================================
print("== 2. Shapley from Moebius == exact Shapley ==")
for name, nu, mi in [
    ("B", dictator_game(n, 0), mi_B),
    ("X", or_game(n, [0, 1]), mi_X),
    ("indole", conjunction_game(n, [0, 1, 2, 3]), mi_ind),
]:
    sv_ref = shapley_from_game(nu, n)
    sv_mi = C.sv_from_mi(mi, n)
    check(f"SV({name}) matches", np.allclose(sv_ref, sv_mi, atol=1e-9),
          f"{sv_ref} vs {sv_mi}")
# efficiency: SV sums to nu(full)-nu(empty)=1 for positive molecules
check("indole SV sums to 1", abs(C.sv_from_mi(mi_ind, n).sum() - 1.0) < 1e-9)
check("indole gt atoms = {0,1,2,3}",
      set(np.nonzero(np.abs(C.sv_from_mi(mi_ind, n)) > 1e-9)[0]) == {0, 1, 2, 3})

# ===========================================================================
print("== 3. Truncation curve: game_rmse == 0 exactly at intrinsic order ==")
pts = C.truncation_curve(mi_ind, n)
io = C.intrinsic_order(mi_ind)
below = [p for p in pts if p.L < io]
at = [p for p in pts if p.L == io][0]
check("game_rmse > 0 for all L < intrinsic order", all(p.game_rmse > 1e-9 for p in below),
      str([(p.L, p.game_rmse) for p in below]))
check("game_rmse == 0 at L == intrinsic order", at.game_rmse < 1e-12, str(at.game_rmse))
check("dropped_mass == 0 only at/above intrinsic order",
      all(p.dropped_mass > 0 for p in below) and at.dropped_mass < 1e-12)
# halogen OR: order 1 truncation drops the -1 pair term -> nonzero error
ptsX = C.truncation_curve(mi_X, n)
check("halogen L=1 truncation has nonzero game error",
      [p for p in ptsX if p.L == 1][0].game_rmse > 1e-9)
check("halogen L=2 exact", [p for p in ptsX if p.L == 2][0].game_rmse < 1e-12)

# ===========================================================================
print("== 4. interaction_mse and by-order ==")
check("identical interactions -> MSE 0", C.interaction_mse(mi_ind, mi_ind) == 0.0)
# a method capped at order 2 sees none of the 4-way term:
mi_trunc2 = C.truncate_mi(mi_ind, 2)
check("order-2 truncated pred misses the 4-way term",
      C.interaction_mse(mi_trunc2, mi_ind) > 0)
byo = C.interaction_error_by_order(mi_trunc2, mi_ind)
check("all interaction error concentrated at order 4",
      set(k for k, v in byo.items() if v > 1e-12) == {4}, str(byo))

# ===========================================================================
print("== 5. Gap decomposition is additive at the vector level ==")
# Build a synthetic on-model scenario for a conjunction task.
ref_del = dict(mi_ind)                                   # rule, deletion masking
ref_base = dict(ref_del); ref_base[frozenset({0, 1})] = 0.15   # masking mismatch term
exact_model = dict(ref_base); exact_model[frozenset({2})] = 0.05  # model-error term
pred_model = C.truncate_mi(exact_model, 2)               # lambda=2 cap on the model game

d = C.decompose_gap(ref_del, pred_model, ref_base, exact_model)
# reconstruct the total difference vector and the sum of the three component vectors
keys = sorted(set(ref_del) | set(ref_base) | set(exact_model) | set(pred_model),
              key=lambda s: (len(s), sorted(s)))
def vec(m): return np.array([m.get(k, 0.0) for k in keys])
total_vec = vec(pred_model) - vec(ref_del)
comp_sum = (vec(ref_base) - vec(ref_del)) + (vec(exact_model) - vec(ref_base)) + (vec(pred_model) - vec(exact_model))
check("component vectors sum to total gap vector", np.allclose(total_vec, comp_sum, atol=1e-12))
check("reported total norm matches", abs(d.total_norm - np.linalg.norm(total_vec)) < 1e-9)
check("masking-mismatch norm correct",
      abs(d.masking_mismatch_norm - np.linalg.norm(vec(ref_base) - vec(ref_del))) < 1e-9)
check("model-error norm correct",
      abs(d.model_error_norm - np.linalg.norm(vec(exact_model) - vec(ref_base))) < 1e-9)
check("lambda-truncation norm correct",
      abs(d.lambda_truncation_norm - np.linalg.norm(vec(pred_model) - vec(exact_model))) < 1e-9)

# unit-test special case: model-agnostic method fed the rule-game directly
d2 = C.decompose_gap(ref_del, C.truncate_mi(ref_del, 2), ref_del, ref_del)
check("unit-test mode: masking mismatch 0", d2.masking_mismatch_norm < 1e-12)
check("unit-test mode: model error 0", d2.model_error_norm < 1e-12)
check("unit-test mode: only lambda-truncation nonzero", d2.lambda_truncation_norm > 0)

# residual path when exact_model not provided
d3 = C.decompose_gap(ref_del, pred_model, ref_base, None)
check("residual path: masking mismatch still exact",
      abs(d3.masking_mismatch_norm - np.linalg.norm(vec(ref_base) - vec(ref_del))) < 1e-9)
check("residual path: model/lambda collapsed to None", d3.model_error_norm is None and d3.lambda_truncation_norm is None)

# ===========================================================================
print("== 6. AUROC numpy fallback + perfect/random behaviour ==")
gt = np.array([1, 1, 0, 0, 0])
perfect = np.array([0.9, 0.8, 0.1, 0.2, 0.05])
check("perfect ranking AUROC == 1", abs(C.auroc(gt, perfect) - 1.0) < 1e-12)
inv = -perfect
check("inverted ranking AUROC == 0", abs(C.auroc(gt, inv) - 0.0) < 1e-12)
# tie handling vs sklearn if present
try:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = rng.integers(0, 2, 12)
        if g.min() == g.max():
            continue
        s = rng.integers(0, 3, 12).astype(float)  # lots of ties
        check("numpy AUROC == sklearn (ties)", abs(C.auroc(g, s) - roc_auc_score(g, s)) < 1e-12,
              f"{C.auroc(g,s)} vs {roc_auc_score(g,s)}")
        break
except Exception as e:
    print(f"  [skip] sklearn not installed ({e})")

# ===========================================================================
print("== 7. Reference CSV round-trip + graph_key identity ==")
# node ordering identical on both sides -> identity key
x = np.eye(11)[[0, 1, 2, 0, 3]]          # 5 atoms, one-hot atom types
edge_index = np.array([[0, 1, 1, 2, 2, 3, 3, 4],
                       [1, 0, 2, 1, 3, 2, 4, 3]])
gk = C.graph_key(x, edge_index)
check("graph_key stable under directed-edge duplication",
      gk == C.graph_key(x, edge_index[:, ::-1]))
mi_small = moebius_from_game(conjunction_game(5, [1, 2, 3]), 5)
entry = C.RefEntry(n_atoms=5, mi=mi_small, sv=C.sv_from_mi(mi_small, 5),
                   gt_atoms=frozenset({1, 2, 3}), intrinsic_order=3, label=1,
                   graph_key=gk, mol_id=42, smiles="CCC")
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "reference_test.csv")
C.write_reference_csv([entry], path)
reloaded = C.load_reference(path)
r = reloaded[gk]
check("round-trip MI equal", r.mi == mi_small)
check("round-trip n_atoms", r.n_atoms == 5)
check("round-trip gt atoms", r.gt_atoms == frozenset({1, 2, 3}))
check("round-trip intrinsic order", r.intrinsic_order == 3)
check("round-trip SV", np.allclose(r.sv, entry.sv))

# ===========================================================================
print("== 8. End-to-end AUROC via NodeExplanation (no torch needed) ==")
# emulate two explainer outputs against the small conjunction reference
ref_map = {gk: r}
good = C.NodeExplanation(node_importance=np.array([0.0, 0.9, 0.8, 0.7, 0.0]),
                         gt_node_mask=r.gt_mask(), graph_key=gk, n_atoms=5)
bad = C.NodeExplanation(node_importance=np.array([0.9, 0.0, 0.1, 0.0, 0.8]),
                        gt_node_mask=r.gt_mask(), graph_key=gk, n_atoms=5)
res_good = C.atom_recovery_auroc(ref_map, [good])
res_bad = C.atom_recovery_auroc(ref_map, [bad])
check("perfect explainer AUROC == 1", abs(res_good.mean - 1.0) < 1e-12)
check("adversarial explainer AUROC == 0", abs(res_bad.mean - 0.0) < 1e-12)
# 2-D node_mask (attributes) reduces via sum over feature dim, like the repo
attr = C.NodeExplanation(
    node_importance=C._reduce_node_importance(np.array([[0, 0], [0.5, 0.4], [0.4, 0.4], [0.3, 0.4], [0, 0]])),
    gt_node_mask=r.gt_mask(), graph_key=gk, n_atoms=5)
check("2-D attribute mask reduces and scores AUROC 1",
      abs(C.atom_recovery_auroc(ref_map, [attr]).mean - 1.0) < 1e-12)

# ===========================================================================
print("== 9. shapiq InteractionValues duck-typing ==")
class FakeIV:
    index = "Moebius"
    def __init__(self, mi):
        self.interaction_lookup = {tuple(sorted(k)): i for i, k in enumerate(mi)}
        self.values = np.array([mi[k] for k in mi])
iv = FakeIV(mi_ind)
check("InteractionValues -> Moebius dict", C.as_moebius_dict(iv) == mi_ind)
try:
    class BadIV(FakeIV):
        index = "k-SII"
    C.as_moebius_dict(BadIV(mi_ind)); check("non-Moebius index rejected", False)
except ValueError:
    check("non-Moebius index rejected", True)

print(f"\n==== {PASS} passed, {FAIL} failed ====")
raise SystemExit(1 if FAIL else 0)
