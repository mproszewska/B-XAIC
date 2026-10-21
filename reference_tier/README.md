# B-XAIC reference tier

Exact rule-game ground truth for molecular XAI, with model-independent unit tests
and on-model evaluation of trained GNNs. Companion code to the B-XAIC benchmark.

The key idea: define a cooperative game `ν(T) = rule(delete V\T)` directly from each
task's labelling rule. Because `ν` *is* the rule, its exact Shapley values and Möbius
interactions are the ground-truth explanation — no trained model, no thresholding.
Any Shapley/interaction explainer can then be unit-tested against an exact target.

## Contents

    compare.py                core library + CLI (Möbius/Shapley, graph_key join,
                              atom-recovery AUROC, interaction MSE, truncation
                              curves, additive gap decomposition)
    run_bxaic_reference.py    build exact reference_<task>.csv from data.csv + explanations.sdf
    run_onmodel.py            exact Shapley/Möbius of a trained GIN (node-deletion
                              masking), resumable; writes onmodel_<task>.csv
    gini_ne.py                continuous null-explanation metric (node-attribution
                              Gini impurity on negative-class molecules)
    summarize_reference.py    full-dataset intrinsic-order + label-match summary
    test_compare.py           44 self-checks for compare.py (no data/torch needed)
    results/
      intrinsic_order_summary.csv   full 50k-molecule summary (below)
      onmodel_results.csv           on-model GIN table (below)

## Install

    pip install -r requirements.txt
    # numpy is pinned <2 for torch 2.2.2 ABI compatibility; the reference tier
    # itself (reference/compare/gini) works with any recent numpy.

## Reproduce

Exact reference per task (writes reference_<task>.csv):

    for t in B P X indole PAINS rings-max; do
      python run_bxaic_reference.py --data_dir data --task $t
    done
    # rings-count is circuit-rank / global: exact only for <=~20 ring atoms;
    # its positives exceed this (sparse-Möbius over ring occurrences is future work).

Full-dataset summary (fast; counts marked atoms):

    python summarize_reference.py            # -> results/intrinsic_order_summary.csv

Model-agnostic unit test of an interaction method against the exact reference:

    python compare.py truncation   --reference reference_indole.csv --lam 2
    python compare.py interactions --reference reference_indole.csv --pred pred_moebius.json
    python compare.py decompose    --reference reference_indole.csv --pred pred.json \
        --baseline reference_indole_bshap.csv --exact-model exact_model.json

On-model table (exact GIN Shapley/Möbius vs reference; resumable, re-run to finish):

    for t in B P X indole PAINS rings-max; do
      python run_onmodel.py --task $t --time_budget 120
    done

Atom-recovery AUROC from a real explanations.pt (repo format):

    python compare.py auroc --reference reference_indole.csv --explanations explanations.pt

Continuous null-explanation (Gini) on negative molecules:

    python gini_ne.py --task PAINS --n_mols 30 --max_atoms 12

Sanity (no data/torch):

    python test_compare.py                   # -> 44 passed

## Results (shipped in results/)

Full dataset (all 50,000 molecules). `ν(V)` reproduces the published label for
100% of molecules on every task below. Intrinsic interaction order = size of the
largest non-zero Möbius term (= number of ground-truth atoms for these tasks):

| Task | #positive | intrinsic order (min / median / mean / max) |
|---|---|---|
| B | 1,116 | 1 / 1 / 1.0 / 5 |
| P | 6,612 | 1 / 1 / 1.5 / 29 |
| X | 27,934 | 1 / 1 / 1.8 / 20 |
| indole | 18,403 | 9 / 9 / 10.3 / 108 |
| PAINS | 16,446 | 2 / 9 / 9.7 / 90 |
| rings-max | 2,841 | 7 / 9 / 18.1 / 490 |

On-model (trained GIN, exact Shapley/Möbius under node-deletion masking, small-molecule
subset so 2^n is exact; masking-mismatch = 0 by construction):

| Task | GIN F1 | #mols | atom-recovery AUROC (mean / min) | model-error ‖·‖ | λ=2 trunc ‖·‖ |
|---|---|---|---|---|---|
| B | 1.00 | 87 | 1.00 / 1.00 | 2.07 | 1.61 |
| P | 1.00 | 100 | 1.00 / 1.00 | 2.42 | 1.96 |
| X | 1.00 | 100 | 1.00 / 0.91 | 1.52 | 0.98 |
| indole | 0.98 | 78 | 0.98 / 0.82 | 1.81 | 1.64 |
| PAINS | 0.93 | 100 | 0.77 / 0.10 | 3.41 | 3.12 |
| rings-max | 0.92 | 22 | 0.76 / 0.54 | 1.84 | 1.47 |

## Scope / known limits

- The pattern reference (indole/PAINS) is exact for disjoint occurrences; when two
  occurrences share atoms the current construction merges them (approximation).
  Plugging the per-occurrence SMARTS matcher into `build_nu` makes it exact.
- `rings-count` has no small positives (>4 rings needs a large molecule), so exact
  on-model is infeasible there; both rings tasks await the sparse-Möbius extension.
- `run_onmodel.py` builds a GIN (train_model.get_model). For GCN/GAT/ProtGNN, swap
  `build_gin()` for the matching constructor.

## License

Code: add your preferred license before release. Dataset: CC-BY-SA (see the B-XAIC
repository). This code is a companion to B-XAIC (Proszewska, Danel, Rymarczyk et al.).
