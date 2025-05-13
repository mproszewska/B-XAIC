import argparse
import copy
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset

from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, subgraph

from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import XAIMolecularDataset
from train_model import set_seed


def args_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--save_path", type=str, default=None, help="path to save results")
    parser.add_argument("--explanations_path", type=str, required=True)
    parser.add_argument("--mask_to_eval", type=str, default="node", choices=["node", "edge"])
    args = parser.parse_args()
    return args


def check_if_eval_possible(e, data, mask_to_eval):
    if not hasattr(data, f"expl_{mask_to_eval}_mask"):
        print(f"No ground truth {mask_to_eval} mask")
        return False

    if mask_to_eval == "node":
        node_mask = e.node_mask if hasattr(e, "node_mask") else None
        if node_mask is None:
            print("No node mask")
            return False
        elif node_mask.shape[0] != e.gt_node_mask.shape[0]:
            print("Invalid node mask shape")
            return False
        else:
            return True

    elif mask_to_eval == "edge":
        edge_mask = e.edge_mask if hasattr(e, "edge_mask") else None
        if edge_mask is None:
            print("No edge mask")
            return False
        else:
            return True

    else:
        assert False, mask_to_eval


def get_node_importance(e, data):
    node_mask = e.node_mask.detach() if hasattr(e, "node_mask") else None
    node_mask = torch.nan_to_num(node_mask)
    if len(node_mask.shape) == 2:
        node_mask = node_mask.sum(-1)
    assert len(node_mask.shape) == 1
    return node_mask


def get_edge_importance(e, data):
    edge_mask = e.edge_mask.detach() if hasattr(e, "edge_mask") else None
    edge_mask = torch.nan_to_num(edge_mask)
    assert len(edge_mask.shape) == 1
    return edge_mask


def no_outliers(x):
    Q1 = torch.quantile(x, 0.25)
    Q3 = torch.quantile(x, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = x[(x < lower_bound) | (x > upper_bound)]
    return float(len(outliers) == 0)


def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)

    loaded = torch.load(args.explanations_path, weights_only=False)
    task = loaded["model_args"]["task"]
    dataset = XAIMolecularDataset("data", task)
    explanations = loaded["explanations"]
    if not check_if_eval_possible(explanations[0], dataset[0], args.mask_to_eval):
        print(f"Eval not possible for {args.explanations_path}")
        exit(0)

    metric_fn = roc_auc_score
    print(f"Task: {task} Eval metric: {metric_fn.__name__}")

    _, _, test_idxs = dataset.get_splits(loaded["model_args"]["split"])
    dataset_test = Subset(dataset, test_idxs)

    non_empty_ex, empty_ex = list(), list()
    for idx, e, data in tqdm(zip(test_idxs, explanations, dataset_test), total=len(explanations)):
        if args.mask_to_eval == "node":
            if hasattr(e, "gt_node_mask"):
                assert (e.gt_node_mask == data.expl_node_mask).all()
            pred_mask = get_node_importance(e, data)
            gt = data.expl_node_mask
        elif args.mask_to_eval == "edge":
            if hasattr(e, "gt_edge_mask"):
                assert (e.gt_edge_mask == data.expl_edge_mask).all()
            pred_mask = get_edge_importance(e, data)
            gt = data.expl_edge_mask
        else:
            assert False, args.mask_to_eval

        assert pred_mask.shape == gt.shape, (pred_mask.shape, gt.shape)
        if gt.min() == gt.max():
            m = no_outliers(pred_mask)
            empty_ex.append(m)
        else:
            m = metric_fn(gt, pred_mask)
            non_empty_ex.append(m)
    metrics = {
        "non_empty_ex": torch.tensor(non_empty_ex),
        "empty_ex": torch.tensor(empty_ex),
        "metric": metric_fn.__name__,
        "f1": loaded["f1"],
        "mask_to_eval": args.mask_to_eval,
    }

    assert len(non_empty_ex) + len(empty_ex) == len(explanations)
    num_non_empty_ex = len(non_empty_ex) / len(explanations)
    num_empty_ex = len(empty_ex) / len(explanations)
    metric_non_empty_ex = np.mean(non_empty_ex)
    metric_empty_ex = np.mean(empty_ex)
    print(f"Non empty: {num_non_empty_ex} | Avg: {metric_non_empty_ex:.4f}")
    print(f"Empty: {num_empty_ex} | Avg: {metric_empty_ex:.4f}")

    if args.save_path is not None:
        torch.save(
            {
                "explainer_args": loaded["args"],
                "model_args": loaded["model_args"],
                "metrics": metrics,
            },
            args.save_path,
        )
        print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
