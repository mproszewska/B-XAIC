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
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import GINConv

from sklearn.metrics import roc_auc_score

from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
    PGExplainer,
    CaptumExplainer,
    GraphMaskExplainer,
)

from dataset import XAIMolecularDataset
from train_model import get_model, set_seed, test

import sys

sys.path.append(f"{os.getcwd()}/libs/ProtGNN")
from libs.ProtGNN.Configures import model_args
from libs.ProtGNN.models import GnnNets

EXPLAINERS = [
    "GNNExplainer",
    "PGExplainer",
    "IntegratedGradients",
    "ShapleyValueSampling",
    "Saliency",
    "InputXGradient",
    "Deconvolution",
    "GuidedBackprop",
    "GraphMaskExplainer",
]


def args_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--trials", type=int, default=1, help="number of trials")
    parser.add_argument("--save_path", type=str, required=True, help="path to save explanations")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument(
        "--explainer_type",
        type=str,
        required=True,
        choices=EXPLAINERS,
        help="explainer type",
    )
    parser.add_argument("--explanation_type", type=str, required=True, choices=["phenomenon", "model"])
    parser.add_argument(
        "--node_mask_type",
        type=str,
        required=True,
        choices=["object", "none", "attributes", "common_attributes"],
    )
    parser.add_argument(
        "--edge_mask_type",
        type=str,
        required=True,
        choices=["object", "none", "attributes", "common_attributes"],
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="n_samples in ShapleyValueSampling",
    )
    parser.add_argument("--save_all", type=bool, default=False)
    args = parser.parse_args()
    return args


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch):
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return self.model(data)[0]


def main():
    args = args_parser()
    if args.node_mask_type == "none":
        args.node_mask_type = None
    if args.edge_mask_type == "none":
        args.edge_mask_type = None
    print(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = torch.load(args.model_path, map_location=torch.device("cpu"), weights_only=False)
    loaded_args = loaded["args"]

    dataset = XAIMolecularDataset("data", loaded_args["task"])
    train_idxs, val_idxs, test_idxs = dataset.get_splits(loaded_args["split"])
    dataset_train, dataset_val, dataset_test = (
        Subset(dataset, train_idxs),
        Subset(dataset, val_idxs),
        Subset(dataset, test_idxs),
    )

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    try:
        model = get_model(**loaded["model_args"]).to(device)
        model.load_state_dict(loaded["state_dict"])
        num_layers = loaded["model_args"]["num_layers"]
        # Setting up params used by an explainer
        for module in model.modules():
            if isinstance(module, MessagePassing):
                if not hasattr(module, "in_channels"):
                    print(module.__dict__)
                    channel_list = module.nn.channel_list
                    module.in_channels = channel_list[0]
                    module.out_channels = channel_list[-1]
    except:
        model_args.model_name = loaded["args"]["model_type"]
        model_args.readout = loaded["args"]["readout"]
        input_dim = dataset.num_node_features
        output_dim = int(dataset.num_classes)
        protgnn = GnnNets(input_dim, output_dim, model_args)
        for layer in protgnn.model.gnn_layers:
            if isinstance(layer, GINConv):
                layer.in_channels = layer.nn[0].in_features
                layer.out_channels = layer.nn[0].out_features
        num_layers = len(protgnn.model.gnn_layers)
        protgnn.load_state_dict(loaded["state_dict"])
        model = ModelWrapper(protgnn.model).to(device)
    model.eval()

    _, f1 = test(model, dataloader_test, device)
    assert np.abs(f1 - loaded["f1"]) < 0.01, f"f1 score: {f1:.2f} != {loaded['f1']:.2f}"

    model_config = dict(mode="multiclass_classification", task_level="graph", return_type="raw")
    if args.explainer_type == "PGExplainer":
        algorithm = PGExplainer(epochs=args.epochs, lr=args.lr).to(device)
    elif args.explainer_type == "GNNExplainer":
        algorithm = GNNExplainer(epochs=args.epochs, lr=args.lr).to(device)
    elif args.explainer_type == "ShapleyValueSampling":
        algorithm = CaptumExplainer("ShapleyValueSampling", n_samples=5)
    elif args.explainer_type == "GraphMaskExplainer":
        algorithm = GraphMaskExplainer(num_layers=num_layers, epochs=args.epochs, lr=args.lr)
    else:
        algorithm = CaptumExplainer(args.explainer_type)

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type=args.explanation_type,
        node_mask_type=args.node_mask_type,
        edge_mask_type=args.edge_mask_type,
        model_config=model_config,
    )

    if args.explainer_type == "PGExplainer":
        for epoch in range(args.epochs):
            for data in dataloader_test:
                data = data.to(device)
                target = data.y if args.explanation_type != "model" else None
                explainer.algorithm.train(
                    epoch,
                    model,
                    data.x,
                    data.edge_index,
                    batch=data.batch,
                    target=target,
                )

    explanations = list()
    for batch in tqdm(dataloader_test):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y if args.explanation_type != "model" else None
        node_mask = list()
        edge_mask = list()
        for _ in range(args.trials):
            e = explainer(batch.x, batch.edge_index, batch=batch.batch, target=target)
            if hasattr(e, "node_mask"):
                node_mask.append(e.node_mask)
            if hasattr(e, "edge_mask"):
                edge_mask.append(e.edge_mask)

        if len(node_mask) > 0:
            node_mask = torch.stack(node_mask)
            node_mask = node_mask.mean(dim=0)
        else:
            node_mask = None
        if len(edge_mask) > 0:
            edge_mask = torch.stack(edge_mask)
            edge_mask = edge_mask.mean(dim=0)
        else:
            edge_mask = None

        if (node_mask is not None) and (node_mask.shape[0] != batch.expl_node_mask.shape[0]):
            assert edge_mask is not None
            node_mask = None

        num_nodes = 0
        for b in range(batch.batch.max() + 1):
            x_mask = batch.batch == b
            edge_index_mask = batch.batch[batch.edge_index[0]] == b
            edge_index = batch.edge_index[:, edge_index_mask] - num_nodes
            data = Data(
                x=batch.x[x_mask].detach().cpu(),
                edge_index=edge_index.detach().cpu(),
                y=batch.y[b].detach().cpu(),
                pred=pred[b].detach().cpu(),
                node_mask=node_mask[x_mask].detach().cpu() if node_mask is not None else None,
                edge_mask=edge_mask[edge_index_mask].detach().cpu() if edge_mask is not None else None,
                gt_node_mask=batch.expl_node_mask[x_mask].detach().cpu() if hasattr(batch, "expl_node_mask") else None,
                gt_edge_mask=(
                    batch.expl_edge_mask[edge_index_mask].detach().cpu() if hasattr(batch, "expl_edge_mask") else None
                ),
            )

            explanations.append(data)
            num_nodes += x_mask.sum()

    print(f"Finished with {len(explanations)}/{len(dataset_test)} explanations")
    torch.save(
        {
            "args": vars(args),
            "model_args": loaded_args,
            "explanations": explanations,
            "f1": loaded["f1"],
        },
        args.save_path,
    )
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
