import argparse
import copy
import random
from tqdm import tqdm

import numpy as np

import torch
from torch.nn import Linear, ReLU, Dropout, ELU
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler

from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv, GATConv, GAT, GCN, GIN
from torch_geometric.nn import global_add_pool, global_max_pool

from sklearn.metrics import f1_score

from dataset import XAIMolecularDataset, TASKS


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--task", type=str, default="n_rings", choices=TASKS)
    parser.add_argument("--split", type=int, default=0, help="split")
    parser.add_argument("--model_type", type=str, default="GCN", choices=["GCN", "GAT", "GIN"])
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--linear_dim", type=int, default=32)
    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_model(model_type, num_node_features, num_classes, hidden_dim, num_layers, linear_dim=32):
    if model_type == "GCN":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GCN(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        out_channels=linear_dim,
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(linear_dim, num_classes),
            ],
        )

    elif model_type == "GAT":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GAT(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        out_channels=linear_dim,
                        act="elu",
                        dropout=0.6,
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(linear_dim, num_classes),
            ],
        )
    elif model_type == "GIN":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GIN(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        out_channels=linear_dim,
                        norm="batch_norm",
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(linear_dim, num_classes),
            ],
        )
    else:
        assert False
    return model


def train(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, batch=data.batch)
        loss = F.cross_entropy(out, data.y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return float(loss) / len(dataloader)


@torch.no_grad()
def test(model, dataloader, device):
    model.eval()
    ys, preds = list(), list()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        out = model(data.x, data.edge_index, batch=data.batch)
        pred = out.argmax(dim=-1)
        ys.append(data.y.cpu())
        preds.append(pred.cpu())
        total_loss += F.cross_entropy(out, data.y)
    ys, preds = torch.cat(ys), torch.cat(preds)
    f1 = f1_score(ys, preds, average="weighted")
    return total_loss / len(dataloader), f1


def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)

    dataset = XAIMolecularDataset("data", args.task)
    train_idxs, val_idxs, test_idxs = dataset.get_splits(args.split)
    dataset_train, dataset_val, dataset_test = (
        Subset(dataset, train_idxs),
        Subset(dataset, val_idxs),
        Subset(dataset, test_idxs),
    )

    y = torch.cat([data.y for data in dataset_train])
    class_counts = torch.bincount(y)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[y]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    dataloader_train_sampler = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = {
        "model_type": args.model_type,
        "num_node_features": dataset.num_node_features,
        "num_classes": dataset.num_classes,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "linear_dim": args.linear_dim,
    }

    model = get_model(**model_args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pbar = tqdm(range(1, args.epochs + 1))
    max_f1_train, max_f1_val, max_f1_test = 0, 0, 0
    bast_state_dict = copy.deepcopy(model.state_dict())
    count = 0

    for epoch in pbar:
        train(model, optimizer, dataloader_train_sampler, device)
        loss_train, f1_train = test(model, dataloader_train, device)
        loss_val, f1_val = test(model, dataloader_val, device)
        loss_test, f1_test = test(model, dataloader_test, device)
        if max_f1_val < f1_val:
            max_f1_train, max_f1_val, max_f1_test = f1_train, f1_val, f1_test
            bast_state_dict = copy.deepcopy(model.state_dict())
            count = 0
        else:
            if epoch > args.warmup_epochs:
                count += 1
        pbar.set_description(
            f"e:{epoch} | train l:{loss_train:.4f} f1:{f1_train:.4f} | val l:{loss_val:.4f} f1:{f1_val:.4f} | test l:{loss_test:.4f} f1:{f1_test:.4f}"
        )
        if count >= 10:
            break
    pbar.set_description(f"Final {max_f1_train:.4f} {max_f1_val:.4f} {max_f1_test:.4f}")
    pbar.close()
    model.load_state_dict(bast_state_dict)
    model.eval()
    _, final_f1 = test(model, dataloader_test, device)
    print(f"Final f1: {final_f1:.4f}")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_args": model_args,
            "args": vars(args),
            "f1": final_f1,
        },
        args.save_path,
    )
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
