import argparse
import copy
import os
import shutil
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Subset
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn import MessagePassing

from sklearn.metrics import f1_score

from dataset import XAIMolecularDataset, TASKS
from train_model import set_seed

import sys

sys.path.append(f"{os.getcwd()}/libs/ProtGNN")
from libs.ProtGNN.models.train_gnns import evaluate_GC, warm_only, joint, test_GC
from libs.ProtGNN.Configures import train_args, model_args
from libs.ProtGNN.my_mcts import mcts
from libs.ProtGNN.models import GnnNets


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--task", type=str, default="n_rings", choices=TASKS)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--node_attrs", action="store_true")
    parser.add_argument("--clst", type=float, default=0.0, help="cluster")
    parser.add_argument("--sep", type=float, default=0.0, help="separation")
    parser.add_argument("--model_type", type=str, default="GIN", choices=["GIN", "GAT", "GCN"])
    parser.add_argument("--readout", type=str, default="max", choices=["max", "sum", "mean"])
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    return args


def get_dataloader(dataset, batch_size, split):
    train_idx, val_idx, test_idx = dataset.get_splits(split)
    train = Subset(dataset, train_idx)
    eval = Subset(dataset, val_idx)
    test = Subset(dataset, test_idx)
    dataloader = dict()
    dataloader["train"] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader["eval"] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader["test"] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_f1, is_best, args):
    print("saving....")
    gnnNets.to("cpu")
    state = {
        "state_dict": gnnNets.state_dict(),
        "epoch": epoch,
        "f1": eval_f1,
        "args": vars(args),
    }
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f"{model_name}_best.pth"
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to(model_args.device)


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    # acc = []
    loss_list = []
    gnnNets.eval()

    with torch.no_grad():
        preds, ys = list(), list()
        for batch in eval_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)
            preds.append(probs.argmax(-1).detach().cpu())
            ys.append(batch.y.detach().cpu())

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            # acc.append(prediction.eq(batch.y).cpu().numpy())

        preds, ys = torch.cat(preds), torch.cat(ys)
        f1 = f1_score(ys, preds, average="weighted")
        eval_state = {
            "loss": np.average(loss_list),
            #'acc': np.concatenate(acc, axis=0).mean(),
            "f1": f1,
        }

    return eval_state


if __name__ == "__main__":
    args = args_parser()
    print(args)
    set_seed(args.seed)
    clst = args.clst
    sep = args.sep
    model_args.model_name = args.model_type.lower()
    model_args.readout = args.readout

    dataset = XAIMolecularDataset("data", args.task)

    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(dataset, train_args.batch_size, args.split)

    gnnNets = GnnNets(input_dim, output_dim, model_args)
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    best_f1 = 0.0
    best_state_dict = None
    data_size = len(dataset)

    early_stop_count = 0
    data_indices = dataloader["train"].dataset.indices
    for epoch in range(train_args.max_epochs):
        loss_list = []
        ld_loss_list = []
        # Prototype projection
        if epoch >= train_args.proj_epochs and epoch % 10 == 0:
            gnnNets.eval()
            for i in range(output_dim * model_args.num_prototypes_per_class):
                count = 0
                best_similarity = 0
                label = i // model_args.num_prototypes_per_class
                proj_prot = None
                for j in range(i * 10, len(data_indices)):
                    data = dataset[data_indices[j]]
                    if data.y == label:
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i])
                        if proj_prot is None:
                            proj_prot = prot
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= 10:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        print("Projection of prototype completed")
                        break

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)
        ys, preds = list(), list()
        for batch in dataloader["train"]:
            logits, probs, _, _, min_distances = gnnNets(batch)
            loss = criterion(logits, batch.y)
            # cluster loss
            prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y.cpu()].bool()).to(
                model_args.device
            )
            cluster_cost = torch.mean(
                torch.min(
                    min_distances[prototypes_of_correct_class].reshape(-1, model_args.num_prototypes_per_class), dim=1
                )[0]
            )

            # seperation loss
            separation_cost = -torch.mean(
                torch.min(
                    min_distances[~prototypes_of_correct_class].reshape(
                        -1, (output_dim - 1) * model_args.num_prototypes_per_class
                    ),
                    dim=1,
                )[0]
            )

            # sparsity loss
            l1_mask = 1 - torch.t(gnnNets.model.prototype_class_identity).to(model_args.device)
            l1 = (gnnNets.model.last_layer.weight * l1_mask).norm(p=1)

            # diversity loss
            ld = 0
            for k in range(output_dim):
                p = gnnNets.model.prototype_vectors[
                    k * model_args.num_prototypes_per_class : (k + 1) * model_args.num_prototypes_per_class
                ]
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3
                matrix2 = torch.zeros(matrix1.shape).to(model_args.device)
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

            loss = loss + clst * cluster_cost + sep * separation_cost + 5e-4 * l1 + 0.00 * ld

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            # acc.append(prediction.eq(batch.y).cpu().numpy())
            ys.append(batch.y.detach().cpu())
            preds.append(probs.argmax(-1).detach().cpu())

        # report train msg
        ys, preds = torch.cat(ys), torch.cat(preds)
        f1 = f1_score(ys, preds, average="weighted")
        print(
            f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | "
            f"F1: {f1:.3f}"
        )

        # report eval msg
        eval_state = evaluate_GC(dataloader["eval"], gnnNets, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | F1: {eval_state['f1']:.3f}")

        # only save the best model
        is_best = eval_state["f1"] > best_f1

        if eval_state["f1"] > best_f1:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_f1 = eval_state["f1"]
            early_stop_count = 0
            best_state_dict = copy.deepcopy(gnnNets.state_dict())

    print(f"The best validation f1 is {best_f1}.")
    gnnNets.update_state_dict(best_state_dict)
    test_state = evaluate_GC(dataloader["test"], gnnNets, criterion)
    print(f"Test: | Loss: {test_state['loss']:.3f} | F1: {test_state['f1']:.3f}")
    torch.save(
        {
            "state_dict": best_state_dict,
            "model_args": model_args,
            "args": vars(args),
            "f1": test_state["f1"],
        },
        args.save_path,
    )
    print(f"Saved to {args.save_path}")
