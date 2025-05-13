import os
import random
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import BondType
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from torch_geometric.utils import subgraph


TASKS = ["B", "P", "X", "indole", "PAINS", "rings-count", "rings-max"]
SYMBOLS = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]


class XAIMolecularDataset(InMemoryDataset):

    molecular_data_path = "data/data.csv"
    explanations_data_path = "data/explanations.sdf"

    def __init__(self, root, name, explanations=True) -> None:
        self.name = name
        super().__init__(root)
        assert self.name in TASKS

        self._num_classes = 2
        property = {
            "P": "P",
            "X": "X",
            "B": "B",
            "indole": "indole",
            "PAINS": "pains",
            "rings-count": "rings",
            "rings-max": "largest_rings",
        }[self.name]

        df = pd.read_csv(self.molecular_data_path)
        self.splits = df[[f"split_{i}" for i in range(5)]]

        dataset = list()
        with Chem.SDMolSupplier(self.explanations_data_path) as suppl:
            ys = torch.tensor(df[name].tolist()).unsqueeze(1)
            for y, mol in tqdm(zip(ys, suppl), total=len(ys)):
                x, edge_index, edge_attr = smiles_to_graph(mol)
                if explanations:
                    p = mol.GetProp(property)
                    expl_node_mask = torch.zeros(len(x), dtype=torch.bool)
                    if p != "":
                        nodes = p.split(",")
                        nodes = torch.tensor([int(n) for n in nodes], dtype=torch.long)
                        expl_node_mask[nodes] = True
                    if property not in ["B", "P", "X"]:
                        p = mol.GetProp(f"{property}_edge")
                        expl_edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
                        if p != "":
                            edges = p.split(",")
                            edges = [e.split("#") for e in edges]
                            edges = {(int(e1), int(e2)) for e1, e2 in edges}
                            for i in range(edge_index.shape[1]):
                                e = edge_index[:, i]
                                e1, e2 = e[0].item(), e[1].item()
                                expl_edge_mask[i] = ((e1, e2) in edges) or ((e2, e1) in edges)
                    else:
                        expl_edge_mask = None

                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        expl_node_mask=expl_node_mask,
                        expl_edge_mask=expl_edge_mask,
                    )
                else:
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                dataset.append(data)
            self.data_lst = dataset
        self._num_node_features = self.__getitem__(0).x.shape[-1]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_node_features(self):
        return self._num_node_features

    def __getitem__(self, idx: int):
        data = self.data_lst[idx]
        return data

    def get_splits(self, split_idx):
        splits = self.splits[f"split_{split_idx}"]
        idx = torch.arange(len(splits))
        train_idx, val_idx, test_idx = (
            idx[splits == "train"],
            idx[splits == "valid"],
            idx[splits == "test"],
        )
        return train_idx, val_idx, test_idx

    def __len__(self):
        return len(self.data_lst)


def atom_label(atom):
    sym = atom.GetSymbol()
    return SYMBOLS.index(sym) if sym in SYMBOLS else len(SYMBOLS) - 1


def bond_type_to_int(bond_type):
    if bond_type == BondType.SINGLE:
        return 0
    elif bond_type == BondType.DOUBLE:
        return 1
    elif bond_type == BondType.TRIPLE:
        return 2
    elif bond_type == BondType.AROMATIC:
        return 3
    else:
        return -1


def smiles_to_graph(mol):
    node_labels = F.one_hot(
        torch.tensor([atom_label(atom) for atom in mol.GetAtoms()], dtype=torch.long),
        len(SYMBOLS),
    ).float()
    x = node_labels
    row, col = [], []
    edge_labels = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        row += [i, j]
        col += [j, i]
        edge_type = bond_type_to_int(bond.GetBondType())
        edge_labels += [
            edge_type,
            edge_type,
        ]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_labels, dtype=torch.long).view(-1, 1)

    return (x, edge_index, edge_attr)
