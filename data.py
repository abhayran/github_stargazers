import torch
from torch_geometric.data import Data, Dataset
import numpy as np


class StargazersDataset(Dataset):
    def __init__(self, edges, target):
        """
        :param edges: dictionary containing edge indices
        :param target: labels
        """
        super().__init__()
        self.data = []
        for i in range(len(target)):
            edge_index = edges[str(i)]
            num_nodes = np.max(edge_index) + 1
            x = torch.zeros((num_nodes, 1), dtype=torch.float)
            for v1, v2 in edge_index:
                x[v1] += 1
                x[v2] += 1
            edge_index = torch.tensor(edge_index, dtype=torch.long).T
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            self.data.append(Data(x=x, edge_index=edge_index, y=torch.tensor(int(target[i]), dtype=torch.float),
                                  num_nodes=num_nodes))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
