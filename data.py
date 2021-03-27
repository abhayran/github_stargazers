import torch
from torch_geometric.data import Data, Dataset


class StargazersDataset(Dataset):
    def __init__(self, edges, target):
        """
        :param edges: dictionary containing edge indices
        :param target: labels
        """
        super().__init__()
        self.data = []
        for i in range(len(target)):
            edge_index = torch.tensor(edges[str(i)], dtype=torch.long).T
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            self.data.append(Data(edge_index=edge_index, y=target[i], num_nodes=torch.max(edge_index).item() + 1))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
