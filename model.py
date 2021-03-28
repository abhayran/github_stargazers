import torch
from torch_geometric.nn import SAGEConv
from torch.nn.functional import relu
import numpy as np


class GIN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.hidden = torch.nn.ModuleList([SAGEConv(1, 10), SAGEConv(10, 100)] + [SAGEConv(100, 100) for _ in range(5)])
        self.output = SAGEConv(100, 1)
        self.device = device

    def forward(self, data):
        x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, torch.max(data.batch).item() + 1
        for layer in self.hidden:
            x = layer(x, edge_index)
            x = relu(x)
        x = self.output(x, edge_index)
        return torch.mean(x).unsqueeze(dim=0)
        # locs = np.cumsum([0] + [len(torch.where(batch == ind)[0]) for ind in range(batch_size)])
        # return torch.tensor([torch.mean(x[locs[i]:locs[i + 1]]) for i in range(len(locs) - 1)],
        #                     dtype=torch.float, device=self.device, requires_grad=True)
