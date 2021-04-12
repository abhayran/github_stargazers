import torch
from torch_geometric.nn import SAGEConv
from torch.nn.functional import relu, dropout
import numpy as np


class GraphNorm(torch.nn.Module):
    def __init__(self, input_dim, preserve_mean=True):
        super(GraphNorm, self).__init__()
        self.preserve_mean = preserve_mean
        if self.preserve_mean:
            self.alpha = torch.nn.Parameter(torch.rand(input_dim, dtype=torch.float))
        self.scale = torch.nn.Parameter(torch.ones(input_dim, dtype=torch.float))
        self.shift = torch.nn.Parameter(torch.zeros(input_dim, dtype=torch.float))

    def forward(self, x, batch):
        batch_size = torch.max(batch).item() + 1
        locs = np.cumsum([0] + [len(torch.where(batch == ind)[0]) for ind in range(batch_size)])
        out = []
        for i in range(batch_size):
            sqrt = int(np.sqrt(locs[i+1] - locs[i]))
            clone = x[locs[i]:locs[i+1], :].clone()
            if self.preserve_mean:
                clone = clone - self.alpha * torch.mean(clone, dim=0)
            else:
                clone = clone - torch.mean(clone, dim=0)
            clone = clone / (torch.norm(clone, dim=0) / sqrt)
            clone = clone * self.scale
            clone = clone + self.shift
            out.append(clone)
        return torch.cat(out)


class GIN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.hidden = torch.nn.ModuleList([SAGEConv(1, 10), SAGEConv(10, 100)] + [SAGEConv(100, 100) for _ in range(5)])
        self.norm = torch.nn.ModuleList([GraphNorm(10)] + [GraphNorm(100) for _ in range(6)])
        self.output = SAGEConv(100, 1)
        self.device = device

    def forward(self, data):
        x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, torch.max(data.batch).item() + 1
        for idx, layer in enumerate(self.hidden):
            x = layer(x, edge_index)
            x = self.norm[idx](x, batch)
            # x = dropout(x, p=0.3)
            x = relu(x)
        x = self.output(x, edge_index)
        locs = np.cumsum([0] + [len(torch.where(batch == ind)[0]) for ind in range(batch_size)])
        ret = torch.zeros(batch_size, device=self.device)
        for i in range(len(locs) - 1):
            ret[i] = torch.mean(x[locs[i]:locs[i + 1]])
        return ret
