import torch
from torch_geometric.nn import MessagePassing, BatchNorm
from torch.nn.functional import relu, dropout
import numpy as np


class GIN(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(GIN, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.epsilon = torch.nn.Parameter(torch.tensor([0], dtype=torch.float))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def update(self, aggr_out, x):
        return torch.nn.functional.normalize(self.linear((1 + self.epsilon) * x + aggr_out), dim=1)


class Model(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.hidden = torch.nn.ModuleList([GIN(1, 10), GIN(10, 100)] + [GIN(100, 100) for _ in range(5)])
        self.output = torch.nn.Linear(100, 1)
        self.device = device

    def forward(self, data):
        x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, torch.max(data.batch).item() + 1
        locs = np.cumsum([0] + [len(torch.where(batch == ind)[0]) for ind in range(batch_size)])

        for idx, layer in enumerate(self.hidden):
            x = layer(x, edge_index)
            # x = dropout(x, p=0.3)
            x = relu(x)

        out = torch.zeros((batch_size, 100), device=self.device)
        for i in range(batch_size):
            out[i, :] = torch.max(x[locs[i]:locs[i + 1]], dim=0)[0]

        out = self.output(out)
        return out.squeeze()
