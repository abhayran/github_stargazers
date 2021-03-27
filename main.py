from data import StargazersDataset
from model import GIN
import csv
import json
import torch
from torch_geometric.data import DataLoader


def train():
    with open('config.json') as f:
        config = json.load(f)

    with open('git_edges.json') as f:
        edges = json.load(f)

    with open('git_target.csv', 'r') as file:
        target = list(map(lambda x: x[1], csv.reader(file)))
    target.pop(0)

    dataset = StargazersDataset(edges, target)
    data_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=config['training']['shuffle'])

    device = torch.device('cuda') if config['use_gpu'] else torch.device('cpu')
    model = GIN()
    model = model.float()
    model = model.to(device)


if __name__ == '__main__':
    train()
