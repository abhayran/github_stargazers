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

    dataset = StargazersDataset(edges, target[:1])
    data_loader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                             shuffle=config['training']['shuffle'])

    device = torch.device('cuda') if config['use_gpu'] else torch.device('cpu')
    model = GIN(device)
    model = model.float()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_function = torch.nn.BCEWithLogitsLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=0,
    #                                                        threshold=1e-8, min_lr=1e-6)

    for epoch in range(config['training']['epochs']):
        training_loss = 0.
        for idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            out = model(data.to(device))
            loss = loss_function(out, data.y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        print(training_loss)
        # scheduler.step(training_loss)


if __name__ == '__main__':
    train()
