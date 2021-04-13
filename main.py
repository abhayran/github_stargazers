from data import StargazersDataset
from model import Model
import csv
import json
import torch
from torch_geometric.data import DataLoader


class GithubStargazers:
    def __init__(self, train_split=0.8):
        assert 0 < train_split < 1, 'Train split must be between 0 and 1'

        with open('config.json') as f:
            config = json.load(f)
            self.config = config
        with open('git_edges.json') as f:
            edges = json.load(f)
        with open('git_target.csv', 'r') as file:
            target = list(map(lambda x: x[1], csv.reader(file)))
        target.pop(0)

        self.device = torch.device('cuda') if config['use_gpu'] else torch.device('cpu')
        model = Model(self.device)
        model = model.float()
        self.model = model.to(self.device)

        self.loss_function = torch.nn.BCEWithLogitsLoss()

        number_of_samples = len(target)
        dataset_train = StargazersDataset(edges, target[:int(0.8 * number_of_samples)])
        dataset_val = StargazersDataset(edges, target[int(0.8 * number_of_samples):int(0.9 * number_of_samples)])
        dataset_test = StargazersDataset(edges, target[int(0.9 * number_of_samples):])

        self.data_loader_train = DataLoader(dataset_train, batch_size=config['training']['batch_size'],
                                            shuffle=config['training']['shuffle'])
        self.data_loader_val = DataLoader(dataset_val, batch_size=config['val']['batch_size'],
                                          shuffle=config['val']['shuffle'])
        self.data_loader_test = DataLoader(dataset_test, batch_size=config['test']['batch_size'],
                                           shuffle=config['test']['shuffle'])

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2,
        #                                                        threshold=1e-8, min_lr=1e-6)
        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            training_loss = 0.
            training_acc = 0.
            for idx, data in enumerate(self.data_loader_train):
                optimizer.zero_grad()
                out = self.model(data.to(self.device))
                loss = self.loss_function(out, data.y)
                training_acc += 1 - torch.sum(torch.logical_xor(out > 0.5, data.y > 0.5)).item() / len(out)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            training_acc /= len(self.data_loader_train)
            training_loss /= len(self.data_loader_train)
            val_loss, val_acc = self.eval(val=True)
            print('Training loss:', f'{training_loss:.6f}', 'Training acc:', f'{training_acc:.6f}',
                  'Val loss:', f'{val_loss:.6f}', 'Val acc:' f'{val_acc:.6f}')
            # scheduler.step(training_loss)
        test_loss, test_acc = self.eval(val=False)
        print('Test loss / acc:', test_loss, test_acc)

    def eval(self, val=True):
        self.model.eval()
        val_loss = 0.
        val_acc = 0.
        data_loader = self.data_loader_val if val else self.data_loader_test
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                out = self.model(data.to(self.device))
                loss = self.loss_function(out, data.y)
                val_acc += 1 - torch.sum(torch.logical_xor(out > 0.5, data.y > 0.5)).item() / len(out)
                val_loss += loss.item()
        return val_loss / len(data_loader), val_acc / len(data_loader)


def main():
    g = GithubStargazers(train_split=0.8)
    g.train()


if __name__ == '__main__':
    main()
