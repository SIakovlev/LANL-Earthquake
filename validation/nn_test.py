import argparse
import json
from torch.nn import Module, Linear, MSELoss, ReLU
from torch.optim import Adam
import torch
import torch.utils.data
import pandas as pd
import numpy as np


class CustomNN(Module):
    def __init__(self, **kwargs):
        super(CustomNN, self).__init__()

        # neural net params
        self.in_features = kwargs["in_features"]
        self.out_features = kwargs['out_features']
        self.num_hidden = kwargs['num_hidden']
        self.linear = Linear(self.in_features, self.num_hidden)
        self.out = Linear(self.num_hidden, self.out_features)
        self.relu = ReLU()

        # train params
        self.optim = Adam(self.parameters(), lr=kwargs['learning_rate'])
        self.minibatch_size = kwargs['minibatch_size']
        self.num_epochs = kwargs['num_epochs']

        self.metrics = {"neg_mean_absolute_error": MSELoss(), "neg_mean_squared_error": MSELoss()}



        self.loss = MSELoss()

    def forward(self, x):
        res = self.out(self.relu(self.linear(x)))
        return res

    def train_model(self, train_data, train_y):
        score_data_list = []
        n_train_steps_per_epoch = train_data.shape[0] // self.minibatch_size

        train_data = torch.tensor(train_data.values.astype(np.float32))
        train_y = torch.tensor(train_y.values.astype(np.float32)).view(-1, 1)

        for e in range(self.num_epochs):
            print(f" epoch: {e} out of {self.num_epochs}")
            for i in range(n_train_steps_per_epoch):
                batch_idx = np.random.randint(low=0, high=train_data.shape[0], size=self.minibatch_size)
                x_batch = train_data[batch_idx]
                y_batch = train_y[batch_idx]
                predict = self(torch.Tensor(x_batch))

                self.optim.zero_grad()
                loss = self.loss(predict, y_batch)
                loss.backward()
                self.optim.step()
                score_data_list.append(loss)
                print(f"\r step: {i} | loss={loss.detach().cpu():.4f}", end="")
            print()
        return score_data_list

    def compute_loss(self, test_data, test_y):
        test_data = torch.tensor(test_data.values.astype(np.float32))
        test_y = torch.tensor(test_y.values.astype(np.float32)).view(-1, 1)

        predict = self(test_data)
        loss = self.loss(predict, test_y).detach()
        return loss



# def main(**kwargs):
#     train_data_fname = kwargs['train_data_fname']
#     train_df = pd.read_hdf(train_data_fname, 'table')
#     train_data = train_df.drop(['time_to_failure'], axis=1)
#     train_y = train_df['time_to_failure']
#
#     test_data_fname = kwargs['test_data_fname']
#     test_df = pd.read_hdf(test_data_fname, 'table')
#     test_data = test_df.drop(['time_to_failure'], axis=1)
#     test_y = test_df['time_to_failure']
#
#     neural_net = CustomNN(**kwargs['neural_net'])
#
#     neural_net.train_model(train_data, train_y)
#
#     loss = neural_net.compute_loss(test_data, test_y)
#     print(f"test loss: {loss:.4f}")
#     return
#
#
# if __name__ == '__main__':
#     params = {
#         "train_data_fname" : "../data/train_short.h5",
#         "test_data_fname": "../data/train_short.h5",
#         "neural_net": {
#             "in_features": 1,
#             "out_features": 1,
#             "num_hidden": 100,
#             "learning_rate": 0.0001,
#             "num_epochs": 2,
#             "minibatch_size": 256
#         }
#     }
#
#     main(**params)