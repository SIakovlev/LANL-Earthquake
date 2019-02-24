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

        self.loss = MSELoss()

    def forward(self, x):
        res = self.out(self.relu(self.linear(x)))
        return res

    def fit(self, train_data, train_y):
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
                print(f"\r step: {i} | loss={loss.detach().cpu():.4f}", end="")
            print()

    def predict(self, test_data):
        test_data = torch.tensor(test_data.values.astype(np.float32))
        y_predict = self(test_data)
        return y_predict.detach().numpy()


