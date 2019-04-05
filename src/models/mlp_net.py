from torch.nn import Module, Linear, MSELoss, ReLU, ModuleList, Dropout, BatchNorm1d
from torch.optim import Adam
import torch
import torch.utils.data
import numpy as np
from models.models import ModelBase


class MLP(Module, ModelBase):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()

        # neural net params
        self.in_features_shape = kwargs["in_features"]

        if isinstance(self.in_features_shape, list) and len(self.in_features_shape) == 2:
            self.in_features = self.in_features_shape[1]
            self.in_seq_len = self.in_features_shape[0]
        elif isinstance(self.in_features_shape, int):
            self.in_features = self.in_features_shape
            self.in_seq_len = 1
        else:
            raise TypeError("input features must either int either list of ints")

        self.out_features = kwargs['out_features']
        self.linears_size = kwargs['hidden_layers']
        self.device = kwargs['device']

        self.dropout = Dropout(p=kwargs['dropout'])

        self.bn = BatchNorm1d(self.in_features)

        self.linears = ModuleList([Linear(self.in_features * self.in_seq_len, self.linears_size[0])])
        self.linears.extend(
            [Linear(self.linears_size[i - 1], self.linears_size[i]) for i in range(1, len(self.linears_size))])

        self.out = Linear(self.linears_size[-1], self.out_features)
        self.relu = ReLU()

        # train params
        self.optim = Adam(self.parameters(), lr=kwargs['learning_rate'])
        self.minibatch_size = kwargs['minibatch_size']
        self.num_epochs = kwargs['num_epochs']

        self.loss = MSELoss()
        self = self.to(self.device)

    def forward(self, x):
        # x = torch.clamp(x, max=0.001)
        orig_shape = x.shape
        x = x.view(-1, self.in_features)
        x = self.bn(x)
        x = x.view(orig_shape)
        x = self.dropout(x)
        for linear in self.linears:
            x = self.relu(linear(x))
        res = self.out(x)
        return res

    def fit(self, train_data, train_y, valid_data, valid_y):
        self.train()
        train_data = torch.tensor(train_data.values.astype(np.float32)).to(self.device)
        train_y = torch.tensor(train_y.values.astype(np.float32)).view(-1, 1).to(self.device)

        valid_y = torch.tensor(valid_y.values.astype(np.float32)).view(-1, 1).to(self.device)

        n_train_steps_per_epoch = train_data.shape[0] // self.minibatch_size

        for e in range(self.num_epochs):
            print(f" epoch: {e} out of {self.num_epochs}")
            for i in range(n_train_steps_per_epoch):
                batch_idx = np.random.randint(low=0, high=train_data.shape[0], size=self.minibatch_size)
                x_batch = train_data[batch_idx]
                y_batch = train_y[batch_idx]
                predict = self(x_batch)

                self.optim.zero_grad()
                loss = self.loss(predict, y_batch)
                loss.backward()
                self.optim.step()

                mae_loss = torch.abs(predict - y_batch).mean()
                print(f"\r step: {i} | mse_loss={loss.detach().cpu():.4f} | mae_loss={mae_loss.detach().cpu():.4f}", end="")
            print()
            # validate
            pred = torch.tensor(self.predict(valid_data)).to(self.device)
            loss = self.loss(pred, valid_y)
            mae_loss = torch.abs(pred - valid_y).mean()
            self.train()
            print(f"validation score: mse_loss={loss.detach().cpu():.4f} | mae_loss={mae_loss.detach().cpu():.4f}")
        del train_data
        del train_y

    def predict(self, test_data):
        self.eval()
        test_data = torch.tensor(test_data.values.astype(np.float32)).to(self.device)
        y_predict = self(test_data)
        del test_data
        return y_predict.detach().cpu().numpy()


