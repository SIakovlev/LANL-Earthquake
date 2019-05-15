from torch.nn import Module, Linear, MSELoss, L1Loss, ReLU, ModuleList, Dropout, BatchNorm1d
from torch.nn.init import xavier_uniform_
from torch.optim import Adam
import torch
import torch.utils.data
import numpy as np
from models.models import ModelBase
import pickle


def init_weights(m):
    if type(m) == Linear:
        xavier_uniform_(m.weight, gain=1)


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

        # TODO: check this
        self.loss = L1Loss()
        self.apply(init_weights)
        self = self.to(self.device)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, self.in_features)
        x = self.bn(x)
        x = x.view(orig_shape)
        # x = self.dropout(x)
        for linear in self.linears:
            x = self.relu(linear(x))
            x = self.dropout(x)
        res = self.out(x)
        return res

    def fit(self, train_data, train_y):
        self.train()
        train_data = torch.tensor(train_data.values.astype(np.float32)).to(self.device)
        train_y = torch.tensor(train_y.values.astype(np.float32)).view(-1, 1).to(self.device)

        # valid_y = torch.tensor(valid_y.values.astype(np.float32)).view(-1, 1).to(self.device)

        n_train_steps_per_epoch = train_data.shape[0] // self.minibatch_size

        for e in range(self.num_epochs):
            # print(f" epoch: {e} out of {self.num_epochs}")
            for i in range(n_train_steps_per_epoch):
                batch_idx = np.random.randint(low=0, high=train_data.shape[0], size=self.minibatch_size)
                x_batch = train_data[batch_idx]
                y_batch = train_y[batch_idx]
                predict = self(x_batch)

                self.optim.zero_grad()
                loss = self.loss(predict, y_batch)
                loss.backward()
                self.optim.step()

                # mae_loss = torch.abs(predict - y_batch).mean()
            #     print(f"\r step: {i} | obj_loss={loss.detach().cpu():.4f}", end="")
            # print()
        del train_data
        del train_y

    def predict(self, test_data):
        self.eval()
        test_data = torch.tensor(test_data.values.astype(np.float32)).to(self.device)
        y_predict = self(test_data)
        del test_data
        return y_predict.detach().cpu().numpy()

    def load(f):
        # with open(fname, 'r') as f:

        net = pickle.load(f)
        return net


class MLPEnsemble(Module, ModelBase):
    def __init__(self, **kwargs):
        super(MLPEnsemble, self).__init__()

        self.num_nets = kwargs["num_nets"]

        self.nets = ModuleList([MLP(**kwargs) for _ in range(self.num_nets)])

        self.device = kwargs['device']
        self.optim = Adam(self.parameters(), lr=kwargs['learning_rate'])
        self.minibatch_size = kwargs['minibatch_size']
        self.num_epochs = kwargs['num_epochs']
        self.loss = L1Loss(reduction='none')
        self = self.to(self.device)


    def forward(self, x):
        res = torch.stack([net(x) for net in self.nets], dim=-1)
        return res

    def fit(self, train_data, train_y):
        self.train()


        train_data = torch.tensor(train_data.values.astype(np.float32)).to(self.device)
        train_y = torch.tensor(train_y.values.astype(np.float32)).view(-1, 1).to(self.device)

        # valid_y = torch.tensor(valid_y.values.astype(np.float32)).view(-1, 1).to(self.device)

        n_train_steps_per_epoch = train_data.shape[0] // self.minibatch_size

        for e in range(self.num_epochs):
            # print(f" epoch: {e} out of {self.num_epochs}")
            for i in range(n_train_steps_per_epoch):
                batch_idx = np.random.randint(low=0, high=train_data.shape[0], size=self.minibatch_size)
                x_batch = train_data[batch_idx]
                y_batch = train_y[batch_idx]
                predict = self(x_batch)
                self.optim.zero_grad()
                loss = self.loss(predict[:, :, :], y_batch.repeat(1,1,self.num_nets).transpose(0,1)).mean(dim=(0,1)).sum()
                loss.backward()
                self.optim.step()

                # mae_loss = torch.abs(predict - y_batch).mean()
            #     print(f"\r step: {i} | obj_loss={loss.detach().cpu():.4f}", end="")
            # print()
        del train_data
        del train_y

    def predict(self, test_data):
        self.eval()
        test_data = torch.tensor(test_data.values.astype(np.float32)).to(self.device)
        y_predict = self(test_data).mean(dim=-1)
        del test_data
        return y_predict.detach().cpu().numpy()

    def load(f):
        # with open(fname, 'r') as f:

        net = pickle.load(f)
        return net


