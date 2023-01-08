import torch


class GlasgowFC(torch.nn.Module):
    def __init__(self, n_feat=300, n_layers=1, n_hidden=64):
        super().__init__()
        self.n_layer = n_layers
        self.n_hidden = n_hidden
        self.fc = torch.nn.Linear(in_features=n_feat, out_features=1)
        # self.fc2 = torch.nn.Linear(in_features=n_hidden, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        out = self.fc(x)
        # out = self.fc2(out)
        out = self.sigmoid(out)
        return out