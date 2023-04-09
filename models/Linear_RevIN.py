import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        x = x * self.affine_weight
        x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        x = x - self.affine_bias
        x = x / self.affine_weight
        x = x * self.stdev
        x = x + self.mean
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.revin = RevIN(self.enc_in)

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x_enc, x_mark_enc, x_mark_dec):
        x_enc = self.revin(x_enc, 'norm')
        output = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.revin(output, 'denorm')
        return output[:, -self.pred_len:, :]
