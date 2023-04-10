import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, enc_in, eps=1e-5):
        super(RevIN, self).__init__()
        self.num_features = enc_in
        self.eps = eps

        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode='normalize'):
        if mode == 'normalize':
            self._preget(x)
            x = self._normalize(x)
        elif mode == 'denormalize':
            x = self._denormalize(x)
        return x

    def _preget(self, x):
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
        x_enc = self.revin(x_enc, 'normalize')

        output = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.revin(output, 'denormalize')
        return output[:, -self.pred_len:, :]
