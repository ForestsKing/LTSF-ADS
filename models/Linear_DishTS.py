import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTS(nn.Module):
    def __init__(self, enc_in, seq_len):
        super().__init__()
        n_series = enc_in
        lookback = seq_len

        self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2) / lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))

    def forward(self, x, mode='normalize'):
        if mode == 'normalize':
            self._preget(x)
            x = self._normalize(x)
        elif mode == 'denormalize':
            x = self._denormalize(x)
        return x

    def _preget(self, batch_x):
        x_transpose = batch_x.permute(2, 0, 1)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        theta = F.gelu(theta)
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
        self.xil = torch.sum(torch.pow(batch_x - self.phil, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)

    def _normalize(self, batch_input):
        temp = (batch_input - self.phil) / torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst

    def _denormalize(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dishts = DishTS(self.enc_in, self.seq_len)

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x_enc, x_mark_enc, x_mark_dec):
        x_enc = self.dishts(x_enc, 'normalize')

        output = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.dishts(output, 'denormalize')
        return output[:, -self.pred_len:, :]
