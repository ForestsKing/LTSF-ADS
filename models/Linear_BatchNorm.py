import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x_enc, x_mark_enc, x_mark_dec):
        means = torch.mean(x_enc, dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc - means
        x_enc /= stdev

        output = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        output = output * stdev
        output = output + means
        return output[:, -self.pred_len:, :]
