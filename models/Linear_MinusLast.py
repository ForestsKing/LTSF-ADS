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
        seq_last = x_enc[:, -1:, :].detach()
        x_enc = x_enc - seq_last

        output = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        output = output + seq_last
        return output[:, -self.pred_len:, :]
