import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # set padding to ensure the same length
        # it is correct only when kernel_size is odd, dilation is 1, stride is 1
        self.conv = nn.Conv1d(input_dim, output_dim,
                              kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        # input shape: [batch_size, node_num, seq_len, input_dim]
        # output shape: [batch_size, node_num, seq_len, input_dim]
        batch_size, node_num, seq_len, input_dim = x.shape
        x = x.view(-1, seq_len, input_dim).permute(0, 2, 1)
        y = self.conv(x)  # [batch_size*node_num, output_dim, conved_seq_len]
        y = y.permute(0, 2, 1).view(batch_size, node_num, -1, self.output_dim)

        return y


class KRNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, dup_num):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num
        self.dup_num = dup_num

        self.rnn_modules = nn.ModuleList()
        for _ in range(dup_num):
            self.rnn_modules.append(
                nn.GRU(input_dim, output_dim, num_layers=2)
            )
        self.attn = nn.Embedding(node_num, dup_num)

    def forward(self, x, n_id):
        # input shape: [batch_size, node_num, seq_len, input_dim]
        # output shape: [batch_size, node_num, seq_len, output_dim]
        batch_size, node_num, seq_len, input_dim = x.shape
        # [seq_len, batch_size*node_num, input_dim]
        x = x.view(-1, seq_len, input_dim).permute(1, 0, 2)

        hids = []
        for rnn in self.rnn_modules:
            h, _ = rnn(x)  # [seq_len, batch_size*node_num, output_dim]
            hids.append(h)
        # [seq_len, batch_size*node_num, output_dim, num_dups]
        hids = torch.stack(hids, dim=-1)

        attn = torch.softmax(self.attn(n_id), dim=-1)  # [node_num, num_dups]

        hids = hids.view(seq_len, batch_size, node_num,
                         self.output_dim, self.dup_num)
        hids = torch.einsum('ijklm,km->ijkl', hids, attn)
        hids = hids.permute(1, 2, 0, 3)

        return hids


class CNNKRNNEncoder(nn.Module):
    def __init__(self, cnn_input_dim, cnn_output_dim, cnn_kernel_size, rnn_output_dim, rnn_node_num, rnn_dup_num):
        super().__init__()

        self.cnn_encoder = CNNEncoder(
            cnn_input_dim, cnn_output_dim, cnn_kernel_size)
        self.bn = nn.BatchNorm2d(cnn_output_dim)
        self.krnn_encoder = KRNNEncoder(
            cnn_output_dim, rnn_output_dim, rnn_node_num, rnn_dup_num)

    def forward(self, x, n_id):
        cnn_out = self.cnn_encoder(x)
        krnn_out = self.krnn_encoder(cnn_out, n_id)

        return krnn_out


class KRNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.month_em = nn.Embedding(12, config.date_emb_dim)
        self.day_em = nn.Embedding(31, config.date_emb_dim)
        self.week_em = nn.Embedding(7, config.date_emb_dim)

        day_input_dim = config.day_fea_dim + 3 * (config.date_emb_dim - 1)
        hour_input_dim = config.hour_fea_dim

        self.day_encoder = CNNKRNNEncoder(day_input_dim,
                                          config.cnn_dim,
                                          config.cnn_kernel_size,
                                          config.rnn_dim,
                                          config.num_nodes,
                                          config.rnn_dups)
        self.hour_encoder = CNNKRNNEncoder(hour_input_dim,
                                           config.cnn_dim,
                                           config.cnn_kernel_size,
                                           config.rnn_dim,
                                           config.num_nodes,
                                           config.rnn_dups)

        self.out_fc = nn.Linear(config.rnn_dim * 2, 1)

    def add_date_embed(self, input_day):
        # last 3 dims correspond to month, day, weekday
        x = input_day[:, :, :, :-3]
        month = self.month_em(input_day[:, :, :, -3].long())
        day = self.day_em(input_day[:, :, :, -2].long())
        weekday = self.week_em(input_day[:, :, :, -1].long())

        return torch.cat([x, month, day, weekday], dim=-1)

    def forward(self, input_day, input_hour, g):
        input_day = self.add_date_embed(input_day)

        day_encode = self.day_encoder(input_day, g['cent_n_id'])
        hour_encode = self.hour_encoder(input_hour, g['cent_n_id'])

        day_pool, _ = day_encode.max(dim=2)
        hour_pool, _ = hour_encode.max(dim=2)

        out = torch.cat([day_pool, hour_pool], dim=-1)
        out = self.out_fc(out).squeeze(dim=-1)

        return out
