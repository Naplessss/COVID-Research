import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GATNet, EGNNNet
from krnn import CNNKRNNEncoder

from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, gcn_type, aggr, normalize):
        super(GCNBlock, self).__init__()
        GCNClass = {
            'gat': GATNet,
            'egnn': EGNNNet,
        }.get(gcn_type)
        self.gcn = GCNClass(in_channels,
                            out_channels,
                            edge_channels,
                            aggr=aggr,
                            normalize=normalize)

    def forward(self, X, g):
        """
        :param X: Input data of shape (batch_size, node_num, num_timesteps,
        num_features=in_channels).
        :param g: graph information.
        :return: Output data of shape (batch_size, node_num,
        num_timesteps_out, num_features=out_channels).
        """
        batch_size, node_num, seq_len, fea_num = X.shape
        t1 = X.permute(1, 0, 2, 3).contiguous().\
            view(node_num, batch_size*seq_len, fea_num)
        t2 = F.relu(self.gcn(t1, g))
        out = t2.view(node_num, batch_size, seq_len, -1).\
            permute(1, 0, 2, 3).contiguous()
        atten_context = [self.gcn.conv1.gate, self.gcn.conv1.x_i, self.gcn.conv1.x_j,
                        self.gcn.conv2.gate, self.gcn.conv2.x_i, self.gcn.conv2.x_j,
                        t2]
        return out, atten_context


class SandwichEncoder(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()

        self.config = config
        self.first_encoder = CNNKRNNEncoder(cnn_input_dim=input_dim,
                                            cnn_output_dim=config.cnn_dim,
                                            cnn_kernel_size=config.cnn_kernel_size,
                                            rnn_output_dim=config.rnn_dim,
                                            rnn_node_num=config.num_nodes,
                                            rnn_dup_num=config.rnn_dups)
        self.gcn = GCNBlock(in_channels=config.rnn_dim,
                            out_channels=config.gcn_dim,
                            edge_channels=config.edge_fea_dim,
                            gcn_type=config.gcn_type,
                            aggr=config.gcn_aggr,
                            normalize=config.gcn_norm)
        self.second_encoder = CNNKRNNEncoder(cnn_input_dim=config.gcn_dim,
                                             cnn_output_dim=config.cnn_dim,
                                             cnn_kernel_size=config.cnn_kernel_size,
                                             rnn_output_dim=config.rnn_dim,
                                             rnn_node_num=config.num_nodes,
                                             rnn_dup_num=config.rnn_dups)

    def forward(self, x, g):
        if g['type'] == 'dataflow':
            first_out = self.first_encoder(x, g['graph_n_id'])
        elif g['type'] == 'subgraph':
            first_out = self.first_encoder(x, g['cent_n_id'])
        else:
            raise Exception('Unsupported graph type: {}'.format(g['type']))

        gcn_out, atten_context = self.gcn(first_out, g)
        second_out = self.second_encoder(gcn_out, g['cent_n_id'])
        encode_out = first_out + second_out

        return encode_out, atten_context


class SandwichModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.week_em = nn.Embedding(7, config.date_emb_dim)

        day_fea_dim = config.day_fea_dim +  config.date_emb_dim - 1
        self.day_encoder = SandwichEncoder(config, day_fea_dim)

        self.out_fc = nn.Linear(config.rnn_dim, 1)

    def add_date_embed(self, input_day):
        # last 3 dims correspond to month, day, weekday
        x = input_day[:, :, :, :-1]
        weekday = self.week_em(input_day[:, :, :, -1].long())

        return torch.cat([x, weekday], dim=-1)

    def forward(self, input_day, g):
        input_day = self.add_date_embed(input_day)

        day_encode, atten_context = self.day_encoder(input_day, g)
        day_pool, _ = day_encode.max(dim=2)

        out = self.out_fc(day_pool)
        atten_context.append(day_pool)
        return out, atten_context