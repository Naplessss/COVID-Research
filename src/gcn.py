import math
import torch
import torch_geometric
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader


class BaseGNNNet(nn.Module):
    def __init__(self):
        super().__init__()

    def dataflow_forward(self, X, g):
        raise NotImplementedError

    def subgraph_forward(self, X, g):
        raise NotImplementedError

    def forward(self, X, g):
        if g['type'] == 'dataflow':
            return self.dataflow_forward(X, g)
        elif g['type'] == 'subgraph':
            return self.subgraph_forward(X, g)
        else:
            raise Exception('Unsupported graph type {}'.format(g['type']))


class MyGATConv(PyG.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='max', normalize='none', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.weight_n = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        self.u = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.v = nn.Parameter(torch.Tensor(out_channels, out_channels))

        self.normalize = normalize

        if normalize == 'bn':
            self.batch_norm = nn.BatchNorm1d(out_channels)
        if normalize == 'ln':
            self.layer_norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x, edge_index, edge_attr=None, edge_norm=None, size=None):
        if isinstance(x, tuple) or isinstance(x, list):
            x = [None if xi is None else torch.matmul(xi, self.weight_n) for xi in x]
        else:
            x = torch.matmul(x, self.weight_n)

        edge_attr = torch.matmul(edge_attr, self.weight_e)

        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr, edge_norm=edge_norm)

    def message(self, edge_index_i, x_i, x_j, edge_attr, edge_norm):
        x_i = torch.matmul(x_i, self.u)
        x_j = torch.matmul(x_j, self.u)
        # gate = torch.sigmoid((x_i * x_j).sum(dim=-1)).unsqueeze(dim=-1)
        gate = torch.sigmoid((x_i * x_j).sum(dim=-1)).unsqueeze(dim=-1)
        self.gate = gate
        self.x_i = x_i
        self.x_j = x_j
        msg = x_j * gate
        if edge_norm is None:
            return msg
        else:
            return msg * edge_norm.reshape(edge_norm.size(0), 1, 1)


    def update(self, aggr_out, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[1]

        if self.normalize == 'bn':
            aggr_out = aggr_out.permute(0, 2, 1)
            aggr_out = self.batch_norm(aggr_out)
            aggr_out = aggr_out.permute(0, 2, 1)
        elif self.normalize == 'ln':
            aggr_out = self.layer_norm(aggr_out)
        elif self.normalize == 'vn':
            mean = aggr_out.mean(dim=[1, 2], keepdim=True)
            std = aggr_out.std(dim=[1, 2], keepdim=True)
            aggr_out = (aggr_out - mean) / (std + 1e-5)

        return x + aggr_out


class GATNet(BaseGNNNet):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='max', normalize='none'):
        super().__init__()
        self.conv1 = MyGATConv(in_channels,
                               out_channels,
                               edge_channels,
                               aggr=aggr,
                               normalize=normalize)
        self.conv2 = MyGATConv(out_channels,
                               out_channels,
                               edge_channels,
                               aggr=aggr,
                               normalize=normalize)

    def dataflow_forward(self, X, g):
        edge_index = g['edge_index']
        edge_attr = g['edge_attr']
        size = g['size']
        res_n_id = g['res_n_id']

        c1 = self.conv1(
            (X, X[res_n_id[0]]), edge_index[0], edge_attr=edge_attr[0], size=size[0]
        )
        c1 = F.leaky_relu(c1)

        c2 = self.conv2(
            (c1, c1[res_n_id[1]]), edge_index[1], edge_attr=edge_attr[1], size=size[1]
        )
        c2 = F.leaky_relu(c2)

        return c2

    def subgraph_forward(self, X, g):
        edge_index = g['edge_index']
        edge_attr = g['edge_attr']
        if 'edge_norm' in g:
            edge_norm = g['edge_norm']
        else:
            edge_norm = None

        c1 = self.conv1(X, edge_index, edge_attr=edge_attr, edge_norm=edge_norm)
        c1 = F.leaky_relu(c1)

        c2 = self.conv2(c1, edge_index, edge_attr=edge_attr, edge_norm=edge_norm)
        c2 = F.leaky_relu(c2)

        return c2


class MyEGNNConv(PyG.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, normalize='none', aggr='max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.weight_n = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        self.query = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.key = nn.Parameter(torch.Tensor(out_channels, out_channels))

        self.linear_att = nn.Linear(3 * out_channels, 1)
        self.linear_out = nn.Linear(2 * out_channels, out_channels)

        self.normalize = normalize

        if normalize == 'bn':
            self.batch_norm = nn.BatchNorm1d(out_channels)
        if normalize == 'ln':
            self.layer_norm = nn.LayerNorm(out_channels)
        if normalize == 'vn':
            max_num_nodes = 3000
            self.value_norm = ValueNorm(max_num_nodes, affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key)

    def forward(self, x, edge_index, edge_attr, size=None, indices=None, edge_norm=None):
        if isinstance(x, tuple) or isinstance(x, list):
            x = [None if xi is None else torch.matmul(xi, self.weight_n) for xi in x]
        else:
            x = torch.matmul(x, self.weight_n)

        edge_attr = torch.matmul(edge_attr, self.weight_e)

        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr, indices=indices, edge_norm=edge_norm)

    def message(self, x_j, x_i, edge_attr, edge_norm):
        # cal att of shape [B, E, 1]
        query = torch.matmul(x_j, self.query)
        key = torch.matmul(x_i, self.key)

        edge_attr = edge_attr.unsqueeze(dim=1).expand_as(query)

        att_feature = torch.cat([query, key, edge_attr], dim=-1)
        att = torch.sigmoid(self.linear_att(att_feature))

        # gate of shape [1, E, C]
        gate = torch.sigmoid(edge_attr)

        msg = att * x_j * gate

        if edge_norm is None:
            return msg
        else:
            return msg * edge_norm.reshape(edge_norm.size(0), 1, 1)

    def update(self, aggr_out, x, indices):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[1]

        aggr_out = self.linear_out(torch.cat([x, aggr_out], dim=-1))

        if self.normalize == 'bn':
            aggr_out = aggr_out.permute(0, 2, 1)
            aggr_out = self.batch_norm(aggr_out)
            aggr_out = aggr_out.permute(0, 2, 1)
        elif self.normalize == 'ln':
            aggr_out = self.layer_norm(aggr_out)
        elif self.normalize == 'vn':
            aggr_out = self.value_norm(aggr_out, indices)

        return x + aggr_out


class EGNNNet(BaseGNNNet):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='max', normalize='none'):
        super().__init__()
        self.conv1 = MyEGNNConv(in_channels, out_channels, edge_channels,
                                aggr=aggr, normalize=normalize)
        self.conv2 = MyEGNNConv(out_channels, out_channels, edge_channels,
                                aggr=aggr, normalize=normalize)

    def dataflow_forward(self, X, g):
        edge_index = g['edge_index']
        edge_attr = g['edge_attr']
        size = g['size']
        n_id = g['n_id']
        res_n_id = g['res_n_id']

        c1 = self.conv1((X, X[res_n_id[0]]), edge_index[0],
                        edge_attr=edge_attr[0],
                        size=size[0], indices=n_id[0][res_n_id[0]])
        c1 = F.leaky_relu(c1)

        c2 = self.conv2((c1, c1[res_n_id[1]]), edge_index[1],
                        edge_attr=edge_attr[1],
                        size=size[1], indices=n_id[1][res_n_id[1]])
        c2 = F.leaky_relu(c2)

        return c2

    def subgraph_forward(self, X, g):
        edge_index = g['edge_index']
        edge_attr = g['edge_attr']
        n_id = g['cent_n_id']
        if 'edge_norm' in g:
            edge_norm = g['edge_norm']
        else:
            edge_norm = None

        c1 = self.conv1(X, edge_index,
                        edge_attr=edge_attr,
                        indices=n_id, edge_norm=edge_norm)
        c1 = F.leaky_relu(c1)

        c2 = self.conv2(c1, edge_index,
                        edge_attr=edge_attr,
                        indices=n_id, edge_norm=edge_norm)
        c2 = F.leaky_relu(c2)

        return c2
