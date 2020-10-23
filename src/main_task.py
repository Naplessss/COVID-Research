import os
import time
import argparse
import json
import math
import torch
import torch_geometric
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import Namespace
from torch_geometric.data import Data, Batch, NeighborSampler, ClusterData, ClusterLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from base_task import add_config_to_argparse, BaseConfig, BasePytorchTask, \
    LOSS_KEY, BAR_KEY, SCALAR_LOG_KEY, VAL_SCORE_KEY

from dataset import SAINTDataset, SimpleDataset
from utils import load_data, load_npz_data, ExpL1Loss
from n_beats import NBeatsModel
from sandwich import SandwichModel
from krnn import KRNNModel

class RNNConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # Reset base variables
        self.max_epochs = 1000
        self.early_stop_epochs = 30
        self.infer = False

        # for data loading
        self.data_fp = '../data/gbm_dataset.npz'
        self.start_date = '2020-04-01'
        self.min_peak_size = -1  # min peak confirmed cases selected by country level
        self.lookback_days = 14  # the number of days before the current day for daily series
        self.lookahead_days = 1
        self.forecast_date = '2020-06-29'
        self.horizon = 7
        self.label = 'deaths_target'
        self.use_mobility = False

        self.model_type = 'krnn'  # choices: krnn, sandwich, nbeats
        self.saint_batch_size = 500
        self.saint_sample_type = 'node'
        self.date_emb_dim = 2

        self.use_gbm = False

        # for krnn
        self.cnn_dim = 256
        self.cnn_kernel_size = 3
        self.rnn_dim = 256
        self.rnn_dups = 10

        # for transformer
        self.tfm_layer_num = 8
        self.tfm_head_num = 8
        self.tfm_hid_dim = 32
        self.tfm_ff_dim = 32
        self.tfm_max_pos = 500
        self.tfm_node_dim = 5
        self.tfm_dropout = 0.1
        self.tfm_block_num = -1
        self.tfm_cnn_kernel_size = 1

        # for n_beats
        self.block_size = 3
        self.hidden_dim = 32
        self.id_emb_dim = 8

        # for gcnn
        self.gcn_dim = 32
        self.gcn_type = 'gat'
        self.gcn_aggr = 'max'
        self.gcn_norm = 'none'

        # per-gpu training batch size, real_batch_size = batch_size * num_gpus * grad_accum_steps
        self.batch_size = 4
        self.lr = 1e-3  # the learning rate
        self.rand_mask = False  # use random mask for training

        # batch sample type
        self.use_saintdataset = True
        self.use_lr = True

class WrapperNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.label2idx = {
        'confirmed_target':-4,
        'deaths_target':-3,
        'recovered_target':-2
        }
        # self.net = Model(config)
        if self.config.model_type == 'krnn':
            self.net = KRNNModel(config)
        elif self.config.model_type == 'nbeats':
            self.net = NBeatsModel(config)
        elif self.config.model_type == 'sandwich':
            self.net = SandwichModel(config)
        else:
            raise Exception(
                'Unsupported model type {}'.format(config.model_type))

        if config.use_lr:
            self.weight_lr = nn.Parameter(torch.Tensor(self.config.lookback_days, self.config.lookahead_days))
            self.b_lr = nn.Parameter(torch.Tensor([0.0] * self.config.lookahead_days))
            self.reset_parameters()  


    def lr(self, input_day):
        sz = input_day.size()
        # print(sz)
        label_idx = self.label2idx.get(self.config.label,-3)
        ts = torch.expm1(input_day[:,:,:,label_idx])     # label ts
        pred = torch.matmul(ts, torch.softmax(self.weight_lr, dim=0)) + self.b_lr 
        pred = torch.log1p(pred)
        pred = pred.view(sz[0],sz[1],self.config.lookahead_days)
        return pred

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_lr)

    def forward(self, input_day, g):
        if config.model_type == 'sandwich':
            out, atten_context = self.net(input_day, g)
        else:
            out = self.net(input_day, g)
            atten_context = None
        if self.config.use_lr:
            out = out + self.lr(input_day)
        return out, atten_context

class RNNTask(BasePytorchTask):
    def __init__(self, config):
        super().__init__(config)
        self.log('Intialize {}'.format(self.__class__))

        self.init_data()
        # self.loss_func = nn.L1Loss(reduction='none')
        self.loss_func = nn.MSELoss(reduction='none')
        # self.loss_func = ExpL1Loss(reduction='none')
        self.log('Config:\n{}'.format(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=4)
        ))

    def init_data(self, data_fp=None):
        if data_fp is None:
            data_fp = self.config.data_fp

        # load data
        if self.config.data_fp.endswith('.npz'):
            day_inputs, gbm_outputs, outputs, edge_index, dates, countries = load_npz_data(data_fp)
        else:
            day_inputs, outputs, edge_index, dates, countries = \
                load_data(data_fp, 
                self.config.start_date, 
                self.config.min_peak_size, 
                self.config.lookback_days, 
                self.config.lookahead_days,
                self.config.label,
                self.config.use_mobility, 
                logger=self.log)
            gbm_outputs = outputs
        # numpy default dtype is float64, but torch default dtype is float32
        self.day_inputs = day_inputs
        self.outputs = outputs
        self.gbm_outputs = gbm_outputs
        self.edge_index = edge_index
        self.edge_attr = torch.ones(edge_index.size(1),1)
        self.dates = dates  # share index with sample id
        self.countries = countries  # share index with node id

        # fulfill config
        self.config.num_nodes = self.day_inputs.shape[1]
        self.config.day_seq_len = self.day_inputs.shape[2]
        self.config.day_fea_dim = self.day_inputs.shape[3]
        self.config.edge_fea_dim = self.edge_attr.shape[1]

        use_dates = [pd.to_datetime(item) for item in dates if pd.to_datetime(item)<=pd.to_datetime(self.config.forecast_date)]
        test_divi = len(use_dates) - 1 
        val_divi = test_divi - self.config.horizon
        train_divi = val_divi - 1
        if self.config.infer:
            # use all achieved train data
            train_divi = val_divi + 1 

        print(train_divi,val_divi,test_divi)
        print(dates[train_divi],dates[val_divi],dates[test_divi])

        self.train_day_inputs = self.day_inputs[:train_divi+1]
        self.train_gbm_outputs = self.gbm_outputs[:train_divi+1]
        self.train_outputs = self.outputs[:train_divi+1]
        self.train_dates = self.dates[:train_divi+1]

        if self.config.infer:
            self.val_day_inputs = self.day_inputs[:train_divi+1]
            self.val_gbm_outputs = self.gbm_outputs[:train_divi+1]
            self.val_outputs = self.outputs[:train_divi+1]
            self.val_dates = self.dates[:train_divi+1]  
        else:          
            self.val_day_inputs = self.day_inputs[val_divi:val_divi+1]
            self.val_gbm_outputs = self.gbm_outputs[val_divi:val_divi+1]
            self.val_outputs = self.outputs[val_divi:val_divi+1]
            self.val_dates = self.dates[val_divi:val_divi+1]

        self.test_day_inputs = self.day_inputs[test_divi:test_divi+1]
        self.test_gbm_outputs = self.gbm_outputs[test_divi:test_divi+1]
        self.test_outputs = self.outputs[test_divi:test_divi+1]
        self.test_dates = self.dates[test_divi:test_divi+1]

    def make_sample_dataloader(self, day_inputs, gbm_outputs, outputs, shuffle=False):
        if self.config.use_saintdataset:
            dataset = SAINTDataset(
                [day_inputs, gbm_outputs, outputs],
                self.edge_index, self.edge_attr,
                self.config.num_nodes, self.config.batch_size,
                shuffle=shuffle, saint_sample_type=self.config.saint_sample_type,
                saint_batch_size=self.config.saint_batch_size,
            )

            return DataLoader(dataset, batch_size=None)
        else:
            dataset = SimpleDataset([day_inputs, gbm_outputs, outputs])
            def collate_fn(samples):
                day_inputs = torch.cat([item[0][0] for item in samples]).unsqueeze(0)   # [1,bs,seq_length,feature_dim]
                gbm_outputs = torch.cat([item[0][1] for item in samples]).unsqueeze(0)  
                outputs = torch.cat([item[0][2] for item in samples]).unsqueeze(0)
                node_ids = torch.LongTensor([item[1] for item in samples])   # [bs]
                date_ids = torch.LongTensor([item[2] for item in samples])   # [bs]
                return [[day_inputs, gbm_outputs, outputs], {'cent_n_id':node_ids,'type':'random'}, date_ids]

            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    def build_train_dataloader(self):
        return self.make_sample_dataloader(
            self.train_day_inputs, self.train_gbm_outputs, self.train_outputs, shuffle=True
        )

    def build_val_dataloader(self):
        return self.make_sample_dataloader(
            self.val_day_inputs, self.val_gbm_outputs, self.val_outputs, shuffle=False
        )

    def build_test_dataloader(self):
        return self.make_sample_dataloader(
            self.test_day_inputs, self.test_gbm_outputs, self.test_outputs, shuffle=False
        )

    def build_optimizer(self, model):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_step(self, batch, batch_idx):
        # input_day, input_hour, output = batch
        inputs, g, _ = batch
        input_day, y_gbm, y = inputs
        if self.config.use_gbm:
            y = y - y_gbm
        y_hat, _ = self.model(input_day, g)
        assert(y.size() == y_hat.size())
        loss = self.loss_func(y_hat, y)

        if self.config.rand_mask:
            mask = torch.randint_like(loss, high=2)
        else:
            mask = torch.ones_like(loss).float()

        loss = (loss * mask).sum() / (mask.sum() + 1e-5) 

        loss_i = loss.item()  # scalar loss

        return {
            LOSS_KEY: loss,
            BAR_KEY: {'train_loss': loss_i},
            SCALAR_LOG_KEY: {'train_loss': loss_i}
        }

    def eval_step(self, batch, batch_idx, tag):
        inputs, g, rows = batch
        input_day, y_gbm, y = inputs
        forecast_length = y.size()[-1]
        y_hat, _ = self.model(input_day, g)
        if self.config.use_gbm:
            y_hat += y_gbm

        assert(y.size() == y_hat.size())

        if g['type'] == 'subgraph' and 'res_n_id' in g:  # if using SAINT sampler
            cent_n_id = g['cent_n_id']
            res_n_id = g['res_n_id']
            # Note: we only evaluate predictions on those initial nodes (per random walk)
            # to avoid duplicated computations
            y = y[:, res_n_id]
            y_hat = y_hat[:, res_n_id]
            cent_n_id = cent_n_id[res_n_id]
        else:
            cent_n_id = g['cent_n_id']

        if self.config.use_saintdataset:
            index_ptr = torch.cartesian_prod(
                torch.arange(rows.size(0)),
                torch.arange(cent_n_id.size(0)),
                torch.arange(forecast_length)
            )

            label = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': cent_n_id[index_ptr[:, 1]].data.cpu().numpy(),
                'forecast_idx': index_ptr[:,2].data.cpu().numpy(),
                'val': y.flatten().data.cpu().numpy()
            })

            pred = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': cent_n_id[index_ptr[:, 1]].data.cpu().numpy(),
                'forecast_idx': index_ptr[:,2].data.cpu().numpy(),
                'val': y_hat.flatten().data.cpu().numpy()
            })

        else:
            index_ptr = torch.cartesian_prod(
                torch.arange(rows.size(0)),
                torch.arange(forecast_length)
            )

            label = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': cent_n_id[index_ptr[:, 0]].data.cpu().numpy(),
                'forecast_idx': index_ptr[:,1].data.cpu().numpy(),
                'val': y.flatten().data.cpu().numpy()
            })

            pred = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': cent_n_id[index_ptr[:, 0]].data.cpu().numpy(),
                'forecast_idx': index_ptr[:,1].data.cpu().numpy(),
                'val': y_hat.flatten().data.cpu().numpy()
            })

        pred = pred.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()

        return {
            'label': label,
            'pred': pred,
            # 'atten': atten_context
        }

    def eval_epoch_end(self, outputs, tag, dates):
        pred = pd.concat([x['pred'] for x in outputs], axis=0)
        label = pd.concat([x['label'] for x in outputs], axis=0)
        pred = pred.groupby(['row_idx', 'node_idx','forecast_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()
        # atten_context = [x['atten'] for x in outputs]

        align_countries = label.reset_index().node_idx.map(lambda x: self.countries[x]).values
        align_dates = label.reset_index().row_idx.map(lambda x: dates[x]).values

        loss = np.mean(np.abs(pred['val'].values - label['val'].values))
        scores = self.produce_score(pred, label, dates)

        log_dict = {
            '{}_loss'.format(tag): loss,
            '{}_total_mistakes'.format(tag): scores['total_mistakes'],
            '{}_total_label'.format(tag): scores['total_label'],
            '{}_total_predict'.format(tag): scores['total_predict']
        }

        out = {
            BAR_KEY: log_dict,
            SCALAR_LOG_KEY: log_dict,
            VAL_SCORE_KEY: -scores['total_mistakes'],
            'pred': pred,
            'label': label,
            'scores': scores,
            'dates':align_dates,
            'countries':align_countries,
            # 'atten': atten_context
        }

        return out

    def produce_score(self, pred, label, dates=None):
        y_hat = pred.apply(lambda x: np.expm1(x))
        y = label.apply(lambda x: np.expm1(x))
        mape_metric = np.abs((y_hat+1)/(y+1)-1).reset_index(drop=False)

        total_mistakes = np.abs(y_hat.values - y.values).sum()
        total_label = np.abs(y.values).sum()
        total_predict = np.abs(y.values).sum()
        eval_df = pd.concat([y_hat.rename(columns={'val': 'pred'}),
                             y.rename(columns={'val': 'label'})],
                            axis=1).reset_index(drop=False)

        eval_df['mape'] = mape_metric['val']
        if dates is not None:
            eval_df['date'] = eval_df.row_idx.map(lambda x: dates[x])
        eval_df['countries'] = eval_df.node_idx.map(lambda x: self.countries[x])

        def produce_percent_count(m_df):
            res = pd.Series()
            res['pred'] = m_df['pred'].sum()
            res['label'] = m_df['label'].sum()
            res['mistake'] = np.abs(m_df['pred'] - m_df['label']).sum()
            return res

        scores = {'total_mistakes':total_mistakes,'total_label':total_label,'total_predict':total_predict}
        for name, metric in [
            ('mistakes', eval_df),
        ]:
            scores[name] = metric.groupby(
                'row_idx').apply(produce_percent_count)
            if dates is not None:
                scores[name]['date'] = dates

        return scores

    def val_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def val_epoch_end(self, outputs):
        val_out = self.eval_epoch_end(outputs, 'val', self.val_dates)
        return val_out

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs):
        test_out = self.eval_epoch_end(outputs, 'test', self.test_dates)
        return test_out


if __name__ == '__main__':
    start_time = time.time()

    # build argument parser and config
    config = RNNConfig()
    parser = argparse.ArgumentParser(description='RNN-Task')
    add_config_to_argparse(config, parser)

    # parse arguments to config
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)

    # build validation task
    task = RNNTask(config)
    # Set random seed before the initialization of network parameters
    # Necessary for distributed training
    task.set_random_seed()
    net = WrapperNet(task.config)
    task.init_model_and_optimizer(net)
    task.log('Build Validation Neural Nets')
    # select epoch with best validation accuracy
    best_epochs = 50
    if not task.config.skip_train:
        task.fit()
        best_epochs = task._best_val_epoch
        print('Best validation epochs: {}'.format(best_epochs))

    # Resume the best checkpoint for evaluation
    task.resume_best_checkpoint()
    val_eval_out = task.val_eval()
    test_eval_out = task.test_eval()
    # dump evaluation results of the best checkpoint to val out
    task.dump(val_out=val_eval_out,
              test_out=test_eval_out,
              epoch_idx=-1,
              is_best=True,
              dump_option=1)
    task.log('Best checkpoint (epoch={}, {}, {})'.format(
        task._passed_epoch, val_eval_out[BAR_KEY], test_eval_out[BAR_KEY]))

    if task.is_master_node:
        for tag, eval_out in [
            ('val', val_eval_out),
            ('test', test_eval_out),
        ]:
            print('-'*15, tag)
            scores = eval_out['scores']['mistakes']
            print('-'*5, 'mistakes')
            print('Average:')
            print(scores.mean().to_frame('mistakes'))
            print('Daily:')
            print(scores)

    task.log('Training time {}s'.format(time.time() - start_time))

    # # build task
    # config.update_by_dict({'max_epochs':best_epochs+2, 'infer': True})
    # task = RNNTask(config)
    # task.set_random_seed()
    # net = WrapperNet(task.config)
    # task.init_model_and_optimizer(net)
    # task.log('Build Neural Nets')
    # if not task.config.skip_train:
    #     task.fit()

    # # Resume the best checkpoint for evaluation
    # task.resume_best_checkpoint()
    # val_eval_out = task.val_eval()
    # test_eval_out = task.test_eval()
    # # dump evaluation results of the best checkpoint to val out
    # task.dump(val_out=val_eval_out,
    #           test_out=test_eval_out,
    #           epoch_idx=-1,
    #           is_best=True,
    #           dump_option=1)
    # task.log('Best checkpoint (epoch={}, {}, {})'.format(
    #     task._passed_epoch, val_eval_out[BAR_KEY], test_eval_out[BAR_KEY]))

    # if task.is_master_node:
    #     for tag, eval_out in [
    #         ('val', val_eval_out),
    #         ('test', test_eval_out),
    #     ]:
    #         print('-'*15, tag)
    #         scores = eval_out['scores']['mistakes']
    #         print('-'*5, 'mistakes')
    #         print('Average:')
    #         print(scores.mean().to_frame('mistakes'))
    #         print('Daily:')
    #         print(scores)

    # task.log('Training time {}s'.format(time.time() - start_time))
