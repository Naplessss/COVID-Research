import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import datetime
import argparse
from scipy.stats import skew,kurtosis
from joblib import Parallel,delayed
from base_task import BaseConfig, add_config_to_argparse
from main_task import load_data
import os
import warnings
warnings.filterwarnings(action='ignore')

class GBDTConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # for data loading
        self.data_fp = '../data/daily_mobility.csv'
        self.start_date = '2020-04-01'
        self.min_peak_size = -1
        self.lookback_days = 14  # the number of days before the current day for daily series
        self.lookahead_days = 1
        self.forecast_date = '2020-06-07'
        self.horizon = 7
        self.label = 'deaths_target'
        self.use_mobility = False


        self.exp_dir = '../gbdt'
        self.nthreads = 12
        self.load_data = True
        self.dump_data = True
        self.dump_fp = '../data/gbm_dataset.npz'

def feature_engineering_raw_data(day_inputs, outputs, dates, ids):
    features_list, label_list, id_list, datetime_list = [], [], [], []
    for i, _date in enumerate(dates):
        for j, _id in enumerate(ids):
            features_list.append(day_inputs[i,j].cpu().numpy().flatten().tolist())
            label_list.append(outputs[i,j].cpu().numpy())
            id_list.append(_id)
            datetime_list.append(_date)

    df = pd.DataFrame(features_list)
    df['datetime'] = [pd.to_datetime(item) for item in datetime_list]
    df['id'] = id_list
    df['label'] = np.array(label_list).reshape(-1,1)
    id2index = {id:i for i,id in enumerate(df['id'].unique())}
    df['id_onehot'] = df['id'].map(id2index)
    print(df.shape)

    return df


def train_valid_test_split(dates, forecast_date, horizon=7):

    forecast_date = pd.to_datetime(forecast_date)
    val_date = forecast_date - datetime.timedelta(days=horizon+7)

    train = df[df.datetime < val_date]
    val = df[df.datetime == val_date]
    test = df[df.datetime > val_date]

    return train, val, test

def mape(preds, label, EPS=1):
    preds = np.expm1(preds)
    label = np.expm1(label)
    return np.abs(preds - label) / (label + EPS)

def eval_func(preds,dtrain):
    labels = dtrain.get_label()
    return 'my_mape', mape(preds,labels), False


def dump(clf, imp, val, test, config):
    os.makedirs(os.path.join(config.exp_dir,'Output'), exist_ok=True)
    node_zonecode2idx = {zonecode:i for i,zonecode in enumerate(countries)}
    for out,fn in zip([val, test],['val','test']):
        datetime2idx = {datetime:i for i,datetime in enumerate(out['datetime'].unique())}
        out = out[['datetime','id','predict','label']]
        out['node_idx'] = out['id'].map(node_zonecode2idx)
        out['row_idx'] = out['datetime'].map(datetime2idx)
        out = out.sort_values(['row_idx','node_idx'])
        out = out.set_index(['row_idx','node_idx'])

        torch.save({'pred':out[['predict']].rename({'predict':'val'},axis=1),
                    'label':out[['label']].rename({'label':'val'},axis=1)}, os.path.join(config.exp_dir,'Output',fn+'.out.cpt'))

    torch.save({'model':clf,'importance':imp}, os.path.join(config.exp_dir,'model.cpt'))

    return True

def train_lgb():
    train, val, test = train_valid_test_split(dates, config.forecast_date, config.horizon)
    fea_names = [item for item in train.columns if item not in ['datetime','id','label']]
    params = {
          "objective": "regression_l1",  #regression,mape
          "boosting_type": "gbdt",
          "learning_rate": 0.03,
          "num_leaves": 31,
          "max_bin": 256,
          "feature_fraction": 0.55,
          "verbosity": 0,
          "subsample": 0.55,
          "num_threads": config.nthreads
          }

    dtrain =lgb.Dataset(train[fea_names],train['label'])
    dval = lgb.Dataset(val[fea_names],val['label'], reference=dtrain)

    clf = lgb.train(params, dtrain,
                    num_boost_round=10000,
                    valid_sets=[dtrain,dval],
                    # feval=eval_func,
                    verbose_eval=100,
                    early_stopping_rounds=200
                   )

    val['predict'] = clf.predict(val[fea_names])
    test['predict'] = clf.predict(test[fea_names])
    all_pred = clf.predict(df[fea_names])

    val['mape'] = mape(val.predict,val.label)
    test['mape'] = mape(test.predict,test.label)

    val['mistakes'] = np.abs(np.expm1(val['predict']) - np.expm1(val['label']))
    test['mistakes'] = np.abs(np.expm1(test['predict']) - np.expm1(test['label']))

    print("overall mistakes:  val={}, test={}".format(val['mistakes'].sum(), test['mistakes'].sum()))

    imp = pd.Series(clf.feature_importance(), index=fea_names).sort_values(ascending=False)

    return clf, imp, val, test, all_pred

if __name__ == '__main__':

    # build argument parser and config
    config = GBDTConfig()
    parser = argparse.ArgumentParser(description='GBDT-Task')
    add_config_to_argparse(config, parser)

    # parse arguments to config
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)

    outputs = load_data(config.data_fp, 
                        config.start_date, 
                        config.min_peak_size, 
                        config.lookback_days, 
                        config.lookahead_days, 
                        config.label,
                        config.use_mobility)
    day_inputs, outputs, edge_index, dates, countries = outputs
    print(day_inputs.shape, outputs.shape, edge_index.shape)
    df = feature_engineering_raw_data(day_inputs, outputs, dates, countries)

    clf, imp, val, test, all_pred = train_lgb()
    dump(clf, imp, val, test, config)

    if config.load_data and config.dump_data:
        gbm_outputs = all_pred.reshape(outputs.shape)
        print('Dump gbm dataset to {}'.format(config.dump_fp))
        np.savez(config.dump_fp,
                 label_date=dates,
                 countries=countries,
                 day_inputs=day_inputs,
                 outputs=outputs,
                 edge_index=edge_index,
                 gbm_outputs=gbm_outputs)