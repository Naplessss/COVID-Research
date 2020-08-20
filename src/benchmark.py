import pandas as pd
import numpy as np
from epiweeks import Week, Year
from base_task import BaseConfig, add_config_to_argparse
import argparse
import datetime
import warnings
import torch
import os
warnings.filterwarnings(action='ignore')

class BenchmarkConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.target_start_date = '2020-07-12'
        self.horizon = 7
        self.model_name = 'sandwich'
        self.type_name = 'US'

        self.baseline_dir = '/home/zhgao/COVID-Research/covid19-forecast-hub/data-processed'
        self.baseline_name = 'GT-DeepCOVID'
        self.location_fp = '/home/zhgao/COVID-Research/covid19-forecast-hub/data-locations/locations.csv'



def get_benchmark(model_dir, model_name, location2name, target='1 wk ahead cum death'):
    baseline_fps = os.listdir(os.path.join(model_dir, model_name))
    baseline_fps = [item for item in baseline_fps if not (item.startswith('metadata') or item.endswith('.txt'))]
    horizon = 7 * int(target.split()[0])
    if len(baseline_fps)==0:
        return -1
    for i,_fp in enumerate(baseline_fps):
        tmp = pd.read_csv(os.path.join(model_dir, model_name, _fp))
        tmp = tmp[tmp['type']=='point'][tmp['target']==target]
        tmp['region'] = tmp.location.map(location2name)
        tmp = tmp[tmp.region!='US'][['forecast_date','target_end_date','value','region']]
        tmp['target_start_date'] = pd.to_datetime(tmp['target_end_date']) - datetime.timedelta(days=horizon-1)
        tmp['target_end_date'] = pd.to_datetime(tmp['target_end_date'])
        if i==0:
            df = tmp.copy()
        else:
            df = pd.concat([df,tmp], axis=0)
        
    return df.sort_values(['target_start_date','region']).reset_index(drop=True)

def get_label(death_fp, target_start_date, horizon=7, group_name='Province_State'):
    deaths = pd.read_csv(death_fp)
    ts_features = [item for item in deaths.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population', 'Province/State', 'Country/Region', 'Long']]

    deaths_us = deaths.groupby(group_name)[ts_features].sum().diff(axis=1).T
    deaths[ts_features] = deaths[ts_features].mask(deaths[ts_features]<0,0)
    deaths_us.index = pd.to_datetime(deaths_us.index)
    use_index = pd.date_range(start=target_start_date,freq='D',periods=horizon)
    deaths_us = deaths_us.loc[use_index]
    deaths_us['target_start_date'] = pd.to_datetime(use_index[0])
    deaths_us['target_end_date'] = pd.to_datetime(use_index[-1])
    deaths_us['horizon'] = horizon
    deaths_us = deaths_us.set_index(['target_start_date','target_end_date','horizon']).stack().reset_index().rename({0:'label', group_name:'region'},
                                                                                          axis=1)
    deaths_us = deaths_us.groupby(['target_start_date','horizon','region'])['label'].sum().reset_index()
    deaths_us.loc[deaths_us['label']<0,'label'] = 0

    deaths_cum =  deaths.groupby(group_name)[ts_features].sum().T
    deaths_cum.index = pd.to_datetime(deaths_cum.index)
    deaths_cum = deaths_cum.loc[pd.to_datetime(target_start_date)- datetime.timedelta(days=1)].reset_index()
    deaths_cum.columns = ['region','cum_label']
    deaths_us = pd.merge(deaths_us,deaths_cum,how='left',on=['region'])
    deaths_us['cum_label'] = deaths_us['cum_label'] + deaths_us['label']
    
    return deaths_us

def get_model_predict(model_fp):
    test = torch.load(os.path.join(model_fp,'Output','test.out.cpt'))
    res_test = pd.DataFrame({'pred':np.expm1(test['pred']['val'].values),
              'label':np.expm1(test['label']['val']).values,
              'forecast_idx':test['label'].reset_index()['forecast_idx'].values,
              'countries':test['countries'],
              'dates':test['dates']})    
    return res_test

if __name__ == "__main__":


    # build argument parser and config
    config = BenchmarkConfig()
    parser = argparse.ArgumentParser(description='Benchmark-Task')
    add_config_to_argparse(config, parser)

    # parse arguments to config
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)    

    if config.horizon == 7:
        config.target = '1 wk ahead cum death'
    elif config.horizon == 14:
        config.target = '2 wk ahead cum death'        
    config.death_fp ='/home/zhgao/COVID19/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_{}.csv'.format(config.type_name)
    config.group_name = {'US':'Province_State','global':'Country/Region'}.get(config.type_name)
    config.model_fp = '/home/zhgao/COVID-Research/{}_{}_{}_{}'.format(config.type_name, config.model_name, config.horizon, '_'.join(config.target_start_date.split('-')[-2:]))

    # benchmark link: https://github.com/reichlab/covid19-forecast-hub
    location = pd.read_csv(config.location_fp)
    location2name = dict(zip(location['location'],location['location_name']))   

    
    # epidemiological week start date: should be sunday of a specific week
    res_test = get_model_predict(config.model_fp)

    gt = get_label(config.death_fp, config.target_start_date, horizon=config.horizon, group_name=config.group_name)   # ['Country/Region', 'Province_State']
    pred = get_benchmark(config.baseline_dir, config.baseline_name, location2name, config.target) 
    pred = pd.merge(gt, pred, on=['target_start_date','region'], how='inner')

    states = list(set(pred.region.unique()) & set(res_test.countries.unique())) 
    print(states)
    pred = pred[pred.region.isin(states)]
    pred = pred.drop_duplicates(['target_start_date','region'], keep='last')
    res_test = res_test[res_test.countries.isin(states)]

    print("{}_MSE: ".format(config.baseline_name), np.sqrt((np.abs(pred['value'] - pred['cum_label'])**2).mean()))
    print("MSE: ", np.sqrt((np.abs(res_test['pred'] - res_test['label'])**2).mean()))

    print("{}_MAE: ".format(config.baseline_name), np.abs(pred['value'] - pred['cum_label']).mean())
    print("MAE: ", np.abs(res_test['pred'] - res_test['label']).mean())

    print(pred.head())
    print(res_test.head())
    print(pred.shape,res_test.shape)




