import torch
import os
import datetime
from datetime import datetime as dt
from glob import glob
import numpy as np
import pandas as pd


def get_locations():
    location = pd.read_csv('/home/zhgao/COVID-Research/covid19-forecast-hub/data-locations/locations.csv')
    location2id = location[['location_name','location']].set_index('location_name')['location'].to_dict()
    return location2id

def get_label(target_date='2020-10-04'):
    target_date = dt.strftime(pd.to_datetime(target_date) - datetime.timedelta(days=1), '%-m/%-d/%y')
    deaths = pd.read_csv('/home/zhgao/COVID19/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
    confirmed = pd.read_csv('/home/zhgao/COVID19/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    deaths = deaths.groupby('Province_State').sum()[target_date]
    confirmed = confirmed.groupby('Province_State').sum()[target_date]
    deaths = deaths.append(pd.Series({'US':deaths.sum()}))
    confirmed = confirmed.append(pd.Series({'US':confirmed.sum()}))
    
    return deaths, confirmed

def get_predict(model_dir='/home/zhgao/COVID-Research/weights_major/', model_fp='US_08_30'):
    fname_pattern = glob(os.path.join(model_dir, model_fp + '_seed_*'))
    res = pd.DataFrame()
    location2id = get_locations()
    forecast_date = model_fp.split('_')[2]
    deaths_label, confirmed_label = get_label(forecast_date)
    deaths_label.index = deaths_label.index.map(location2id)
    confirmed_label.index = confirmed_label.index.map(location2id)
    
    for fname in fname_pattern:
        test = torch.load(os.path.join(fname,'Output','test.out.cpt'))
        predict_value = [item if item>0 else 0 for item in np.expm1(test['pred']['val'].values)]
        res_test = pd.DataFrame({'pred':predict_value,
                  'label':np.expm1(test['label']['val']).values,
                  'forecast_idx':test['label'].reset_index()['forecast_idx'].values,
                  'countries':test['countries'],
                  'dates':test['dates']})   
        res_test = res_test.rename({'countries':'region','dates':'target_start_date','pred':'value'},axis=1)
        res_test['region'] = res_test['region'].map(location2id)
        res_test = res_test[res_test.region!='11001']
        res_test = res_test[['value','region']].set_index('region',drop=True).rename({'value':'_'.join(fname.split('_')[-2:])},
                                                                                    axis=1)
        
        res = pd.concat([res, res_test], axis=1)
    
    return res, deaths_label, confirmed_label

def get_predict_list(date='2020-10-04', save=False):
    predict_deaths_list, predict_confirmed_list = [], []
    for type_name in ['deaths','confirmed']:
        for days in [7,14,21,28]:
            fp = 'US_{}_{}_{}'.format(type_name, date, days)
            print(fp)
            predict, deaths_label, confirmed_label = get_predict(model_fp=fp)
            if type_name == 'deaths':           
                predict_deaths_list.append(predict)
            if type_name == 'confirmed':
                predict_confirmed_list.append(predict)
    
    if save:
        pd.to_pickle([predict_deaths_list, predict_confirmed_list, deaths_label, confirmed_label],
                '../output/gnn.predict.{}.pkl'.format(date))
    return predict_deaths_list, predict_confirmed_list, deaths_label, confirmed_label


def get_ensemble_results(date = forecast_start_date):
    predict_deaths_list, predict_confirmed_list, deaths_label, confirmed_label = get_predict_list(date)
    gbm_predict = pd.read_csv('../output/gbm.predict.{}.csv'.format(date))
    location2id = get_locations()
    gbm_predict['region'] = gbm_predict['Location'].map(location2id)
    for i in [0,1,2,3]:
        _deaths = predict_deaths_list[i]
        _gbm_deaths = gbm_predict[(gbm_predict['region'].isin(_deaths.index)) &\
                                  (gbm_predict['k'] == (i*7+7)) &\
                                  (gbm_predict['TYPE'] == 'Deaths')]
        _gbm_deaths = _gbm_deaths[['region','PREDICTION']]
        _gbm_deaths = _gbm_deaths.append({'region':'US','PREDICTION':_gbm_deaths['PREDICTION'].sum()},
                       ignore_index=True).set_index('region')

        _deaths = pd.merge(_deaths,_gbm_deaths,left_index=True, right_index=True)
        seeds_cols = [item for item in _deaths.columns if item.startswith('seed')]
        _deaths = _deaths[seeds_cols].add(_deaths['PREDICTION'], axis=0) / 2.0
        predict_deaths_list[i] = _deaths

        _confirmed = predict_confirmed_list[i]
        _gbm_confirmed = gbm_predict[(gbm_predict['region'].isin(_confirmed.index)) &\
                                  (gbm_predict['k'] == (i*7+7)) &\
                                  (gbm_predict['TYPE'] == 'Confirmed')]
        _gbm_confirmed = _gbm_confirmed[['region','PREDICTION']]
        _gbm_confirmed = _gbm_confirmed.append({'region':'US','PREDICTION':_gbm_confirmed['PREDICTION'].sum()},
                       ignore_index=True).set_index('region')

        _confirmed = pd.merge(_confirmed,_gbm_confirmed,left_index=True, right_index=True)
        seeds_cols = [item for item in _confirmed.columns if item.startswith('seed')]
        _confirmed = _confirmed[seeds_cols].add(_confirmed['PREDICTION'], axis=0) / 2.0
        predict_confirmed_list[i] = _confirmed    
    
    return predict_deaths_list, predict_confirmed_list, deaths_label, confirmed_label

use_quantile_list = [0.01,0.025,0.05,0.1,0.15,0.2,0.25,0.3,
                     0.35,0.4,0.45,0.5,0.55,0.6,0.65,
                     0.7,0.75,0.8,0.85,0.9,0.95,0.975,0.99]

def generate_gnn_cdc_format(save_dir = './',
                        model_name = 'MSRA-DeepST',
                        forecast_date = '2020-10-04',
                        predict_date = '2020-10-10',
                        quantile = use_quantile_list,
                        use_ensemble = True
                       ):
    
    forecast_start_date = dt.strftime(pd.to_datetime(predict_date) - datetime.timedelta(days=6),'%Y-%m-%d')
    if not use_ensemble:
        predict_deaths_list, predict_confirmed_list, deaths_label, confirmed_label = get_predict_list(date=forecast_start_date)
    else:
        predict_deaths_list, predict_confirmed_list, deaths_label, confirmed_label = get_ensemble_results(date=forecast_start_date)
        

    results = []
    out_fp = '-'.join([forecast_date, model_name]) + '.csv'
    
    for predict_list,label,type_name in [[predict_deaths_list, deaths_label,'death'],
                                    [predict_confirmed_list, confirmed_label,'case']]:
        for ahead_weeks,predict in enumerate(predict_list):
            target_end_date = pd.to_datetime(predict_date) + datetime.timedelta(days=7*ahead_weeks)
            target_end_date = dt.strftime(target_end_date, '%Y-%m-%d')
            ahead_weeks = ahead_weeks + 1
            if type_name == 'death':
                for region_id in predict.index:
                    _cum_value = label[region_id]
                    _point_value = predict.loc[region_id].mean()
                    _quantile_value = predict.loc[region_id].quantile(use_quantile_list).values
                    _quantile_value = [item if item>0 else 0 for item in _quantile_value]
                    _value = [_point_value] + list(_quantile_value)

                    tmp_cum = pd.DataFrame({
                                 'forecast_date': [forecast_date] * len(_value),
                                 'target': ['{} wk ahead cum {}'.format(ahead_weeks, type_name)] * len(_value),
                                 'target_end_date': [target_end_date] * len(_value),
                                 'location': [region_id] * len(_value),
                                 'type':['point'] + ['quantile']*len(use_quantile_list),
                                 'quantile':['NA'] + use_quantile_list,
                                 'value':_value + _cum_value})
                    results.append(tmp_cum)
                    
    for i in [3,2,1]:
        predict_deaths_list[i] = predict_deaths_list[i] - predict_deaths_list[i-1]
        predict_confirmed_list[i] = predict_confirmed_list[i] - predict_confirmed_list[i-1]
    
    for predict_list,label,type_name in [[predict_deaths_list, deaths_label,'death'],
                                    [predict_confirmed_list, confirmed_label,'case']]:
        for ahead_weeks,predict in enumerate(predict_list):
            target_end_date = pd.to_datetime(predict_date) + datetime.timedelta(days=7*ahead_weeks)
            target_end_date = dt.strftime(target_end_date, '%Y-%m-%d')
            ahead_weeks = ahead_weeks + 1
            if type_name == 'death':
                for region_id in predict.index:
                    _cum_value = label[region_id]
                    _point_value = predict.loc[region_id].mean()
                    _quantile_value = predict.loc[region_id].quantile(use_quantile_list).values
                    _quantile_value = [item if item>0 else 0 for item in _quantile_value]
                    _value = [_point_value] + list(_quantile_value)
                    tmp_inc = pd.DataFrame({
                                 'forecast_date': [forecast_date] * len(_value),
                                 'target': ['{} wk ahead inc {}'.format(ahead_weeks, type_name)] * len(_value),
                                 'target_end_date': [target_end_date] * len(_value),
                                 'location': [region_id] * len(_value),
                                 'type':['point'] + ['quantile']*len(use_quantile_list),
                                 'quantile':['NA'] + use_quantile_list,
                                 'value':_value})

                    results.append(tmp_inc)
                    
            if type_name == 'case':
                tmp = pd.DataFrame({
                                 'forecast_date': [forecast_date] * len(predict),
                                 'target': ['{} wk ahead inc {}'.format(ahead_weeks, type_name)] * len(predict),
                                 'target_end_date': [target_end_date] * len(predict),
                                 'location': predict.index.values,
                                 'type':['point'] * len(predict),
                                 'quantile':['NA'] * len(predict),
                                 'value': [item if item>0 else 0 for item in predict.mean(axis=1).values]})
                
                results.append(tmp)
                
    results = pd.concat(results, axis=0, ignore_index=True)
    # results = results.sort_values(['target','location'])
    results.to_csv(os.path.join(save_dir, out_fp),index=False,header=True)
    
    return results

if __name__=='__main__':

    results = generate_gnn_cdc_format(save_dir = '../output',
                            model_name = 'MSRA-DeepST',
                            forecast_date = '2020-10-19',
                            predict_date = '2020-10-24',
                            quantile = use_quantile_list,
                            use_ensemble= True) 