import pandas as pd
import numpy as np
from epiweeks import Week, Year
import datetime
import warnings
import torch
import os
warnings.filterwarnings(action='ignore')

def get_benchmark(model_dir, model_name, location2name, target='1 wk ahead cum death'):
    baseline_fps = os.listdir(os.path.join(baseline_dir, baseline_name))
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

def get_label_v2(death_fp):
    deaths = pd.read_csv(death_fp)
    ts_features = [item for item in deaths.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']]

    deaths_us = deaths.groupby('Province_State')[ts_features].sum().diff(axis=1).T
    deaths_us.index = pd.to_datetime(deaths_us.index)
    deaths_us['epiweek'] = [Week.fromdate(item) for item in deaths_us.index]
    deaths_us['forecast_date'] = deaths_us['epiweek'].map(lambda x:x.startdate())
    deaths_us['target_end_date'] = deaths_us['epiweek'].map(lambda x:x.enddate())
    deaths_us = deaths_us.set_index(['forecast_date','target_end_date']).stack().reset_index().rename({0:'label','Province_State':'region'},
                                                                                          axis=1)
    deaths_us = deaths_us[deaths_us.region!='epiweek']
    deaths_us = deaths_us.groupby(['forecast_date','target_end_date','region'])['label'].sum().reset_index()
    deaths_us.loc[deaths_us['label']<0,'label'] = 0
    deaths_us['cum_label'] = deaths_us.groupby('region')['label'].cumsum()
    
    return deaths_us

def get_label(death_fp, target_start_date, horizon=7):
    deaths = pd.read_csv(death_fp)
    ts_features = [item for item in deaths.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']]

    deaths[ts_features] = deaths[ts_features].mask(deaths[ts_features]<0,0)
    deaths_us = deaths.groupby('Province_State')[ts_features].sum().diff(axis=1).T
    deaths_us.index = pd.to_datetime(deaths_us.index)
    use_index = pd.date_range(start=target_start_date,freq='D',periods=horizon)
    deaths_us = deaths_us.loc[use_index]
    deaths_us['target_start_date'] = pd.to_datetime(use_index[0])
    deaths_us['target_end_date'] = pd.to_datetime(use_index[-1])
    deaths_us['horizon'] = horizon
    deaths_us = deaths_us.set_index(['target_start_date','target_end_date','horizon']).stack().reset_index().rename({0:'label','Province_State':'region'},
                                                                                          axis=1)
    deaths_us = deaths_us.groupby(['target_start_date','horizon','region'])['label'].sum().reset_index()
    deaths_us.loc[deaths_us['label']<0,'label'] = 0

    deaths_cum =  deaths.groupby('Province_State')[ts_features].sum().T
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
    # benchmark link: https://github.com/reichlab/covid19-forecast-hub
    baseline_dir = '/home/zhgao/COVID-Research/covid19-forecast-hub/data-processed'
    baseline_name = 'IHME-CurveFit'
    location_fp = '/home/zhgao/COVID-Research/covid19-forecast-hub/data-locations/locations.csv'
    death_fp ='/home/zhgao/COVID19/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    location = pd.read_csv(location_fp)
    location2name = dict(zip(location['location'],location['location_name']))   

    
    # epidemiological week start date: should be sunday of a specific week
    target_start_date = '2020-07-12'
    horizon = 7
    target = '1 wk ahead cum death'
    model_name = 'sandwich'
    model_fp = '/home/zhgao/COVID-Research/US_{}_{}_{}'.format(model_name, horizon,'_'.join(target_start_date.split('-')[-2:]))
    res_test = get_model_predict(model_fp)

    gt = get_label(death_fp, target_start_date, horizon=horizon)
    pred = get_benchmark(baseline_dir, baseline_name, location2name, target) 
    pred = pd.merge(gt, pred, on=['target_start_date','region'], how='inner')
    print("{}_MSE: ".format(baseline_name), np.sqrt((np.abs(pred['value'] - pred['cum_label'])**2).mean()))
    print("MSE: ", np.sqrt((np.abs(res_test['pred'] - res_test['label'])**2).mean()))

    print("{}_MAE: ".format(baseline_name), np.abs(pred['value'] - pred['cum_label']).mean())
    print("MAE: ", np.abs(res_test['pred'] - res_test['label']).mean())

    print(pred.head())
    print(res_test.head())
    print(pred.shape,res_test.shape)




