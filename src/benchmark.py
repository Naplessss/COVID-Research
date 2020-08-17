import pandas as pd
import numpy as np
from epiweeks import Week, Year
import datetime
import warnings
import os
warnings.filterwarnings(action='ignore')

def get_benchmark(model_dir, model_name, forecast_date, location2name):
    df = pd.read_csv(os.path.join(model_dir, model_name,'{}-{}.csv'.format(forecast_date, model_name)))
    df = df[df.type=='point'][df.target=='1 wk ahead cum death']
    df['region'] = df.location.map(location2name)
    df = df[df.region!='US'][['forecast_date','target_end_date','value','region']]
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])

    return df

def get_label(death_fp,forecast_date,horizon=7):
    deaths = pd.read_csv(death_fp)
    ts_features = [item for item in deaths.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']]

    deaths[ts_features] = deaths[ts_features].mask(deaths[ts_features]<0,0)
    deaths_us = deaths.groupby('Province_State')[ts_features].sum().diff(axis=1).T
    deaths_us.index = pd.to_datetime(deaths_us.index)
    use_index = pd.date_range(forecast_date,freq='D',periods=horizon)
    deaths_us = deaths_us.loc[use_index]
    deaths_us['forecast_date'] = pd.to_datetime(forecast_date)
    deaths_us['horizon'] = horizon
    deaths_us = deaths_us.set_index(['forecast_date','horizon']).stack().reset_index().rename({0:'label','Province_State':'region'},
                                                                                          axis=1)
    deaths_us = deaths_us.groupby(['forecast_date','horizon','region'])['label'].sum().reset_index()
    deaths_us.loc[deaths_us['label']<0,'label'] = 0

    deaths_cum =  deaths.groupby('Province_State')[ts_features].sum().T
    deaths_cum.index = pd.to_datetime(deaths_cum.index)
    deaths_cum = deaths_cum.loc[pd.to_datetime(forecast_date)- datetime.timedelta(days=1)].reset_index()
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
    baseline_name = 'UT-Mobility'
    forecast_date = '2020-06-08'
    location_fp = '/home/zhgao/COVID-Research/covid19-forecast-hub/data-locations/locations.csv'
    death_fp ='/home/zhgao/COVID19/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    location = pd.read_csv(location_fp)
    location2name = dict(zip(location['location'],location['location_name']))   

    model_fp = '/home/zhgao/COVID-Research/US_sandwich_7_6_29'
    res_test = get_model_predict(model_fp)

    gt = get_label(death_fp, forecast_date, horizon=7)
    pred = get_benchmark(baseline_dir, baseline_name, forecast_date, location2name) 
    pred = pd.merge(pred, gt, on=['forecast_date','region'], how='left') 
    print("{}_MAE: ".format(baseline_name), np.abs(pred['value'] - pred['cum_label']).mean())
    print("MAE: ", np.abs(res_test['pred'] - res_test['label']).mean())
    print(pred.head())




