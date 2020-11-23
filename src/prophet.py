import pandas as pd
import numpy as np
import datetime as dt
import json
import os
from fbprophet import Prophet

def get_label():
    confirmed_fp ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    confirmed = pd.read_csv(confirmed_fp)
    ts_features = [item for item in confirmed.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population', 'Province/State', 'Country/Region', 'Long']]
    confirmed = confirmed.groupby('Province_State')[ts_features].sum()
    confirmed[ts_features] = confirmed[ts_features].diff(axis=1)
    confirmed = confirmed.T
    confirmed.index = pd.to_datetime(confirmed.index)
    
    deaths_fp ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    deaths = pd.read_csv(deaths_fp)
    ts_features = [item for item in deaths.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population', 'Province/State', 'Country/Region', 'Long']]
    deaths = deaths.groupby('Province_State')[ts_features].sum()
    deaths[ts_features] = deaths[ts_features].diff(axis=1)
    deaths = deaths.T
    deaths.index = pd.to_datetime(deaths.index)
    
    return deaths, confirmed

def get_single_date_predict(df, state, label_end_date, lookback_days=28, lookahead_days=28):
    tmp = df[state].reset_index()
    tmp.columns = ['ds','y']
    tmp['ds'] = pd.to_datetime(tmp['ds'])
    tmp = tmp[tmp['ds']<=pd.to_datetime(label_end_date)]
    tmp = tmp.iloc[(0-lookback_days):]
    print(tmp)
    m = Prophet()
    m.fit(tmp)
    future = m.make_future_dataframe(periods=lookahead_days)
    forecast = m.predict(future)
    forecast = forecast[['ds','yhat']]
    forecast['location_name'] = state
    return forecast

def dump_results(label_end_date='2020-11-16', forecast_start_date='2020-11-15'):
    all_results = []
    for state in confirmed.columns:
        print(state)
        forecast = get_single_date_predict(confirmed, state, label_end_date)
        forecast['TYPE'] = 'confirmed'
        all_results.append(forecast)

        forecast = get_single_date_predict(deaths, state, label_end_date)
        forecast['TYPE'] = 'deaths'
        all_results.append(forecast)

    all_results = pd.concat(all_results)

    all_results['predict_week'] = all_results.groupby('location_name')['yhat'].apply(lambda x:x.rolling(7).sum())
    dates = pd.date_range(start=forecast_start_date, periods=4, freq=pd.offsets.Week(n=1,weekday=5))   # end day of each epidemic week
    all_results = all_results[all_results.ds.isin(dates)]
    all_results['predict_week'] = all_results['predict_week'].map(lambda x: x if x>0 else 0)

    def get_locations():
        location = pd.read_csv('../data/locations.csv')
        location2id = location[['location_name','location']].set_index('location_name')['location'].to_dict()
        return location2id
    location2id = get_locations()
    US = all_results.groupby(['ds','TYPE'])[['yhat','predict_week']].sum().reset_index()
    US['location_name'] = 'US'
    all_results = pd.concat([all_results, US], axis=0)
    all_results['region'] = all_results['location_name'].map(location2id).astype(str)
    all_results.to_csv('../output/prophet.predict.{}.csv'.format(forecast_start_date),index=False)

    return all_results

if __name__=='__main__':
    # newest date of labeling data
    LABEL_END_DATE = '2020-11-22'    
    # test start day to infer next epidemic weeks (included this day)
    # prefer to be sunday of this epidemic week (the same as LABEL_DATE_END)
    FORECAST_START_DATE = '2020-11-22'   

    deaths, confirmed = get_label()
    dump_results(label_end_date=LABEL_END_DATE, forecast_start_date=FORECAST_START_DATE)
    # dump_results(forecast_date='2020-11-09', target_start_date='2020-11-07')
    # dump_results(forecast_date='2020-11-02', target_start_date='2020-10-31')
    # dump_results(forecast_date='2020-10-26', target_start_date='2020-10-24')
    # dump_results(forecast_date='2020-10-19', target_start_date='2020-10-17')
    # dump_results(forecast_date='2020-10-12', target_start_date='2020-10-10')
    print(deaths.tail(5))
    print(confirmed.tail(5))