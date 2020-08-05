import pandas as pd 
import numpy as np
import torch
from torch import nn
import os

def raw_data_preprocessing():

    # data link
    # mobility https://github.com/midas-network/COVID-19/tree/master/data/mobility/global/google_mobility
    # time series https://github.com/CSSEGISandData/COVID-19
    mobility = pd.read_csv('../mobility/Global_Mobility_Report_20200721.csv')
    daily_confirmed = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    daily_deaths = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    daily_recovered = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    daily_features = [item for item in daily_confirmed.columns if item not in ['Province/State', 'Country/Region', 'Lat', 'Long']]


    daily_confirmed[daily_features] = daily_confirmed[daily_features].diff(axis=1)
    daily_deaths[daily_features] = daily_deaths[daily_features].diff(axis=1)
    daily_recovered[daily_features] = daily_recovered[daily_features].diff(axis=1)

    daily_confirmed[daily_features] = daily_confirmed[daily_features].mask(daily_confirmed[daily_features]<0,0)
    daily_deaths[daily_features] = daily_deaths[daily_features].mask(daily_deaths[daily_features]<0,0)
    daily_recovered[daily_features] = daily_recovered[daily_features].mask(daily_recovered[daily_features]<0,0)      


    daily_confirmed = daily_confirmed[['Country/Region'] + daily_features].set_index('Country/Region').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'confirmed'},
                                                                                                                axis=1).groupby(['Country/Region','date'])['confirmed'].sum().reset_index()

    daily_deaths = daily_deaths[['Country/Region'] + daily_features].set_index('Country/Region').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'deaths'},
                                                                                                                axis=1).groupby(['Country/Region','date'])['deaths'].sum().reset_index()

    daily_recovered = daily_recovered[['Country/Region'] + daily_features].set_index('Country/Region').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'recovered'},
                                                                                                                axis=1).groupby(['Country/Region','date'])['recovered'].sum().reset_index()  

    daily_confirmed['confirmed'] = np.log1p(daily_confirmed.rolling(7,axis=1)['confirmed'].mean())
    daily_deaths['deaths'] = np.log1p(daily_deaths.rolling(7,axis=1)['deaths'].mean())
    daily_recovered['recovered'] = np.log1p(daily_recovered.rolling(7,axis=1)['recovered'].mean()) 

    daily_ts = pd.merge(daily_confirmed, daily_deaths, on=['Country/Region','date'],how='left')
    daily_ts = pd.merge(daily_ts, daily_recovered, on=['Country/Region','date'],how='left')    
    mobility_features = ['retail_and_recreation_percent_change_from_baseline',
                        'grocery_and_pharmacy_percent_change_from_baseline',
                        'parks_percent_change_from_baseline',
                        'transit_stations_percent_change_from_baseline',
                        'workplaces_percent_change_from_baseline',
                        'residential_percent_change_from_baseline']
    mobility = mobility[mobility.sub_region_1.isnull()]\
                        [['country_region','date'] + mobility_features]\
                        .rename({'country_region':'Country/Region'},axis=1)\
                        .replace({'United States':'US'})

    use_countries = set(mobility['Country/Region'].unique()) & set(daily_ts['Country/Region'].unique()) 
    mobility[mobility_features] = mobility[mobility_features] / 100.0
    mobility = mobility[mobility['Country/Region'].isin(use_countries)]
    mobility['date'] = pd.to_datetime(mobility.date)

    daily_ts = daily_ts[daily_ts['Country/Region'].isin(use_countries)]
    daily_ts['date'] = pd.to_datetime(daily_ts.date)

    df = pd.merge(mobility, daily_ts, on=['Country/Region','date'], how='left')

    df.to_csv('../data/daily_mobility.csv',index=False,header=True)


def load_data(data_fp, start_date, min_peak_size, lookback_days, lookahead_days, logger=print):
    logger('Load Data from ' + data_fp)
    logger('lookback_days={}, lookahead_days={}, '.format(
        lookback_days, lookahead_days))
    data = pd.read_csv(data_fp, parse_dates=['date'])
    data = data[data.date>=pd.to_datetime(start_date)].reset_index(drop=True)
    min_confirmed = data.groupby('Country/Region')['confirmed'].max()
    use_countries = min_confirmed[min_confirmed>=np.log1p(min_peak_size)].index.values
    data = data[data['Country/Region'].isin(use_countries)].reset_index(drop=True)
    data['weekday'] = data['date'].map(lambda x:x.weekday())
    dates = data['date'].unique()

    countries = data['Country/Region'].unique()
    use_features = [
        'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 'confirmed', 'deaths',
       'recovered',
       'weekday']
    use_features = ['confirmed', 'deaths','recovered','weekday']

    df = pd.DataFrame(index=pd.MultiIndex.from_product([countries, dates],
                      names=['Country/Region','date'])).reset_index()
    
    df = pd.merge(df, data, on=['Country/Region','date'], how='left').fillna(0.0)
    df = df[use_features].values.reshape(len(countries), len(dates), len(use_features))
    
    day_inputs = []
    outputs = []
    for day_idx in range(lookback_days, len(dates) - lookahead_days):
        day_input = df[:, day_idx-lookback_days:day_idx].copy()
        output = df[:, day_idx:day_idx + lookahead_days,-4].copy()            # confirmed, deaths, recovered

        day_inputs.append(day_input)
        outputs.append(output)
    
    # [num_samples, num_nodes, lookback_days, day_feature_dim]
    day_inputs = np.stack(day_inputs, axis=0)
    # [num_samples, num_nodes, 3]ies
    outputs = np.stack(outputs, axis=0)
    day_inputs = torch.from_numpy(day_inputs).float()
    outputs = torch.from_numpy(outputs).float()
    A = torch.ones(len(countries),len(countries)).to_sparse()
    edge_index = A._indices().long()

    logger("Input size: {}, Output size: {}, Edge size: {}".format(
        day_inputs.size(), outputs.size(), edge_index.size()))


    return day_inputs, outputs, edge_index, dates[lookback_days: -lookahead_days], countries

def load_npz_data(data_fp):

    data = np.load(data_fp, allow_pickle=True)
    day_inputs = torch.from_numpy(data['day_inputs']).float()
    countries = data['countries'].copy()
    dates = data['label_date'].copy()
    outputs = torch.from_numpy(data['outputs']).float()
    edge_index = torch.from_numpy(data['edge_index']).long()
    gbm_outputs = torch.from_numpy(data['gbm_outputs']).float()

    return day_inputs, gbm_outputs, outputs, edge_index, dates, countries

class ExpL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ExpL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        input = torch.expm1(input)
        target = torch.expm1(target)
        return F.l1_loss(input, target, reduction=self.reduction) 