import pandas as pd 
import numpy as np
import datetime 
import torch
from torch import nn
import os

def raw_data_preprocessing(data_fp='../data/daily_mobility.csv', horizon=7, only_china=False):

    # data link
    # mobility https://github.com/midas-network/COVID-19/tree/master/data/mobility/global/google_mobility
    # time series https://github.com/CSSEGISandData/COVID-19

    mobility_dir = '/home/zhgao/COVID19/mobility'
    ts_dir = '/home/zhgao/COVID19/COVID-19'
    mobility = pd.read_csv(os.path.join(mobility_dir, 'Global_Mobility_Report_20201006.csv'))
    daily_confirmed = pd.read_csv(os.path.join(ts_dir, 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'))
    daily_deaths = pd.read_csv(os.path.join(ts_dir, 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'))
    daily_recovered = pd.read_csv(os.path.join(ts_dir, 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'))
    daily_features = [item for item in daily_confirmed.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
        'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Country/Region', 'Province/State']]

    if only_china:
        daily_confirmed = daily_confirmed[daily_confirmed['Country/Region']=='China']
        daily_deaths = daily_deaths[daily_deaths['Country/Region']=='China']
        daily_recovered = daily_recovered[daily_recovered['Country/Region']=='China']
        daily_confirmed['Country/Region'] = daily_confirmed['Province/State']
        daily_deaths['Country/Region'] = daily_deaths['Province/State']
        daily_recovered['Country/Region'] = daily_recovered['Province/State']

    daily_confirmed[daily_features] = daily_confirmed[daily_features].diff(axis=1)
    daily_deaths[daily_features] = daily_deaths[daily_features].diff(axis=1)
    daily_recovered[daily_features] = daily_recovered[daily_features].diff(axis=1)

    # daily_confirmed[daily_features] = daily_confirmed[daily_features].mask(daily_confirmed[daily_features]<0,0)
    # daily_deaths[daily_features] = daily_deaths[daily_features].mask(daily_deaths[daily_features]<0,0)
    # daily_recovered[daily_features] = daily_recovered[daily_features].mask(daily_recovered[daily_features]<0,0)      


    daily_confirmed = daily_confirmed[['Country/Region'] + daily_features].set_index('Country/Region').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'confirmed'},
                                                                                                                axis=1).groupby(['Country/Region','date'])['confirmed'].sum().reset_index()

    daily_deaths = daily_deaths[['Country/Region'] + daily_features].set_index('Country/Region').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'deaths'},
                                                                                                                axis=1).groupby(['Country/Region','date'])['deaths'].sum().reset_index()

    daily_recovered = daily_recovered[['Country/Region'] + daily_features].set_index('Country/Region').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'recovered'},
                                                                                                                axis=1).groupby(['Country/Region','date'])['recovered'].sum().reset_index()  


    daily_confirmed_Global = daily_confirmed.groupby('date')['confirmed'].sum().reset_index().rename({0:'confirmed'})
    daily_confirmed_Global['Country/Region'] = 'Global'
    daily_confirmed = pd.concat([daily_confirmed, daily_confirmed_Global], axis=0, ignore_index=True)
    daily_confirmed['date'] = pd.to_datetime(daily_confirmed['date'])  
    daily_confirmed = daily_confirmed.sort_values(['Country/Region','date'])

    daily_deaths_global = daily_deaths.groupby('date')['deaths'].sum().reset_index().rename({0:'deaths'})
    daily_deaths_global['Country/Region'] = 'Global'
    daily_deaths = pd.concat([daily_deaths, daily_deaths_global], axis=0, ignore_index=True)
    daily_deaths['date'] = pd.to_datetime(daily_deaths['date'])  
    daily_deaths = daily_deaths.sort_values(['Country/Region','date'])

    daily_recovered_global = daily_recovered.groupby('date')['recovered'].sum().reset_index().rename({0:'recovered'})
    daily_recovered_global['Country/Region'] = 'Global'
    daily_recovered = pd.concat([daily_recovered, daily_recovered_global], axis=0, ignore_index=True)
    daily_recovered['date'] = pd.to_datetime(daily_recovered['date'])  
    daily_recovered = daily_recovered.sort_values(['Country/Region','date'])
    
    # refactor -inf 
    daily_confirmed.loc[daily_confirmed['confirmed']<0,'confirmed']=0
    daily_deaths.loc[daily_deaths['deaths']<0,'deaths']=0
    daily_recovered.loc[daily_recovered['recovered']<0,'recovered']=0

    ## weekly rolling mean for confirmed, deaths, recovered
    daily_confirmed['confirmed_rolling'] = np.log1p(daily_confirmed.groupby('Country/Region').apply(lambda x:x.rolling(horizon, axis=1)['confirmed'].mean()).values)
    daily_deaths['deaths_rolling'] = np.log1p(daily_deaths.groupby('Country/Region').apply(lambda x:x.rolling(horizon, axis=1)['deaths'].mean()).values)
    daily_recovered['recovered_rolling'] = np.log1p(daily_recovered.groupby('Country/Region').apply(lambda x:x.rolling(horizon, axis=1)['recovered'].mean()).values) 

    # target, we need to forecast the cumsum of next horizon days.
    daily_confirmed['confirmed_target'] = np.log1p(daily_confirmed.groupby('Country/Region')['confirmed'].apply(lambda x:x.rolling(horizon).sum().shift(1-horizon)))
    daily_deaths['deaths_target'] = np.log1p(daily_deaths.groupby('Country/Region')['deaths'].apply(lambda x:x.rolling(horizon).sum().shift(1-horizon)))
    daily_recovered['recovered_target'] = np.log1p(daily_recovered.groupby('Country/Region')['recovered'].apply(lambda x:x.rolling(horizon).sum().shift(1-horizon)))
    daily_confirmed['confirmed'] = np.log1p(daily_confirmed['confirmed'])
    daily_deaths['deaths'] = np.log1p(daily_deaths['deaths'])
    daily_recovered['recovered'] = np.log1p(daily_recovered['recovered'])
   
    daily_ts = pd.merge(daily_confirmed, daily_deaths, on=['Country/Region','date'],how='left')
    daily_ts = pd.merge(daily_ts, daily_recovered, on=['Country/Region','date'],how='left')    
    mobility_features = ['retail_and_recreation_percent_change_from_baseline',
                        'grocery_and_pharmacy_percent_change_from_baseline',
                        'parks_percent_change_from_baseline',
                        'transit_stations_percent_change_from_baseline',
                        'workplaces_percent_change_from_baseline',
                        'residential_percent_change_from_baseline']
    mobility = mobility[mobility.sub_region_1.isnull()][mobility.sub_region_2.isnull()]\
                        [mobility.metro_area.isnull()]\
                        [['country_region','date'] + mobility_features]\
                        .rename({'country_region':'Country/Region'},axis=1)\
                        .replace({'United States':'US'})

    mobility_global = mobility.groupby('date')[mobility_features].mean().reset_index()
    mobility_global['Country/Region'] = 'Global'
    mobility = pd.concat([mobility, mobility_global], axis=0, ignore_index=True)
    print(mobility.tail())

    use_countries = set(mobility['Country/Region'].unique()) & set(daily_ts['Country/Region'].unique()) 
    # use_countries = daily_ts['Country/Region'].unique()
    mobility[mobility_features] = mobility[mobility_features] / 100.0
    mobility = mobility[mobility['Country/Region'].isin(use_countries)]
    mobility['date'] = pd.to_datetime(mobility.date)

    daily_ts = daily_ts[daily_ts['Country/Region'].isin(use_countries)]
    daily_ts['date'] = pd.to_datetime(daily_ts.date)
    df = pd.merge(daily_ts, mobility, on=['Country/Region','date'], how='left')
    df = df.fillna(0.0)

    save_fp = '.'.join(data_fp.split('.')[:-1]) + '_' + str(horizon) + '.csv'
    df.to_csv(save_fp, index=False, header=True)

def raw_data_preprocessing_US(data_fp='../data/daily_mobility_US.csv', horizon=7):
    mobility_dir = '/home/zhgao/COVID19/mobility'
    ts_dir = '/home/zhgao/COVID19/COVID-19'
    mobility = pd.read_csv(os.path.join(mobility_dir, 'Global_Mobility_Report_20201006.csv'))
    daily_confirmed = pd.read_csv(os.path.join(ts_dir, 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'))
    daily_deaths = pd.read_csv(os.path.join(ts_dir, 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'))
    # placeholder for daily_recovered, because no recovered data for US.
    daily_recovered = pd.read_csv(os.path.join(ts_dir, 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'))
    daily_features = [item for item in daily_confirmed.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
        'Country_Region', 'Lat', 'Long_', 'Combined_Key']]

    daily_confirmed[daily_features] = daily_confirmed[daily_features].diff(axis=1)
    daily_deaths[daily_features] = daily_deaths[daily_features].diff(axis=1)
    daily_recovered[daily_features] = daily_recovered[daily_features].diff(axis=1)

    #daily_confirmed[daily_features] = daily_confirmed[daily_features].mask(daily_confirmed[daily_features]<0,0)
    #daily_deaths[daily_features] = daily_deaths[daily_features].mask(daily_deaths[daily_features]<0,0)
    #daily_recovered[daily_features] = daily_recovered[daily_features].mask(daily_recovered[daily_features]<0,0)  
  
    daily_confirmed = daily_confirmed[['Province_State'] + daily_features].set_index('Province_State').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'confirmed'},
                                                                                                                axis=1).groupby(['Province_State','date'])['confirmed'].sum().reset_index()

    daily_deaths = daily_deaths[['Province_State'] + daily_features].set_index('Province_State').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'deaths'},
                                                                                                                axis=1).groupby(['Province_State','date'])['deaths'].sum().reset_index()

    daily_recovered = daily_recovered[['Province_State'] + daily_features].set_index('Province_State').stack().reset_index().rename({'level_1':'date',
                                                                                                                0:'recovered'},
                                                                                                                axis=1).groupby(['Province_State','date'])['recovered'].sum().reset_index()  

    daily_confirmed_US = daily_confirmed.groupby('date')['confirmed'].sum().reset_index().rename({0:'confirmed'})
    daily_confirmed_US['Province_State'] = 'US'
    daily_confirmed = pd.concat([daily_confirmed, daily_confirmed_US], axis=0, ignore_index=True)
    daily_confirmed['date'] = pd.to_datetime(daily_confirmed['date'])  
    daily_confirmed = daily_confirmed.sort_values(['Province_State','date'])

    daily_deaths_US = daily_deaths.groupby('date')['deaths'].sum().reset_index().rename({0:'deaths'})
    daily_deaths_US['Province_State'] = 'US'
    daily_deaths = pd.concat([daily_deaths, daily_deaths_US], axis=0, ignore_index=True)
    daily_deaths['date'] = pd.to_datetime(daily_deaths['date'])  
    daily_deaths = daily_deaths.sort_values(['Province_State','date'])

    daily_recovered_US = daily_recovered.groupby('date')['recovered'].sum().reset_index().rename({0:'recovered'})
    daily_recovered_US['Province_State'] = 'US'
    daily_recovered = pd.concat([daily_recovered, daily_recovered_US], axis=0, ignore_index=True)
    daily_recovered['date'] = pd.to_datetime(daily_recovered['date'])  
    daily_recovered = daily_recovered.sort_values(['Province_State','date'])
    
    # refactor -inf 
    daily_confirmed.loc[daily_confirmed['confirmed']<0,'confirmed']=0
    daily_deaths.loc[daily_deaths['deaths']<0,'deaths']=0
    daily_recovered.loc[daily_recovered['recovered']<0,'recovered']=0

    # feature engineering: weekly rolling mean for confirmed, deaths, recovered
    daily_confirmed['confirmed_rolling'] = np.log1p(daily_confirmed.groupby('Province_State')['confirmed'].apply(lambda x:x.rolling(horizon).sum()))
    daily_deaths['deaths_rolling'] = np.log1p(daily_deaths.groupby('Province_State')['deaths'].apply(lambda x:x.rolling(horizon).sum()))
    daily_recovered['recovered_rolling'] = np.log1p(daily_recovered.groupby('Province_State')['recovered'].apply(lambda x:x.rolling(horizon).sum()))

    # target, we need to forecast the cumsum of next horizon days.(including today)
    daily_confirmed['confirmed_target'] = np.log1p(daily_confirmed.groupby('Province_State')['confirmed'].apply(lambda x:x.rolling(horizon).sum().shift(1-horizon)))
    daily_deaths['deaths_target'] = np.log1p(daily_deaths.groupby('Province_State')['deaths'].apply(lambda x:x.rolling(horizon).sum().shift(1-horizon)))
    daily_recovered['recovered_target'] = np.log1p(daily_recovered.groupby('Province_State')['recovered'].apply(lambda x:x.rolling(horizon).sum().shift(1-horizon)))

    daily_confirmed['confirmed'] = np.log1p(daily_confirmed['confirmed'])
    daily_deaths['deaths'] = np.log1p(daily_deaths['deaths'])
    daily_recovered['recovered'] = np.log1p(daily_recovered['recovered'])

    daily_ts = pd.merge(daily_confirmed, daily_deaths, on=['Province_State','date'],how='left')
    daily_ts = pd.merge(daily_ts, daily_recovered, on=['Province_State','date'],how='left').rename({'Province_State':'Country/Region'},axis=1)    

    mobility_features = ['retail_and_recreation_percent_change_from_baseline',
                        'grocery_and_pharmacy_percent_change_from_baseline',
                        'parks_percent_change_from_baseline',
                        'transit_stations_percent_change_from_baseline',
                        'workplaces_percent_change_from_baseline',
                        'residential_percent_change_from_baseline']
    mobility = mobility[mobility.country_region=='United States'][mobility.sub_region_1.notnull()][mobility.sub_region_2.isnull()]
    mobility = mobility[['sub_region_1','date'] + mobility_features].rename({'sub_region_1':'Country/Region'},axis=1)

    mobility_US = mobility.groupby('date')[mobility_features].mean().reset_index()
    mobility_US['Country/Region'] = 'US'
    mobility = pd.concat([mobility, mobility_US], axis=0, ignore_index=True)
    print(mobility.tail())
    use_countries = set(mobility['Country/Region'].unique()) & set(daily_ts['Country/Region'].unique()) 
    # use_countries = daily_ts['Country/Region'].unique()
    print(use_countries)
    mobility[mobility_features] = mobility[mobility_features] / 100.0
    mobility = mobility[mobility['Country/Region'].isin(use_countries)]
    mobility['date'] = pd.to_datetime(mobility['date'])

    daily_ts = daily_ts[daily_ts['Country/Region'].isin(use_countries)]
    daily_ts['date'] = pd.to_datetime(daily_ts['date'])
    print(daily_ts['date'].unique())

    df = pd.merge(daily_ts, mobility, on=['Country/Region','date'], how='left')
    df = df.fillna(0.0)

    save_fp = '.'.join(data_fp.split('.')[:-1]) + '_' + str(horizon) + '.csv'
    df.to_csv(save_fp, index=False, header=True)

def process_government_data():
    gov_index = pd.read_csv('https://storage.googleapis.com/covid19-open-data/v2/index.csv')
    gov_index =gov_index.loc[(gov_index['aggregation_level']==1) & (gov_index['country_code']=='US'),
                            ['key','subregion1_name']].rename({'subregion1_name':'Country/Region'},axis=1).reset_index(drop=True)
    gov_df = pd.read_csv('https://storage.googleapis.com/covid19-open-data/v2/oxford-government-response.csv')
    gov = pd.merge(gov_index, gov_df, on='key', how='left').drop('key', axis=1).fillna(0.0)
    feature_cols = [item for item in gov.columns if item not in ['Country/Region','date']]
    for col in feature_cols:
        gov[col] = np.log1p(gov[col])
    # gov_df['date'] = gov_df['date'].map(lambda x: datetime.datetime.strptime(x,'%Y-%M-%d'))
    # gov_df['date'] = pd.to_datetime(gov_df['date'])
    # print(gov_df.dtypes)

    return gov

def load_data(data_fp, start_date, min_peak_size, lookback_days, lookahead_days, label='deaths_target', use_mobility=True, logger=print):
    logger('Load Data from ' + data_fp)
    logger('lookback_days={}, lookahead_days={}, '.format(
        lookback_days, lookahead_days))
    data = pd.read_csv(data_fp, parse_dates=['date'])
    data = data[data.date>=pd.to_datetime(start_date)].reset_index(drop=True)
    min_confirmed = data.groupby('Country/Region')['confirmed'].max()
    min_peak_size = max(0, min_peak_size)
    use_countries = min_confirmed[min_confirmed>=np.log1p(min_peak_size)].index.values
    data = data[data['Country/Region'].isin(use_countries)].reset_index(drop=True)

    gov = process_government_data()
    gov['date'] = pd.to_datetime(gov['date'])
    data = pd.merge(data, gov, on=['Country/Region', 'date'], how='left').fillna(0.0)

    data['weekday'] = data['date'].map(lambda x:x.weekday())
    dates = data['date'].unique()
    countries = [item for item in data['Country/Region'].unique() if item not in ['US']]
    print(countries)

    gov_features = ['school_closing', 'workplace_closing', 'cancel_public_events', 
                    'restrictions_on_gatherings', 'public_transport_closing', 'stay_at_home_requirements', 
                    'restrictions_on_internal_movement', 'international_travel_controls', 'income_support', 
                    'debt_relief', 'fiscal_measures', 'international_support', 'public_information_campaigns',
                    'testing_policy', 'contact_tracing', 'emergency_investment_in_healthcare', 
                    'investment_in_vaccines', 'stringency_index']
    use_features = [
       'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 
       'confirmed', 'deaths','recovered',
       'confirmed_rolling', 'deaths_rolling','recovered_rolling',
       'weekday']
    if not use_mobility:
        use_features = [
       'confirmed', 'deaths','recovered',
       'confirmed_rolling', 'deaths_rolling','recovered_rolling',
       'weekday']
    
    target_features = [
        'confirmed_target','deaths_target','recovered_target'
    ]

    df = pd.DataFrame(index=pd.MultiIndex.from_product([countries, dates],
                      names=['Country/Region','date'])).reset_index()
    
    df = pd.merge(df, data, on=['Country/Region','date'], how='left').fillna(0.0)

    df_gov = df[gov_features].values.reshape(len(countries), len(dates), len(gov_features)).copy()
    # df[gov_features].to_csv('../features/gov.csv')
    df = df[use_features + target_features].values.reshape(len(countries), len(dates), len(use_features)+len(target_features))    
    # dates = list(dates)
    # dates.append(datetime.timedelta(days=1) + pd.to_datetime(dates[-1]))
    print(df_gov.shape, df.shape)
    dates = list(map(lambda x: pd.to_datetime(x), dates))
    day_inputs = []
    day_gov_inputs = []
    outputs = []
    label_dates = []
    label2idx = {
        'confirmed_target':-3,
        'deaths_target':-2,
        'recovered_target':-1
    }
    label_idx = label2idx.get(label, -2)
    for day_idx in range(lookback_days, len(dates)):
        day_input = df[:, day_idx-lookback_days+1:day_idx+1, :-3].copy()
        day_gov_input = df_gov[:, day_idx-lookback_days+1:day_idx+1, :].copy()
        if day_idx + lookahead_days > len(dates):
            tmp = df[:, day_idx:day_idx + lookahead_days, label_idx].copy()
            sz = tmp.shape
            tmp_empty = np.zeros((sz[0], lookahead_days - sz[1]))
            output = np.concatenate([tmp, tmp_empty], axis=1)
        else:
            output = df[:, day_idx:day_idx + lookahead_days, label_idx].copy() 

        day_inputs.append(day_input)
        day_gov_inputs.append(day_gov_input)
        outputs.append(output)
        label_dates.append(dates[day_idx])     
    
    # [num_samples, num_nodes, lookback_days, day_feature_dim]
    day_inputs = np.stack(day_inputs, axis=0)
    # [num_samples, num_nodes, lookback_days, day_gov_feature_dim]
    day_gov_inputs = np.stack(day_gov_inputs, axis=0)
    # [num_samples, num_nodes, lookahead_days]
    outputs = np.stack(outputs, axis=0)

    day_inputs = torch.from_numpy(day_inputs).float()
    day_gov_inputs = torch.from_numpy(day_gov_inputs).float()
    outputs = torch.from_numpy(outputs).float()
    A = torch.ones(len(countries),len(countries)).to_sparse()
    edge_index = A._indices().long()

    logger("Input size: {}; {},Output size: {}, Edge size: {}".format(
        day_inputs.size(), day_gov_inputs.size(), outputs.size(), edge_index.size()))


    return day_inputs, day_gov_inputs, outputs, edge_index, label_dates, countries

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

def load_data_ahead(data_fp, start_date, min_peak_size, lookback_days, lookahead_days, label='deaths_target', 
        use_mobility=True,select_features=[], set_values=[], logger=print):
    logger('Load Data from ' + data_fp)
    logger('lookback_days={}, lookahead_days={}, '.format(
        lookback_days, lookahead_days))
    data = pd.read_csv(data_fp, parse_dates=['date'])
    data = data[data.date>=pd.to_datetime(start_date)].reset_index(drop=True)
    min_confirmed = data.groupby('Country/Region')['confirmed'].max()
    min_peak_size = max(0, min_peak_size)
    use_countries = min_confirmed[min_confirmed>=np.log1p(min_peak_size)].index.values
    data = data[data['Country/Region'].isin(use_countries)].reset_index(drop=True)

    gov = process_government_data()
    gov['date'] = pd.to_datetime(gov['date'])
    data = pd.merge(data, gov, on=['Country/Region', 'date'], how='left').fillna(0.0)

    data['weekday'] = data['date'].map(lambda x:x.weekday())
    dates = data['date'].unique()
    countries = [item for item in data['Country/Region'].unique() if item not in ['US']]
    print(countries)

    gov_features = ['school_closing', 'workplace_closing', 'cancel_public_events', 
                    'restrictions_on_gatherings', 'public_transport_closing', 'stay_at_home_requirements', 
                    'restrictions_on_internal_movement', 'international_travel_controls', 'income_support', 
                    'debt_relief', 'fiscal_measures', 'international_support', 'public_information_campaigns',
                    'testing_policy', 'contact_tracing', 'emergency_investment_in_healthcare', 
                    'investment_in_vaccines', 'stringency_index']
    #### no intervention
    # gov[gov_features] = 0.0
    print('###change####')
    print(select_features, set_values)
    for col,val in zip(select_features, set_values):
        print(col, val)
        data[col] = val
    print(data[select_features].head())

    use_features = [
       'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 
       'confirmed', 'deaths','recovered',
       'confirmed_rolling', 'deaths_rolling','recovered_rolling',
       'weekday']
    if not use_mobility:
        use_features = [
       'confirmed', 'deaths','recovered',
       'confirmed_rolling', 'deaths_rolling','recovered_rolling',
       'weekday']
    
    target_features = [
        'confirmed_target','deaths_target','recovered_target'
    ]

    df = pd.DataFrame(index=pd.MultiIndex.from_product([countries, dates],
                      names=['Country/Region','date'])).reset_index()
    
    df = pd.merge(df, data, on=['Country/Region','date'], how='left').fillna(0.0)

    df_gov = df[gov_features].values.reshape(len(countries), len(dates), len(gov_features)).copy()
    # df[gov_features].to_csv('../features/gov.csv')
    df = df[use_features + target_features].values.reshape(len(countries), len(dates), len(use_features)+len(target_features))    
    # dates = list(dates)
    # dates.append(datetime.timedelta(days=1) + pd.to_datetime(dates[-1]))
    print(df_gov.shape, df.shape)
    dates = list(map(lambda x: pd.to_datetime(x), dates))
    day_inputs = []
    day_gov_inputs = []
    outputs = []
    label_dates = []
    label2idx = {
        'confirmed_target':-3,
        'deaths_target':-2,
        'recovered_target':-1,
        'confirmed':0,
        'deaths':1,
    }
    label_idx = label2idx.get(label, -2)
    for day_idx in range(lookback_days, len(dates)):
        day_input = df[:, day_idx-lookback_days+1:day_idx+1, :-3].copy()
        # day_gov_input = df_gov[:, day_idx-lookback_days+1:day_idx+1, :].copy()
        day_gov_input = df_gov[:, day_idx: day_idx + lookahead_days, :].copy()
        if day_idx + lookahead_days > len(dates):
            tmp = df_gov[:, day_idx: day_idx + lookahead_days, :].copy()
            mean_val = tmp.mean(axis=1)
            sz = tmp.shape
            tmp_empty = np.zeros((sz[0], lookahead_days - sz[1], sz[2]))
            day_gov_input = np.concatenate([tmp, tmp_empty], axis=1)
            day_gov_input = np.expand_dims(mean_val, axis=1).repeat(lookahead_days, axis=1)

            tmp = df[:, day_idx:day_idx + lookahead_days, label_idx].copy()
            sz = tmp.shape
            tmp_empty = np.zeros((sz[0], lookahead_days - sz[1]))
            output = np.concatenate([tmp, tmp_empty], axis=1)
        else:
            output = df[:, day_idx:day_idx + lookahead_days, label_idx].copy()
            day_gov_input = df_gov[:, day_idx: day_idx + lookahead_days, :].copy()
            mean_val = day_gov_input.mean(axis=1)
            day_gov_input = np.expand_dims(mean_val, axis=1).repeat(lookahead_days, axis=1)


        day_inputs.append(day_input)
        day_gov_inputs.append(day_gov_input)
        outputs.append(output)
        label_dates.append(dates[day_idx])     
    
    # [num_samples, num_nodes, lookback_days, day_feature_dim]
    day_inputs = np.stack(day_inputs, axis=0)
    # [num_samples, num_nodes, lookback_days, day_gov_feature_dim]
    day_gov_inputs = np.stack(day_gov_inputs, axis=0)
    # [num_samples, num_nodes, lookahead_days]
    outputs = np.stack(outputs, axis=0)

    day_inputs = torch.from_numpy(day_inputs).float()
    day_gov_inputs = torch.from_numpy(day_gov_inputs).float()
    outputs = torch.from_numpy(outputs).float()
    A = torch.ones(len(countries),len(countries)).to_sparse()
    edge_index = A._indices().long()

    logger("Input size: {}; {},Output size: {}, Edge size: {}".format(
        day_inputs.size(), day_gov_inputs.size(), outputs.size(), edge_index.size()))


    return day_inputs, day_gov_inputs, outputs, edge_index, label_dates, countries

if __name__ == "__main__":
    # raw_data_preprocessing(data_fp='../data/daily_mobility_china.csv', horizon=7, only_china=True)
    #raw_data_preprocessing(data_fp='../data/daily_mobility_global.csv',horizon=7,only_china=False)
    # raw_data_preprocessing_US(horizon=7)
    # raw_data_preprocessing_US(horizon=14)
    # raw_data_preprocessing_US(horizon=21)
    # raw_data_preprocessing_US(horizon=28)

    process_government_data()