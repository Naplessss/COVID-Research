import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn import metrics
import datetime as dt
from functools import partial
import lightgbm as lgb
import warnings
warnings.filterwarnings(action='ignore')

def process_train():
    df_deaths = pd.read_csv(os.path.join(DATA_PATH, 'time_series_covid19_deaths_{}.csv'.format(LEVEL)))
    df_confirmed = pd.read_csv(os.path.join(DATA_PATH, 'time_series_covid19_confirmed_{}.csv'.format(LEVEL)))    
    
    df_deaths['Location'] = df_deaths['Province_State']
    df_confirmed['Location'] = df_confirmed['Province_State']
    
    pop = df_deaths[['Location','Population']].groupby('Location')['Population'].sum()
    geo = df_deaths[['Location','Lat','Long_']].groupby('Location')[['Lat','Long_']].median()
    ts_cols = [item for item in df_deaths.columns if item not in ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
        'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Country/Region', 'Province/State', 'Location','Population']]
    df_deaths[ts_cols] = df_deaths[ts_cols].diff(axis=1)
    df_deaths = df_deaths[['Location'] + ts_cols].set_index('Location').stack().reset_index().rename({0:'Deaths',
                                                                                        'level_{}'.format(1):'Date'},
                                                                                      axis=1).groupby(['Location','Date'])['Deaths'].sum().reset_index()    
    df_confirmed[ts_cols] = df_confirmed[ts_cols].diff(axis=1)
    df_confirmed = df_confirmed[['Location'] + ts_cols].set_index('Location').stack().reset_index().rename({0:'Confirmed',
                                                                                        'level_{}'.format(1):'Date'},
                                                                                      axis=1).groupby(['Location','Date'])['Confirmed'].sum().reset_index() 
    df = pd.merge(df_confirmed, df_deaths, on = ['Location','Date'], how='left')
    df['Date'] = pd.to_datetime(df['Date'])  
    
    df['DayOfWeek'] = df.Date.dt.weekday
    df = df.merge(pop,on='Location')
    df = df.merge(geo,on='Location')
    
    return df


def extract_timeseries_features(single_location):
    df = single_location.copy()
    df = df.sort_values(by='Date')
    for target in ['Confirmed', 'Deaths']:
        df[f'{target}CumSum'] = df[target].cumsum()
        for k in [3, 7, 14, 21]:
            df[f'{target}RollingMean{k}'] = df[target].rolling(k).mean()
            df[f'{target}RollingMean{k}PerK'] = df[target].rolling(k).mean() / df.Population * 10000
        df[f'{target}RollingStd21'] = df[target].rolling(21).std().round(0)
        df[f'{target}DaysSince100'] = (df[f'{target}CumSum'] > 100).cumsum()
        df[f'{target}DaysSince1000'] = (df[f'{target}CumSum'] > 1000).cumsum()
        df[f'{target}DaysSince5000'] = (df[f'{target}CumSum'] > 5000).cumsum()


        df[f'{target}RollingMeanDiff2w'] = df[f'{target}RollingMean7'] / (df[f'{target}RollingMean14'] + 1) - 1
        df[f'{target}RollingMeanDiff3w'] = df[f'{target}RollingMean7'] / (df[f'{target}RollingMean21'] + 1) - 1

    df['DeathRate'] = 100 * df.DeathsCumSum.clip(0, None) / (df.ConfirmedCumSum.clip(0, None) + 1)
    df['DeathRateRolling3w'] = 100 * df.DeathsRollingMean7.clip(0, None) / (
            df.ConfirmedRollingMean21.clip(0, None) + 1)
    df['ConfirmedPerK'] = 1000 * df.ConfirmedCumSum.clip(0, None) / df.Population
    df['DeathsPerK'] = 1000 * df.DeathsCumSum.clip(0, None) / df.Population
    return df    

def get_nearby_features(features, rank):
    closest = pd.read_csv(CLOSEST_PATH)

    to_aggregate = ['ConfirmedCumSum',
                    'ConfirmedRollingMean21',
                    'ConfirmedRollingMean14',
                    'ConfirmedRollingMean7',

                    'DeathsCumSum',
                    'DeathsRollingMean21',
                    'DeathsRollingMean14',
                    'DeathsRollingMean7']

    subset = features[['Date', 'Location','Population'] + to_aggregate].copy()
    subset = subset.rename(columns={'Location': 'Location_1'})

    nearby = features[['Date', 'Location']].merge(closest[closest['rank'] <= rank], on='Location')
    nearby = nearby.merge(subset, on=['Date', 'Location_1'])

    nearby_sum = nearby.groupby(['Date', 'Location']).sum()
    nearby_mean = nearby.groupby(['Date', 'Location'])[['distance']].mean().round(0)
    for c in to_aggregate:
        nearby_sum[f'Nearby{rank}{c}'] = 1000 * nearby_sum[c] / nearby_sum['Population']

    nearby_features = pd.merge(nearby_sum, nearby_mean, on=['Date', 'Location'])
    nearby_features = nearby_features.rename(columns={'distance_y': f'Nearby{rank}Distance'})
    return nearby_features[[f for f in nearby_features.columns if f.startswith('Nearby')]]

def w5_loss(preds, data, q=0.5):
    y_true = data.get_label()
    weights = data.get_weight()
    diff = (y_true - preds) * weights
    gt_is_higher = np.sum(diff[diff >= 0] * q)
    gt_is_lower = np.sum(- diff[diff < 0] * (1 - q))
    return 'w5', (gt_is_higher + gt_is_lower) / len(preds), False

def wmae_loss(preds, data):
    y_true = data.get_label()
    weights = data.get_weight()
    wmae = (np.abs(y_true - preds) * weights).mean()
    return 'wmae', wmae, False

def mae_loss(preds, data):
    y_true = data.get_label()
    
    _mae = (np.abs(y_true - preds)).mean()
    return 'mae', _mae, False

def mae(y_preds, y_true):
    try:
        return (np.abs(y_true - y_preds)).mean()
    except:
        return np.nan

def apply_lgb(features, target, q, params, k, num_round=1000):
    #lb = LabelEncoder()
    features['Date'] = pd.to_datetime(features['Date'])
    #features['Location2id'] = LabelEncoder().fit_transform(features['Location'])
    features['TARGET'] = features.groupby('Location')[target].apply(lambda x:x.rolling(k).sum().shift(1-k))  # included today
    features['DaysTillEnd'] = (pd.to_datetime(FORECAST_START_DATE) - pd.to_datetime(features['Date'])).map(lambda x:x.days) + 1
    TRAIN_END_DATE = pd.to_datetime(FORECAST_START_DATE) - dt.timedelta(days=(k-1))
    features['Weight'] = 1.0 / features['DaysTillEnd'] ** 0.5
    do_not_use = [
                     'Location', 'Date', 'TARGET', 'Weight',  'DaysTillEnd', 'DateTime', 
                 ] + ['Confirmed', 'Deaths']
    feature_names = [f for f in features.columns if f not in do_not_use]

    print(features[['TARGET','Date']].tail(20))
    train = features.loc[(~features.TARGET.isnull()) & 
                         (features.Date > TRAIN_START) &  
                         (features.Date <= TRAIN_END_DATE)].reset_index(drop=True)
    
    ### only forecast one day(start date of epidemic week)
    test = features.loc[features.Date == FORECAST_START_DATE].reset_index(drop=True)

    test['PREDICTION'] = 0
    train['PREDICTION'] = 0
    feature_importances = []
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr_ind,te_ind in gkf.split(train, train['TARGET'], train['Location']):
        tr = train.loc[tr_ind]
        val = train.loc[te_ind]

        train_set = lgb.Dataset(tr[feature_names], label=tr.TARGET, weight=tr.Weight)
        valid_set = lgb.Dataset(val[feature_names], label=val.TARGET, weight=val.Weight)
        test_set = lgb.Dataset(test[feature_names], label=test.TARGET, weight=test.Weight)

        model = lgb.train(params, train_set, 
                          num_round, 
                          valid_sets=[train_set, valid_set],
                          # early_stopping_rounds=50, 
                          feval=partial(mae_loss), 
                          verbose_eval=100)

        train.loc[te_ind, 'PREDICTION'] = model.predict(val[feature_names])
        test['PREDICTION'] += model.predict(test[feature_names]) / N_FOLDS

        fimp = pd.DataFrame({'f': feature_names, 'imp': model.feature_importance()})
        feature_importances.append(fimp)

    cv_error = mae(train['PREDICTION'], train['TARGET'])
    val_error = mae(test['PREDICTION'], test['TARGET'])
    
    feature_importances = pd.concat(feature_importances)
    feature_importances = feature_importances.groupby('f').sum().reset_index().sort_values(by='imp', ascending=False)
    feature_importances['target'] = target
    feature_importances['k'] = k
    feature_importances['q'] = q

    train_preds = train[['Date', 'Location', 'PREDICTION', 'TARGET']]
    test_preds = test[['Date', 'Location', 'PREDICTION', 'TARGET']]
    train_preds['TYPE'] = target
    test_preds['TYPE'] = target
    train_preds['k'] = k
    test_preds['k'] = k
    train_preds['quantile'] = q
    test_preds['quantile'] = q
    return cv_error, val_error, train_preds, test_preds, feature_importances

    def get_locations():
        location = pd.read_csv('/home/zhgao/COVID-Research/covid19-forecast-hub/data-locations/locations.csv')
        location2id = location[['location_name','location']].set_index('location_name')['location'].to_dict()
        return location2id

if __name__=='__main__':
    # newest date of labeling data
    LABEL_END_DATE = '2020-10-18'    
    # test start day to infer next epidemic weeks (included this day)
    # prefer to be sunday of this epdimic week (the same as LABEL_DATE_END)
    FORECAST_START_DATE = '2020-10-11'   

    DAYS = 7
    LEVEL = 'US'
    PRECISION = 2
    N_FOLDS = 5
    RELOAD_FEATURES = True
    N_SEEDS = 20
    TRAIN_START = '2020-03-31'
    CLOSEST_PATH = '/home/zhgao/COVID-Research/data/us_geo_closest.csv'
    FEATURE_FILE_PATH = f'/home/zhgao/COVID-Research/data/features_{LABEL_END_DATE}.csv'
    DATA_PATH = '/home/zhgao/COVID19/COVID-19/csse_covid_19_data/csse_covid_19_time_series'

    targets = process_train() 
    if os.path.exists(FEATURE_FILE_PATH) and RELOAD_FEATURES:
        features = pd.read_csv(FEATURE_FILE_PATH)
    else:
        features = []
        for loc, df in tqdm(targets.groupby('Location')):
            df = extract_timeseries_features(df)
            features.append(df)
        features = pd.concat(features)
        for rank in [5, 10, 20]:
            nearby_features = get_nearby_features(features, rank)
            features = features.merge(nearby_features, on=['Date', 'Location'])

        to_log = ['ConfirmedCumSum', 'Population', 'DeathsCumSum']
        for c in to_log:
            features.loc[:, c] = np.log(features[c].values + 1).round(2)

        features.loc[:, 'DeathRate'] = features.loc[:, 'DeathRate'].clip(0, 50)
        features.loc[:, 'DeathRateRolling3w'] = features.loc[:, 'DeathRateRolling3w'].clip(0, 50)

        round_1_digit = [
            'ConfirmedRollingMean21', 'ConfirmedRollingMean14', 'ConfirmedRollingMean7', 'ConfirmedRollingMean3',
        ]
        for c in round_1_digit:
            features.loc[:, c] = features[c].round(1)

        features = features.round(PRECISION)
        features.to_csv(FEATURE_FILE_PATH, index=False)

    features = features[features.Date <= LABEL_END_DATE]
    print(f'Features: {features.shape}')
    print(f'Features: {features.count()}')

    train_results, test_results = [], []
    for target in ['Deaths', 'Confirmed']:
    # for target in ['Confirmed']:
        for k in [7, 14, 21, 28]:
        # for k in [7]:
            for q in [0.01,0.025,0.05,0.1,0.15,0.2,0.25,0.3,
                         0.35,0.4,0.45,0.5,0.55,0.6,0.65,
                         0.7,0.75,0.8,0.85,0.9,0.95,0.975,0.99]:
            # for q in [0.5]:
                if q == 0.5:
                    N_SEEDS_USE = N_SEEDS
                else:
                    N_SEEDS_USE = 1
                train_preds = pd.DataFrame()
                test_preds = pd.DataFrame()
                cv_error, val_error = 0, 0
                for seed in range(N_SEEDS_USE):
                    params = dict(
                        objective='quantile',
                        alpha=q,
                        metric='mae',
                        max_depth=np.random.choice([14,15,16]),
                        learning_rate=np.random.choice([0.06,0.07,0.08,0.09]),
                        feature_fraction=np.random.choice([0.5, 0.6, 0.7, 0.8]),
                        bagging_freq=np.random.choice([2, 3, 5]),
                        bagging_fraction=np.random.choice([0.7, 0.8]),
                        min_data_in_leaf=np.random.choice([5, 15]),
                        num_leaves=np.random.choice([127, 255]),
                        verbosity=0,
                        random_seed=seed,
                        n_jobs=12
                    )
                    if target == 'Deaths':
                        num_round = np.random.choice([35,40,45,50])
                    if target == 'Confirmed':
                        num_round = np.random.choice([50,60,70,80])
                    _cv_error, _val_error, _train_preds, _test_preds, feature_importances = apply_lgb(
                        features, target, q, params, k, num_round=num_round)
                    print(target,k,q,seed,_cv_error,_val_error)
                    if seed == 0:
                        train_preds = _train_preds.copy()
                        test_preds = _test_preds.copy()
                        train_preds['PREDICTION'] = _train_preds['PREDICTION'] / N_SEEDS_USE
                        test_preds['PREDICTION'] = _test_preds['PREDICTION'] / N_SEEDS_USE
                        cv_error = _cv_error / N_SEEDS_USE
                        val_error = _val_error / N_SEEDS_USE
                    else:
                        train_preds['PREDICTION'] += _train_preds['PREDICTION'] / N_SEEDS_USE
                        test_preds['PREDICTION'] += _test_preds['PREDICTION'] / N_SEEDS_USE
                        cv_error += _cv_error / N_SEEDS_USE
                        val_error += _val_error / N_SEEDS_USE
                        
                    
                train_results.append(train_preds)
                test_results.append(test_preds)

    location2id = get_locations()
    gbm_predict = pd.concat(test_results)
    gbm_predict = gbm_predict[gbm_predict['quantile']==0.5]
    gbm_predict.to_csv('../output/gbm.predict.{}.csv'.format(FORECAST_START_DATE))