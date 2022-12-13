from utils import standardize
from models.losses import *
from ts2vec import TS2Vec
from pathlib import Path
from tqdm import tqdm
from os import walk
import pandas as pd
import numpy as np


def save_checkpoint_callback(
    run_dir,
    save_every=1,
    unit='epoch', 
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


def get_data(base_path, timeframe, lookback_window=7, standardize_lookback_window=21, pct_train=0.85):
    mypath      = base_path + "CFT/datasets/STOCKS/" + timeframe + "/"
    ffd_path    = base_path + "CFT/datasets/META_FEATURES/FFD/" + timeframe + "/"
    cpd_path    = base_path + "CFT/datasets/META_FEATURES/CPD/" + timeframe + "/"

    filenames   = next(walk( mypath), (None, None, []))[2]  

    cols_to_perform_ffd = ['open', 'high', 'low', 'close', 'vwap']

    data_dict = {}

    for file in tqdm(filenames[:]):
        df                  = pd.read_csv(mypath + file).set_index('timestamp')[cols_to_perform_ffd]
        df_ffd              = pd.read_csv(ffd_path + file).set_index('timestamp')

        if not Path(cpd_path + file[:-4] + "_short.csv").is_file() or not Path(cpd_path + file[:-4] + "_long.csv").is_file():  continue 
        df_cpd_short        = pd.read_csv(cpd_path + file[:-4] + "_short.csv").set_index('date').drop(['t', 'cp_location'], axis=1).fillna(method='ffill')
        df_cpd_long         = pd.read_csv(cpd_path + file[:-4] + "_long.csv").set_index('date').drop(['t', 'cp_location'], axis=1).fillna(method='ffill')

        df_cpd_long.index     = df[-df_cpd_long.shape[0]:].index 
        df_cpd_short.index    = df[-df_cpd_short.shape[0]:].index 

        df_cpd_long.columns   = [x + "_long" for x in df_cpd_long.columns]
        df_cpd_short.columns  = [x + "_short" for x in df_cpd_short.columns]

        meta_features         = pd.concat([df_ffd, df_cpd_short, df_cpd_long], axis=1).sort_index().dropna()
        
        tasks                                 = pd.DataFrame(index=df.index)
        tasks['one_day_price_pred']           = ( df.close - df.close.shift(-1) ) / df.close.shift(-1)
        tasks['two_day_price_pred']           = ( df.close - df.close.shift(-2) ) / df.close.shift(-2)
        tasks['three_day_price_pred']         = ( df.close - df.close.shift(-3) ) / df.close.shift(-3)
        tasks['four_day_price_pred']          = ( df.close - df.close.shift(-4) ) / df.close.shift(-4)
        tasks['five_day_price_pred']          = ( df.close - df.close.shift(-5) ) / df.close.shift(-5)

        ### Check if volatilty is being created correctly
        tasks['one_week_vol_pred']            = ( df.close.rolling(5).std() - df.close.rolling(5).std().shift(-1) )
        tasks['one_month_vol_pred']           = ( df.close.rolling(21).std() - df.close.rolling(21).std().shift(-1) )

        tasks['one_week_skew_pred']            = ( df.close.rolling(5).skew() - df.close.rolling(5).skew().shift(-1) )
        tasks['one_month_skew_pred']           = ( df.close.rolling(21).skew() - df.close.rolling(21).skew().shift(-1) )

        # tasks.dropna(inplace=True)

        df_raw_price    = df.copy()

        # Experiment between this and using raw values 
        df                  = standardize(df,     look_back=standardize_lookback_window)
        meta_features       = standardize(meta_features, look_back=standardize_lookback_window)

        idx                 = meta_features.index.intersection(df.index).intersection(tasks.index)
        df, meta_features,tasks    = df.loc[idx], meta_features.loc[idx], tasks.loc[idx]
        df_raw_price= df_raw_price.loc[idx]

        X_data_array, X_data_dict                = [], {}       
        EXP_FEAT_data_array                      = []


        for i in range(lookback_window, len(df)+1):
            X_data_array.append( df.iloc[i - lookback_window:i].values )
            EXP_FEAT_data_array.append( meta_features.iloc[i-1].values ) # Without minus one, the exp features lead X_DATA by one tiemstamp
            X_data_dict[df.index[i-1]]  = df.iloc[i - lookback_window:i].values
            # EXP_FEAT_data_array[meta_features.index[i-1]] = meta_features.iloc[i-1].values
        

            assert meta_features.iloc[i-1].name == df.iloc[i - lookback_window:i].iloc[-1].name 
        # X_data_array        = np.array( X_data_array )
        # EXP_FEAT_data_array = np.array( EXP_FEAT_data_array )

        data_dict[file[:-4]]                      = {}
        data_dict[file[:-4]]['X_DATA']            = X_data_array
        data_dict[file[:-4]]['EXP_FEAT_DATA']     = EXP_FEAT_data_array
        data_dict[file[:-4]]['Y_DATA']            = tasks 
        data_dict[file[:-4]]['X_DATA_DICT']       = X_data_dict 
        data_dict[file[:-4]]['RAW_TIME_SERIES']   = df_raw_price
        data_dict[file[:-4]]['RAW_TIME_SERIES_STD']   = df 
        
        data_dict[file[:-4]]['RAW_EXP_FEAT']      = meta_features 
        # print(file, X_data_array.shape, EXP_FEAT_data_array.shape)


    train_data         = np.concatenate( [data_dict[x]['X_DATA'][:int(pct_train*len(data_dict[x]['X_DATA']))] for x in data_dict.keys()])
    exp_train_data     = np.concatenate( [data_dict[x]['EXP_FEAT_DATA'][:int(pct_train*len(data_dict[x]['X_DATA']))] for x in data_dict.keys()])
    # train_labels       = np.concatenate( [data_dict[x]['Y_DATA'][:int(pct_train*len(data_dict[x]['X_DATA']))] for x in data_dict.keys()])
    assert train_data.shape[0] == exp_train_data.shape[0] 
    print(train_data.shape, exp_train_data.shape)


    test_data         = np.concatenate( [data_dict[x]['X_DATA'][int(pct_train*len(data_dict[x]['X_DATA'])):] for x in data_dict.keys()])
    exp_test_data     = np.concatenate( [data_dict[x]['EXP_FEAT_DATA'][int(pct_train*len(data_dict[x]['X_DATA'])):] for x in data_dict.keys()])
    # test_labels       = np.concatenate( [data_dict[x]['Y_DATA'][:int(pct_train*len(data_dict[x]['X_DATA']))] for x in data_dict.keys()])
    assert test_data.shape[0] == exp_test_data.shape[0] 
    print(test_data.shape, exp_test_data.shape)
    # # (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)


    return data_dict, train_data, exp_train_data, test_data, exp_test_data
        

def train_model(train_data, exp_train_data, model_name, **config):
    use_expclr_loss = False
    if exp_train_data is not None: use_expclr_loss = True

    model = TS2Vec(
        input_dims=train_data.shape[-1],
        **config
    )

    loss_log = model.fit(
        train_data,
        expert_features=exp_train_data, # train_data.reshape(100, -1)[:,40:],
        verbose=True,
        use_expclr_loss=use_expclr_loss,
        n_epochs=config['n_epochs']
    )

    model.save("saved_models/model_name/" + model_name + ".pkl")
    

