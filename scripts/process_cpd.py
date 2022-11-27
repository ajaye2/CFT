
import sys
sys.path.insert(1,'/home/ec2-user/CFT')

from multiprocessing import Pool, freeze_support
from matplotlib import pyplot as plt
from changepoint_detection import *
from utils import standardize
from datetime import datetime
from functools import partial
from itertools import repeat
from models.losses import *
from ts2vec import TS2Vec
from pathlib import Path
from tqdm import tqdm
from os import walk
import pandas as pd
import numpy as np
import os


BASE_PATH                           = "/home/ec2-user/CFT/datasets/"
stock_data_path_hour                = BASE_PATH + "STOCKS/hour/"
ffd_path_hour                       = BASE_PATH + "META_FEATURES/FFD/hour/"
cpd_path_hour                       = BASE_PATH + "META_FEATURES/CPD/hour/"

stock_data_path_day                 = BASE_PATH + "STOCKS/day/"
ffd_path_day                        = BASE_PATH + "META_FEATURES/FFD/day/"
cpd_path_day                        = BASE_PATH + "META_FEATURES/CPD/day/"

cpd_path_test                       = BASE_PATH + "META_FEATURES/CPD/test/"
cpd_path_test2                      = BASE_PATH + "META_FEATURES/CPD/test2/"

short_cpd_lookback_window_length    = 12
long_cpd_lookback_window_length     = 126

filenames_hour                      = next(walk(stock_data_path_hour), (None, None, []))[2] 
filenames_day                       = next(walk(stock_data_path_day), (None, None, []))[2]  

cols_to_perform_ffd = ['open', 'high', 'low', 'close', 'vwap']




def prep_data_for_cpd(file, folder_path):

    temp            = pd.read_csv(folder_path + file).set_index('timestamp')[cols_to_perform_ffd]
    temp.index      = pd.to_datetime(temp.index)
    temp            = temp[['close']].pct_change().reset_index().rename({'timestamp': 'date', 'close': 'daily_returns'}, axis=1).dropna()
    temp            = temp.set_index('date')
    
    return temp
    

# files = filenames_day[:]
# cpd_args_day_short = [( prep_data_for_cpd(x, stock_data_path_day),  short_cpd_lookback_window_length, cpd_path_day  + x[:-4] + "_short.csv" ) for x in files]
# cpd_args_day_long  = [( prep_data_for_cpd(x, stock_data_path_day),  long_cpd_lookback_window_length,  cpd_path_day  + x[:-4] + "_long.csv"  ) for x in files]

# with Pool() as pool:
#     results = pool.starmap(run_module, cpd_args_day_short)
#     results = pool.starmap(run_module, cpd_args_day_long)


def file_should_be_processed(path, file):
    file_path = Path(path + file)
    if file_path.is_file():
        df = pd.read_csv(file_path)
        if 'hour' in (path + file)  and df.shape[0] > 20000:
            return False
        if 'day' in (path + file)  and df.shape[0] > 2000:
            return False
    return True

files = filenames_hour[:]
cpd_args_hour_short = [( prep_data_for_cpd(x, stock_data_path_hour),  short_cpd_lookback_window_length, cpd_path_hour  + x[:-4] + "_short.csv"  ) for x in files if file_should_be_processed(cpd_path_hour, x[:-4] + "_short.csv") ]
cpd_args_hour_long  = [( prep_data_for_cpd(x, stock_data_path_hour),  short_cpd_lookback_window_length, cpd_path_hour  + x[:-4] + "_long.csv"   ) for x in files if file_should_be_processed(cpd_path_hour, x[:-4] + "_long.csv")]

with Pool() as pool:
    results = pool.starmap(run_module, cpd_args_hour_short)
    results = pool.starmap(run_module, cpd_args_hour_long)