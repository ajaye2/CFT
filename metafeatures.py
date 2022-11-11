from KEYS import MLFINLABKEY

import os
os.environ['MLFINLAB_API_KEY'] = MLFINLABKEY
import mlfinlab as ml
from mlfinlab.features.fracdiff import (frac_diff_ffd, plot_min_ffd)

import numpy as np
import pandas as pd





def get_frac_diff_series(df, alpha_column, coef_d=0.25, plot=False):

    # TODO: Add functionality for deriving optimal differentiation
    data_for_frac_diff = df[[alpha_column]].copy().replace([np.inf, -np.inf], np.nan).dropna()
    data_for_frac_diff = data_for_frac_diff.rename({alpha_column: 'close'}, axis=1)
    frac_diff_series = frac_diff_ffd(data_for_frac_diff, coef_d)
    ret_alpha = frac_diff_series.rename({'close': 'val'}, axis=1)
    if plot:
        plot_min_ffd(data_for_frac_diff)

    return ret_alpha

def standardize(df, column, look_back):
    #look into when 0 in values
    x       = df[column].dropna()
    x_bar   = df[column].rolling(look_back).mean()
    z_std   = df[column].rolling(look_back).std()
    z_score = (x - x_bar) / z_std
    z_score = z_score.replace([np.inf, -np.inf], np.nan)
    return 'z_score_' + column, z_score