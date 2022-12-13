from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import linear_model
import pandas as pd
import numpy as np



from statsmodels.regression.rolling import RollingOLS

def get_classification_acc(x_data, y_data, test_size=0.2):
    idx            = x_data.dropna().index.intersection(y_data.dropna().index)
    x_data, y_data =  x_data.loc[idx], y_data.loc[idx]

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42, shuffle=False)

    pca = PCA(n_components=int(max( 0.4* X_train.shape[1], 5)))
    X_train = pd.DataFrame( pca.fit_transform(X_train.values), index=X_train.index )
    X_test  = pd.DataFrame( pca.transform(X_test.values), index=X_test.index )

    reg = linear_model.LinearRegression()
    reg = RidgeClassifier()

    reg.fit( X_train, np.sign( y_train ) )

    train_acc = reg.score(X_train, np.sign( y_train ) ) 
    test_acc  = reg.score(X_test,  np.sign( y_test ) ) 

    return train_acc, test_acc


def get_rolling_ols_results(x_data, y_data, test_size=0.2):
    
    idx            = x_data.dropna().index.intersection(y_data.dropna().index)
    x_data, y_data = x_data.loc[idx].sort_index().astype(np.float64), y_data.loc[idx].sort_index()
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42, shuffle=False)

    pca = PCA(n_components=int(max( 0.4* X_train.shape[1], 5)))
    X_train = pd.DataFrame( pca.fit_transform(X_train.values), index=X_train.index )
    X_test  = pd.DataFrame( pca.transform(X_test.values), index=X_test.index )

    mod             = RollingOLS(y_train.values, X_train, window=41)
    rolling_results = mod.fit()
    params_fs       = rolling_results.params.dropna()


    y_pred          = np.sign( (params_fs * X_train.loc[params_fs.index]).sum(axis=1) )
    y_true          = np.sign( y_train.loc[params_fs.index] )
    acc             = accuracy_score(y_true, y_pred)

    y_pred          = np.sign( (X_test*params_fs.iloc[-1].values).sum(axis=1) )
    y_true          = np.sign(y_test )
    acc_test        = accuracy_score(y_true, y_pred)
    
    acc_series      = y_true == y_pred

    return rolling_results, acc, acc_series, acc_test


def get_price_plot_with_clusters(repr_data, raw_time_series, asset, type_of_repr, n_components=2):
    assert n_components <= 4
    gm_x_data    = repr_data
    gm_x_data    = StandardScaler().fit_transform(gm_x_data)
    gm_clusters  = KMeans(n_clusters=n_components, random_state=0).fit_predict(gm_x_data)

    gm_clusters_repr_fs         = pd.Series( gm_clusters , index=repr_data.index)#.iloc[-500:]
    gm_clusters_repr_fs         = (gm_clusters_repr_fs.ewm(10).mean() > (1/n_components)).astype(np.int32)
    idx                         = raw_time_series['close'].index.intersection(gm_clusters_repr_fs.index)
    gm_clusters_repr_fs         = gm_clusters_repr_fs.loc[idx]
    price_data                  = raw_time_series[['close']].loc[idx]
    price_data['cluster']       = gm_clusters_repr_fs
    price_data.index            = pd.to_datetime(price_data.index)
    ax1                         = price_data.close.plot(figsize=(15, 4))

    markers = {
        0: '|r',
        1: '|g',
        2: '|b',
        3: '|y',
    }
    patch_color = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'yellow',
    }

    patches = []
    for cluster in set(gm_clusters_repr_fs):
        plt.plot(price_data[price_data['cluster'] == cluster].index, price_data[price_data['cluster'] == cluster].close, markers[cluster]);

        patches.append(mpatches.Patch(color=patch_color[cluster], label='Cluster ' + str(cluster) ))

    plt.xlabel('Date')
    plt.ylabel('Daily Closing Price') 
    plt.title('Daily Close price for ' + asset + ". Clusters made using " + type_of_repr)

    plt.legend(handles=patches)
    plt.show()
