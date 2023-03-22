"""
This file is part of the accompanying code to our papers
Jiang, S., Zheng, Y., Wang, C., & Babovic, V. (2022) Uncovering flooding mecha-
nisms across the contiguous United States through interpretive deep learning on
representative catchments. Water Resources Research, 57, e2021WR030185.
https://doi.org/10.1029/2021WR030185.

Jiang, S., Bevacqua, E., & Zscheischler, J. (2022) River flooding mechanisms 
and their changes in Europe revealed by explainable machine learning, Hydrology 
and Earth System Sciences, 26, 6339–6359. https://doi.org/10.5194/hess-26-6339-2022.

Copyright (c) 2023 Shijie Jiang. All rights reserved.

You should have received a copy of the MIT license along with the code. If not,
see <https://opensource.org/licenses/MIT>
"""

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold

def get_station_data(fname):
    """
    Obtain the pandas dataframe from MOPEX dataset.

    Parameters
    ----------
    fname: the MOPEX filename that ends with extension "dly".

    Returns
    ----------
    dataset: the pandas dataframe for the MOPEX file.
    """
    dataset = pd.read_fwf(
        fname,
        header=None,
        widths=[4, 2, 2, 10, 10, 10, 10, 10],
        names=["year", "month", "day", "prcp", "pet", "flow", "tmax", "tmin"],
    )
    dataset["date"] = pd.to_datetime(dataset["year"].astype(str) + "-" + dataset["month"].astype(str) + "-" + dataset["day"].astype(str))
    dataset["tmean"] = (dataset["tmin"] + dataset["tmax"]) / 2
    dataset = dataset.drop(["year", "month", "day", "tmax", "tmin", "pet"], axis=1)
    dataset = dataset.replace(-99, np.NaN)
    dataset = dataset.set_index("date")
    dataset = dataset.dropna()

    return dataset


def get_wrapped_data(dataset, wrap_length=365):
    """
    Wrap the data for the shape requirement of LSTM.

    Parameters
    ----------
    dataset: the pandas dataframe obtained from the function get_station_data().
    wrap_length: the number of time steps to be considered for the LSTM layer.

    Returns
    ----------
    data_x_dict: the input dictionary whose key is the date and value is the corresponding wrapped input matrix of each sample.
    data_y_dict: the output dictionary whose key is the date and value is the corresponding target of each sample.
    """
    data_x_dict, data_y_dict = {}, {}

    for date_i in tqdm(dataset.index, desc=f'Preparing data with wrap length = {wrap_length}'):
        try:
            data_x = dataset.loc[pd.date_range(end=date_i,
                                               periods=wrap_length + 1,
                                               freq="d")[:-1], ["prcp", "tmean"], ].to_numpy(dtype='float16')
            data_y = dataset.loc[pd.date_range(end=date_i,
                                               periods=wrap_length + 1,
                                               freq="d")[-1:], "flow", ].to_numpy(dtype='float16')

            data_x_dict[date_i] = data_x
            data_y_dict[date_i] = data_y
        except KeyError:
            continue

    return data_x_dict, data_y_dict


def split_train_test(dataset, data_x_dict, data_y_dict, frac=0.7, random_state=100, scale=True):
    """
    Randomly split the dataset for training and testing.

    Parameters
    ----------
    dataset: the pandas dataframe obtained from the function get_station_data().
    data_x_dict: the input dictionary obtained from the function get_wrapped_data().
    data_y_dict: the output dictionary obtained from the function get_wrapped_data().
    frac: the fraction of samples to be trained (default: 0.7).
    random_state: the random seed (default: 100).
    scale: [bool] whether scale the split dataset by the mean and std values of the training data (default: True).

    Returns
    ----------
    train_dates: the dates of the picked training data.
    test_dates: the dates of the picked testing data.
    train_x: the (scaled) inputs for training.
    train_y: the (scaled) outputs for training.
    test_x: the (scaled) inputs for testing.
    test_y: the (scaled) outputs for testing.
    scale_params: the mean and std values of the training data (available when scale is True)
    """
    train_dates = (dataset.loc[data_x_dict.keys()].sample(frac=frac, random_state=random_state).index)
    test_dates  = dataset.loc[data_x_dict.keys()].drop(train_dates).index

    train_x = np.stack([data_x_dict.get(i) for i in train_dates.to_list()])
    train_y = np.stack([data_y_dict.get(i) for i in train_dates.to_list()])
    test_x  = np.stack([data_x_dict.get(i) for i in test_dates.to_list()])
    test_y  = np.stack([data_y_dict.get(i) for i in test_dates.to_list()])

    scale_params = {"train_x_mean": 0, "train_x_std": 1, "train_y_mean": 0, "train_y_std": 1}

    if scale is False:
        return train_dates, test_dates, train_x, train_y, test_x, test_y, scale_params
    else:
        scale_params["train_x_mean"] = (dataset.loc[train_dates, ["prcp", "tmean"]].mean().values)
        scale_params["train_x_std"]  = (dataset.loc[train_dates, ["prcp", "tmean"]].std().values)
        scale_params["train_y_mean"] = dataset.loc[train_dates, ["flow"]].mean().values
        scale_params["train_y_std"]  = dataset.loc[train_dates, ["flow"]].std().values

        train_x = (train_x - scale_params["train_x_mean"][None, None, :]) / scale_params["train_x_std"][None, None, :]
        train_y = (train_y - scale_params["train_y_mean"][None, :]) / scale_params["train_y_std"][None, :]
        test_x  = (test_x - scale_params["train_x_mean"][None, None, :]) / scale_params["train_x_std"][None, None, :]
        test_y  = (test_y - scale_params["train_y_mean"][None, :]) / scale_params["train_y_std"][None, :]

        return train_dates, test_dates, train_x, train_y, test_x, test_y, scale_params

def daylength(dayOfYear, lat):
    """
    Computes the length of the day (the time between sunrise and sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
        return 2.0*hourAngle/15.0    

def get_station_data_eu(fname, lat):
    dataset = pd.read_csv(fname, index_col=0, parse_dates=True)
    dataset['dl'] = pd.Series({x: daylength(x.dayofyear, lat) for x in dataset.index})
    
    dataset = dataset[['tg', 'rr', 'dl', 'fl']]
    dataset = dataset.interpolate(method='linear', axis=0, limit=1, inplace=False)

    return dataset

def get_wrapped_data_eu(dataset, wrap_length=365):
    data_x_dict, data_y_dict = {}, {}
    
    dataset_np = dataset.to_numpy().astype(np.float32)
    dateset_tm = dataset.index
    
    for date_i in range(wrap_length, dataset_np.shape[0]):
        data_x = dataset_np[date_i-wrap_length+1:date_i+1, 0:3]
        data_y = dataset_np[date_i, 3:4]
        date_value = dateset_tm[date_i]
        
        if (~np.isnan(data_y)) and (~np.isnan(data_x).any()):
            data_x_dict[date_value] = data_x
            data_y_dict[date_value] = data_y

    return data_x_dict, data_y_dict

def split_train_test_eu(dataset, data_x_dict, data_y_dict, k=10, scale=True):
    train_dates_list  = []
    test_dates_list   = []
    train_x_list      = []
    train_y_list      = []
    test_x_list       = []
    test_y_list       = []
    scale_params_list = []

    for train_index, test_index in KFold(n_splits=k, shuffle=False).split(dataset.loc[data_x_dict.keys()].index):


        train_dates = dataset.loc[data_x_dict.keys()].iloc[train_index].index
        test_dates  = dataset.loc[data_x_dict.keys()].iloc[test_index].index

        train_x = np.stack([data_x_dict.get(i) for i in train_dates.to_list()])
        train_y = np.stack([data_y_dict.get(i) for i in train_dates.to_list()])
        test_x  = np.stack([data_x_dict.get(i) for i in test_dates.to_list()])
        test_y  = np.stack([data_y_dict.get(i) for i in test_dates.to_list()])

        scale_params = {"train_x_a": 0, "train_x_b": 1, "train_y_a": 0, "train_y_b": 1}

        if scale is False:
            return train_dates, test_dates, train_x, train_y, test_x, test_y, scale_params

        else:
            scale_params["train_x_a"] = (dataset.loc[train_dates, ['tg', 'rr', 'dl']].mean().values)
            scale_params["train_x_b"] = (dataset.loc[train_dates, ['tg', 'rr', 'dl']].std().values)

            scale_params["train_x_a"][None, None, 0] = 0 # we only scale the tg with its std.
            scale_params["train_x_a"][None, None, 1] = 0 # we only scale the rr with its std.

            scale_params["train_y_a"] = dataset.loc[train_dates, ["fl"]].mean().values * 0 # we only scale the fl with its std.
            scale_params["train_y_b"] = dataset.loc[train_dates, ["fl"]].std().values

            train_x = (train_x - scale_params["train_x_a"][None, None, :]) / scale_params["train_x_b"][None, None, :]
            train_y = (train_y - scale_params["train_y_a"][None, :]) / scale_params["train_y_b"][None, :]
            test_x  = (test_x - scale_params["train_x_a"][None, None, :]) / scale_params["train_x_b"][None, None, :]
            test_y  = (test_y - scale_params["train_y_a"][None, :]) / scale_params["train_y_b"][None, :]
        
        train_dates_list.append(train_dates)
        test_dates_list.append(test_dates)
        train_x_list.append(train_x)
        train_y_list.append(train_y)
        test_x_list.append(test_x)
        test_y_list.append(test_y)
        scale_params_list.append(scale_params)
        
    return train_dates_list, test_dates_list, train_x_list, train_y_list, test_x_list, test_y_list, scale_params_list