"""
This file is part of the accompanying code to our paper
Jiang, S., Zheng, Y., Wang, C., & Babovic, V. (2021). Uncovering flooding mecha-
nisms across the contiguous United States through interpretive deep learning on
representative catchments. Water Resources Research, 57, e2021WR030185.
https://doi.org/10.1029/2021WR030185.

Copyright (c) 2021 Shijie Jiang. All rights reserved.

You should have received a copy of the MIT license along with the code. If not,
see <https://opensource.org/licenses/MIT>
"""

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

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
