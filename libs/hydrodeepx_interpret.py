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

import os
import shap
import pickle
import warnings
import collections
import numpy as np
from tqdm.notebook import tqdm

def interpret_eg(model, background, x_dict, scale_params, path, nsamples=1000, overwrite=True):
    """
    Explains a model using expected gradients.

    Parameters
    ----------
    model: a differentiable model (tf.keras.Model) to be explained
    background: the background dataset to use for integrating out features, such as the training dataset.
    x_dict: the input x for which the explainer will return feature importance values.
       It should be a subset of data_x_dict obtained from the function get_wrapped_data().
    scale_params: the mean and std values of the training data obtained from split_train_test().
    path: the path to save the results.
    nsamples: number of samples to approximate the integral (default: 1000).
    overwrite: whether overwrite the existed file in the given path (default: True).

    Returns
    -------
    peak_eg_dict: a dictionary of the resulting expected gradients whose key is the same as the key of x.
        The expected gradients have the same shape as the corresponding x.

    """
    if (os.path.exists(path)) and (overwrite is False):
        print(f"File exists! Loading the saved file from {os.path.relpath(path)}")
        with open(path, 'rb') as f:
            peak_eg_dict, peak_eg_var_dict = pickle.load(f)
        print("Done!")

    else:
        if (os.path.exists(path)) and (overwrite is True):
            print(f"File exists in {os.path.relpath(path)}! Now overwrite the file.")
            print("You can set overwrite=False to reload the saved file.\n")

        eval_x_data   = np.array(list(x_dict.values()))
        eval_x_scaled = (eval_x_data - scale_params['train_x_mean'][None, None, :]) \
                        / scale_params['train_x_std'][None, None, :]

        print("Create a GradientExplainer and calculate EG values. Please be patient, it would take some time...")

        explainer_grad = shap.GradientExplainer(model=model, data=background, batch_size=10000, local_smoothing=0)

        peak_eg_values, peak_eg_values_var = [], []
        for i in tqdm(range(eval_x_scaled.shape[0]), ''):
            eg_values_var_min = 1e6

            count = 0
            while (count < 10):
                _eg_values, _eg_values_var = explainer_grad.shap_values(eval_x_scaled[i:i+1, :, :], nsamples=nsamples, return_variances=True)

                if np.sum(_eg_values_var) < eg_values_var_min:
                    eg_values, eg_values_var = _eg_values, _eg_values_var
                    eg_values_var_min = np.sum(_eg_values_var)
                count = count + 1

            peak_eg_values.append(eg_values[0])
            peak_eg_values_var.append(eg_values_var[0])

        peak_eg_dict     = dict(zip(x_dict.keys(), peak_eg_values))
        peak_eg_var_dict = dict(zip(x_dict.keys(), peak_eg_values_var))

        print(f"Done! The file is saved to {os.path.relpath(path)}")

        with open(path, 'wb') as f:
            pickle.dump([peak_eg_dict, peak_eg_var_dict], f)

    return peak_eg_dict, peak_eg_var_dict

class interpret_decomp():
    def __init__(self, model):
        lstm_W  = model.get_layer('lstm').get_weights()[0]
        lstm_U  = model.get_layer('lstm').get_weights()[1]
        lstm_b  = model.get_layer('lstm').get_weights()[2]

        self.model = model
        self.UNITS = model.get_layer('lstm').units

        self.lstm_W_i = lstm_W[:,:self.UNITS]
        self.lstm_W_f = lstm_W[:, self.UNITS * 1 : self.UNITS * 2]
        self.lstm_W_c = lstm_W[:, self.UNITS * 2 : self.UNITS * 3]
        self.lstm_W_o = lstm_W[:, self.UNITS * 3 :]

        self.lstm_U_i = lstm_U[:,:self.UNITS]
        self.lstm_U_f = lstm_U[:, self.UNITS * 1 : self.UNITS * 2]
        self.lstm_U_c = lstm_U[:, self.UNITS * 2 : self.UNITS * 3]
        self.lstm_U_o = lstm_U[:, self.UNITS * 3 :]

        self.lstm_b_i = lstm_b[:self.UNITS]
        self.lstm_b_f = lstm_b[self.UNITS * 1 : self.UNITS * 2]
        self.lstm_b_c = lstm_b[self.UNITS * 2 : self.UNITS * 3]
        self.lstm_b_o = lstm_b[self.UNITS * 3 :]

        self.dense_W  = model.get_layer('dense').get_weights()[0]

    @staticmethod
    def __sigmoid(arr):
        c = np.clip(arr, -1000, 1000)
        return 1 / (1 + np.exp(-c))

    @staticmethod
    def __tanh(arr):
        c = np.clip(arr, -1000, 1000)
        return (np.exp(c) - np.exp(-c)) / (np.exp(c) + np.exp(-c))

    def __forward_lstm(self, x_dict, scale_params):

        x          = np.stack([i for i in x_dict.values()])

        self.x     = (x - scale_params['train_x_mean'][None, None, :]) / scale_params['train_x_std'][None, None, :]
        self.dates = np.stack([i for i in x_dict.keys()])
        self.batch_num = self.x.shape[0]
        self.steps = self.x.shape[1]

        h_states = [np.zeros([self.batch_num, self.UNITS])]
        c_states = [np.zeros([self.batch_num, self.UNITS])]

        hf_arr, hi_arr, ho_arr, hc_arr = [], [], [], []

        for t in range(self.steps):
            x = self.x[:, t, :]
            c_prev, h_prev = c_states[-1], h_states[-1]

            hf = self.__sigmoid(np.dot(x, self.lstm_W_f) + np.dot(h_prev, self.lstm_U_f) + self.lstm_b_f)
            hi = self.__sigmoid(np.dot(x, self.lstm_W_i) + np.dot(h_prev, self.lstm_U_i) + self.lstm_b_i)
            ho = self.__sigmoid(np.dot(x, self.lstm_W_o) + np.dot(h_prev, self.lstm_U_o) + self.lstm_b_o)
            hc = self.__tanh(np.dot(x, self.lstm_W_c)    + np.dot(h_prev, self.lstm_U_c) + self.lstm_b_c)

            c = hf * c_prev + hi * hc
            h = ho * self.__tanh(c)

            h_states.append(h)
            c_states.append(c)
            hf_arr.append(hf)
            hi_arr.append(hi)
            ho_arr.append(ho)
            hc_arr.append(hc)

        h_states = np.stack(h_states, axis=1)
        c_states = np.stack(c_states, axis=1)

        hf_arr   = np.stack(hf_arr, axis=1)
        hi_arr   = np.stack(hi_arr, axis=1)
        ho_arr   = np.stack(ho_arr, axis=1)
        hc_arr   = np.stack(hc_arr, axis=1)

        out_arr  = np.dot(h_states[:,-1,:], self.dense_W)

        return h_states, c_states, hf_arr, hi_arr, ho_arr, hc_arr, out_arr

    @staticmethod
    def __cal_h_i_j(f, o, start, end):
        if start == end:
            return 1
        else:
            return np.prod(f[:, start:end, :] * o[:, start:end, :] / o[:, start-1:end-1, :], axis=1)

    def __cal_h_T(self, f, o, h, T):
        h_forget = np.zeros_like(f[:, 0:T, :])
        h_update = np.zeros_like(f[:, 0:T, :])

        for t in range(0, T):
            h_forget[:, t, :] = self.__cal_h_i_j(f, o, t+1, T)
            if t == 0:
                h_update[:, t, :] = h[:, t, :]
            else:
                h_update[:, t, :] = h[:, t+1, :] - h[:, t, :] * f[:, t, :] * o[:, t, :] / o[:, t-1, :]

        return h_forget, h_update

    @staticmethod
    def __cal_c_i_j(f, start, end):
        if start == end:
            return 1
        else:
            return np.prod(f[:, start:end, :], axis=1)

    def __cal_c_T(self, f, T):
        e_vec = np.zeros_like(f[:, 0:T, :])
        for t in range(T):
            e_vec[:, t, :] = self.__cal_c_i_j(f, t+1, T)
        e_vec = [e_vec[i, :] for i in range(e_vec.shape[0])]

        return np.stack(e_vec)

    def get_internals(self, x_dict, scale_params, validate=True, val_threshold=0.01):
        self.h_states, self.c_states, self.hf_arr, self.hi_arr, self.ho_arr, self.hc_arr, self.out_arr = self.__forward_lstm(x_dict, scale_params)
        self.h_forget, self.h_update = self.__cal_h_T(f=self.hf_arr, o=self.ho_arr, h=self.h_states, T=self.steps)
        self.c_evolve                = self.__cal_c_T(self.hf_arr, T=self.steps)

        if validate:
            count = 0
            error_index, error_list = [], []
            for i in range(self.batch_num):
                a = np.sum(self.h_forget[i, :] * self.h_update[i, :], axis=0)
                b = self.h_states[i, -1, :]
                if np.sum(a - b) > val_threshold:
                    count = count + 1
                    error_index.append(i)
                    error_list.append(np.sum(a - b))

            if count != 0:
                warnings.warn(f'Warning! Errors in {count} samples are larger than the val_threshold ({val_threshold}).')
                for i, index in enumerate(error_index):
                    warnings.warn(f'{self.dates[index]}: sum[sum(h_forget*h_update) - h_state] = {error_list[i]:.3f}')

        result_dict = collections.defaultdict(dict)

        for i, date in enumerate(self.dates):
            result_dict[date]['x']        = self.x[i, :]
            result_dict[date]['h_states'] = self.h_states[i, :]
            result_dict[date]['c_states'] = self.c_states[i, :]
            result_dict[date]['hf_arr']   = self.hf_arr[i, :]
            result_dict[date]['hi_arr']   = self.hi_arr[i, :]
            result_dict[date]['ho_arr']   = self.ho_arr[i, :]
            result_dict[date]['hc_arr']   = self.hc_arr[i, :]
            result_dict[date]['out_arr']  = self.out_arr[i, :]
            result_dict[date]['h_forget'] = self.h_forget[i, :]
            result_dict[date]['h_update'] = self.h_update[i, :]
            result_dict[date]['c_evolve'] = self.c_evolve[i, :]

            result_dict[date]['dense_W']  = self.dense_W

        return result_dict
