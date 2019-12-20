# -*- coding: utf-8 -*-
__all__ = ['data_normalize', 'get_list_files']
import os
import numpy as np


def data_normalize(x_train, x_test, opts='min-max'):
    """ Normalize train and test directly."""
    if opts == 'l2':  # l2 normalization method
        for i in range(len(x_train)):
            for j in range(len(x_train[i])):
                vector = x_train[i][j, :]
                if np.linalg.norm(vector) != 0.0:
                    x_train[i][j, :] = vector / np.linalg.norm(vector)
                else:
                    x_train[i][j, :] = vector
        for i in range(len(x_test)):
            for j in range(len(x_test[i])):
                vector = x_test[i][j, :]
                if np.linalg.norm(vector) != 0.0:
                    x_test[i][j, :] = vector / np.linalg.norm(vector)
                else:
                    x_test[i][j, :] = vector
    elif opts == 'min-max':  # min-max scaling method to [0,1]
        for ii in range(len(x_train)):
            for jj in range(len(x_train[ii])):
                vector = x_train[ii][jj, :]
                max_ = np.max(vector, axis=0)
                min_ = np.min(vector, axis=0)
                if max_ == min_:
                    x_train[ii][jj, :] = vector
                else:
                    x_train[ii][jj, :] = (vector - min_) / (max_ - min_)
        for ii in range(len(x_test)):
            for jj in range(len(x_test[ii])):
                vector = x_test[ii][jj, :]
                max_ = np.max(vector, axis=0)
                min_ = np.min(vector, axis=0)
                if max_ == min_:
                    x_test[ii][jj, :] = vector
                else:
                    x_test[ii][jj, :] = (vector - min_) / (max_ - min_)
    # standardization ( or Z-score normalization) normalize x to
    # [mu=0,std=1.] search:
    # [Sebastian Raschka About Feature Scaling and Normalization]
    # often used in logistic regression, SVMs, perceptrons, NNs
    elif opts == 'std':
        for i in range(len(x_train)):
            for j in range(len(x_train[i])):
                vector = x_train[i][j, :]
                mean_v = np.mean(vector)
                std_v = np.std(vector)
                if std_v != 0.0:
                    x_train[i][j, :] = (vector - mean_v) / std_v
                else:
                    x_train[i][j, :] = vector
        for i in range(len(x_test)):
            for j in range(len(x_test[i])):
                vector = x_test[i][j, :]
                mean_v = np.mean(vector)
                std_v = np.mean(vector)
                if np.linalg.norm(vector) != 0.0:
                    x_test[i][j, :] = (vector - mean_v) / std_v
                else:
                    x_test[i][j, :] = vector


def get_list_files(input_folder, prefix=None):
    if not os.path.exists(input_folder):
        file_list = []
        return file_list
    for (dir_path, dir_names, file_names) in os.walk(input_folder):
        file_list = sorted([os.path.join(input_folder, filename)
                            for filename in file_names])
        if prefix is None or prefix == '':
            return file_list
        file_list = [_ for _ in file_list if _.find(prefix) != -1]
        return file_list
