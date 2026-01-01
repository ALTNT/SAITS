"""
Utility functions for data processing.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


import os

import h5py
import numpy as np


def window_truncate(feature_vectors, seq_len, sliding_len=None):
    """ Generate time series samples, truncating windows from time-series data with a given sequence length.
    Parameters
    ----------
    feature_vectors: time series data, len(shape)=2, [total_length, feature_num]
    seq_len: sequence length
    sliding_len: size of the sliding window
    """
    sliding_len = seq_len if sliding_len is None else sliding_len
    total_len = feature_vectors.shape[0]
    start_indices = np.asarray(range(total_len // sliding_len)) * sliding_len
    if total_len - start_indices[-1] * sliding_len < seq_len:  # remove the last one if left length is not enough
        start_indices = start_indices[:-1]
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx: idx + seq_len])
    return np.asarray(sample_collector).astype('float32')


def random_mask(vector, artificial_missing_rate):#(1136640,) 0.1
    """generate indices for random mask"""
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()#len()= 227426
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate))#(22742,)
    return indices


def add_artificial_mask(X, artificial_missing_rate, set_name):#(2557, 48, 37)
    """Add artificial missing values.
    Parameters
    ----------
    X: feature vectors
    artificial_missing_rate: rate of artificial missing values that are going to be create
    set_name: dataset name, train/val/test
    """
    sample_num, seq_len, feature_num = X.shape# 2557 48 37
    if set_name == "train":
        # if this is train set, we don't need add artificial missing values right now.
        # 如果这是训练集，我们现在不需要添加人为缺失值。
        # If we want to apply MIT during training, dataloader will randomly mask out some values to generate X_hat
        # 如果要在训练过程中应用MIT，数据加载器会随机屏蔽一些值以生成X_hat。

        # calculate empirical mean for model GRU-D, refer to paper
        # 计算模型 GRU-D 的经验均值，请参阅论文。
        mask = (~np.isnan(X)).astype(np.float32)#(2557, 48, 37)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(
            mask, axis=(0, 1)
        )#(37,)  逻辑上等价于 np.nanmean(X, axis=(0, 1)) ，但当前实现更显式地展示了“只统计观测值的和与计数”。  分子 np.sum(mask * X_filledWith0, axis=(0, 1)) 是每个特征的观测值总和。 分母 np.sum(mask, axis=(0, 1)) 是每个特征的观测值次数。 两者相除得到每个特征的经验均值向量，形状 (feature_num,)  逻辑上等价于 np.nanmean(X, axis=(0, 1)) ，但当前实现更显式地展示了“只统计观测值的和与计数”。
        data_dict = {
            "X": X,
            "empirical_mean_for_GRUD": empirical_mean_for_GRUD,
        }
    else:
        # if this is val/test set, then we need to add artificial missing values right now,
        # because we need they are fixed
        X = X.reshape(-1)#(1136640,)
        indices_for_holdout = random_mask(X, artificial_missing_rate)# (22742,) 从非 nan 的元素中选择 artificial_missing_rate 个索引
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
        missing_mask = (~np.isnan(X_hat)).astype(np.float32) #(1136640,) 缺失值掩码 1表示非缺失值，0表示缺失值
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32) #(1136640,) 表示需要被填充的缺失值位置，1表示缺失值，0表示非缺失值

        data_dict = {
            "X": X.reshape([sample_num, seq_len, feature_num]),#(800, 48, 37)
            "X_hat": X_hat.reshape([sample_num, seq_len, feature_num]),#(800, 48, 37)
            "missing_mask": missing_mask.reshape([sample_num, seq_len, feature_num]),#(800, 48, 37)
            "indicating_mask": indicating_mask.reshape(
                [sample_num, seq_len, feature_num]
            ),#(800, 48, 37)    
        }

    return data_dict


def saving_into_h5(saving_dir, data_dict, classification_dataset):
    """Save data into h5 file.
    Parameters
    ----------
    saving_dir: path of saving dir
    data_dict: data dictionary containing train/val/test sets
    classification_dataset: boolean, if this is a classification dataset
    """

    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        if classification_dataset:
            single_set.create_dataset("labels", data=data["labels"].astype(int))
        single_set.create_dataset("X", data=data["X"].astype(np.float32))
        if name in ["val", "test"]:
            single_set.create_dataset("X_hat", data=data["X_hat"].astype(np.float32))
            single_set.create_dataset(
                "missing_mask", data=data["missing_mask"].astype(np.float32)
            )
            single_set.create_dataset(
                "indicating_mask", data=data["indicating_mask"].astype(np.float32)
            )

    saving_path = os.path.join(saving_dir, "datasets.h5")
    with h5py.File(saving_path, "w") as hf:
        hf.create_dataset(
            "empirical_mean_for_GRUD",
            data=data_dict["train"]["empirical_mean_for_GRUD"],
        )
        save_each_set(hf, "train", data_dict["train"])
        save_each_set(hf, "val", data_dict["val"])
        save_each_set(hf, "test", data_dict["test"])
