"""
The script for generating PhysioNet-2012 dataset.

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


import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump

sys.path.append(".")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    add_artificial_mask,
    saving_into_h5,
)

np.random.seed(26)


def process_each_set(set_df, all_labels):#set_df: (122736, 38) all_labels: (12000, 1)
    # gene labels, y
    sample_ids = set_df["RecordID"].to_numpy().reshape(-1, 48)[:, 0]#(2557,)
    y = all_labels.loc[sample_ids].to_numpy().reshape(-1, 1)#(2557, 1)   array([[0],
    #    [0],
    #    [0],
    #    ...,
    #    [0],
    #    [1],
    #    [0]], shape=(2557, 1))
    # gene feature vectors, X
    set_df = set_df.drop("RecordID", axis=1)#(122736, 37)
    feature_names = set_df.columns.tolist()#len()=37 ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH']
    X = set_df.to_numpy()#(122736, 37)
    X = X.reshape(len(sample_ids), 48, len(feature_names))#(2557, 48, 37)
    return X, y, feature_names#(2557, 48, 37)  (2557, 1) len()=37


def keep_only_features_to_normalize(all_feats, to_remove):#['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RecordID', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH']
    for i in to_remove:#to_remove = ['RecordID]
        all_feats.remove(i)
    return all_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PhysioNet2012 dataset")
    parser.add_argument(
        "--raw_data_path", help="path of physio 2012 raw dataset", type=str
    )
    parser.add_argument(
        "--outcome_files_dir", help="dir path of raw dataset's outcome file", type=str
    )
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--train_frac", help="fraction of train set", type=float, default=0.8
    )
    parser.add_argument(
        "--val_frac", help="fraction of validation set", type=float, default=0.2
    )
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)#'./test'
    # create saving dir
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    # set up logger
    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
        "Generate PhysioNet2012 dataset",
        mode="w",
    )
    logger.info(args)

    outcome_files = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]
    outcome_collector = []
    args.outcome_files_dir = "/home/al/gitdep/STraTS/data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0"
    for o_ in outcome_files:
        outcome_file_path = os.path.join(args.outcome_files_dir, o_)
        with open(outcome_file_path, "r") as f:
            outcome = pd.read_csv(f)[["In-hospital_death", "RecordID"]]
        outcome = outcome.set_index("RecordID")
        outcome_collector.append(outcome)
    all_outcomes = pd.concat(outcome_collector)

    all_recordID = []
    df_collector = []
    args.raw_data_path = "/home/al/gitdep/STraTS/data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set-a"
    for filename in os.listdir(args.raw_data_path):#遍历set-a目录下的所有文件
        recordID = int(filename.split(".txt")[0])
        with open(os.path.join(args.raw_data_path, filename), "r") as f:
            df_temp = pd.read_csv(f)#(301, 3)
        df_temp["Time"] = df_temp["Time"].apply(lambda x: int(x.split(":")[0]))#将Time列中的时间转换为整数
        df_temp = df_temp.pivot_table("Value", "Time", "Parameter")#(40, 36) 将原始的“长表”( Time , Parameter , Value )转换为“宽表”，行是每个小时( Time )，列是各生理指标( Parameter )，单元格是对应的测量值( Value )。
        df_temp = df_temp.reset_index()  # take Time from index as a col  (40, 37) 将行索引 Time 还原为普通列，索引重置为默认的 RangeIndex ，因此可以用 df_temp['Time'] 做后续运算
        if len(df_temp) == 1:
            logger.info(
                f"Pass {recordID}, because its len==1, having no time series data"
            )
            continue
        all_recordID.append(recordID)  # only count valid recordID
        if df_temp.shape[0] != 48:#(40, 37) 缺失值填充
            missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))#[32, 34, 46, 22, 24, 26, 28, 30]
            missing_part = pd.DataFrame({"Time": missing})
            df_temp = pd.concat([df_temp, missing_part], ignore_index=False, sort=False)
            df_temp = df_temp.set_index("Time").sort_index().reset_index()#(48, 37) 缺失值填充后，将 Time 列设为索引，按时间排序，再重置索引为默认的 RangeIndex ，确保按时间顺序排列。
        df_temp = df_temp.iloc[
            :48
        ]  # only take 48 hours, some samples may have more records, like 49 hours
        df_temp["RecordID"] = recordID
        df_temp["Age"] = df_temp.loc[0, "Age"]#(48, 37) 缺失值填充后，将第0行的Age值复制到所有行，确保每个样本的年龄一致。
        df_temp["Height"] = df_temp.loc[0, "Height"]#(48, 37) 缺失值填充后，将第0行的Height值复制到所有行，确保每个样本的身高一致。
        df_collector.append(df_temp)
    df = pd.concat(df_collector, sort=True)#(191856, 43)
    df = df.drop(["Age", "Gender", "ICUType", "Height"], axis=1)#(191856, 39)
    df = df.reset_index(drop=True)#(191856, 39) 缺失值填充后，重置索引为默认的 RangeIndex ，确保按时间顺序排列。
    df = df.drop("Time", axis=1)  # dont need Time col. (191856, 38)

    train_set_ids, test_set_ids = train_test_split(
        all_recordID, train_size=args.train_frac
    )
    train_set_ids, val_set_ids = train_test_split(
        train_set_ids, test_size=args.val_frac
    )
    logger.info(f"There are total {len(train_set_ids)} patients in train set.")#2557
    logger.info(f"There are total {len(val_set_ids)} patients in val set.")#640
    logger.info(f"There are total {len(test_set_ids)} patients in test set.")#800

    all_features = df.columns.tolist()#len()=38 ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RecordID', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH']
    feat_no_need_to_norm = ["RecordID"]
    feats_to_normalize = keep_only_features_to_normalize(
        all_features, feat_no_need_to_norm
    )

    train_set = df[df["RecordID"].isin(train_set_ids)]#(122736, 38)
    val_set = df[df["RecordID"].isin(val_set_ids)]#(30720, 38)
    test_set = df[df["RecordID"].isin(test_set_ids)]#(38400, 38)

    # standardization
    scaler = StandardScaler()
    train_set.loc[:, feats_to_normalize] = scaler.fit_transform(
        train_set.loc[:, feats_to_normalize]
    )
    val_set.loc[:, feats_to_normalize] = scaler.transform(#仅使用训练集得到的 μ 和 σ 进行转换（不重新计算验证集 / 测试集的统计量）；
        val_set.loc[:, feats_to_normalize]
    )
    test_set.loc[:, feats_to_normalize] = scaler.transform(#仅使用训练集得到的 μ 和 σ 进行转换（不重新计算验证集 / 测试集的统计量）；
        test_set.loc[:, feats_to_normalize]
    )

    train_set_X, train_set_y, feature_names = process_each_set(train_set, all_outcomes)
    val_set_X, val_set_y, _ = process_each_set(val_set, all_outcomes)##(640, 48, 37)  (640, 1) len()=37
    test_set_X, test_set_y, _ = process_each_set(test_set, all_outcomes)

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )
    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    train_set_dict["labels"] = train_set_y
    val_set_dict["labels"] = val_set_y
    test_set_dict["labels"] = test_set_y

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }

    logger.info(f"All saved features: {feature_names}")
    saved_df = df.loc[:, feature_names]#(191856, 37)

    total_sample_num = 0
    total_positive_num = 0
    for set_name, rec in zip(
        ["train", "val", "test"], [train_set_dict, val_set_dict, test_set_dict]
    ):
        total_sample_num += len(rec["labels"])
        total_positive_num += rec["labels"].sum()
        logger.info(
            f'Positive rate in {set_name} set: {rec["labels"].sum()}/{len(rec["labels"])}='
            f'{(rec["labels"].sum() / len(rec["labels"])):.3f}'
        )
    logger.info(
        f"Dataset overall positive rate: {(total_positive_num / total_sample_num):.3f}"
    )

    missing_part = np.isnan(saved_df.to_numpy())#(191856, 37)
    logger.info(
        f"Dataset overall missing rate of original feature vectors (without any artificial mask): "
        f"{(missing_part.sum() / missing_part.shape[0] / missing_part.shape[1]):.3f}"
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=True)
    pickle_dump(scaler, os.path.join(dataset_saving_dir, 'scaler'))
    logger.info(f"All done. Saved to {dataset_saving_dir}.")
