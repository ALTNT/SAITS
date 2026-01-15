import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from configparser import ConfigParser, ExtendedInterpolation

from run_models3_shared_memory_train_and_val import CropAttriMappingDatasetBin, mean2, std2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_idx", type=int, required=True)
    p.add_argument("--nir_idx", type=int, default=6)
    p.add_argument("--red_idx", type=int, default=2)
    p.add_argument("--imputations_h5", type=str, default="./NIPS_results/PhysioNet2012_SAITS_best/step_313/imputations.h5")
    p.add_argument("--dataset_root", type=str)
    p.add_argument("--config_path", type=str)
    p.add_argument("--phase", type=str, default="val")
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--output_png", type=str)
    p.add_argument("--label", type=int)
    return p.parse_args()


def resolve_dataset_root(args):
    if args.dataset_root:
        return Path(args.dataset_root)
    if args.config_path:
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(args.config_path)
        base = cfg.get("file_path", "dataset_base_dir")
        name = cfg.get("dataset", "dataset_name")
        return Path(os.path.join(base, name)) / "US-dataset"
    raise ValueError("必须提供 --dataset_root 或 --config_path")


def load_imputed_array(h5_path):
    with h5py.File(h5_path, "r") as hf:
        if "imputed_test_set" in hf:
            return np.array(hf["imputed_test_set"])
        elif "imputed_val_set" in hf:
            return np.array(hf["imputed_val_set"])
        elif "imputed_train_set" in hf:
            return np.array(hf["imputed_train_set"])
        else:
            raise KeyError("imputations.h5 中未找到 imputed_*_set 数据集")


def inv_norm(x, idx):
    return x * std2[idx] + mean2[idx]


def find_idx_by_label(dataset, label):
    for i in range(len(dataset)):
        sample = dataset.__getitem__(i, need_y=True)
        y_val = sample[-1].item() if torch.is_tensor(sample[-1]) else int(sample[-1])
        if y_val == label:
            return i
    raise ValueError(f"未在数据集中找到标签为 {label} 的样本")





def main():
    args = parse_args()
    ds_root = resolve_dataset_root(args)#PosixPath('/home/al/gitdep/SAITS/data/data-no-mask-cloud/US-dataset')

    dataset = CropAttriMappingDatasetBin(
        args.phase.lower(), args.year, ds_root, None
    )

    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        pass  # 若按标签选择，将在下方覆盖索引

    idx = args.sample_idx
    if args.label is not None:
        idx = find_idx_by_label(dataset, args.label)

    index_t, X_hat_t, missing_mask_t, X_t, indicating_mask_t, doy_t, y_t = dataset.__getitem__(idx, need_y=True)

    X = X_t.numpy()
    doy = doy_t.numpy()

    imputed_all = load_imputed_array(args.imputations_h5)
    if idx >= imputed_all.shape[0]:
        raise IndexError(f"imputed 数组行数不足: {imputed_all.shape[0]}，需要 {idx+1}")
    imputed = imputed_all[idx]

    nir_idx = args.nir_idx
    red_idx = args.red_idx

    nir_orig = inv_norm(X[:, nir_idx], nir_idx)#(75,)
    red_orig = inv_norm(X[:, red_idx], red_idx)#(75,)
    nir_imp = inv_norm(imputed[:, nir_idx], nir_idx)#(75,)
    red_imp = inv_norm(imputed[:, red_idx], red_idx)#(75,)

    threshold = 1e-8
    nir_orig = np.where(np.abs(nir_orig) < threshold, 0.0, nir_orig)
    red_orig = np.where(np.abs(red_orig) < threshold, 0.0, red_orig)
    nir_imp = np.where(np.abs(nir_imp) < threshold, 0.0, nir_imp)
    red_imp = np.where(np.abs(red_imp) < threshold, 0.0, red_imp)

    orig_ndvi = (nir_orig - red_orig) / (nir_orig + red_orig + 1e-10)
    imputed_ndvi = (nir_imp - red_imp) / (nir_imp + red_imp + 1e-10)
    orig_ndvi = np.nan_to_num(orig_ndvi)
    imputed_ndvi = np.nan_to_num(imputed_ndvi)



    x_axis = doy

    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, orig_ndvi, "-", color="C0", label="Origin NDVI")
    plt.plot(x_axis, imputed_ndvi, "-", color="C1", label="Imputed NDVI")

    plt.xlabel("DOY")
    plt.ylabel("NDVI (0–1 scale)")
    plt.title(f"Sample {args.sample_idx} NDVI Origin vs Imputed")
    plt.legend()
    plt.tight_layout()

    if args.output_png:
        plt.savefig(args.output_png, dpi=150)
        print(f"已保存图像到: {args.output_png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()