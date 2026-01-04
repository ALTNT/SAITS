"""
The script for running (including training and testing) all models in this repo.

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

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023.

"""


import argparse
import math
import os
import time
import warnings
import atexit, signal, sys, hashlib   # 新增：清理共享内存
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
from pathlib import Path
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import DistributedSampler
# 文件头部新增
import multiprocessing as mp
from multiprocessing import shared_memory  # PyTorch 1.8+ 自带

import pandas as pd

try:
    import nni
except ImportError:
    pass

from Global_Config import RANDOM_SEED
from modeling.saits import SAITS
from modeling.saits_for_CACM import SAITS_for_CACM
from modeling.transformer import TransformerEncoder
from modeling.brits import BRITS
from modeling.mrnn import MRNN
from modeling.unified_dataloader import UnifiedDataLoader
import modeling.misc as misc
from modeling.utils import (
    Controller,
    setup_logger,
    save_model,
    load_model,
    check_saving_dir_for_model,
    masked_mae_cal,
    masked_rmse_cal,
    masked_mre_cal,
)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
warnings.filterwarnings("ignore")  # if to ignore warnings

mean=np.array([0.0592465,0.08257256,0.07940245,0.122184,0.26104507,0.32686248,0.33226496,0.35085383,0.25254872,0.16759971])
std = np.array([0.03050405, 0.03372968, 0.05003787, 0.05067676, 0.07917259,0.11817433, 0.11635097, 0.12101864, 0.09216624, 0.08916556])

mean2= np.array([0.33384848099779335, 0.33743598254332613, 0.3313643807274545, 0.3738025112940193, 0.43651407997201386, 0.46520171460177606, 0.47281375925359626, 0.4763394446147105, 0.3112739897772981, 0.24385958955982265])
std2 = np.array([0.36185714564075394, 0.3371488493517183, 0.33266536803207, 0.33175146328580024, 0.2805921329391628, 0.2656709362852829, 0.2702308822869307, 0.25237716313205766, 0.1477668850798491, 0.13772583877397532])
# ## tmn tmx srad
# mean_c=np.array([51.9398428,  169.45926927, 1877.14472879])#气候数据的均值
# std_c = np.array([115.44607918 , 123.6412339, 718.64770224])#气候数据的标准差
mean_c=np.array([78.02576446969697, 198.9981634469697, 1850.3222502462122])
std_c = np.array([120.325428898157, 124.14974885223911, 680.851609868019])
MODEL_DICT = {
    # Self-Attention (SA) based
    "Transformer": TransformerEncoder,
    "SAITS": SAITS,
    "SAITS_for_CACM":SAITS_for_CACM,
    # RNN based
    "BRITS": BRITS,
    "MRNN": MRNN,
}
OPTIMIZER = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def read_arguments(arg_parser, cfg_parser):
    # file path
    arg_parser.dataset_base_dir = cfg_parser.get("file_path", "dataset_base_dir")
    arg_parser.result_saving_base_dir = cfg_parser.get(
        "file_path", "result_saving_base_dir"
    )
    # dataset info
    arg_parser.seq_len = cfg_parser.getint("dataset", "seq_len")
    arg_parser.batch_size = cfg_parser.getint("dataset", "batch_size")
    arg_parser.num_workers = cfg_parser.getint("dataset", "num_workers")
    arg_parser.feature_num = cfg_parser.getint("dataset", "feature_num")
    arg_parser.dataset_name = cfg_parser.get("dataset", "dataset_name")
    arg_parser.dataset_path = os.path.join(
        arg_parser.dataset_base_dir, arg_parser.dataset_name
    )
    arg_parser.eval_every_n_steps = cfg_parser.getint("dataset", "eval_every_n_steps")
    # training settings
    arg_parser.MIT = cfg_parser.getboolean("training", "MIT")
    arg_parser.ORT = cfg_parser.getboolean("training", "ORT")
    arg_parser.lr = cfg_parser.getfloat("training", "lr")
    arg_parser.optimizer_type = cfg_parser.get("training", "optimizer_type")
    arg_parser.weight_decay = cfg_parser.getfloat("training", "weight_decay")
    arg_parser.device = cfg_parser.get("training", "device")
    arg_parser.epochs = cfg_parser.getint("training", "epochs")
    arg_parser.early_stop_patience = cfg_parser.getint(
        "training", "early_stop_patience"
    )
    arg_parser.model_saving_strategy = cfg_parser.get(
        "training", "model_saving_strategy"
    )
    arg_parser.max_norm = cfg_parser.getfloat("training", "max_norm")
    arg_parser.imputation_loss_weight = cfg_parser.getfloat(
        "training", "imputation_loss_weight"
    )
    arg_parser.reconstruction_loss_weight = cfg_parser.getfloat(
        "training", "reconstruction_loss_weight"
    )
    # model settings
    arg_parser.model_name = cfg_parser.get("model", "model_name")
    arg_parser.model_type = cfg_parser.get("model", "model_type")
    return arg_parser


def summary_write_into_tb(summary_writer, info_dict, step, stage):
    """write summary into tensorboard file"""
    summary_writer.add_scalar(f"total_loss/{stage}", info_dict["total_loss"], step)
    summary_writer.add_scalar(
        f"imputation_loss/{stage}", info_dict["imputation_loss"], step
    )
    summary_writer.add_scalar(
        f"imputation_MAE/{stage}", info_dict["imputation_MAE"], step
    )
    summary_writer.add_scalar(
        f"reconstruction_loss/{stage}", info_dict["reconstruction_loss"], step
    )
    summary_writer.add_scalar(
        f"reconstruction_MAE/{stage}", info_dict["reconstruction_MAE"], step
    )


def result_processing(results):
    """process results and losses for each training step"""
    results["total_loss"] = torch.tensor(0.0, device=args.device)
    if args.model_type == "BRITS":#False
        results["total_loss"] = (
            results["consistency_loss"] * args.consistency_loss_weight
        )
    results["reconstruction_loss"] = (
        results["reconstruction_loss"] * args.reconstruction_loss_weight
    )
    results["imputation_loss"] = (
        results["imputation_loss"] * args.imputation_loss_weight
    )
    if args.MIT:#True
        results["total_loss"] += results["imputation_loss"]
    if args.ORT:#True
        results["total_loss"] += results["reconstruction_loss"]
    return results


def process_each_training_step(
    results, optimizer, val_dataloader, training_controller, summary_writer, logger
):
    """process each training step and return whether to early stop"""
    state_dict = training_controller(stage="train")
    # apply gradient clipping if args.max_norm != 0
    if args.max_norm != 0:#False
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    results["total_loss"].backward()
    optimizer.step()

    summary_write_into_tb(summary_writer, results, state_dict["train_step"], "train")
    if state_dict["train_step"] % args.eval_every_n_steps == 0:#30
        state_dict_from_val = validate(
            model, val_dataloader, summary_writer, training_controller, logger
        )
        if state_dict_from_val["should_stop"]:
            logger.info(f"Early stopping worked, stop now...")
            return True
    return False


def model_processing(
    data,
    model,
    stage,
    # following arguments are only required in the training stage
    optimizer=None,
    val_dataloader=None,
    summary_writer=None,
    training_controller=None,
    logger=None,
):
    if stage == "train":
        optimizer.zero_grad()
        if not args.MIT:#False
            if args.model_type in ["BRITS", "MRNN"]:#False
                (
                    indices,
                    X,
                    missing_mask,
                    deltas,
                    back_X,
                    back_missing_mask,
                    back_deltas,
                ) = map(lambda x: x.to(args.device), data)
                inputs = {
                    "indices": indices,
                    "forward": {"X": X, "missing_mask": missing_mask, "deltas": deltas},
                    "backward": {
                        "X": back_X,
                        "missing_mask": back_missing_mask,
                        "deltas": back_deltas,
                    },
                }
            else:  # then for self-attention based models, i.e. Transformer/SAITS
                indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                inputs = {"indices": indices, "X": X, "missing_mask": missing_mask}
            results = result_processing(model(inputs, stage))
            early_stopping = process_each_training_step(
                results,
                optimizer,
                val_dataloader,
                training_controller,
                summary_writer,
                logger,
            )
        else:#True
            if args.model_type in ["BRITS", "MRNN"]:#False
                (
                    indices,
                    X,
                    missing_mask,
                    deltas,
                    back_X,
                    back_missing_mask,
                    back_deltas,
                    X_holdout,
                    indicating_mask,
                ) = map(lambda x: x.to(args.device), data)
                inputs = {
                    "indices": indices,
                    "X_holdout": X_holdout,
                    "indicating_mask": indicating_mask,
                    "forward": {"X": X, "missing_mask": missing_mask, "deltas": deltas},
                    "backward": {
                        "X": back_X,
                        "missing_mask": back_missing_mask,
                        "deltas": back_deltas,
                    },
                }
            else:#True
                indices, X, missing_mask, X_holdout, indicating_mask, doy = map(
                    lambda x: x.to(args.device), data
                )
                inputs = {
                    "indices": indices,#torch.Size([128])
                    "X": X,#torch.Size([128, 48, 37])
                    "missing_mask": missing_mask,#torch.Size([128, 48, 37])
                    "X_holdout": X_holdout,#torch.Size([128, 48, 37])
                    "indicating_mask": indicating_mask,#torch.Size([128, 48, 37])
                    "doy":doy,#torch.Size([128, 48])
                }
            results = result_processing(model(inputs, stage))
            early_stopping = process_each_training_step(
                results,
                optimizer,
                val_dataloader,
                training_controller,
                summary_writer,
                logger,
            )
        return early_stopping

    else:  # in val/test stage
        if args.model_type in ["BRITS", "MRNN"]:
            (
                indices,
                X,
                missing_mask,
                deltas,
                back_X,
                back_missing_mask,
                back_deltas,
                X_holdout,
                indicating_mask,
            ) = map(lambda x: x.to(args.device), data)
            inputs = {
                "indices": indices,
                "X_holdout": X_holdout,
                "indicating_mask": indicating_mask,
                "forward": {"X": X, "missing_mask": missing_mask, "deltas": deltas},
                "backward": {
                    "X": back_X,
                    "missing_mask": back_missing_mask,
                    "deltas": back_deltas,
                },
            }
            inputs["missing_mask"] = inputs["forward"][
                "missing_mask"
            ]  # for error calculation in validation stage
        else:
            indices, X, missing_mask, X_holdout, indicating_mask, doy = map(
                lambda x: x.to(args.device), data
            )
            inputs = {
                "indices": indices,
                "X": X,
                "missing_mask": missing_mask,
                "X_holdout": X_holdout,
                "indicating_mask": indicating_mask,
                "doy":doy,
            }
        results = model(inputs, stage)
        results = result_processing(results)
        return inputs, results


def train(
    model,
    optimizer,
    train_dataloader,
    test_dataloader,
    summary_writer,
    training_controller,
    logger,
):
    for epoch in range(args.epochs):#10000
        early_stopping = False
        args.final_epoch = True if epoch == args.epochs - 1 else False#False
        
        start_time = time.time()
        for idx, data in enumerate(train_dataloader):
            if idx == 599:
                tmp = 1
            # print(idx)
            data_time = time.time() - start_time
            
            model.train()
            
            process_start_time = time.time()
            early_stopping = model_processing(
                data,
                model,
                "train",
                optimizer,
                test_dataloader,
                summary_writer,
                training_controller,
                logger,
            )
            torch.cuda.synchronize()
            process_time = time.time() - process_start_time
            
            if idx % 200 == 0:
                logger.info(f"idx: {idx}")
                logger.info(f"Epoch {epoch}: Data loading time: {data_time:.4f}s, Model forward/backward time: {process_time:.4f}s")

            if early_stopping:
                break
            
            start_time = time.time()
        if early_stopping:
            break
        training_controller.epoch_num_plus_1()
    logger.info("Finished all epochs. Stop training now.")

class RandomAddNoise: 
    def __call__(self, x, valid_positions):
        # x: (T, C), valid_positions: (T, 1) or (T,)
        t, c = x.shape
        noise_mask = np.zeros((t, 1), dtype=bool)#(75, 1)
        
        # 获取有效位置的索引
        valid_indices = np.where(valid_positions.reshape(-1))[0]#(16,)
        num_noise = int(len(valid_indices) * 0.15)#数量
        
        if num_noise > 0:
            chosen_indices = np.random.choice(valid_indices, num_noise, replace=False)
            for idx in chosen_indices:
                # 模拟云(增亮)或云阴影(变暗)，各50%概率
                # 使用绝对值高斯噪声确保方向一致性
                noise = np.abs(np.random.normal(0, 0.5, size=c))
                if np.random.rand() < 0.5:
                    x[idx, :] -= noise # 变暗
                else:
                    x[idx, :] += noise # 增亮
                noise_mask[idx] = True
                
        return x, noise_mask

class CropAttriMappingDatasetBin(Dataset):
    """
    Dataset for loading pre-aligned data from binary files using memory mapping.
    支持单文件与分片二进制文件加载。
    """
    def __init__(self,phase,year,root, dataaug= None, target_region=None, need_x_length=False, need_growing_season_masking=False):
    # def __init__(self, config, phase, year, root, dataaug=None, target_region=None):
        super().__init__()
        phase = phase.lower()
        assert phase in ["train", "val", "test"]
        self.root = root#'data/data-no-mask-cloud/US-dataset'
        self.sequencelength = 75 # Fixed length
        # self.mean = mean
        # self.std = std
        self.mean = mean2
        self.std = std2
        self.mean_c = mean_c
        self.std_c = std_c

        kernel_size = 3
        confidence_threshold = 90
        self.target_region = target_region  # 保存目标区域
        mode_name = 'val' if phase == 'valid' else phase  # 修正命名差异

        # 支持分片文件优先，其次是合并文件
        ds_dir = Path(root) / (str(year) + "_valid_kernel" + str(kernel_size) + "_conf" + str(confidence_threshold))
        merged_bin = ds_dir / f"mode_{mode_name}_merged.bin"
        shard_glob = ds_dir.glob(f"mode_{mode_name}_merged_shard*.bin")
        shard_paths = sorted([str(p) for p in shard_glob])
        if shard_paths:
            self.bin_paths = shard_paths
            self.is_sharded = True
        else:
            if not merged_bin.exists():
                raise FileNotFoundError(f"合并后的二进制文件不存在: {merged_bin}")
            self.bin_paths = [str(merged_bin)]
            self.is_sharded = False

        # 二进制布局常量
        self.X_FEATURES = self.sequencelength * 10
        self.DOY_FEATURES = self.sequencelength
        self.COND_FEATURES = 8 * 3
        self.SCL_FEATURES = 75
        # self.SAMPLE_SIZE_BYTES = 4 * (1 + self.X_FEATURES + self.DOY_FEATURES + self.COND_FEATURES)
        self.SAMPLE_SIZE_BYTES = 4 * (1 + self.X_FEATURES + self.DOY_FEATURES + self.COND_FEATURES + self.SCL_FEATURES)
        # self.floats_count = 1 + self.X_FEATURES + self.DOY_FEATURES + self.COND_FEATURES
        self.floats_count = 1 + self.X_FEATURES + self.DOY_FEATURES + self.COND_FEATURES + self.SCL_FEATURES
        self.dtype = np.float32

        # 预加载路径并计算样本数，初始化memmap
        self._preload_bin_paths()
        self._init_memmap()

        # 按区域过滤样本并构建指针（覆盖 n_samples）
        self.sample_ptrs = None
        if self.target_region is not None:
            self.sample_ptrs = self._load_index_and_build_ptrs(ds_dir, mode_name, self.bin_paths)
            self.n_samples = len(self.sample_ptrs)
            print(f"按区域 {self.target_region} 过滤后样本数: {self.n_samples}")

        print(f'Loaded {phase} dataset with {self.n_samples} samples.')
        self.tempaug = TwoCropsTransform(dataaug) if dataaug is not None else None

        self.rc = False  # whether to random choice the time series data
        self.interp = False  # whether to interpolate the time series data
        self.mask_cloud_seq = False
        self.cloud_pad_0 = True
        self.need_weight = False

        # 新增：生长季内模拟掩蔽训练参数
        self.growing_season_masking = False
        self.min_mapping_doy = 180  # 最早制图日期（7月初）
        # self.max_mapping_doy = config.DATASET.MAX_MAPPING_DOY  # 最晚制图日期（8月底）
        self.masking_probability = 0.5  # 掩蔽概率

        self.add_noise = RandomAddNoise()

    def __len__(self):
        return self.n_samples

    def _load_index_and_build_ptrs(self, ds_dir, mode_name, bin_paths):
        """读取 index.csv，按 mode+region 过滤，返回样本指针列表"""
        idx_path = ds_dir / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"未找到索引文件: {idx_path}，无法按区域筛选")
        df = pd.read_csv(str(idx_path))
        required_cols = {'region', 'mode', 'bin_path', 'sample_idx'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"index.csv 缺少必需列 {required_cols}")
        df = df[(df['mode'] == mode_name) & (df['region'] == self.target_region)]
        if len(df) == 0:
            raise ValueError(f"区域 {self.target_region} 在阶段 {mode_name} 无样本")
        path_to_idx = {str(p): i for i, p in enumerate(bin_paths)}
        ptrs = []
        for _, row in df.iterrows():
            bp = str(row['bin_path'])
            if bp in path_to_idx:
                shard_idx = path_to_idx[bp]#0
            else:
                # 兼容不同绝对/相对路径：仅按文件名匹配
                bn = Path(bp).name
                candidates = [i for i, p in enumerate(bin_paths) if Path(p).name == bn]
                if not candidates:
                    raise FileNotFoundError(f"索引指向的分片不存在: {bp}")
                shard_idx = candidates[0]
            ptrs.append({'shard_idx': int(shard_idx), 'sample_idx': int(row['sample_idx'])})
        print(f"区域 {self.target_region} 指向 {len(set([p['shard_idx'] for p in ptrs]))} 个分片文件")
        return ptrs

    def _preload_bin_paths(self):
        """预加载二进制文件路径并计算样本数（支持分片）"""
        print("Preloading binary file paths...")
        self.shard_sample_counts = []
        for bp in self.bin_paths:
            file_size = os.path.getsize(bp)#536870200
            if file_size % self.SAMPLE_SIZE_BYTES != 0:
                raise ValueError(f"文件大小 {file_size} 不是每样本字节数 {self.SAMPLE_SIZE_BYTES} 的整数倍: {bp}")
            self.shard_sample_counts.append(file_size // self.SAMPLE_SIZE_BYTES)
        self.shard_sample_counts = np.array(self.shard_sample_counts, dtype=np.int64)#array([145100, 145100,   9800])
        self.shard_ends = np.cumsum(self.shard_sample_counts)#array([145100, 290200, 300000])
        self.n_samples = int(self.shard_ends[-1])#300000
        print(f"Preloaded {len(self.bin_paths)} binary file(s) with total {self.n_samples} samples")

    def _init_memmap(self):
        """只在主进程创建共享内存，worker 进程直接 attach"""
        self._shm_list = [None] * len(self.bin_paths)
        self._buffers = [None] * len(self.bin_paths)

        # 主进程：提前为所有分片创建共享内存
        if mp.current_process().name == 'MainProcess':
            print(f"MainProcess: Initializing SharedMemory for {len(self.bin_paths)} shards...")
            for i, path in enumerate(self.bin_paths):
                name = f"saits_shm_{hashlib.md5(path.encode()).hexdigest()}"#'saits_shm_9d71c4218e5b02881f27ee28a234ab98'
                
                # 清理残留（如果有）
                try:
                    existing = shared_memory.SharedMemory(name=name)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                
                # 创建并写入数据
                tmp = np.memmap(path, dtype=self.dtype, mode='r')#使用 numpy.memmap 为磁盘上的二进制文件创建一个“内存映射” 此时数据 并没有 被完全读入内存。 tmp 只是一个指向磁盘文件的“视图”（View）
                shm = shared_memory.SharedMemory(create=True, size=tmp.nbytes, name=name)#调用系统 API（在 Linux 下通常对应 /dev/shm ）申请一块 真实的物理内存区域
                # 拷贝数据到共享内存
                np.ndarray(tmp.shape, dtype=self.dtype, buffer=shm.buf)[:] = tmp#[:] = tmp ：执行 全量数据复制
                self._shm_list[i] = shm#将共享内存对象保存在主进程的列表中 原因 ：防止 Python 的垃圾回收机制（GC）把 shm 对象销毁
                
                # 主进程也顺便初始化 buffer
                self._buffers[i] = np.ndarray(
                    tmp.shape, dtype=self.dtype, buffer=shm.buf
                )
            print("MainProcess: SharedMemory initialized.")

    def _ensure_shm(self, shard_idx):
        """返回可直接 numpy 使用的内存视图"""
        if self._buffers[shard_idx] is not None:
            return self._buffers[shard_idx]

        path = self.bin_paths[shard_idx]
        name = f"saits_shm_{hashlib.md5(path.encode()).hexdigest()}"
        
        # Worker 进程：attach 现有的共享内存
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            raise RuntimeError(
                f"Worker 进程找不到共享内存 {name} (shard {shard_idx})。\n"
                "请确保主进程已完全执行 _init_memmap 并创建了共享内存。"
            )

        # 存入列表防止被 GC 导致 buffer 失效（Worker 只 close 不 unlink）
        self._shm_list[shard_idx] = shm
        
        size = os.path.getsize(path)
        buf = np.ndarray(
            (size // np.dtype(self.dtype).itemsize,),
            dtype=self.dtype,
            buffer=shm.buf
        )
        self._buffers[shard_idx] = buf
        return buf

    def close_shm(self):
        """释放共享内存资源"""
        is_main = (mp.current_process().name == 'MainProcess')
        for shm in getattr(self, '_shm_list', []):
            if shm is not None:
                try:
                    shm.close()
                    if is_main:
                        shm.unlink()   # 只有主进程负责销毁
                except FileNotFoundError:
                    pass
        self._shm_list = []
        self._buffers = []

    def __getitem__(self, index):
        # 计算所在分片及片内索引
        # if index < 0 or index >= self.n_samples:
        #     raise IndexError(f'Index {index} out of range [0, {self.n_samples})')

        # 若启用区域筛选，使用预构建的指针定位
        if self.sample_ptrs is not None:
            ptr = self.sample_ptrs[index]
            shard_idx = int(ptr['shard_idx'])
            local_sample_idx = int(ptr['sample_idx'])
        else:
            shard_idx = int(np.searchsorted(self.shard_ends, index, side='right'))  # 找到第一个大于index的分片索引
            prev_end = int(self.shard_ends[shard_idx - 1]) if shard_idx > 0 else 0
            local_sample_idx = index - prev_end

        #old
        # mm = self._ensure_memmap(shard_idx)
        # start_f = local_sample_idx * self.floats_count
        # end_f = start_f + self.floats_count
        # data = mm[start_f:end_f]
        #new 
        buf = self._ensure_shm(shard_idx)
        start_f = local_sample_idx * self.floats_count
        data = buf[start_f:start_f + self.floats_count]

        n_features = self.X_FEATURES // self.sequencelength
        data = np.array(data, copy=True)
        x = data[1:1 + self.X_FEATURES].reshape(self.sequencelength, n_features)
        doy = data[1 + self.X_FEATURES:1 + self.X_FEATURES + self.DOY_FEATURES].reshape(self.sequencelength)
        # cond = data[1 + self.X_FEATURES + self.DOY_FEATURES:1 + self.X_FEATURES + self.DOY_FEATURES + self.COND_FEATURES].reshape(8, 3)
        scl = data[1 + self.X_FEATURES + self.DOY_FEATURES + self.COND_FEATURES:].reshape(self.sequencelength)#(75,)

        valid_timestep_mask = (scl != 3) & (scl != 8) & (scl != 9) & (scl != 10)#(75,)
        x[x == 0] = np.nan
        x_hat = x.copy()
        obs_per_timestep = np.any(~np.isnan(x_hat), axis=1)#(75,)是观测值则为 True
        candidate_timestep_mask = valid_timestep_mask & obs_per_timestep#(75,) 是观测值且不是云 和shadow则为 True
        candidate_timestep_idx = np.where(candidate_timestep_mask)[0]#（20，）是观测值且不是云 和shadow的索引
        
        masked_rows = np.zeros(self.sequencelength, dtype=bool)#(75,)
        if candidate_timestep_idx.size > 0:#进行人工掩码
            num_mask_ts = int(round(candidate_timestep_idx.size * 0.2))
            if num_mask_ts > 0:
                chosen_ts = np.random.choice(candidate_timestep_idx, num_mask_ts, replace=False)#需要进行人工掩码的索引
                x_hat[chosen_ts, :] = np.nan
                masked_rows[chosen_ts] = True

        valid_positions = valid_timestep_mask[:, None]#(75, 1)
        missing_mask = ((~np.isnan(x_hat)) & valid_positions).astype(np.float32)#(75, 10) x_hat(实际要输入的时序）中对应位置有观测值且不是云 和shadow,则对应位置为 True
        indicating_mask_imputation = ((~np.isnan(x)) & valid_positions & np.isnan(x_hat))
        
        # 噪声添加的有效位置：原始有效观测且非云/阴影，且未被人工掩码
        valid_for_noise_mask = candidate_timestep_mask & (~masked_rows)
        
        x = np.nan_to_num(x)#(75, 10)
        x_hat = np.nan_to_num(x_hat)#(75, 10)
        cond = None

        x = self.transform(x, cond)
        x_hat = self.transform(x_hat, cond)

        # 添加噪声并获取噪声掩码
        x_hat, noise_mask = self.add_noise(x_hat, valid_positions=valid_for_noise_mask)
        
        # 合并指示掩码：人工掩码位置 OR 噪声添加位置
        indicating_mask = (indicating_mask_imputation | noise_mask).astype(np.float32)

        sample = (
            torch.tensor(index),
            torch.from_numpy(x_hat.astype("float32")),
            torch.from_numpy(missing_mask.astype("float32")),
            torch.from_numpy(x.astype("float32")),
            torch.from_numpy(indicating_mask.astype("float32")),
            torch.from_numpy(doy).type(torch.LongTensor)
        )
        return sample

    def close_mmaps(self):
        """释放memmap引用"""
        if hasattr(self, "_memmaps"):
            for mm in self._memmaps:
                try:
                    del mm
                except Exception:
                    pass
            self._memmaps = []

    def __del__(self):
        self.close_mmaps()

    def target_transform(self, y):
        return torch.tensor(y, dtype=torch.long)
    
    def transform(self, x, cond):
        # 生长季内模拟掩蔽训练（在云数据处理之前进行）
        x = x * 1e-4
        # if hasattr(self, 'growing_season_masking') and self.growing_season_masking:
            # x, doy, cond = self._apply_growing_season_masking_in_transform(x, doy, cond)

        # if self.need_weight:
            # weight = getWeight(x)

        x = (x - self.mean) / self.std
        # if cond is not None:
        #     cond = (cond - self.mean_c) / self.std_c

        # mask = np.ones((x.shape[0],), dtype=int)

        # if self.need_weight:
            # return torch.from_numpy(x).type(torch.FloatTensor), torch.from_numpy(mask==0), torch.from_numpy(doy).type(torch.LongTensor), torch.from_numpy(cond).type(torch.FloatTensor), weight
        # else:
            # return torch.from_numpy(x).type(torch.FloatTensor), torch.from_numpy(mask==0), torch.from_numpy(doy).type(torch.LongTensor), torch.from_numpy(cond).type(torch.FloatTensor)
        return x

    def _apply_growing_season_masking_in_transform(self, x, doy, cloud_doy, cond=None):
        """在transform方法中应用生长季内模拟掩蔽训练"""
        # 随机决定是否应用掩蔽
        if np.random.random() > self.masking_probability:
            return x, doy, cond

        # 确保有足够的观测数据
        valid_mapping_mask = doy >= self.min_mapping_doy
        if not np.any(valid_mapping_mask):
            return x, doy, cond

        # 在指定范围内随机选择制图日期
        max_mapping_doy = max(270, doy.max())  # 不超过实际观测的最大日期
        cutoff_doy = np.random.uniform(self.min_mapping_doy, max_mapping_doy)

        # 创建保留数据的掩码（基于观测数据doy）
        keep_mask = doy <= cutoff_doy
        keep_indices = np.where(keep_mask)[0]

        # 应用掩蔽：只保留选定日期之前（包含当天）的观测
        x_masked = x[keep_indices]
        doy_masked = doy[keep_indices]

        # 处理气候数据
        cond_masked = cond
        if cond is not None:
            # 根据cutoff_doy确定应该保留哪些月份的气候数据
            # 气候数据是1-8月，对应索引0-7
            # cutoff_month = int(cutoff_doy // 30.44) + 1  # 粗略估算月份（30.44天/月）
            
            # 更精确的月份计算：基于一年中的天数
            if cutoff_doy <= 59:  # 2月
                cutoff_month = 1
            elif cutoff_doy <= 90:  # 3月
                cutoff_month = 2
            elif cutoff_doy <= 120:  # 4月
                cutoff_month = 3
            elif cutoff_doy <= 151:  # 5月
                cutoff_month = 4
            elif cutoff_doy <= 181:  # 6月
                cutoff_month = 5
            elif cutoff_doy <= 212:  # 7月
                cutoff_month = 6
            elif cutoff_doy <= 243:  # 8月
                cutoff_month = 7
            else:  # 8月及以后
                cutoff_month = 8
            
            # 确保cutoff_month在有效范围内（1-8月）
            # cutoff_month = min(max(cutoff_month, 1), 8)
            
            # 创建气候数据掩码：保留cutoff_month及之前的月份
            # cond的形状应该是 (8, 3) 或 (8,) 等，第一维是月份
            if len(cond.shape) == 2:  # (8, features)
                cond_masked = cond[:cutoff_month].copy()
                # 如果需要保持原始维度，可以用零填充剩余月份
                if cond_masked.shape[0] < 8:
                    padding_shape = (8 - cond_masked.shape[0], cond.shape[1])
                    padding = np.zeros(padding_shape, dtype=cond.dtype)
                    cond_masked = np.concatenate([cond_masked, padding], axis=0)
            elif len(cond.shape) == 1:  # (8,)
                cond_masked = cond[:cutoff_month].copy()
                # 如果需要保持原始维度，可以用零填充剩余月份
                if cond_masked.shape[0] < 8:
                    padding = np.zeros(8 - cond_masked.shape[0], dtype=cond.dtype)
                    cond_masked = np.concatenate([cond_masked, padding])

        return x_masked, doy_masked, cond_masked

# 创建数据加载器，支持对比学习（返回两个增强版本的样本）
def create_contrastive_dataloader(args, gpus, phase='train', augment=True):
    dataset_cls = CropAttriMappingDatasetBin
    # train_dataaug =transforms.Compose([
    #     # DataAugmentation(config),
    #     DataAugmentation2(config),
    # ])
    if phase == 'train':
        dataset = dataset_cls(
            # config,
            'test',#TODO: to change 
            2021, 
            Path(args.dataset_path)/'EU-dataset',
            # dataaug=DataAugmentation(config) if augment else None,
            None,
            # dataaug=None, 
        )
        batch_size = 128 * len(gpus)
        shuffle = True
    elif phase == 'val':
        #TODO: to change
        dataset = dataset_cls(
            config,
            'val', 
            config.DATASET.VAL_YEAR, 
            config.DATASET.ROOT,
            dataaug=DataAugmentation(config) if augment else None, #TODO: to change
        )
        # dataset = CropAttriMappingDataset7(
        #     'test', 
        #     config.DATASET.VAL_YEAR, 
        #     config.DATASET.TEST_ROOT,
        #     dataaug=None, 
        # )
        batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(gpus)
        shuffle = False

    def seed_worker(worker_id):
        worker_seed = 50 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        # 若子进程中使用GPU，需设置CUDA种子，但通常数据加载在CPU完成，可省略

    num_tasks = misc.get_world_size()#1 当前参与训练的总进程数
    global_rank = misc.get_rank()#0 当前进程的唯一标识符（排名）
    sampler = DistributedSampler(dataset,num_replicas=num_tasks, rank=global_rank, shuffle=shuffle, seed=50)
    
    dataloader = DataLoader(
        dataset,sampler=sampler,
        batch_size=batch_size,
        shuffle=False,#使用DistributedSampler时必须设置shuffle=False，否则会与sampler的shuffle逻辑冲突‌
        # generator=torch.Generator().manual_seed(config.SEED),
        num_workers=4,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=(phase == 'train'),
        persistent_workers=True,
    )
    
    return dataloader, dataset

def validate(model, val_iter, summary_writer, training_controller, logger):
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    (
        total_loss_collector,
        imputation_loss_collector,
        reconstruction_loss_collector,
        reconstruction_MAE_collector,
    ) = ([], [], [], [])

    with torch.no_grad():
        for idx, data in enumerate(val_iter):
            inputs, results = model_processing(data, model, "val")
            evalX_collector.append(inputs["X_holdout"])
            evalMask_collector.append(inputs["indicating_mask"])
            imputations_collector.append(results["imputed_data"])

            total_loss_collector.append(results["total_loss"].data.cpu().numpy())
            reconstruction_MAE_collector.append(
                results["reconstruction_MAE"].data.cpu().numpy()
            )
            reconstruction_loss_collector.append(
                results["reconstruction_loss"].data.cpu().numpy()
            )
            imputation_loss_collector.append(
                results["imputation_loss"].data.cpu().numpy()
            )

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(
            imputations_collector, evalX_collector, evalMask_collector
        )
    info_dict = {
        "total_loss": np.asarray(total_loss_collector).mean(),
        "reconstruction_loss": np.asarray(reconstruction_loss_collector).mean(),
        "imputation_loss": np.asarray(imputation_loss_collector).mean(),
        "reconstruction_MAE": np.asarray(reconstruction_MAE_collector).mean(),
        "imputation_MAE": imputation_MAE.cpu().numpy().mean(),
    }
    state_dict = training_controller("val", info_dict, logger)
    summary_write_into_tb(summary_writer, info_dict, state_dict["val_step"], "val")
    if args.param_searching_mode:#False
        nni.report_intermediate_result(info_dict["imputation_MAE"])
        if args.final_epoch or state_dict["should_stop"]:
            nni.report_final_result(state_dict["best_imputation_MAE"])

    if (
        state_dict["save_model"] and args.model_saving_strategy
    ) or args.model_saving_strategy == "all":
        saving_path = os.path.join(
            args.model_saving,
            "model_trainStep_{}_valStep_{}_imputationMAE_{:.4f}".format(
                state_dict["train_step"],
                state_dict["val_step"],
                info_dict["imputation_MAE"],
            ),
        )
        save_model(model, optimizer, state_dict, args, saving_path)
        logger.info(f"Saved model -> {saving_path}")
    return state_dict


def test_trained_model(model, test_dataloader):
    logger.info(f"Start evaluating on whole test set...")
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            inputs, results = model_processing(data, model, "test")
            # collect X_holdout, indicating_mask and imputed data
            evalX_collector.append(inputs["X_holdout"])
            evalMask_collector.append(inputs["indicating_mask"])
            imputations_collector.append(results["imputed_data"])

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(
            imputations_collector, evalX_collector, evalMask_collector
        )
        imputation_RMSE = masked_rmse_cal(
            imputations_collector, evalX_collector, evalMask_collector
        )
        imputation_MRE = masked_mre_cal(
            imputations_collector, evalX_collector, evalMask_collector
        )

    assessment_metrics = {
        "imputation_MAE on the test set": imputation_MAE,
        "imputation_RMSE on the test set": imputation_RMSE,
        "imputation_MRE on the test set": imputation_MRE,
        "trainable parameter num": args.total_params,
    }
    with open(
        os.path.join(args.result_saving_path, "overall_performance_metrics.out"), "w"
    ) as f:
        logger.info("Overall performance metrics are listed as follows:")
        for k, v in assessment_metrics.items():
            logger.info(f"{k}: {v}")
            f.write(k + ":" + str(v))
            f.write("\n")


def impute_all_missing_data(model, train_data, val_data, test_data):
    logger.info(f"Start imputing all missing data in all train/val/test sets...")
    model.eval()
    imputed_data_dict = {}
    with torch.no_grad():
        for dataloader, set_name in zip(
            [train_data, val_data, test_data], ["train", "val", "test"]
        ):
            indices_collector, imputations_collector = [], []
            for idx, data in enumerate(dataloader):
                if args.model_type in ["BRITS", "MRNN"]:
                    (
                        indices,
                        X,
                        missing_mask,
                        deltas,
                        back_X,
                        back_missing_mask,
                        back_deltas,
                    ) = map(lambda x: x.to(args.device), data)
                    inputs = {
                        "indices": indices,
                        "forward": {
                            "X": X,
                            "missing_mask": missing_mask,
                            "deltas": deltas,
                        },
                        "backward": {
                            "X": back_X,
                            "missing_mask": back_missing_mask,
                            "deltas": back_deltas,
                        },
                    }
                else:  # then for self-attention based models, i.e. Transformer/SAITS
                    indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                    inputs = {"indices": indices, "X": X, "missing_mask": missing_mask}
                imputed_data, _ = model.impute(inputs)
                indices_collector.append(indices)
                imputations_collector.append(imputed_data)

            indices_collector = torch.cat(indices_collector)
            indices = indices_collector.cpu().numpy().reshape(-1)
            imputations_collector = torch.cat(imputations_collector)
            imputations = imputations_collector.data.cpu().numpy()
            ordered = imputations[np.argsort(indices)]  # to ensure the order of samples
            imputed_data_dict[set_name] = ordered

    imputation_saving_path = os.path.join(args.result_saving_path, "imputations.h5")
    with h5py.File(imputation_saving_path, "w") as hf:
        hf.create_dataset("imputed_train_set", data=imputed_data_dict["train"])
        hf.create_dataset("imputed_val_set", data=imputed_data_dict["val"])
        hf.create_dataset("imputed_test_set", data=imputed_data_dict["test"])
    logger.info(f"Done saving all imputed data into {imputation_saving_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path of config file")
    parser.add_argument(
        "--test_mode",
        dest="test_mode",
        action="store_true",
        help="test mode to test saved model",
    )
    parser.add_argument(
        "--param_searching_mode",
        dest="param_searching_mode",
        action="store_true",
        help="use NNI to help search hyper parameters",
    )
    args = parser.parse_args()
    assert os.path.exists(
        args.config_path
    ), f'Given config file "{args.config_path}" does not exists'
    # load settings from config file
    cfg = ConfigParser(interpolation=ExtendedInterpolation())#就当是默认写法吧，这是一种更强大、更直观的语法（类似于 zc.buildout ），允许你在配置文件中使用 ${section:option} 的格式来引用其他配置项的值。
    cfg.read(args.config_path)
    args = read_arguments(args, cfg)

    if args.model_type in ["Transformer", "SAITS", "SAITS_for_CACM"]:  #True if SA-based model
        args.input_with_mask = cfg.getboolean("model", "input_with_mask")#True
        args.n_groups = cfg.getint("model", "n_groups")#5
        args.n_group_inner_layers = cfg.getint("model", "n_group_inner_layers")#1
        args.param_sharing_strategy = cfg.get("model", "param_sharing_strategy")#'inner_group'
        assert args.param_sharing_strategy in [
            "inner_group",
            "between_group",
        ], 'only "inner_group"/"between_group" sharing'
        args.d_model = cfg.getint("model", "d_model")#256
        args.d_inner = cfg.getint("model", "d_inner")#512
        args.n_head = cfg.getint("model", "n_head")#8
        args.d_k = cfg.getint("model", "d_k")#32
        args.d_v = cfg.getint("model", "d_v")#32
        args.dropout = cfg.getfloat("model", "dropout")#0.0
        args.diagonal_attention_mask = cfg.getboolean(
            "model", "diagonal_attention_mask"
        )#True

        dict_args = vars(args)
        if args.param_searching_mode:#False
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f"{args.model_name}/{experiment_id}/{trial_id}"
            dict_args["d_k"] = dict_args["d_model"] // dict_args["n_head"]
        model_args = {
            "device": args.device,
            "MIT": args.MIT,
            # imputer args
            "n_groups": dict_args["n_groups"],
            "n_group_inner_layers": args.n_group_inner_layers,
            "d_time": args.seq_len,
            "d_feature": args.feature_num,
            "dropout": dict_args["dropout"],
            "d_model": dict_args["d_model"],
            "d_inner": dict_args["d_inner"],
            "n_head": dict_args["n_head"],
            "d_k": dict_args["d_k"],
            "d_v": dict_args["d_v"],
            "input_with_mask": args.input_with_mask,
            "diagonal_attention_mask": args.diagonal_attention_mask,
            "param_sharing_strategy": args.param_sharing_strategy,
        }
    elif args.model_type in ["BRITS", "MRNN"]:  # if RNN-based model
        if args.model_type == "BRITS":
            args.consistency_loss_weight = cfg.getfloat(
                "training", "consistency_loss_weight"
            )
        args.rnn_hidden_size = cfg.getint("model", "rnn_hidden_size")

        dict_args = vars(args)
        if args.param_searching_mode:
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f"{args.model_name}/{experiment_id}/{trial_id}"
        model_args = {
            "device": args.device,
            "MIT": args.MIT,
            # imputer args
            "seq_len": args.seq_len,
            "feature_num": args.feature_num,
            "rnn_hidden_size": dict_args["rnn_hidden_size"],
        }
    else:
        assert (
            ValueError
        ), f"Given model_type {args.model_type} is not in {MODEL_DICT.keys()}"

    # parameter insurance
    assert args.model_saving_strategy.lower() in [
        "all",
        "best",
        "none",
    ], "model saving strategy must be all/best/none"
    if args.model_saving_strategy.lower() == "none":#False
        args.model_saving_strategy = False
    assert (
        args.optimizer_type in OPTIMIZER.keys()
    ), f"optimizer type should be in {OPTIMIZER.keys()}, but get{args.optimizer_type}"
    assert args.device in ["cpu", "cuda"], "device should be cpu or cuda"

    time_now = datetime.now().__format__("%Y-%m-%d_T%H:%M:%S")
    args.model_saving, args.log_saving = check_saving_dir_for_model(args, time_now)
    logger = setup_logger(args.log_saving + "_" + time_now, "w")
    logger.info(f"args: {args}")
    logger.info(f"Config file path: {args.config_path}")
    logger.info(f"Model name: {args.model_name}")#PhysioNet2012_SAITS_best

    gpus = [0]


    # train_dataloader, train_dataset = create_contrastive_dataloader(
    #     args, gpus, phase='train', augment=False)

    # val_dataloader, val_dataset = create_contrastive_dataloader(
    #     args, gpus, phase='train', augment=False)
    # 手动创建完整数据集（复用 create_contrastive_dataloader 中 phase='train' 的逻辑）
    full_dataset = CropAttriMappingDatasetBin(
        'train', # 保持原代码中的 'test' 参数 (尽管是训练阶段)
        2019, 
        Path(args.dataset_path)/'US-dataset',
        None,
    )

    # 计算拆分长度 (80% 训练, 20% 验证)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    
    # 拆分数据集
    # 设置种子以保证可复现性
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=generator)

    def create_loader(dataset, shuffle):
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        # DistributedSampler 可以正常处理 Subset (它会使用 len(subset))
        sampler = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle, seed=50)
        
        def seed_worker(worker_id):
            worker_seed = 50 + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size, # Use batch_size from args
            shuffle=False, # Must be False with DistributedSampler
            num_workers=args.num_workers, # Use num_workers from args
            worker_init_fn=seed_worker,
            pin_memory=True,
            drop_last=(shuffle is True), # 训练集丢弃最后不完整的 batch，验证集保留
            persistent_workers=True,
        )

    train_dataloader = create_loader(train_dataset, shuffle=True)
    val_dataloader = create_loader(val_dataset, shuffle=False)

    # unified_dataloader = UnifiedDataLoader(
    #     args.dataset_path,
    #     args.seq_len,
    #     args.feature_num,
    #     args.model_type,
    #     args.batch_size,
    #     args.num_workers,
    #     args.MIT,
    # )
    model = MODEL_DICT[args.model_type](**model_args)#'SAITS'
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num of total trainable params is: {args.total_params}")

    # if utilize GPU and GPU available, then move
    if "cuda" in args.device and torch.cuda.is_available():
        model = model.to(args.device)

    if args.test_mode:
        logger.info("Entering testing mode...")
        args.model_path = cfg.get("test", "model_path")
        args.save_imputations = cfg.getboolean("test", "save_imputations")
        args.result_saving_path = cfg.get("test", "result_saving_path")
        os.makedirs(args.result_saving_path) if not os.path.exists(
            args.result_saving_path
        ) else None
        model = load_model(model, args.model_path, logger)
        test_dataloader = unified_dataloader.get_test_dataloader()
        test_trained_model(model, test_dataloader)
        if args.save_imputations:
            (
                train_data,
                val_data,
                test_data,
            ) = unified_dataloader.prepare_all_data_for_imputation()
            impute_all_missing_data(model, train_data, val_data, test_data)
    else:  # in the training mode
        logger.info(f"Creating {args.optimizer_type} optimizer...")

        optimizer = OPTIMIZER[args.optimizer_type](
            model.parameters(), lr=dict_args["lr"], weight_decay=args.weight_decay
        )#adam
        logger.info("Entering training mode...")
        # train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()
        training_controller = Controller(args.early_stop_patience)

        train_set_size = train_dataloader.sampler.total_size
        logger.info(#train set len is 2557, batch size is 128,so each epoch has 20 steps
            f"train set len is {train_set_size}, batch size is {args.batch_size},"
            f"so each epoch has {math.ceil(train_set_size / args.batch_size)} steps"
        )

        tb_summary_writer = SummaryWriter(
            os.path.join(args.log_saving, "tensorboard_" + time_now)
        )
        #Python 3.11+ 引入了Zero-cost exception handling（零开销异常处理） 机制，进入 try 块没有任何运行时开销。
        #  编译器会生成一个静态跳转表，只有当异常真正发生时，解释器才会去查表
        try:
            train(
                model,
                optimizer,
                train_dataloader,
                val_dataloader,
                tb_summary_writer,
                training_controller,
                logger,
            )
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt. Exiting...")
        finally:
            logger.info("Cleaning up shared memory...")
            # 优先清理 full_dataset (涵盖了 train 和 val 的所有数据)
            if 'full_dataset' in locals() and full_dataset is not None:
                 if hasattr(full_dataset, 'close_shm'):
                    full_dataset.close_shm()
            
            # 兼容旧逻辑或处理未定义 full_dataset 的情况
            if 'train_dataset' in locals() and train_dataset is not None:
                # 如果是 Subset，尝试清理底层的 dataset (前提是 full_dataset 没被清理过)
                if hasattr(train_dataset, 'close_shm'):
                    train_dataset.close_shm()
                elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'close_shm'):
                     if 'full_dataset' not in locals():
                        train_dataset.dataset.close_shm()

            if 'val_dataset' in locals() and val_dataset is not None:
                 if hasattr(val_dataset, 'close_shm'):
                    val_dataset.close_shm()
                 # 验证集通常共享训练集的底层数据，无需重复清理底层 dataset
                 
            logger.info("SharedMemory cleanup done.")

    logger.info("All Done.")
