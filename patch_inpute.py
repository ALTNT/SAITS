#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch-based Crop Mapping Script
基于patch的作物制图脚本

数据结构:
- S2数据: fugou_S2_2020/fugou_2020_{indice}-{x_coord}-{y_coord}.tif
- Terra数据: 2020_fugou_Terra_1-9/
- WorldCereal数据: fugou_WorldCereal_2020/
"""

import os
# from signal import valid_signals
import sys
import gc
import logging
import argparse
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import warnings
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import torch
import torch.nn.parallel

from modeling.saits_for_CACM import SAITS_for_CACM
from modeling.utils import load_model

# 在文件开头添加导入
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
# from config import get_active_config, ALL_REGIONS
# site_start_doy = 120
# site_end_doy = 330
site_start_doy = 90
site_end_doy = 300

mean2= np.array([0.33384848099779335, 0.33743598254332613, 0.3313643807274545, 0.3738025112940193, 0.43651407997201386, 0.46520171460177606, 0.47281375925359626, 0.4763394446147105, 0.3112739897772981, 0.24385958955982265])
std2 = np.array([0.36185714564075394, 0.3371488493517183, 0.33266536803207, 0.33175146328580024, 0.2805921329391628, 0.2656709362852829, 0.2702308822869307, 0.25237716313205766, 0.1477668850798491, 0.13772583877397532])

mean_c=np.array([78.02576446969697, 198.9981634469697, 1850.3222502462122])
std_c = np.array([120.325428898157, 124.14974885223911, 680.851609868019])

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatchBasedMapping:
    """基于patch的作物制图处理器"""
    
    def __init__(self, s2_dir, output_dir, config_path, model_path, device=None, file_glob=None, output_steps=None):
        self.s2_dir = Path(s2_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = str(config_path)
        self.model_path = str(model_path)

        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(self.config_path)

        self.sequencelength = cfg.getint("dataset", "seq_len")
        self.feature_num = cfg.getint("dataset", "feature_num")

        self.device = device if device is not None else cfg.get("training", "device")
        self.device = "cuda" if ("cuda" in self.device and torch.cuda.is_available()) else "cpu"

        self.mean = mean2
        self.std = std2

        self.file_glob = file_glob#"Indiana_2019_*-*-*.tif"
        self.output_steps = output_steps

        self.model = None
        logger.info('加载 SAITS_for_CACM 模型...')

        model_args = {
            "device": self.device,#"cuda"
            "MIT": cfg.getboolean("training", "MIT"),#True
            "n_groups": cfg.getint("model", "n_groups"),#6
            "n_group_inner_layers": cfg.getint("model", "n_group_inner_layers"),#1
            "d_time": self.sequencelength,#75
            "d_feature": self.feature_num,#10
            "dropout": cfg.getfloat("model", "dropout"),#0.0
            "d_model": cfg.getint("model", "d_model"),#256
            "d_inner": cfg.getint("model", "d_inner"),#512
            "n_head": cfg.getint("model", "n_head"),#8
            "d_k": cfg.getint("model", "d_k"),#32
            "d_v": cfg.getint("model", "d_v"),#32
            "input_with_mask": cfg.getboolean("model", "input_with_mask"),#False
            "diagonal_attention_mask": cfg.getboolean("model", "diagonal_attention_mask"),#True
            "param_sharing_strategy": cfg.get("model", "param_sharing_strategy"),#"inner_group"
        }

        self.model = SAITS_for_CACM(**model_args)
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        self.model = load_model(self.model, self.model_path, logger)
        self.model.eval()
        logger.info('模型加载完成')
    
    def get_patch_coordinates(self):
        """获取所有 patch 的坐标信息"""
        if not self.s2_dir.exists():
            logger.error(f"S2数据目录不存在: {self.s2_dir}")
            return {}

        pattern = self.file_glob if self.file_glob else "*.tif"
        s2_files = list(self.s2_dir.glob(pattern))

        if not s2_files:
            logger.warning(f"未找到S2文件: {self.s2_dir}")
            return {}

        patch_coords = defaultdict(list)

        for file_path in s2_files:
            # 解析文件名: fugou_2020_1-0000000000-0000000000.tif
            filename = file_path.stem
            parts = filename.split('-')
            if len(parts) < 3:
                continue

            try:
                x_coord = int(parts[-2])
                y_coord = int(parts[-1])
                indice_part = parts[0].split('_')[-1]
                indice = int(indice_part)
            except ValueError as e:
                logger.warning(f"解析文件名失败: {filename}, 错误: {e}")
                continue

            patch_coords[(x_coord, y_coord)].append({
                "file": file_path,
                "indice": indice,
            })

        logger.info(f"找到 {len(patch_coords)} 个patch坐标")
        return dict(patch_coords)
    
    def load_climate_data(self, coord):
        """
        加载气候数据
        
        Args:
            coord: 坐标
        
        Returns:
            numpy.ndarray or None: 气候数据数组
        """
        # 构建气候数据文件路径
        x_coord, y_coord = coord
        climate_file = self.terra_dir / f"fugou_{self.year}_terra-{x_coord:010d}-{y_coord:010d}.tif"
        
        if climate_file.exists():
            try:
                with rasterio.open(climate_file) as src:
                    climate_data = src.read()  # 读取所有波段#(112, 512, 512)
                    logger.debug(f"成功读取气候数据: {climate_file}")
                    return climate_data
            except Exception as e:
                logger.error(f"读取气候数据失败: {climate_file}, 错误: {e}")
                return None
        else:
            logger.warning(f"气候数据文件不存在: {climate_file}")
            return None
    
    def load_patch_timeseries(self, coord, file_info_list):
        """
        加载单个patch的时间序列数据
        
        Args:
            coord: (x_coord, y_coord)
            file_info_list: [{'file': path, 'indice': int}]
        
        Returns:
            dict: patch数据
        """
        x_coord, y_coord = coord
        
        # 按月份排序
        file_info_list.sort(key=lambda x: x['indice'])
        
        timeseries_data = []
        indices = []
        doy_list = []
        
        # 读取时间序列数据
        for info in file_info_list:
            try:
                with rasterio.open(info['file']) as src:
                    # 读取所有波段
                    data = src.read()  # (bands, height, width)
                    
                    # 第一个文件时保存地理信息
                    if not timeseries_data:
                        transform = src.transform
                        crs = src.crs
                        height, width = data.shape[1], data.shape[2]
                        profile = src.profile.copy()
                        dtype = src.dtypes[0]
                        band_count = src.count
                        profile = src.profile.copy()
                        dtype = src.dtypes[0]
                        band_count = src.count
                        profile = src.profile.copy()
                        dtype = src.dtypes[0]
                        band_count = src.count
                        profile = src.profile.copy()
                        dtype = src.dtypes[0]
                        band_count = src.count
                        profile = src.profile.copy()
                        dtype = src.dtypes[0]
                        band_count = src.count
                        profile = src.profile.copy()
                        dtype = src.dtypes[0]
                        band_count = src.count
                        profile = src.profile.copy()
                        dtype = src.dtypes[0]
                        band_count = src.count
                    
                    unique_doys = np.unique(data[-1, data[-1]!=0])#array([92], dtype=uint16)
                    if len(unique_doys) == 0:
                        # logger.warning(f"{info['file']} 无法提取DOY")
                        continue
                    doy = unique_doys[0]#TODO: 不用+1，在GEE 脚本中已经+1了
                    
                    timeseries_data.append(data)
                    indices.append(info['indice'])
                    doy_list.append(doy)
                    
            except Exception as e:
                logger.warning(f"读取文件失败 {info['file']}: {e}")
                return None
        
        if not timeseries_data:
            return None
        
        # 堆叠时间序列 (time, bands, height, width)
        timeseries_array = np.stack(timeseries_data, axis=0)#(83, 13, 327, 219)
        
        return {
            'coord': coord,
            'timeseries': timeseries_array,
            'indices': np.array(indices),
            'doy_list': doy_list,
            'transform': transform,
            'crs': crs,
            'height': height,
            'width': width,
            'profile': profile,
            'dtype': dtype,
            'band_count': band_count,
            'file_info_list': file_info_list,
        }
    
    # def extract_samples_from_patch(self, patch_data, climate_data=None, pixel_coord=None):
    #     """
    #     从patch中提取样本进行制图

    #     Args:
    #         patch_data: patch数据字典

    #     Returns:
    #         list: 样本列表
    #     """
    #     s2_data = patch_data['timeseries']  # (T, C, H, W)#(42, 13, 512, 512)
    #     indices = patch_data['indices']##(42,)当前patch 有图像的索引 array([ 1,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66,68, 70, 72, 74, 75, 77, 79, 81])
    #     doy_list = patch_data['doy_list']#len()=42 [92, 97, 102, 107, 112, 117, 122, 127, 132, 137, 142, 147, 152, 157, 162, 167, 172, 177, 182, 187, 192, 197, 202, 207, 212, 217, 222, 227, 232, 237, 242, 247, 252, 257, 262, 267, 272, 277, 282, 287, 292, 297]
    #     T, C, H, W = s2_data.shape

    #     samples = []
        
    #     if True: # for OurNet
    #         #过滤出 120到 270 天日期的样本
    #         doy_list_mask = (site_start_doy+30 <= np.array(doy_list)) & (np.array(doy_list) <= site_end_doy-30)
    #         s2_data = s2_data[doy_list_mask]#(30, 13, 512, 512)
    #         T, C, H, W = s2_data.shape
    #         indices = indices[doy_list_mask]#(T,)
    #         doy_list = np.array(doy_list)[doy_list_mask]#(T,)
    #         del doy_list_mask
            
    #         valid_doy_valid = s2_data[:,-2,:,:]#(T, 512, 512)
    #         valid_doy_valid = valid_doy_valid.transpose(1, 2, 0)#(H,W,T)(512, 512, 30)
    #         s2_data = s2_data[:,:-3,:,:]*1e-4#(T, C, H, W)#(30, 10, 512, 512)
    #         s2_data_masked = s2_data.transpose(0, 2, 3, 1)#(T, H, W, C)
    #         # s2_data_masked = s2_data.astype(np.float32).transpose(0, 2, 3, 1)#(T, H, W, C)
    #         del s2_data
    #         gc.collect()

    #         t, h, w, c = s2_data_masked.shape

    #         for row in range(H):
    #             for col in range(W):
    #                 # 提取像素的时间序列 (T, C)
    #                 pixel_ts = s2_data_masked[:, row, col,:]  # (T, C)
    #                 doy_valid = valid_doy_valid[row, col]#(T,)
    #                 # 获取有效的时间点
    #                 valid_indices = np.where(doy_valid == 1)[0]
    #                 pixel_ts = pixel_ts[doy_valid == 1]
    #                 # 只选择有效时间点对应的doy值
    #                 valid_doy_list = doy_list[valid_indices]

    #                 x_pad = np.zeros((self.sequencelength, c))
    #                 x_pad[:pixel_ts.shape[0], :] = pixel_ts
    #                 x_pad = (x_pad - self.mean) / self.std
    #                 doy_pad = np.zeros((self.sequencelength,), dtype=int)
    #                 doy_pad[:pixel_ts.shape[0]] = valid_doy_list  # 使用筛选后的doy列表
    #                 mask = np.zeros((self.sequencelength,), dtype=int)
    #                 mask[:pixel_ts.shape[0]] = 1

    #                 sample = {
    #                     'row': row,
    #                     'col': col,
    #                     's2_timeseries': x_pad,  # (valid_T, C)
    #                     'mask': mask==0,
    #                     'doy_timeseries': doy_pad,
    #                     'sequence_length': self.sequencelength,
    #                     'indices': indices,
    #                     'coord': patch_data['coord']
    #                 }

    #                 samples.append(sample)
    #     else:   #for CACM
    #         s2_data = s2_data[:,:-3,:,:]*1e-4#(T, C, H, W)
    #         s2_data = s2_data.transpose(0, 2, 3, 1)#(T, H, W, C)
    #         region_config = ALL_REGIONS.get(self.site, {})
    #         s2_data = self.create_15day_composite_and_interpolate(s2_data, doy_list, region_config)#(10, 512, 512, 10)
            
    #         # 添加插值后NDVI曲线绘制功能
    #         if pixel_coord is not None:
    #             self.plot_interpolated_ndvi_curve(s2_data, pixel_coord, self.output_dir)
            
    #         gc.collect()
    #         climate_data = climate_data.transpose(1, 2, 0)#(512, 512, 112)
    #         climate_data = self.process_climate_data(climate_data, self.climate_time_steps)
    #         # 提取所有有效样本
    #         x_samples, cond_samples, spatial_indices = self.extract_all_samples_for_mapping(
    #             s2_data, climate_data
    #         )
    #         # 标准化 - 修正维度问题
    #         x_samples = (x_samples - self.mean) / self.std
    #         cond_samples = (cond_samples - self.mean_c) / self.std_c
    #         # 创建坐标到样本索引的映射
    #         coord_to_index = {}
    #         for i, (row, col) in enumerate(spatial_indices):
    #             coord_to_index[(row, col)] = i
            
    #         # 遍历所有像素
    #         for row in range(H):
    #             for col in range(W):
    #                 # 检查该坐标是否在有效样本中
    #                 if (row, col) in coord_to_index:
    #                     idx = coord_to_index[(row, col)]
    #                     pixel_ts = x_samples[idx]  # 获取对应的时间序列
    #                     cond_ts = cond_samples[idx]  # 获取对应的条件时间序列
                        
    #                     # 创建样本
    #                     sample = {
    #                         'row': row,
    #                         'col': col,
    #                         's2_timeseries': pixel_ts,  # (C, T)
    #                         'cond_timeseries': cond_ts,  # (features, T)
    #                         'indices': indices,
    #                         'coord': patch_data['coord']
    #                     }
                        
    #                     samples.append(sample)
        
    #     logger.info(f"从patch {patch_data['coord']} 提取了 {len(samples)} ({H}x{W})个样本")
    #     return samples
    
    # 确保标准化参数形状正确
    def _prepare_normalization_params(self):
        """预处理标准化参数，确保形状正确"""
        # 处理遥感数据标准化参数
        if isinstance(self.mean, np.ndarray):
            if self.mean.ndim > 1:
                self.mean = self.mean.squeeze()
            self.mean = self.mean.reshape(-1, 1)  # (10, 1)
        
        if isinstance(self.std, np.ndarray):
            if self.std.ndim > 1:
                self.std = self.std.squeeze()
            self.std = self.std.reshape(-1, 1)  # (10, 1)
            # 防止除零
            self.std = np.maximum(self.std, 1e-8)
        
        # 处理气候数据标准化参数
        if isinstance(self.mean_c, np.ndarray):
            if self.mean_c.ndim > 1:
                self.mean_c = self.mean_c.squeeze()
            self.mean_c = self.mean_c.reshape(-1, 1)  # (3, 1)
        
        if isinstance(self.std_c, np.ndarray):
            if self.std_c.ndim > 1:
                self.std_c = self.std_c.squeeze()
            self.std_c = self.std_c.reshape(-1, 1)  # (3, 1)
            # 防止除零
            self.std_c = np.maximum(self.std_c, 1e-8)

    def predict_samples(self, samples):
        """
        对样本进行预测
        
        Args:
            samples: 样本列表
        
        Returns:
            list: 预测结果列表
        """

        

        
        predictions = []
        

        
        try:
            # 准备批量数据
            batch_size = 4096*10  # 可调整
            num_samples = len(samples)
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            logger.info(f'开始预测 {num_samples} 个样本，分 {num_batches} 批处理')
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    batch_samples = samples[start_idx:end_idx]
                    
                    if self.model_config.MODEL.MODEL_NAME == 'CACMNet':
                        # 提取时间序列和条件数据
                        input_data = []
                        cond_data = []
                        for sample in batch_samples:
                            input_data.append(sample['s2_timeseries'])
                            cond_data.append(sample['cond_timeseries'])
                        
                        # 转换为张量
                        input_tensor = torch.FloatTensor(np.array(input_data))
                        cond_tensor = torch.FloatTensor(np.array(cond_data))
                        
                        # 移至GPU
                        if torch.cuda.is_available():
                            input_tensor = input_tensor.cuda()
                            cond_tensor = cond_tensor.cuda()
                        
                        # 模型预测
                        outputs, _, _, _ = self.model(input_tensor, cond_tensor)
                    else:
                        x_pad = []
                        mask = []
                        doy_pad = []
                        cond = []
                        for sample in batch_samples:
                            x_pad.append(sample['s2_timeseries'])
                            mask.append(sample['mask'])
                            doy_pad.append(sample['doy_timeseries'])
                            cond.append(sample['cond_timeseries'])
                        
                        if self.model_config.MODEL.MODEL_NAME != 'CACMNet':
                            doy_pad = np.array(doy_pad) - 30#TODO: to modify 
                        # 转换为张量
                        x_pad = torch.FloatTensor(np.array(x_pad)).cuda()#torch.Size([40960, 30, 10])
                        mask = torch.from_numpy(np.array(mask)).cuda()#torch.Size([40960, 60])
                        doy_pad = torch.from_numpy(np.array(doy_pad)).type(torch.LongTensor).cuda()#torch.Size([40960, 60])
                        cond = torch.from_numpy(np.array(cond)).type(torch.FloatTensor).cuda()#torch.Size([40960, 8, 3])
                        X = x_pad,mask,doy_pad,cond
                        outputs,_,_,_,_ = self.model(X)

                    # 获取预测类别和置信度
                    probs = torch.softmax(outputs, dim=1)#torch.Size([40960, 5])
                    confidences, pred_classes = torch.max(probs, dim=1)#torch.Size([40960]) torch.Size([40960]) 最大概率值（confidences）和对应的类别索引（pred_classes）
                    
                    # 处理预测结果
                    for j, sample in enumerate(batch_samples):
                        pred_class = pred_classes[j].item()
                        confidence = confidences[j].item()
                        
                        prediction = {
                            'row': sample['row'],
                            'col': sample['col'],
                            'coord': sample['coord'],
                            'predicted_class': pred_class,
                            'confidence': confidence
                        }
                        
                        predictions.append(prediction)
            
            logger.info(f'预测完成，共 {len(predictions)} 个结果')
            return predictions
            
        except Exception as e:
            logger.error(f'预测过程中出错: {e}')
            import traceback
            traceback.print_exc()
            # 如果预测失败，使用随机预测作为备选
            return self._random_predict(samples)
    
    def _random_predict(self, samples):
        """随机预测（作为备选方案）"""
        logger.warning('使用随机预测作为备选方案')
        predictions = []
        
        for sample in samples:
            # 随机生成预测结果
            pred_class = np.random.randint(0, 5)  # 假设有5个类别
            confidence = np.random.random()
            
            prediction = {
                'row': sample['row'],
                'col': sample['col'],
                'coord': sample['coord'],
                'predicted_class': pred_class,
                'confidence': confidence
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def create_prediction_maps(self, predictions, patch_data):
        """
        将预测结果转换为空间图像
        
        Args:
            predictions: 预测结果列表
            patch_data: patch数据
        
        Returns:
            tuple: (prediction_map, confidence_map)
        """
        height = patch_data['height']
        width = patch_data['width']
        
        # 初始化预测图和置信度图
        pred_map = np.full((height, width), -1, dtype=np.int16)  # -1表示无预测
        conf_map = np.zeros((height, width), dtype=np.float32)
        
        # 填充预测结果
        for pred in predictions:
            row, col = pred['row'], pred['col']
            pred_map[row, col] = pred['predicted_class']
            conf_map[row, col] = pred['confidence']
        
        # 应用类别映射（如果需要）
        try:
            # 构建CSV路径 - 根据实际情况调整
            country = 'CN'  # 默认为中国，可以从配置中获取
            csv_path = Path("data")/f"{country}-dataset"/"classmapping.csv"
            
            if csv_path.exists():
                # 加载类别映射
                id_to_code = self.load_id_to_code_mapping(csv_path)
                # 应用类别映射
                pred_map = self.apply_id_to_code_mapping(pred_map, id_to_code)
                logger.info(f"已应用类别映射，从CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"应用类别映射时出错: {e}，使用原始预测结果")
        
        return pred_map, conf_map
    
    def load_id_to_code_mapping(self, csv_path):
        """从CSV文件加载id到code的映射关系"""
        df = pd.read_csv(csv_path)
        id_to_code = {}
        for _, row in df.iterrows():
            class_id = int(row['id'])
            code = int(row['code'])
            id_to_code[class_id] = code
        return id_to_code
    
    def apply_id_to_code_mapping(self, data, id_to_code):
        """安全地应用id到code的映射"""
        # 创建副本避免修改原数据
        result = np.copy(data)
        
        # 创建映射数组，处理所有可能的值
        max_val = max(max(id_to_code.keys()), np.max(data)) if len(id_to_code) > 0 else np.max(data)
        mapping_array = np.arange(max_val + 1, dtype=data.dtype)  # 使用与输入数据相同的数据类型
        
        # 应用CSV中定义的映射
        for id_val, code_val in id_to_code.items():
            if id_val <= max_val:
                mapping_array[id_val] = code_val
        
        # 一次性应用所有映射
        result = mapping_array[result]
        # 确保返回的数组类型与输入数据类型一致
        return result.astype(data.dtype)
        
    def save_patch_results(self, pred_map, conf_map, patch_data):
        """
        保存patch的预测结果
        
        Args:
            pred_map: 预测图
            conf_map: 置信度图
            patch_data: patch数据
        
        Returns:
            tuple: (prediction_file, confidence_file)
        """
        coord = patch_data['coord']
        transform = patch_data['transform']
        crs = patch_data['crs']
        
        # 创建输出文件名
        coord_str = f"{coord[0]:010d}-{coord[1]:010d}"
        pred_file = self.output_dir / f"patch_{coord_str}_prediction.tif"
        conf_file = self.output_dir / f"patch_{coord_str}_confidence.tif"
        
        # 保存预测图
        with rasterio.open(
            pred_file, 'w',
            driver='GTiff',
            height=pred_map.shape[0],
            width=pred_map.shape[1],
            count=1,
            dtype=pred_map.dtype,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(pred_map, 1)
        
        # 保存置信度图
        with rasterio.open(
            conf_file, 'w',
            driver='GTiff',
            height=conf_map.shape[0],
            width=conf_map.shape[1],
            count=1,
            dtype=conf_map.dtype,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(conf_map, 1)
        
        logger.info(f"预测结果已保存: {pred_file}, {conf_file}")
        return pred_file, conf_file
    
    def plot_pixel_ndvi_curve(self, patch_data, pixel_coord=None, output_dir=None):
        """
        绘制指定像素点的全年NDVI曲线
        
        Args:
            patch_data: patch数据字典
            pixel_coord: 像素坐标 (x, y)，如果为None则使用patch中心点
            output_dir: 输出目录，如果为None则显示图像
        """
        timeseries = patch_data['timeseries']  # (T, C, H, W)
        doy_list = patch_data['doy_list']
        coord = patch_data['coord']
        
        T, C, H, W = timeseries.shape
        
        # 如果没有指定像素坐标，使用patch中心点
        if pixel_coord is None:
            pixel_x, pixel_y = H // 2, W // 2
        else:
            pixel_x, pixel_y = pixel_coord
            
        # 检查坐标是否有效
        if pixel_x < 0 or pixel_x >= H or pixel_y < 0 or pixel_y >= W:
            logger.warning(f"像素坐标 ({pixel_x}, {pixel_y}) 超出范围 ({H}, {W})")
            return
            
        # 提取指定像素的时间序列数据
        pixel_timeseries = timeseries[:, :, pixel_x, pixel_y]  # (T, C)
        
        # 计算NDVI (假设红光波段在索引2，近红外波段在索引6)
        # 根据Sentinel-2波段顺序：B2(蓝), B3(绿), B4(红), B5(红边1), B6(红边2), B7(红边3), B8(近红外), B8A(窄近红外), B11(短波红外1), B12(短波红外2)
        red_band = pixel_timeseries[:, 2] * 1e-4  # B4 红光波段
        nir_band = pixel_timeseries[:, 6] * 1e-4  # B8 近红外波段

        threshold = 1e-8
        nir_band = np.where(np.abs(nir_band) < threshold, 0.0, red_band)#高版本的numpy在处理接近零的浮点数时，保留了更多的数值精度
        red_band = np.where(np.abs(red_band) < threshold, 0.0, red_band)
        
        # 计算NDVI
        ndvi_values = (nir_band - red_band) / (nir_band + red_band + 1e-8)
        
        # 过滤无效值
        valid_mask = (ndvi_values > 0) & (ndvi_values < 1) & (~np.isnan(ndvi_values))
        valid_doy = np.array(doy_list)[valid_mask]
        valid_ndvi = ndvi_values[valid_mask]
        
        if len(valid_ndvi) == 0:
            logger.warning(f"像素 ({pixel_x}, {pixel_y}) 没有有效的NDVI值")
            return
            
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制NDVI曲线
        plt.plot(valid_doy, valid_ndvi, 'o-', color='green', linewidth=2, markersize=4)
        
        # 添加DOY数值标注
        for i, (doy, ndvi) in enumerate(zip(valid_doy, valid_ndvi)):
            plt.annotate(f'{int(doy)}', (doy, ndvi), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8, alpha=0.7)
        
        plt.xlabel('Day of Year (DOY)')
        plt.ylabel('NDVI')
        plt.title(f'NDVI Time Series for Pixel ({pixel_x}, {pixel_y}) in Patch {coord}')
        plt.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        plt.xlim(80, 330)  # 设置横坐标范围
        plt.ylim(-0.2, 1.0)
        
        # 添加季节标记
        # plt.axvline(x=80, color='orange', linestyle='--', alpha=0.5, label='Spring')
        # plt.axvline(x=172, color='red', linestyle='--', alpha=0.5, label='Summer')
        # plt.axvline(x=266, color='brown', linestyle='--', alpha=0.5, label='Autumn')
        # plt.axvline(x=355, color='blue', linestyle='--', alpha=0.5, label='Winter')
        
        # plt.legend()
        plt.tight_layout()
        
        # 保存或显示图像
        if self.output_dir:
            filename = f"ndvi_curve_patch_{coord[0]}_{coord[1]}_pixel_{pixel_x}_{pixel_y}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"NDVI曲线已保存到: {filepath}")
        else:
            plt.show()
            
        plt.close()
    
    def plot_interpolated_ndvi_curve(self, interpolated_data, pixel_coord=None, output_dir=None):
        """
        绘制插值后时序数据的NDVI曲线
        
        Args:
            interpolated_data: 插值后的数据 (T, H, W, C) - (10, 512, 512, 10)
            pixel_coord: 像素坐标 (x, y)，如果为None则使用中心点
            output_dir: 输出目录，如果为None则显示图像
        """
        T, H, W, C = interpolated_data.shape
        
        # 如果没有指定像素坐标，使用中心点
        if pixel_coord is None:
            pixel_x, pixel_y = H // 2, W // 2
        else:
            pixel_x, pixel_y = pixel_coord
            
        # 检查坐标是否有效
        if pixel_x < 0 or pixel_x >= H or pixel_y < 0 or pixel_y >= W:
            logger.warning(f"像素坐标 ({pixel_x}, {pixel_y}) 超出范围 ({H}, {W})")
            return
            
        # 提取指定像素的时间序列数据
        pixel_timeseries = interpolated_data[:, pixel_x, pixel_y, :]  # (T, C)
        
        # 插值后的DOY列表 [120, 135, 150, 165, 180, 195, 210, 225, 240, 255]
        interpolated_doys = list(range(site_start_doy+30, site_end_doy-30-15+1, 15))
        
        # 计算NDVI (红光波段在索引2，近红外波段在索引6)
        red_band = pixel_timeseries[:, 2]  # B4 红光波段
        nir_band = pixel_timeseries[:, 6]  # B8 近红外波段
        threshold = 1e-8
        nir_band = np.where(np.abs(nir_band) < threshold, 0.0, red_band)#高版本的numpy在处理接近零的浮点数时，保留了更多的数值精度
        red_band = np.where(np.abs(red_band) < threshold, 0.0, red_band)
        
        # 计算NDVI
        ndvi_values = (nir_band - red_band) / (nir_band + red_band + 1e-8)
        
        # 过滤无效值
        valid_mask = (ndvi_values > 0) & (ndvi_values < 1) & (~np.isnan(ndvi_values))
        valid_doy = np.array(interpolated_doys)[valid_mask]
        valid_ndvi = ndvi_values[valid_mask]
        
        if len(valid_ndvi) == 0:
            logger.warning(f"像素 ({pixel_x}, {pixel_y}) 没有有效的插值NDVI值")
            return
            
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制插值后的NDVI曲线
        plt.plot(valid_doy, valid_ndvi, 'o-', color='blue', linewidth=2, markersize=6, label='Interpolated NDVI')
        
        # 添加DOY数值标注
        for i, (doy, ndvi) in enumerate(zip(valid_doy, valid_ndvi)):
            plt.annotate(f'{int(doy)}', (doy, ndvi), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8, alpha=0.7)
        
        plt.xlabel('Day of Year (DOY)')
        plt.ylabel('NDVI')
        plt.title(f'Interpolated NDVI Time Series for Pixel ({pixel_x}, {pixel_y})')
        plt.grid(True, alpha=0.3)
        
        # 设置y轴范围
        plt.xlim(site_start_doy-10, site_end_doy+10)  # 设置横坐标范围
        plt.ylim(-0.2, 1.0)
        
        # 添加季节标记
        # plt.axvline(x=152, color='orange', linestyle='--', alpha=0.5, label='夏初')
        # plt.axvline(x=213, color='red', linestyle='--', alpha=0.5, label='夏末')
        # plt.axvline(x=244, color='brown', linestyle='--', alpha=0.5, label='秋季')
        
        plt.legend()
        plt.tight_layout()
        
        # 保存或显示图像
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"interpolated_ndvi_curve_pixel_{pixel_x}_{pixel_y}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"插值后NDVI曲线已保存到: {filepath}\n")
        else:
            plt.show()
            
        plt.close()

    @staticmethod
    def align_timeseries(x, doy, scl, cloud_prob, target_length=75):
        """
        Aligns the time series to a target length by interpolating and padding.
        This function is extracted from CropAttriMappingDataset7's transform logic.
        """
        x_length = x.shape[0]#71
        if x_length == 0:
            return np.zeros((target_length, x.shape[1])), np.zeros(target_length), np.zeros((target_length, scl.shape[1]))

        if x_length >= target_length:
            # x.shape = (108, 10)
            #先找出 x为 0 的时间点
            non_zero_indices = np.where(~np.all(x == 0, axis=1))[0]
            x = x[non_zero_indices]#(72, 10)
            scl = scl[non_zero_indices]#(72, 1)
            cloud_prob = cloud_prob[non_zero_indices]#(72, 1)
            doy = doy[non_zero_indices]#(72,)
            # 如果长度仍然超过 target_length，则进行均匀采样
            if x.shape[0] > target_length:
                return x[:target_length], doy[:target_length], scl[:target_length], cloud_prob[:target_length]
            x_length = x.shape[0]
            

        points_to_add = target_length - x_length#4
        
        # Pre-allocate arrays
        new_doy = np.zeros(target_length, dtype=doy.dtype)
        new_x = np.zeros((target_length, x.shape[1]), dtype=x.dtype)
        new_scl = np.zeros((target_length, scl.shape[1]), dtype=scl.dtype)
        new_cloud_prob = np.zeros((target_length, cloud_prob.shape[1]), dtype=cloud_prob.dtype)
        
        new_doy[:x_length] = doy
        new_x[:x_length] = x
        new_scl[:x_length] = scl
        new_cloud_prob[:x_length] = cloud_prob
        
        current_length = x_length
        points_added = 0

        # First pass: insert into 5-day gaps
        intervals = np.diff(new_doy[:current_length])
        interval_5_positions = np.where(intervals == 5)[0]
        
        for pos in reversed(interval_5_positions):
            if points_added >= points_to_add:
                break
            
            insert_pos = pos + 1
            start_doy = new_doy[pos]
            
            prev_interval = intervals[pos-1] if pos > 0 else 0
            next_interval = 2 if abs(prev_interval - 3.0) < 0.1 else 3
            mid_doy = start_doy + next_interval
            
            # Shift data to make space
            new_doy[insert_pos+1:current_length+1] = new_doy[insert_pos:current_length]
            new_x[insert_pos+1:current_length+1] = new_x[insert_pos:current_length]
            new_scl[insert_pos+1:current_length+1] = new_scl[insert_pos:current_length]
            new_cloud_prob[insert_pos+1:current_length+1] = new_cloud_prob[insert_pos:current_length]
            
            # Insert new data
            new_doy[insert_pos] = mid_doy
            new_x[insert_pos] = 0  # Zero-padded features
            new_scl[insert_pos] = 9 #高概率云
            new_cloud_prob[insert_pos] = 100.0
            
            current_length += 1
            points_added += 1

        # Second pass: append to the end if still needed
        while points_added < points_to_add:
            insert_pos = current_length
            if current_length > 0:
                last_doy = new_doy[current_length - 1]
                if current_length >= 2:
                    last_interval = new_doy[current_length - 1] - new_doy[current_length - 2]
                    next_interval = 2 if abs(last_interval - 3.0) < 0.1 else 3
                else:
                    next_interval = 3
                new_doy_value = last_doy + next_interval
            else:
                new_doy_value = 120  # Start DOY if empty

            new_doy[insert_pos] = new_doy_value
            new_x[insert_pos] = 0
            new_scl[insert_pos] = 9#表示高概率云
            new_cloud_prob[insert_pos] = 100.0
            
            current_length += 1
            points_added += 1
            
        return new_x, new_doy, new_scl, new_cloud_prob
    # all_samples_data = []

    # spatial_indices = np.zeros((num_valid, 2), dtype=np.int32)  # 存储空间位置
    
    # for pid in tqdm(range(num_valid), desc=f"Processing samples for block {id} ({site})"):
    #     row, col = valid_indices[0][pid], valid_indices[1][pid]
    #     # 记录空间位置
    #     spatial_indices[pid] = [row, col]

    #     img = valid_s2[pid]#(71, 10)
    #     scl = valid_scl[pid]#(71, 1)
    #     cond = valid_cond[pid]#(8, 3)
    #     lb = int(valid_lb[pid])
    #     if Country == "US":
    #         conf = int(valid_conf[pid])#100

    #     # valid_ind_t = np.all(img == 0, axis=1)
    #     # tmp1= len(img)
    #     # temp2 = len(np.sum(valid_ind_t!=True))
    #     # if temp2 < min_sequence_length:
    #     #     continue

    #     # img_val = img[img_val_ind]
    #     # doy_val = doy[img_val_ind]
    #     img_val = img
    #     doy_val = doy
    #     scl_val = scl

    #     # Align time series data to target_length
    #     x_aligned, doy_aligned, scl_aligned = align_timeseries(img_val, doy_val, scl_val, target_length=75)#(75, 10)
    def extract_samples_from_patch(self, patch_data):

        timeseries = patch_data["timeseries"]
        file_info_list = patch_data["file_info_list"]
        indices = patch_data["indices"].astype(np.int64)
        doy_list = np.array(patch_data["doy_list"], dtype=np.int64)

        feat = int(self.feature_num)#10
        seq_len = int(self.sequencelength)#75

        site_start_doy = 60
        site_end_doy = 300
        s2_data = timeseries[:, :feat]#(83, 10, 327, 219)
        scl = timeseries[:, 10]#(83, 327, 219)
        # 筛选出 90 到 270 天的时序
        doy_list_mask = (site_start_doy+30 <= np.array(doy_list)) & (np.array(doy_list) <= site_end_doy-30)#输入为 90 到 270 天的时序
        s2_data = s2_data[doy_list_mask]#(71, 10, 327, 219)
        # indices = indices[doy_list_mask]#(71,)
        doy_list = np.array(doy_list)[doy_list_mask]#(T,)
        scl = scl[doy_list_mask]#(T, H, W)(71, 327, 219)
        T, C, H, W = s2_data.shape

        # valid_doy_valid = timeseries[:, 11].astype(np.float32) != 0
        # valid_doy_valid = valid_doy_valid.transpose(1, 2, 0)#(H,W,T)(512, 512, 30)
        s2_data = s2_data[:,:10,:,:].transpose(0, 2, 3, 1)*1e-4#(T, H, W, C)#(30, 512, 512, 10)
        scl = scl.transpose(1, 2, 0)#(H,W,T)(512, 512, 71)

        # 遍历所有像素
        samples = []
        for row in range(H):
            for col in range(W):
                # 提取像素的时间序列 (T, C)
                img_val = s2_data[:, row, col,:10]  # (T, C)(71, 10)
                doy_val = doy_list#(T,)#(71,)
                scl_val = scl[row, col, :][:, np.newaxis]#(T, 1)
                cloud_prob = np.zeros((T, 1), dtype=np.float32)
                x_aligned, doy_aligned, scl_aligned,  cloud_prob = self.align_timeseries(img_val, doy_val, scl_val,cloud_prob, target_length=seq_len)#(75, 10)
                
                # Mask out cloudy observations in x_aligned based on scl_aligned
                # SCL values: 3 (Cloud Shadows), 7 (Unclassified), 8 (Cloud Medium Probability), 9 (Cloud High Probability), 10 (Cirrus) are considered invalid/cloudy.
                # Note: scl_aligned is (75, 1)
                if True:
                    scl_flat = scl_aligned.flatten()
                    cloud_mask = (scl_flat == 3) | (scl_flat == 7) | (scl_flat == 8) | (scl_flat == 9) | (scl_flat == 10)
                    # Set x_aligned to 0 where cloud_mask is True
                    x_aligned[cloud_mask] = 0
                
                mask = np.zeros((self.sequencelength,), dtype=int)
                mask[:x_aligned.shape[0]] = 1
                sample = {
                    "row": row,
                    "col": col,
                    "s2_timeseries": x_aligned,
                    "mask": mask==0,
                    "doy_timeseries": doy_aligned,
                    "scl_timeseries": scl_aligned,
                    'coord':patch_data['coord'],
                }
                samples.append(sample)

        logger.info(f"从patch {patch_data['coord']} 提取了 {len(samples)} ({H}x{W})个样本")
        return samples

    def _impute_flat_pixels(self, samples, feat, seq_len):
        num_samples = len(samples)
        out_flat = np.empty((num_samples, seq_len, feat), dtype=np.float32)

        mean = self.mean.reshape(1, 1, -1).astype(np.float32)#(1, 1, 10)
        std = self.std.reshape(1, 1, -1).astype(np.float32)

        # 准备批量数据
        batch_size = 128 * 10
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        logger.info(f'开始插补 {num_samples} 个样本，分 {num_batches} 批处理')

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch_samples = samples[start_idx:end_idx]
                
                x_pad = []
                doy_pad = []
                
                for sample in batch_samples:
                    x_pad.append(sample['s2_timeseries'])
                    doy_pad.append(sample['doy_timeseries'])
                
                x_arr = np.array(x_pad, dtype=np.float32)#(40960, 75, 10)
                doy_arr = np.array(doy_pad, dtype=np.int64)#(40960, 75)
                
                # 计算 mask (1 for observed, 0 for missing)
                # 假设 x_arr 中全 0 为缺失/padding
                # mask_arr = ~np.all(x_arr == 0, axis=2)
                # mask_arr = mask_arr[:, :, None].astype(np.float32)
                
                x_norm = (x_arr - mean) / std
                
                X = torch.from_numpy(x_norm).to(self.device)
                # masks = torch.from_numpy(mask_arr).to(self.device)
                doys = torch.from_numpy(doy_arr).to(self.device).long()
                
                _, learned = self.model.impute({"X": X, "missing_mask": None, "doy": doys})
                # imputed_norm = masks * X + (1 - masks) * learned
                imputed = learned.detach().cpu().numpy() * std + mean#(1280, 75, 10)
                
                out_flat[start_idx:end_idx] = imputed[:, :seq_len, :]

        return out_flat#(71613, 75, 10)

    def process_single_patch(self, coord, file_info_list, pixel_coord=None, output_steps=None):
        logger.info(f"处理patch {coord}")

        try:
            patch_data = self.load_patch_timeseries(coord, file_info_list)
            if patch_data is None:
                logger.error(f"加载patch数据失败: {coord}")
                return None

            samples = self.extract_samples_from_patch(patch_data)

            out_flat = self._impute_flat_pixels(samples, self.feature_num, self.sequencelength)#(71613, 75, 10)
            
            # Extract DOY timeseries for all samples
            doy_flat = np.array([s['doy_timeseries'] for s in samples]) # (N, T)
            
            _,_,H,W = patch_data["timeseries"].shape
            imputed_patch = rearrange(out_flat, "(h w) t c -> t c h w", h=H, w=W)#(75, 10, 327, 219)
            doy_patch = rearrange(doy_flat, "(h w) t -> t h w", h=H, w=W) # (75, 327, 219)

            out_dir = self.output_dir / "patch_imputed"
            out_dir.mkdir(parents=True, exist_ok=True)

            profile = patch_data["profile"].copy()
            dtype = patch_data["dtype"]
            band_count = self.feature_num

            imputed_files = {}
            out_profile = profile.copy()
            # Add 1 band for DOY
            out_profile.update({"count": band_count + 1, "dtype": dtype, "compress": "lzw"})

            for step in range(self.sequencelength):
                if output_steps is not None and step not in output_steps:
                    continue

                src_out = np.zeros((band_count + 1, H, W), dtype=dtype)#(11, 327, 219)
                dn = np.clip(imputed_patch[step] * 1e4, 0, 10000)
                if np.issubdtype(np.dtype(dtype), np.integer):
                    dn = np.rint(dn).astype(dtype)
                else:
                    dn = dn.astype(dtype)
                src_out[:int(self.feature_num)] = dn#(10, 327, 219)
                
                # Add DOY band
                src_out[int(self.feature_num)] = doy_patch[step].astype(dtype)
                
                # Calculate representative DOY (mode) for filename
                vals, counts = np.unique(doy_patch[step], return_counts=True)
                if vals.shape[0] != 1:
                    raise ValueError(f"Falled to calculate representative DOY (mode) for file")
                rep_doy = vals[np.argmax(counts)]

                out_path = out_dir / f"patch_{coord[0]}_{coord[1]}_step_{step:03d}_doy_{int(rep_doy)}_imputed.tif"
                with rasterio.open(out_path, "w", **out_profile) as dst:
                    dst.write(src_out)

                imputed_files[int(step)] = out_path

            del patch_data, out_flat, imputed_patch, doy_flat, doy_patch
            gc.collect()

            logger.info(f"patch {coord} 处理完成")
            return {
                "coord": coord,
                "imputed_files": imputed_files,
                "success": True,
            }

        except Exception as e:
            logger.error(f"处理patch {coord} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                "coord": coord,
                "success": False,
                "error": str(e),
            }
    
    def merge_patch_results(self, patch_results):
        logger.info("开始合并patch结果")

        successful_results = [r for r in patch_results if r and r.get("success")]
        if not successful_results:
            logger.error("没有成功的patch结果可以合并")
            return None

        all_indices = set()
        for r in successful_results:
            all_indices.update(r.get("imputed_files", {}).keys())
        all_indices = sorted(all_indices)

        mosaic_dir = self.output_dir / "mosaic_imputed"
        mosaic_dir.mkdir(parents=True, exist_ok=True)

        merged = {}
        for indice in all_indices:
            patch_files = []
            for r in successful_results:
                fp = r.get("imputed_files", {}).get(indice)
                if fp is not None:
                    patch_files.append(fp)

            if not patch_files:
                continue

            src_files = [rasterio.open(str(fp)) for fp in patch_files]
            mosaic, out_trans = merge(src_files)

            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw",
            })

            out_path = mosaic_dir / f"imputed_{int(indice):03d}_merged.tif"
            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            for src in src_files:
                src.close()

            merged[int(indice)] = out_path

        logger.info(f"合并完成，生成 {len(merged)} 个日期的 mosaics 到 {mosaic_dir}")
        return {"mosaics": merged, "num_patches": len(successful_results)}
    
    def run(self):
        logger.info(f"开始处理目录: {self.s2_dir}")

        patch_coords = self.get_patch_coordinates()
        if not patch_coords:
            logger.error("未找到patch数据")
            return None

        patch_results = []
        for coord, file_info_list in tqdm(patch_coords.items(), desc="处理patches"):
            result = self.process_single_patch(coord, file_info_list, output_steps=self.output_steps)
            patch_results.append(result)

        merged_result = self.merge_patch_results(patch_results)

        successful_patches = len([r for r in patch_results if r and r.get("success")])
        total_patches = len(patch_coords)

        logger.info(f"处理完成:")
        logger.info(f"  总patch数: {total_patches}")
        logger.info(f"  成功处理: {successful_patches}")
        logger.info(f"  成功率: {successful_patches/total_patches*100:.1f}%")

        return {
            "total_patches": total_patches,
            "successful_patches": successful_patches,
            "patch_results": patch_results,
            "merged_result": merged_result,
        }
    
    def create_15day_composite_and_interpolate(self, s2_data, doy_list, region_config):#(59, 1349, 1753, 11) (59,)
        """
        实现与 test_code 一致的 15-day composite 和线性插值处理 TODO: 内存优化版本：分块处理大图像
        """
        time_steps, h, w, bands = s2_data.shape  
        site = region_config.get('field_name', None)
        
        #创建掩码：将无效值（0值、异常值）设置为NaN
        s2_data_masked = s2_data.astype(np.float32)
        del s2_data
        gc.collect()
        # 假设0值和过大/过小的值为无效值
        # 修改：只要任意波段有0值就设置为NaN，形状为(time_steps, h, w)
        # invalid_mask = np.any(s2_data_masked <= 0, axis=3)  # (time_steps, h, w) 这个执行时间太久了
        if NUMBA_AVAILABLE:
            @jit(nopython=True)
            def create_invalid_mask_numba(data):
                time_steps, h, w, bands = data.shape
                invalid_mask = np.zeros((time_steps, h, w), dtype=np.uint8)
                
                for t in range(time_steps):
                    for i in range(h):
                        for j in range(w):
                            for b in range(bands):
                                val = float(data[t, i, j, b])  # 显式转换为float
                                if val <= 0.0:
                                    invalid_mask[t, i, j] = 1
                                    break
                return invalid_mask.astype(np.bool_)
        
            # 使用numba加速版本
            invalid_mask = create_invalid_mask_numba(s2_data_masked)#(144, 1268, 2318)
        else:
            # 使用原始版本或分块版本
            # 优化后：分块处理
            def create_invalid_mask_chunked(s2_data_masked, chunk_size=512):
                """
                分块创建无效值掩码，减少内存占用和计算时间
                """
                time_steps, h, w, bands = s2_data_masked.shape
                invalid_mask = np.zeros((time_steps, h, w), dtype=bool)
                
                # 分块处理
                for i in tqdm(range(0, h, chunk_size), desc="创建无效值掩码"):
                    for j in range(0, w, chunk_size):
                        i_end = min(i + chunk_size, h)
                        j_end = min(j + chunk_size, w)
                        
                        # 处理当前块
                        chunk = s2_data_masked[:, i:i_end, j:j_end, :]
                        chunk_mask = np.any(chunk <= 0, axis=3)
                        invalid_mask[:, i:i_end, j:j_end] = chunk_mask
                        
                        # 清理内存
                        del chunk, chunk_mask
                
                return invalid_mask
            invalid_mask = create_invalid_mask_chunked(s2_data_masked)
        # 将invalid_mask扩展到所有波段
        invalid_mask_expanded = np.expand_dims(invalid_mask, axis=3)  # (time_steps, h, w, 1)
        invalid_mask_expanded = np.repeat(invalid_mask_expanded, s2_data_masked.shape[3], axis=3)  # (time_steps, h, w, bands)
        s2_data_masked[invalid_mask_expanded] = np.nan
        del invalid_mask_expanded
        del invalid_mask
        gc.collect()

        # 1. 创建 15-day composite
        composites = []#len()=14
        # site_start_doy = region_config.get("start_doy", min(doy_list))
        # site_end_doy = region_config.get("end_doy", max(doy_list))
        
        # 计算总的时间窗口数量
        total_windows = len(range(site_start_doy, site_end_doy, 15))
        
        for start_doy in tqdm(range(site_start_doy, site_end_doy, 15), 
                                desc=f"创建{site}的15天合成图像", 
                                total=total_windows):
            end_doy = min(start_doy + 15, site_end_doy+1)
            
            # 找到该时间窗口内的所有图像
            # 优化后：使用布尔索引，避免列表
            mask = (np.array(doy_list) >= start_doy) & (np.array(doy_list) < end_doy)#(144,)
            if np.any(mask):
                selected_data = s2_data_masked[mask]# 直接选择，避免循环和列表 (6, 1268, 2318, 10)
                # 抑制 All-NaN slice 警告
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                    composite = np.nanmedian(selected_data, axis=0)#(1268, 2318, 10) 当所有输入值都是NaN时， np.nanmedian() 函数会返回 NaN 。
                # 验证composite中是否存在NaN值
                # nan_count = np.sum(np.isnan(composite))
                # total_pixels = composite.size
                # nan_percentage = (nan_count / total_pixels) * 100
                # print(f"Composite NaN统计: {nan_count}/{total_pixels} ({nan_percentage:.2f}%) 像素为NaN")#Composite NaN统计: 2687060/37020820 (7.26%) 像素为NaN
                composites.append((composite, start_doy))
            else:# 如果窗口内没有图像，添加空值(应该不可能进这里)
                composite = np.full((h, w, bands), np.nan)
                composites.append((composite, start_doy))
            
        del s2_data_masked
        del composite
        del selected_data
        del mask
        gc.collect()
        
        # 2. 线性插值填补缺失值
        interpolated = []
        for i in tqdm(range(2,len(composites)-2), desc=f"线性插值{site}的15天合成图像", total=len(composites)-4):
            current_composite, current_doy = composites[i]#(1268, 2318, 10) 120
            interpolated_composite = current_composite.copy()#(1268, 2318, 10)
            
            # 寻找前后有效的合成图像（模拟JavaScript的搜索范围）
            before_values = None
            after_values = None
            before_doy = -1
            after_doy = -1
            
            def create_mosaic(composite_list, reverse=False):
                """创建 mosaic，模拟 GEE 的 mosaic() 功能"""
                if not composite_list:
                    return None, None
                
                # 如果需要反向（模拟 .reverse()）
                if reverse:
                    composite_list = composite_list[::-1]
                
                result = None#(2059, 1798, 10)
                result_doy = None#(2059, 1798)
                
                for comp, doy in composite_list:#(2059, 1798, 10) 135
                    if not np.all(np.isnan(comp)):
                        if result is None:
                            result = comp.copy()
                            # 初始化 DOY 数组，形状为 (h, w)，所有像素都使用当前 DOY
                            result_doy = np.full(comp.shape[:2], doy, dtype=np.float32)
                        else:
                            # Mosaic: 后面的有效像素覆盖前面的
                            valid_mask = ~np.isnan(comp)# (h, w, bands)
                            # 对于任何波段有效的像素，都认为该像素有效
                            pixel_valid_mask = np.any(valid_mask, axis=2)  # (h, w)
                            result = np.where(valid_mask, comp, result)
                            # 更新对应像素的 DOY
                            result_doy = np.where(pixel_valid_mask, doy, result_doy)
                
                return result, result_doy
            # 使用新的 mosaic 函数
            before_list = [(composites[j][0], composites[j][1]) #composites[j][0] ：第j个时间步的像素值数据 composites[j][1] ：第j个时间步的DOY（Day of Year）日期
                            for j in range(max(0, i-2), i) #从当前位置 i 向前搜索最多2个时间步
                            if not np.all(np.isnan(composites[j][0]))]
            before_values, before_doy = create_mosaic(before_list, reverse=False)#(2059, 1798, 10) (2059, 1798)
            # print(f"before_doy.unique(): {np.unique(before_doy)}")# [105. 120.]
            # nan_count = np.sum(np.isnan(before_values))
            
            after_list = [(composites[j][0], composites[j][1]) 
                        for j in range(i+1, min(len(composites), i+3)) #[150. 165.]
                        if not np.all(np.isnan(composites[j][0]))]
            after_values, after_doy = create_mosaic(after_list, reverse=True)
            # print(f"after_doy.unique(): {np.unique(after_doy)}")
            # nan_count = np.sum(np.isnan(after_values)) 
            del after_list, before_list
            gc.collect()
            
            # 执行线性插值
            if before_values is not None and after_values is not None:
                # 计算插值权重 (h, w)
                weight = (current_doy - before_doy) / (after_doy - before_doy)#(2059, 1798)
                # print(f"weight.unique():{np.unique(weight)}")#[0.33333334 0.5        0.6666667 ]
                # 避免除零错误
                # weight = np.where(np.abs(after_doy - before_doy) < 1e-6, 0.5, weight)
                # 将权重扩展到三维以匹配光谱数据的形状 (h, w, bands)
                weight_3d = np.expand_dims(weight, axis=2)#(2059, 1798, 1)

                 # 检查before_values和after_values中的NaN值
                before_nan_mask = np.isnan(before_values)
                after_nan_mask = np.isnan(after_values)
                both_valid_mask = ~(before_nan_mask | after_nan_mask)#(2059, 1798, 10)
                # 对光谱波段进行线性插值
                interpolated_result = np.full_like(before_values, np.nan)
                # 只对两个值都有效的像素进行插值
                interpolated_result = np.where(
                    both_valid_mask,
                    before_values + weight_3d * (after_values - before_values),
                    0#这里得改成 0
                    # np.where(
                    #     ~before_nan_mask,  # 如果before_values有效，使用before_values
                    #     before_values,
                    #     np.where(
                    #         ~after_nan_mask,  # 如果after_values有效，使用after_values
                    #         after_values,
                    #         np.nan  # 两者都无效，保持NaN
                    #     )
                    # )
                )
                # nan_count = np.sum(np.isnan(interpolated_result))
                # 对光谱波段进行线性插值
                # interpolated_result = before_values + weight_3d * (after_values - before_values)#(2059, 1798, 10)
                
                # 根据原图像是否有效决定使用原值还是插值
                nan_mask = np.isnan(current_composite)
                interpolated_composite = np.where(nan_mask, interpolated_result, current_composite)
                # nan_count = np.sum(np.isnan(interpolated_composite))
                del interpolated_result
            elif before_values is not None:
                # 只有前值，使用前值填充NaN位置
                nan_mask = np.isnan(current_composite)
                interpolated_composite = np.where(nan_mask, before_values, current_composite)
            elif after_values is not None:
                # 只有后值，使用后值填充NaN位置
                nan_mask = np.isnan(current_composite)
                interpolated_composite = np.where(nan_mask, after_values, current_composite)
            else:
                # 如果前后都没有有效值，保持原状
                interpolated_composite = current_composite
            
            interpolated.append([interpolated_composite, current_doy])#len()=10
        del interpolated_composite
        if nan_mask is not None:
            del nan_mask
        if current_composite is not None:
            del current_composite
        del before_values
        del after_values
        del after_doy
        del before_doy
        gc.collect()
        
        # 3. 月度采样（DOY 120-255，步长15）
        target_doys = list(range(site_start_doy+30, site_end_doy-45+1, 15))  # [120, 135, 150, ..., 255]
        sampled_images = []#len(sampled_images) = 10
        
        for target_doy in target_doys:
            # 找到最接近的合成图像
            best_idx = 0
            min_diff = abs(interpolated[0][1] - target_doy)
            
            for j, (_, comp_doy) in enumerate(interpolated):
                diff = abs(comp_doy - target_doy)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = j
            
            if best_idx < len(interpolated):
                sampled_images.append(interpolated[best_idx][0])
            else:
                # 如果索引超出范围，使用最后一个有效图像
                sampled_images.append(interpolated[-1][0])
        
        del composites
        del interpolated
        gc.collect()
        
        # 4. 选择前10个波段并处理剩余的NaN值
        # ['blue','green','red','red1','red2','red3','nir','red4','swir1','swir2']
        selected_bands = min(10, bands)#10
        final_images = []
        
        # 5. 合并为最终的时间序列数据
        # 形状：[10_timepoints , height, width,  selected_bands]
        final_images = []
        for img in tqdm(sampled_images, desc=f"合成{site}的10个波段", total=len(sampled_images)):
            # 选择前selected_bands个波段
            img_selected = img[:, :, :selected_bands]  # (h, w, selected_bands)
            # 将剩余的NaN值设为0#TODO: 这里应该要注释掉
            # selected_img = np.nan_to_num(img_selected, nan=0.0)
            # final_images.append(selected_img)
            final_images.append(img_selected)
        
        # 6. 合并为最终的时间序列数据
        if final_images:
            time_series = np.array(final_images)
            return time_series
        else:
            return np.zeros((len(target_doys),h, w, selected_bands))

    def process_climate_data(self, climate_data, target_time_steps=8):
        """处理气候数据以适配CACMNet
        
        Args:
            climate_data: 形状为 (h, w, 112) 的气候数据，其中 112 = 8个月 × 14个TerraClimate变量
            target_time_steps: 目标时间步数，默认为8
        
        Returns:
            处理后的气候数据，形状为 (h, w, target_time_steps, 3)
        """
        try:
            h, w, total_vars = climate_data.shape
            
            # 验证输入数据格式 TODO: to modify
            # if total_vars != 112:  # 8个月 × 14个变量
            #     raise ValueError(f"期望气候数据有112个变量 (8月×14变量)，但得到 {total_vars}")
            
            # 重塑数据：(h, w, 112) -> (h, w, 8, 14)
            climate_data = rearrange(climate_data, 'h w (t c) -> h w t c',  c=14)#(2059, 1798, 8, 14)
            
            # 选择特定的气候变量：索引 [9, 10, 7] 对应 TerraClimate 的特定变量
            # 根据 load_terra_data 函数，这些索引对应重要的气候变量
            selected_indices = [9, 10, 7]  # 选择3个关键气候变量
            climate_data = climate_data[:, :, 1:, selected_indices]  # (h, w, 8, 3)TODO : to modify
            
            # 处理时间维度调整
            months = climate_data.shape[2]  # 当前是8个月
            
            if months > target_time_steps:
                # 如果月份数超过目标，选择关键月份
                indices = np.linspace(0, months-1, target_time_steps, dtype=int)
                climate_data = climate_data[:, :, indices, :]  # (h, w, target_time_steps, 3)
            elif months < target_time_steps:
                # 如果月份数不足，重复最后一个月的数据
                padding_needed = target_time_steps - months
                last_month = climate_data[:, :, -1:, :]  # (h, w, 1, 3)
                padding = np.repeat(last_month, padding_needed, axis=2)  # (h, w, padding_needed, 3)
                climate_data = np.concatenate([climate_data, padding], axis=2)  # (h, w, target_time_steps, 3)
            
            # 验证输出形状
            expected_shape = (h, w, target_time_steps, len(selected_indices))#(2059, 1798, 8, 3)
            if climate_data.shape != expected_shape:
                raise ValueError(f"输出形状不匹配：期望 {expected_shape}，得到 {climate_data.shape}")
            
            print(f"气候数据处理完成：{(h, w, total_vars)} -> {climate_data.shape}")
            return climate_data#(h, w, 8, 3)
            
        except Exception as e:
            print(f"处理气候数据时出错：{str(e)}")
            import traceback
            traceback.print_exc()
            print(f"输入数据形状：{climate_data.shape}")
            raise

    def extract_all_samples_for_mapping(self, s2_data, climate_data):
        """
        提取区域内所有有效样本用于作物制图
        与extract_samples的区别：不限制样本数量，返回所有有效样本及其空间位置
        
        Args:
            s2_data: 形状为 (time_steps, h, w, bands) 的遥感数据
            climate_data: 形状为 (h, w, time_steps, features) 的气候数据
            label_data: 形状为 (h, w) 的标签数据
        
        Returns:
            tuple: (x_samples, cond_samples, y_samples, spatial_indices)
                - x_samples: (N, bands, time_steps) 遥感样本
                - cond_samples: (N, features, time_steps) 气候样本
                - y_samples: (N,) 标签样本
                - spatial_indices: (N, 2) 空间位置索引 [row, col]
        """
        try:
            t, h, w, c = s2_data.shape#(10, 2200, 2217, 10)
            
            # 验证数据维度一致性
            if climate_data.shape[:2] != (h, w):
                raise ValueError(f"气候数据和遥感数据的空间维度不匹配: {climate_data.shape[:2]} vs {(h, w)}")
            # if label_data.shape[:2] != (h, w):
            #     raise ValueError(f"标签数据和遥感数据的空间维度不匹配: {label_data.shape} vs {(h, w)}")
            
            # # 获取有效标签位置
            # if len(label_data.shape) > 2:
            #     label_data = label_data[:,:,0]  # 如果是3D，取第一个通道
            
            # if self.Country != 'EU' and self.Country != 'CN':
            #     # valid_mask = label_data > 0  # 移除标签过滤，保留所有像素用于制图
            #     valid_mask = np.ones_like(label_data, dtype=bool)  # 保留所有像素
            # else:
            #     # valid_mask = label_data >= 0  # 移除标签过滤
            #     valid_mask = np.ones_like(label_data, dtype=bool)  # 保留所有像素
            # valid_mask = np.ones_like(s2_data[0,...], dtype=bool)  # 保留所有像素
            valid_mask = np.ones((h, w), dtype=bool)
            
            # 添加数据质量检查
            # s2_valid_mask = np.any(s2_data != 0, axis=(0, 3)) & ~np.any(np.isnan(s2_data), axis=(0, 3))#(2200, 2217)
            # climate_valid_mask = ~np.any(np.isnan(climate_data), axis=(2, 3)) & ~np.any(climate_data == 0, axis=(2, 3))
            
            # 综合有效性掩码
            # combined_valid_mask = valid_mask & s2_valid_mask & climate_valid_mask
            combined_valid_mask = valid_mask
            
            valid_indices = np.where(combined_valid_mask)
            num_valid = len(valid_indices[0])
            
            if num_valid == 0:
                print("警告：没有找到有效的样本")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            print(f"找到 {num_valid} 个有效样本用于作物制图")
            
            # 预分配数组
            x_samples = np.zeros((num_valid, c, t))
            cond_samples = np.zeros((num_valid, climate_data.shape[3], climate_data.shape[2]))
            # y_samples = np.zeros(num_valid, dtype=np.int32)
            spatial_indices = np.zeros((num_valid, 2), dtype=np.int32)  # 存储空间位置
            
            # 批量提取样本
            for i in range(num_valid):
                row, col = valid_indices[0][i], valid_indices[1][i]
                
                # 提取遥感时间序列
                x_sample = s2_data[:, row, col, :].T#
                cond_sample = climate_data[row, col, :, :].T#3,8
                
                x_samples[i] = x_sample
                cond_samples[i] = cond_sample
                
                # 应用类别映射
                # original_label = label_data[row, col]
                # mapped_label = self.map_label_to_class_id(original_label)
                # y_samples[i] = mapped_label
                
                # 记录空间位置
                spatial_indices[i] = [row, col]
            
            # 移除无效样本
            # valid_sample_mask = ~np.all(x_samples == 0, axis=(1, 2))
            # x_samples = x_samples[valid_sample_mask]
            # cond_samples = cond_samples[valid_sample_mask]
            # y_samples = y_samples[valid_sample_mask]
            # spatial_indices = spatial_indices[valid_sample_mask]
            
            final_count = len(x_samples)
            print(f"最终提取到 {final_count} 个有效样本用于作物制图")
            
            if final_count == 0:
                print("错误：所有样本都无效")
                return np.array([]), np.array([]), np.array([])
            
            return x_samples, cond_samples, spatial_indices
            
        except Exception as e:
            print(f"提取作物制图样本时出错：{str(e)}")
            import traceback
            traceback.print_exc()
            raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--file_glob", type=str, default=None)
    parser.add_argument("--output_steps", type=int, nargs='+', help="Steps to output (space separated)")
    args = parser.parse_args()

    mapper = PatchBasedMapping(
        s2_dir=args.s2_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        model_path=args.model_path,
        device=args.device,
        file_glob=args.file_glob,
        output_steps = args.output_steps,
    )

    result = mapper.run()

    if result:
        logger.info("处理成功完成!")
    else:
        logger.error("处理失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()