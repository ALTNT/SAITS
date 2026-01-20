import shutil
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from torch import chunk
from torch.utils import data
from tqdm import tqdm
import logging
import json
import os
import random
from einops import rearrange
import sys
import argparse

from rasterio.mask import mask
from shapely.geometry import mapping
from config import get_active_config, get_config, ALL_REGIONS

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------- Config --------------------------- #
# year = '2019'
# # sites = ['Indiana', 'Iowa', 'Kansas', 'Missouri','Minnesota','Mississippi','Arkansas','Wisconsin']
# # sites = ['Indiana']
# sites = ['CAN2']
# # sites = ['Arkansas']
# Country = 'CAN'
# 从统一配置文件获取参数
config = get_active_config()
year = config['year']
Country = config['Country']
sites = config['sites']
data_to_mapping = config['data_to_mapping']
kernel_size = config['kernel_size']
confidence_threshold = config['confidence_threshold']
bandnames = config['bandnames']
# -------------------------- Sharding Controls --------------------------- #
RECORD_SIZE_FLOATS = 1 + 75*10 + 75 + 8*3 + 75 + 75  # 1 + 75*10 + 75 + 8*3 + 75 + 75 = 1000
RECORD_SIZE_BYTES = 4 * RECORD_SIZE_FLOATS  # float32
MAX_SHARD_BYTES = 0.5 * 1024 ** 3  # 每片大小上限（默认 0.5GB）
# SAMPLES_PER_SHARD = max(1, MAX_SHARD_BYTES // RECORD_SIZE_BYTES)
SAMPLES_PER_SHARD  = int(max(1, MAX_SHARD_BYTES // RECORD_SIZE_BYTES))
CHUNK_SAMPLES      = int(32768)                    # 每次写入的样本数
# ------------------------------------------------------------- #

# 新增：加载坐标索引（data/US-dataset 下的旧流程输出）
def load_spatial_indices_for_block(sample_index_root, site, block_id, year, kernel_size, confidence_threshold):
    base = Path(sample_index_root) / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)) / site
    candidates = [
        base / f"spatial_indices_block_{int(block_id)}.npy",
        base / "spatial_indices.npy",
        base / "chunks" / f"spatial_indices_block_{int(block_id)}.npy",
        base / "chunks" / "spatial_indices.npy",
    ]
    for p in candidates:
        if p.exists():
            try:
                arr = np.load(p)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return arr.astype(np.int32)
            except Exception as e:
                logger.warning(f"读取空间索引失败: {p} => {e}")
    logger.warning(f"未找到空间索引: site={site}, block={block_id}, root={sample_index_root}")
    return None

# 新增：根据坐标从当前区块的有效样本中挑选
def choose_indices_by_spatial(valid_indices, spatial_coords, max_samples=None, seed=None):
    rows = valid_indices[0]#(1277642,)
    cols = valid_indices[1]#(1277642,)
    if spatial_coords is None:
        return None
    coords_set = set(map(tuple, spatial_coords.tolist()))#len()=150000
    mask = np.array([ (int(r), int(c)) in coords_set for r, c in zip(rows, cols) ], dtype=bool)#(1277642,)
    selected = np.where(mask)[0]
    if selected.size == 0:
        return None
    if max_samples is not None and selected.size > max_samples:
        if seed is not None:
            np.random.seed(seed)
        selected = np.random.choice(selected, size=max_samples, replace=False)
    return selected

class OnlineFeatureStats:
    def __init__(self, n_features=10, scale=1e-4):
        self.n_features = n_features
        self.scale = scale
        self.count = 0
        self.sum = np.zeros(n_features, dtype=np.float64)
        self.sum_sq = np.zeros(n_features, dtype=np.float64)

    def update(self, x):
        # x: (T, n_features)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            return
        # 仅统计非全零的时序（排除由对齐/云遮挡插入的零）
        valid = ~np.all(x == 0, axis=1)
        if not np.any(valid):
            return
        rows = x[valid].astype(np.float64) * self.scale
        self.count += rows.shape[0]
        self.sum += rows.sum(axis=0)
        self.sum_sq += (rows ** 2).sum(axis=0)

    def finalize(self):
        if self.count == 0:
            return np.zeros(self.n_features, dtype=np.float64), np.ones(self.n_features, dtype=np.float64), 0
        mean = self.sum / self.count
        var = self.sum_sq / self.count - mean ** 2
        var[var < 0] = 0.0
        std = np.sqrt(var)
        return mean, std, self.count

class OnlineClimateStats:
    def __init__(self, n_features=3):
        self.n_features = n_features
        self.count = 0
        self.sum = np.zeros(n_features, dtype=np.float64)
        self.sum_sq = np.zeros(n_features, dtype=np.float64)

    def update(self, cond):
        # cond: (M, n_features) 例如 M=8（月数）
        if cond.ndim != 2 or cond.shape[1] != self.n_features:
            return
        # 跳过全零的月份行
        valid = ~np.all(cond == 0, axis=1)
        if not np.any(valid):
            return
        rows = cond[valid].astype(np.float64)
        self.count += rows.shape[0]
        self.sum += rows.sum(axis=0)
        self.sum_sq += (rows ** 2).sum(axis=0)

    def finalize(self):
        if self.count == 0:
            return np.zeros(self.n_features, dtype=np.float64), np.ones(self.n_features, dtype=np.float64), 0
        mean = self.sum / self.count
        var = self.sum_sq / self.count - mean ** 2
        var[var < 0] = 0.0
        std = np.sqrt(var)
        return mean, std, self.count


root = Path(r"data")
root = root / Country / "GEE"/"Mosaic"
# out_dir = Path(r"data/US-dataset")
if data_to_mapping:
    out_dir = Path(r"data/"+ Country + "-dataset-mapping")
else:
    if config['cloud_no_mask'] == False:
        out_dir = Path(r"data/"+ Country + "-dataset")
    else:
        out_dir = Path(r"data/data-no-mask-cloud/"+ Country + "-dataset")
#----US-dataset/classmapping.csv----
# ,id,classname,code,Red,Green,Blue
# 1,0, other,0,233,255,190
# 2,4, soybean,5,38,115,0
# 3,1, maize,1,255,212,0
# 4,3, rice,3,0,169,230
# 5,2, cotton,2,255,38,38
#----CAN-dataset/classmapping.csv----
# ,id,classname,code,Red,Green,Blue
# 1,0, other,0,233,255,190
# 2,4, soybean,158,38,115,0
# 3,1, maize,147,255,212,0
#code列是原始 CDL 或 ACI 数据中的土地覆盖类型代码，例如,在美国CDL数据中，代码1通常表示玉米，代码5表示大豆等。
# id 列：模型训练使用的类别ID，这是经过重新映射的类别编号，通常从0开始连续编号。
classmapping = out_dir / "classmapping.csv"
if Country == 'EU':
    classmapping = out_dir / "classmapping-hda.csv"
shp_fn = Path(r"data/"+ Country + "/Boundary/blockpartition.shp")
min_sequence_length = 7  # 最小有效时序长度 TODO: to experiment
random_seed = 42  # 随机种子
# ------------------------------------------------------------- #

# def getWeight(x):
#     score = np.ones(x.shape[0])
#     score = np.minimum(score, (x[:, 0] / 10000 - 0.1) / 0.4)  # blue
#     score = np.minimum(score, (x[:, [0, 1, 2]].sum(1) / 10000 - 0.2) / 0.6)  # rgb
#     cloud = score * 100 > 20  # TODO

#     dark = x[:, [6, 8, 9]].sum(1) < 3500  # TODO

#     ndvi = (x[:, 6] - x[:, 2]) / (x[:, 6] + x[:, 2] + 1e-8)
#     ndvi[cloud] = -1
#     ndvi[dark] = -1

#     weight = np.exp(ndvi)
#     weight /= weight.sum()

#     return weight

def load_cdl_data(sites, root, year):
    """加载CDL数据"""
    labels = {}
    logger.info('=================== Open CDL ===================')
    for site in sites:
        try:
            if Country == 'US':
                cdl_pth = root / site / (site + '_CDL_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
            elif Country == 'EU' or Country == 'CN':
                if site == 'Ukraine' or Country == 'CN':
                    cdl_pth = root / site / (site + '_WorldCereal_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
                else:
                    cdl_pth = root / site / (site + '_HDA_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
            else:
                cdl_pth = root / site / (site + '_label_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
            cdl_dataset = rasterio.open(cdl_pth)
            labels[site] = cdl_dataset
            logger.info(f"Loaded CDL data for {site}")
        except Exception as e:
            logger.error(f"Error loading CDL data for {site}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    logger.info('Done.\n')
    return labels

def load_satellite_images(img_pth, site):
    """加载卫星图像数据"""
    region_config = ALL_REGIONS[site]
    start_doy = region_config.get('start_doy', None)
    end_doy = region_config.get('end_doy', None)
    if start_doy is None or end_doy is None:
        raise ValueError(f"start_doy or end_doy is not set for {site}")
    if end_doy < start_doy:
        raise ValueError(f"end_doy ({end_doy}) is less than start_doy ({start_doy}) for {site}")
    img_fns = [
            file for file in img_pth.glob('*.tif')
            if not file.name.endswith('_scl.tif')
        ]
    
    if not img_fns:
        raise FileNotFoundError(f"No image files found in {img_pth}")
        
    img_fns.sort(key=lambda x: int(x.stem.split('_')[-1]))
    # 过滤日期范围内的图像
    filtered_img_fns = []
    for img_fn in img_fns:
        doy_value = int(img_fn.stem.split('_')[-1])
        if start_doy <= doy_value <= end_doy:
            filtered_img_fns.append(img_fn)
    if not filtered_img_fns:
        logger.warning(f"No images found in date range {start_doy}-{end_doy} for {site}")
        return np.array([]), np.array([])
    doy = np.array([int(img_fn.stem.split('_')[-1]) for img_fn in filtered_img_fns])
    
    data = []
    first_shape = None
    for img_fn in tqdm(filtered_img_fns, desc=f"Loading images for {site} (DOY {start_doy}-{end_doy})"):
        with rasterio.open(img_fn) as f:
            s2_image = f.read().transpose(1, 2, 0)#(2200, 2217, 12)
            
            if first_shape is None:
                first_shape = s2_image.shape
            elif s2_image.shape != first_shape:
                logger.error(f"Image shape mismatch! Expected {first_shape}, but got {s2_image.shape} for file: {img_fn}")
                # You might want to skip this image or raise an error immediately
                # continue 
        
        data.append(s2_image)
    
    logger.info(f"Loaded {len(data)} images for {site} in date range {start_doy}-{end_doy}")
    return np.array(data), doy

def load_scl_images(img_pth, site):
    """加载卫星图像数据"""
    region_config = ALL_REGIONS[site]
    start_doy = region_config.get('start_doy', None)
    end_doy = region_config.get('end_doy', None)
    if start_doy is None or end_doy is None:
        raise ValueError(f"start_doy or end_doy is not set for {site}")
    if end_doy < start_doy:
        raise ValueError(f"end_doy ({end_doy}) is less than start_doy ({start_doy}) for {site}")
    img_fns = list(img_pth.glob('*_scl.tif'))
    
    if not img_fns:
        raise FileNotFoundError(f"No image files found in {img_pth}")
        
    img_fns.sort(key=lambda x: int(x.stem.split('_')[-2]))
    # 过滤日期范围内的图像
    filtered_img_fns = []
    for img_fn in img_fns:
        doy_value = int(img_fn.stem.split('_')[-2])
        if start_doy <= doy_value <= end_doy:
            filtered_img_fns.append(img_fn)
    if not filtered_img_fns:
        logger.warning(f"No images found in date range {start_doy}-{end_doy} for {site}")
        return np.array([]), np.array([])
    doy = np.array([int(img_fn.stem.split('_')[-2]) for img_fn in filtered_img_fns])
    
    data = []
    for img_fn in tqdm(filtered_img_fns, desc=f"Loading scl images for {site} (DOY {start_doy}-{end_doy})"):
        with rasterio.open(img_fn) as f:
            s2_image = f.read().transpose(1, 2, 0)#(2200, 2217, 1)
        data.append(s2_image)
    
    logger.info(f"Loaded {len(data)} images for {site} in date range {start_doy}-{end_doy}")
    return np.array(data), doy

def load_terra_data(cond_img_pth):
    """加载Terra数据"""
    with rasterio.open(cond_img_pth) as f:
        cond_img = f.read().transpose(1, 2, 0)
    
    cond_img = rearrange(cond_img, "h w (t c) -> h w t c", c=14)
    ind = [9, 10, 7]
    cond_img = cond_img[:, :, :, ind]
    return cond_img.transpose(2, 0, 1, 3)

def load_site_image_data(site, root, year):
    """
    按站点加载图像数据（延迟加载优化版本）
    
    Args:
        site (str): 站点名称
        root (Path): 数据根目录
        year (str): 年份
    
    Returns:
        tuple: (images, doys, cond_imgs) - 该站点的图像数据、DOY数组和Terra数据
    
    Raises:
        FileNotFoundError: 当找不到必要的数据文件时
        RuntimeError: 当数据加载或处理失败时
    """
    logger.info(f"========== 开始加载站点 {site} 的图像数据 ==========")
    
    try:
        # 1. 加载 CDL 数据以获取参考尺寸
        logger.info(f"正在加载 {site} 的 CDL 参考数据...")
        if Country == 'US':
            cdl_pth = root / site / (site + '_CDL_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
        elif Country == 'EU' or Country == 'CN':
            if site == 'Ukraine' or Country == 'CN':
                cdl_pth = root / site / (site + '_WorldCereal_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
            else:
                cdl_pth = root / site / (site + '_HDA_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
        else:
            cdl_pth = root / site / (site + '_label_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
        
        if not cdl_pth.exists():
            raise FileNotFoundError(f"CDL文件不存在: {cdl_pth}")
            
        with rasterio.open(cdl_pth) as src:
            cdl_height, cdl_width = src.height, src.width
        logger.info(f"CDL 图像尺寸: {cdl_height}x{cdl_width}")
        
        # 2. 加载卫星图像数据
        logger.info(f"正在加载 {site} 的 S2 卫星图像...")
        region_config = ALL_REGIONS[site]
        start_doy = region_config.get('start_doy', None)
        end_doy = region_config.get('end_doy', None)
        if start_doy is None or end_doy is None:
            logger.error(f"区域 {site} 缺少 start_doy 或 end_doy")
            sys.exit(1)
        # img_pth = root / site / ('images_aligned_'+ 'kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'_'+str(start_doy)+'_'+str(end_doy))
        if config['cloud_no_mask'] == False:
            img_pth = root / site / ('images_aligned_'+ 'kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'_'+str(start_doy)+'_300')
        else:
            img_pth = root / site / ('images_aligned_'+ 'kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'_'+str(start_doy)+'_300_no_mask_cloud')
        
        if not img_pth.exists():
            raise FileNotFoundError(f"S2图像目录不存在: {img_pth}")
            
        images, doys = load_satellite_images(img_pth, site)#(37, 1574, 2322, 11)
        scl_imags, doys2 = load_scl_images(img_pth, site)#(37, 1574, 2322, 1)
        if len(images) != len(scl_imags):
            raise RuntimeError(f"S2图像和SCL图像数量不匹配: {len(images)} vs {len(scl_imags)}")
        if not np.array_equal(doys, doys2):
            raise RuntimeError(f"S2图像和SCL图像DOY不匹配: {doys} vs {doys2}")

        
        # 3. 调整 S2 图像尺寸以匹配 CDL
        # 注意：由于下载过程中的精度问题，S2图像可能比CDL图像大几个像素
        # 虽然这种调整不是最优解，但考虑到在valid_cdl.py中已进行了精度筛选，
        # 这样的调整不会对最终结果产生显著影响
        t, h, w, c = images.shape
        if h != cdl_height or w != cdl_width:
            logger.warning(f"正在调整 S2 图像尺寸: {h}x{w} -> {cdl_height}x{cdl_width}")
            logger.warning(f"尺寸差异可能由下载过程中的坐标精度问题导致")
            
            # 创建新的数组并复制有效区域
            adjusted_s2 = np.zeros((t, cdl_height, cdl_width, c), dtype=images.dtype)
            min_h, min_w = min(h, cdl_height), min(w, cdl_width)
            adjusted_s2[:, :min_h, :min_w, :] = images[:, :min_h, :min_w, :]
            images = adjusted_s2
            
            logger.info(f"S2 图像尺寸调整完成")
        
        logger.info(f"成功加载 {len(images)} 张 S2 图像")
        
        # 4. 加载Terra数据
        logger.info(f"正在加载 {site} 的 Terra 数据...")
        cond_img_pth = root / site / (site + '_Terra_aligned_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
        
        if not cond_img_pth.exists():
            raise FileNotFoundError(f"Terra数据文件不存在: {cond_img_pth}")
            
        cond_imgs = load_terra_data(cond_img_pth)
        
        # 5. 检查并调整 Terra 数据尺寸以匹配 CDL
        t_terra, h_terra, w_terra, c_terra = cond_imgs.shape
        if h_terra != cdl_height or w_terra != cdl_width:
            logger.warning(f"正在调整 Terra 图像尺寸: {h_terra}x{w_terra} -> {cdl_height}x{cdl_width}")
            
            # 创建新的数组并复制有效区域
            adjusted_terra = np.zeros((t_terra, cdl_height, cdl_width, c_terra), dtype=cond_imgs.dtype)
            min_h, min_w = min(h_terra, cdl_height), min(w_terra, cdl_width)
            adjusted_terra[:, :min_h, :min_w, :] = cond_imgs[:, :min_h, :min_w, :]
            cond_imgs = adjusted_terra
            
            logger.info(f"Terra 数据尺寸调整完成")
        
        logger.info(f"成功加载并调整 Terra 数据")
        logger.info(f"========== 站点 {site} 数据加载完成 ==========")
        
        return images, doys, cond_imgs, scl_imags
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到错误 - 站点 {site}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"加载站点 {site} 数据时发生未知错误: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        raise RuntimeError(f"无法加载站点 {site} 的数据") from e

def load_site_cdl_data(site, root, year):
    """
    按站点加载CDL标签数据
    
    Args:
        site (str): 站点名称
        root (Path): 数据根目录
        year (str): 年份
    
    Returns:
        rasterio.DatasetReader: CDL数据集对象
    
    Raises:
        FileNotFoundError: 当CDL文件不存在时
        RuntimeError: 当数据加载失败时
    """
    try:
        logger.info(f"正在加载站点 {site} 的 CDL 标签数据...")
        
        # 根据不同国家/地区构建CDL文件路径
        if Country == 'US':
            cdl_pth = root / site / (site + '_CDL_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
        elif Country == 'EU' or Country == 'CN':
            if site == 'Ukraine' or Country == 'CN':
                cdl_pth = root / site / (site + '_CDL_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
            else:
                cdl_pth = root / site / (site + '_HDA_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
        else:
            cdl_pth = root / site / (site + '_label_' + year + '_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)+'_'+year+'.tif')
        
        if not cdl_pth.exists():
            raise FileNotFoundError(f"CDL文件不存在: {cdl_pth}")
            
        cdl_dataset = rasterio.open(cdl_pth)
        logger.info(f"成功加载站点 {site} 的 CDL 数据")
        
        return cdl_dataset
        
    except FileNotFoundError as e:
        logger.error(f"CDL文件未找到 - 站点 {site}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"加载站点 {site} CDL数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"无法加载站点 {site} 的CDL数据") from e



# def process_block(row, labels, images, doys, cmapping, out_dir, year, pidx, cond_imgs, max_samples=None):
#     """处理单个区块的数据"""
#     geom = row.geometry
#     feature = [mapping(geom)]
#     site = row.origin
#     id = row.ID
#     mode = row['name']
    
#     # 检查站点是否在标签中
#     if site not in labels:
#         logger.warning(f"Site {site} not found in labels")
#         return [], pidx, []
    
#     # 创建输出目录
#     out_dir_s = out_dir / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)) / site
#     out_dir_s.mkdir(parents=True, exist_ok=True)

#     # 获取标签和confidence
#     out_label, out_transform = mask(labels[site], feature, crop=False)
#     if Country == "US" or Country == "EU" or Country == "CN":
#         out_conf = out_label[1].reshape(1, *out_label.shape[1:])#(1, 2200, 2217)
#     out_label = out_label[0].reshape(1, *out_label.shape[1:])#(1, 1349, 1753)
#     if data_to_mapping:
#         metadata = {
#             'site': site,
#             'year': year,
#             'country': Country,
#             'kernel_size': kernel_size,
#             'confidence_threshold': confidence_threshold,
#             'image_height': out_label.shape[1],
#             'image_width': out_label.shape[2],
#             'purpose': 'crop_mapping',
#             'all_samples_extracted': True
#         }
#         with open(out_dir_s/"metadata.json","w") as f:
#             json.dump(metadata, f, indent=2)

#     # 获取图像数据
#     s2_image = images[site]#(59, 1349, 1753, 11)
#     doy_valid = s2_image[:,:,:,-1]#(59, 1349, 1753)
#     s2_image = s2_image[:,:,:,:-1]#(59, 1349, 1753, 10)
#     doy = doys[site]#(59, )
#     t, w, h, c = s2_image.shape

#     # 筛选有效标签
#     # if Country != 'EU':
#     #     valid_ind = out_label > 0#(1, 1151, 1784)
#     # else:# EU
#     #     valid_ind = out_label >= 0
#         # 筛选有效标签
#     if data_to_mapping:
#         # 农作物制图模式：提取整张图的所有样本（包括背景像素）
#         if Country != 'EU' and Country != 'CN':
#             valid_ind = out_label >= 0  # 包括所有像素，包括标签为0的背景像素
#         else:  # EU
#             if Country == 'EU' and site != 'Ukraine':
#                 valid_ind = out_label > 0
#             else:
#                 valid_ind = out_label >= 0
#         logger.info(f"制图模式：提取整张图所有样本，区块 {id} ({site})")
#     else:
#         # 训练模式：只提取置信度高的样本
#         if Country != 'EU' and Country != 'CN':
#             valid_ind = out_label > 0  # 只提取有效标签（排除背景）
#         else:  # EU
#             if Country == 'EU' and site != 'Ukraine':
#                 valid_ind = out_label > 0
#             else:
#                 valid_ind = out_label >= 0
#         logger.info(f"训练模式：只提取有效标签样本，区块 {id} ({site})")
#     valid_lb = out_label[valid_ind]#(1156598,)
#     if Country == "US":
#         valid_conf = out_conf[valid_ind]
#     num_valid = valid_ind.sum()
#     # valid_indices = np.where(valid_ind[0])
    
#     # 如果没有有效标签，返回空列表
#     if num_valid == 0:
#         logger.warning(f"No valid labels found for block {id} ({site})")
#         return [], pidx, []
    
#     # 如果设置了最大样本数，随机选择样本点
#     if not data_to_mapping and max_samples is not None and max_samples < num_valid:
#         logger.info(f"限制区块 {id} ({site}) 的样本数量: {num_valid} -> {max_samples}")
#         # 随机选择索引
#         np.random.seed(random_seed + id)  # 使用区块ID作为随机种子的一部分，确保可重复性
#         sample_indices = np.random.choice(num_valid, max_samples, replace=False)
        
#         # 创建新的有效索引掩码
#         new_valid_ind = np.zeros_like(valid_ind, dtype=bool)
#         flat_indices = np.where(valid_ind.flatten())[0]
#         selected_flat_indices = flat_indices[sample_indices]
#         # new_valid_ind.flat[selected_flat_indices] = True
#         # 将平坦索引转换为多维索引
#         multi_indices = np.unravel_index(selected_flat_indices, valid_ind.shape)
#         new_valid_ind[multi_indices] = True
        
#         valid_ind = new_valid_ind
#         valid_lb = out_label[valid_ind]
#         if Country == "US":
#             valid_conf = out_conf[valid_ind]
#         num_valid = valid_ind.sum()

#     valid_indices = np.where(valid_ind[0])

#     # 释放不再需要的大型数组
#     del out_label
#     if Country == "US" and 'out_conf' in locals():
#         del out_conf
#     import gc
#     gc.collect()

#     valid_ind = np.repeat(valid_ind, t, axis=0)#(59, 1349, 1753)
#     valid_s2 = s2_image[valid_ind].reshape(t, -1, c)#(59, 1059188, 10)
#     valid_s2 = valid_s2.transpose(1, 0, 2).astype(float)#(1059188, 59, 10)
#     valid_doy_valid = doy_valid[valid_ind].reshape(t, -1)#(59, 1059188)
#     valid_doy_valid = valid_doy_valid.transpose(1, 0)#(1059188, 59)

#     # 筛选有效时间点
#     valid_ind_t = np.all(valid_s2 == 0, axis=2)#(1059188, 59) valid_s2 == 0 会返回一个布尔数组，标记每个元素是否为0 np.all(..., axis=2) 沿着第三个维度（波段维度）进行"与"操作，只有当某个时间点的所有波段都为0时，结果才为True
#     valid_s2[valid_ind_t] = np.nan#(1059188, 59, 10) 将所有波段都为0的时间点替换为 np.nan

#     # 获取气候数据
#     cond_img = cond_imgs[site]
#     valid_cond = cond_img[:, valid_ind[0, ...]]
#     valid_cond = valid_cond.transpose(1, 0, 2)

#     # 处理每个样本点
#     pindices = []
#     valid_doy_counts = []

#     # 创建数据收集器
#     x_data = []
#     cloud_doy_data = []
#     cond_data = []
#     y_data = []

#     chunk_dir = out_dir / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)) / site / 'chunks'
#     if chunk_dir.exists():
#         import shutil
#         shutil.rmtree(chunk_dir)
#     chunk_dir.mkdir(parents=True, exist_ok=True)
#     chunk_idx = row.ID
    
#     # 每处理一定数量的样本就保存一次，以减少内存占用
#     batch_size = 100000  # 可以根据实际情况调整
#     current_batch = 0
#     spatial_indices = np.zeros((num_valid, 2), dtype=np.int32)  # 存储空间位置

#     for pid in tqdm(range(num_valid), desc=f"Processing samples for block {id} ({site})"):
#         row, col = valid_indices[0][pid], valid_indices[1][pid]
#         # 记录空间位置
#         spatial_indices[pid] = [row, col]

#         #这里可以删掉保存为csv的操作，这里写在这里是为了观察数据
#         name = str(id) + str(pid + 1).zfill(5)#'100001'
#         out_fn = out_dir_s / (name + '.csv')#PosixPath('data/CAN-dataset/2019/CAN2/100001.csv')
#         out_cond_fn = out_dir_s / (name + '_cond.csv') 
#         out_cloud_doy_fn = out_dir_s/ (name + '_cloud_doy.csv')

#         img = valid_s2[pid]#(61, 10)
#         cond = valid_cond[pid]  # (8, 3)
#         valid_doy = valid_doy_valid[pid]#(61,)  valid_doy为 1表示当前像素点当前时序是拍摄到了照片点，即使后面可能因为云概率产品被 mask 掉当前时序的值为 nan
#         lb = int(valid_lb[pid])#
#         if Country == "US":
#             conf = int(valid_conf[pid])#100

#         # 筛选有效时间点（非 nan）
#         img_val_ind = np.all(~np.isnan(img), axis=1)#(61,)
#         if not data_to_mapping and img_val_ind.sum() < min_sequence_length:
#             continue

#         # 创建云遮挡掩码，而不是直接将值设为0,为了节省存储空间
#         # 如果当前时序波段为 nan，但是当前时序拍摄到了照片点，那么将当前时序的波段值设置为 0
#         # 这样做是为了保证时序数据的连续性
#         cloud_mask = np.all(np.isnan(img), axis=1) & (valid_doy == 1)#(59,) 

#         img_val = img[img_val_ind]#(12, 10)
#         doy_val = doy[img_val_ind].reshape(-1, 1)#(12, 1)
#         doy_cloud = doy[cloud_mask].reshape(-1,1)#(18, 1)

#         # 保存遥感数据CSV
#         # out_df = pd.DataFrame(np.hstack([img_val, doy_val]), columns=bandnames + ['doy'])
#         # out_df.to_csv(out_fn, index=False)
#         x_data.append(np.hstack([img_val, doy_val]))

#         #保存有云的 doy csv
#         # cloud_doy_df = pd.DataFrame(doy_cloud,columns = ['doy_cloud'])
#         # cloud_doy_df.to_csv(out_cloud_doy_fn, index = False)
#         cloud_doy_data.append(doy_cloud)

#         # 保存气候数据CSV
#         # cond_df = pd.DataFrame(cond, columns=['tmn', 'tmx', 'srad'])#'tmn'（最低温度）, 'tmx'（最高温度）, 'srad'（太阳辐射）
#         # cond_df.to_csv(out_cond_fn, index=False)
#         cond_data.append(cond)

#         # 获取类别信息
#         if lb in cmapping.index:
#             classid = cmapping.loc[lb, 'id']
#             classname = cmapping.loc[lb, 'classname']
#         else:
#             classid = 0
#             classname = 'other'
#         y_data.append(classid)

#         # 创建样本索引
#         if Country == "US":
#             pindex = {
#                 'idx': pidx, 
#                 'id': name, 
#                 'code': lb, 
#                 'confidence': conf, 
#                 'chunk_path': str(chunk_dir / f'chunk_{chunk_idx:04d}.npy'),
#                 'csv_path': out_fn,
#                 'chunk_idx': len(x_data) - 1,  # 在chunk中的索引
#                 'sequencelength': img_val.shape[0],
#                 'sequencelength_all': img_val.shape[0] + doy_cloud.shape[0],
#                 'classid': classid, 
#                 'classname': classname, 
#                 'region': site,
#                 'mode': mode,
#             }
#         else:
#             pindex = {
#                 'idx': pidx, 
#                 'id': name, 
#                 'code': lb, 
#                 'chunk_path': str(chunk_dir / f'chunk_{chunk_idx:04d}.npy'),
#                 'csv_path': out_fn,
#                 'chunk_idx': len(x_data) - 1,  # 在chunk中的索引
#                 'sequencelength': img_val.shape[0],
#                 'sequencelength_all': img_val.shape[0] + doy_cloud.shape[0],
#                 'classid': classid, 
#                 'classname': classname, 
#                 'region': site,
#                 'mode': mode,
#             }
#         pindices.append(pindex)
#         pidx += 1
#         valid_doy_counts.append(img_val.shape[0])

#         # 每处理batch_size个样本保存一次数据并清空内存
#         if len(x_data) >= batch_size:
#             batch_chunk_path = chunk_dir / f'chunk_{chunk_idx:04d}_batch_{current_batch:04d}'#PosixPath('data/US-dataset/2019/Arkansas/chunks/chunk_0007_batch_0000')
#             np.save(batch_chunk_path, np.array([x_data, cloud_doy_data,cond_data, y_data], dtype=object))
            
#             # 更新chunk路径
#             for i in range(len(pindices) - len(x_data), len(pindices)):
#                 pindices[i]['chunk_path'] = str(batch_chunk_path) + '.npy'
#                 pindices[i]['chunk_idx'] = i % batch_size
            
#             # 清空数据收集器
#             x_data = []
#             cloud_doy_data = []
#             cond_data = []
#             y_data = []
#             current_batch += 1
            
#             # 手动触发垃圾回收
#             gc.collect()
#     # 保存数据块
#     if x_data: # 只在有数据时保存
#         final_chunk_path = chunk_dir / f'chunk_{chunk_idx:04d}_batch_{current_batch:04d}'#PosixPath('data/US-dataset/2019_valid_kernel3_conf90/Texas2/chunks/chunk_0015_batch_0001')
#         np.save(final_chunk_path, np.array([x_data, cloud_doy_data, cond_data,  y_data], dtype=object))
#         # 更新最后一批的chunk路径
#         for i in range(len(pindices) - len(x_data), len(pindices)):
#             pindices[i]['chunk_path'] = str(final_chunk_path) + '.npy'
#             pindices[i]['chunk_idx'] = i % batch_size
#         np.save(chunk_dir / "spatial_indices.npy", spatial_indices)  # 保存空间位置
#     # 释放大型数组
#     del valid_s2, valid_doy_valid, valid_cond, valid_ind, spatial_indices
#     gc.collect()

#     return pindices, pidx, valid_doy_counts


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

def process_block2(row, labels, images, doys, scl_imags, cmapping, out_dir, year, pidx, cond_imgs, max_samples=None, use_sample_index=False, sample_index_root=None, stats=None, climate_stats=None):
    """
    Processes a single block of data, performs time series alignment, 
    and saves the output to a binary file.
    """
    geom = row.geometry
    feature = [mapping(geom)]
    site = row.origin
    id = row.ID
    mode = row['name']#'train'
    
    if site not in labels:
        logger.warning(f"Site {site} not found in labels")
        return [], pidx
    
    out_dir_s = out_dir / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold)) / site
    out_dir_s.mkdir(parents=True, exist_ok=True)

    out_label, _ = mask(labels[site], feature, crop=False)#(2, 1349, 1753) 
    if Country == "US" or Country == "EU" or Country == "CN":
        out_conf = out_label[1].reshape(1, *out_label.shape[1:])#(1, 1349, 1753)
    out_label = out_label[0].reshape(1, *out_label.shape[1:])#(1, 1349, 1753)

    if data_to_mapping: # 农作物制图模式：提取整张图的所有样本（包括背景像素）
        metadata = {
            'site': site,
            'year': year,
            'country': Country,
            'kernel_size': kernel_size,
            'confidence_threshold': confidence_threshold,
            'image_height': out_label.shape[1],
            'image_width': out_label.shape[2],
            'purpose': 'crop_mapping',
            'all_samples_extracted': True
        }
        with open(out_dir_s/"metadata.json","w") as f:
            json.dump(metadata, f, indent=2)

    s2_image = images[site]#(71, 1349, 1753, 12)
    doy_valid = s2_image[:,:,:,10]#(71, 1349, 1753)
    cloud_probability = s2_image[:,:,:,11]#(71, 1349, 1753)
    s2_image = s2_image[:,:,:,:10]#(71, 1349, 1753, 10)
    scl_image = scl_imags[site]#(1349, 1753,1)
    doy = doys[site]#(71,)
    t, w, h, c = s2_image.shape

    if data_to_mapping:
        # 农作物制图模式：提取整张图的所有样本（包括背景像素）
        if Country != 'EU' and Country != 'CN':
            valid_ind = out_label >= 0  # 包括所有像素，包括标签为0的背景像素(1, 1304, 2216)
        else:  # EU
            if Country == 'EU' and site != 'Ukraine':
                valid_ind = out_label > 0
            else:
                valid_ind = out_label >= 0
        logger.info(f"制图模式：提取整张图所有样本，区块 {id} ({site})")
    else:
        # 训练模式：只提取置信度高的样本
        if Country != 'EU' and Country != 'CN':
            valid_ind = out_label > 0  # 只提取有效标签（排除背景）(1, 1304, 2216)
        else:  # EU
            if Country == 'EU' and site != 'Ukraine':
                valid_ind = out_label > 0
            else:
                valid_ind = out_label >= 0
        logger.info(f"训练模式：只提取有效标签样本，区块 {id} ({site})")
    valid_lb = out_label[valid_ind]#(1142060,)  (1277642,)
    if Country == "US":
        valid_conf = out_conf[valid_ind]#(1142060,)
    num_valid = valid_ind.sum()#np.int64(1142060)
    
    if num_valid == 0:
        logger.warning(f"No valid labels found for block {id} ({site})")
        return [], pidx

    # 如果设置了最大样本数，随机选择样本点（新增：可按坐标优先筛选）
    if not data_to_mapping and max_samples is not None and max_samples < num_valid:
        logger.info(f"Sampling block {id} ({site}): {num_valid} -> {max_samples}")
        # 新增：优先按坐标选择
        if use_sample_index and sample_index_root is not None:
            # 先基于当前有效掩码取出坐标列表
            valid_indices0 = np.where(valid_ind[0])#[(1277642,),(1277642,)]
            spatial_coords = load_spatial_indices_for_block(sample_index_root, site, id, year, kernel_size, confidence_threshold)
            selected_idx = None
            if spatial_coords is not None:
                logger.info(f"Using spatial indices for block {id} ({site})")
                selected_idx = choose_indices_by_spatial(valid_indices0, spatial_coords, max_samples=max_samples, seed=(random_seed + id))
            if selected_idx is not None:
                new_valid_ind = np.zeros_like(valid_ind, dtype=bool)#(1, 1304, 2216)
                rows_sel = valid_indices0[0][selected_idx]
                cols_sel = valid_indices0[1][selected_idx]
                new_valid_ind[0, rows_sel, cols_sel] = True
                valid_ind = new_valid_ind#(1, 1304, 2216)
            else:
                logger.info(f"Using random sampling for block {id} ({site})")
                # 回退到原来的随机抽样
                np.random.seed(random_seed + id)
                sample_indices = np.random.choice(num_valid, max_samples, replace=False)
                new_valid_ind = np.zeros_like(valid_ind, dtype=bool)
                flat_indices = np.where(valid_ind.flatten())[0]
                selected_flat_indices = flat_indices[sample_indices]
                multi_indices = np.unravel_index(selected_flat_indices, valid_ind.shape)
                new_valid_ind[multi_indices] = True
                valid_ind = new_valid_ind#(1, 1304, 2216)
        else:
            # 原逻辑：随机抽样
            np.random.seed(random_seed + id)
            sample_indices = np.random.choice(num_valid, max_samples, replace=False)
            new_valid_ind = np.zeros_like(valid_ind, dtype=bool)
            flat_indices = np.where(valid_ind.flatten())[0]
            selected_flat_indices = flat_indices[sample_indices]
            multi_indices = np.unravel_index(selected_flat_indices, valid_ind.shape)
            new_valid_ind[multi_indices] = True
            valid_ind = new_valid_ind

        # 同步更新有效标签/置信度/数量
        valid_lb = out_label[valid_ind]
        if Country == "US":
            valid_conf = out_conf[valid_ind]#(600000,)
        num_valid = valid_ind.sum()

    valid_indices = np.where(valid_ind[0])
    del out_label
    if Country == "US" and 'out_conf' in locals():
        del out_conf
    import gc
    gc.collect()

    valid_ind = np.repeat(valid_ind, t, axis=0)#(71, 1349, 1753)
    valid_s2 = s2_image[valid_ind].reshape(t, -1, c).transpose(1, 0, 2).astype(float)#(600000, 71, 10)
    valid_scl = scl_image[valid_ind].reshape(t, -1, 1).transpose(1, 0, 2).astype(float)#(600000, 71, 1)
    valid_cloud_probability = cloud_probability[valid_ind].reshape(t, -1, 1).transpose(1, 0, 2).astype(float)#(600000, 71, 1)

    valid_doy_valid = doy_valid[valid_ind].reshape(t, -1)#(71, 600000)
    valid_doy_valid = valid_doy_valid.transpose(1, 0)#(600000, 71)#未使用，以前的代码有用，现在用不到了
    
    # 筛选有效时间点 发现不需要
    # valid_ind_t = np.all(valid_s2 == 0, axis=2)##(1059188, 59) valid_s2 == 0 会返回一个布尔数组，标记每个元素是否为0 np.all(..., axis=2) 沿着第三个维度（波段维度）进行"与"操作，只有当某个时间点的所有波段都为0时，结果才为True
    # valid_s2[valid_ind_t] = np.nan##(1059188, 59, 10) 将所有波段都为0的时间点替换为 np.nan

    # 获取气候数据
    cond_img = cond_imgs[site]#(8, 1349, 1753, 3)
    valid_cond = cond_img[:, valid_ind[0, ...]].transpose(1, 0, 2)#(600000, 8, 3)

    pindices = []
    all_samples_data = []

    spatial_indices = np.zeros((num_valid, 2), dtype=np.int32)  # 存储空间位置
    
    for pid in tqdm(range(num_valid), desc=f"Processing samples for block {id} ({site})"):
        row, col = valid_indices[0][pid], valid_indices[1][pid]
        # 记录空间位置
        spatial_indices[pid] = [row, col]

        img = valid_s2[pid]#(71, 10)
        scl = valid_scl[pid]#(71, 1)
        cloud_prob = valid_cloud_probability[pid]#(71, 1)
        cond = valid_cond[pid]#(8, 3)
        lb = int(valid_lb[pid])
        if Country == "US":
            conf = int(valid_conf[pid])#100

        # valid_ind_t = np.all(img == 0, axis=1)
        # tmp1= len(img)
        # temp2 = len(np.sum(valid_ind_t!=True))
        # if temp2 < min_sequence_length:
        #     continue

        # img_val = img[img_val_ind]
        # doy_val = doy[img_val_ind]
        img_val = img
        doy_val = doy
        scl_val = scl

        # Align time series data to target_length
        x_aligned, doy_aligned, scl_aligned, cloud_prob_aligned = align_timeseries(img_val, doy_val, scl_val, cloud_prob, target_length=75)#(75, 10)

        # 仅在训练模式且非制图时统计
        if (not data_to_mapping) and (mode == "train"):
            if stats is not None:
                stats.update(x_aligned)
            if climate_stats is not None:
                climate_stats.update(cond)
        
        # Align climate data（如需按 DOY 展开，可在此扩展）
        
        if lb in cmapping.index:
            classid = cmapping.loc[lb, 'id']
            classname = cmapping.loc[lb, 'classname']
        else:
            classid = 0
            classname = 'other'
        
        # Combine into a single flat array for binary storage
        # [label (1)] [x_features (75*10)] [cond_features (75*3)]
        sample_data = np.concatenate([
            np.array([classid], dtype=np.float32),
            x_aligned.flatten().astype(np.float32),
            doy_aligned.flatten().astype(np.float32),
            cond.flatten().astype(np.float32),
            scl_aligned.flatten().astype(np.float32),
            cloud_prob_aligned.flatten().astype(np.float32),
        ])#(850+75+75=1000,) 1+75*10+75+8*3+75+75
        all_samples_data.append(sample_data)

        if Country == "US":
            pindex = {
                'idx': pidx,
                'id': f"{id}{str(pid + 1).zfill(5)}",
                'code': lb,
                'confidence': conf,
                'bin_path': str(out_dir_s / f'block_{id}.bin'),
                'sample_idx': len(all_samples_data) - 1,
                'classid': classid,
                'classname': classname,
                'region': site,
                'mode': mode,
            }
        else:
            pindex = {
                'idx': pidx,
                'id': f"{id}{str(pid + 1).zfill(5)}",
                'code': lb,
                'bin_path': str(out_dir_s / f'block_{id}.bin'),
                'sample_idx': len(all_samples_data) - 1,
                'classid': classid,
                'classname': classname,
                'region': site,
                'mode': mode,
            }            
        pindices.append(pindex)
        pidx += 1

    if all_samples_data:
        bin_path = out_dir_s / f'block_{id}.bin'
        # Stack all samples and save to a single binary file
        final_data = np.stack(all_samples_data, axis=0)
        with open(bin_path, 'wb') as f:
            f.write(final_data.tobytes())
        logger.info(f"Saved {len(all_samples_data)} samples to {bin_path}")
        np.save(out_dir_s/"spatial_indices.npy", spatial_indices)

    del valid_s2, valid_cond, valid_ind
    gc.collect()

    return pindices, pidx


def load_class_mapping():
    """加载类别映射"""
    try:
        if not classmapping.exists():
            #如果 classmapping 所在的目录不存在，创建它
            classmapping.parent.mkdir(parents=True, exist_ok=True)
            #将scripts/preproces2/{Country}-classmapping.csv复制到classmapping
            shutil.copyfile(f"scripts/preproces2/{Country}-classmapping.csv", classmapping)

        cmapping = pd.read_csv(classmapping)
        cmapping = cmapping.set_index("code")
        classes = cmapping["id"].unique()
        classname = cmapping["classname"].unique()
        nclasses = len(classes)
        logger.info(f"Loaded {nclasses} classes: {classname}")
        return cmapping
    except Exception as e:
        logger.error(f"Error loading class mapping: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        return cmapping




def main():
    """
    主函数 - 使用按站点延迟加载优化内存使用
    
    该函数实现了内存优化的数据处理流程：
    1. 只在需要时加载特定站点的数据
    2. 处理完一个站点后立即释放其内存
    3. 避免同时在内存中保存所有站点的数据
    
    内存优化效果：
    - 显著减少峰值内存占用（从所有站点数据总和降低到单个站点数据大小）
    - 提高程序启动速度（无需等待所有数据加载完成）
    - 增强可扩展性（可处理更多站点而不受内存限制）
    - 提升错误恢复能力（单个站点错误不影响其他站点处理）
    """
    
    logger.info("========== 开始执行主处理流程 ==========")
    
    # 解析可选参数：按坐标选样的开关与索引根目录
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--use_sample_index', action='store_true', help='使用 data/US-dataset 下的空间坐标来选样')
    parser.add_argument('--sample_index_root', type=str, default='data/US-dataset', help='旧流程生成索引所在根目录')
    args, _ = parser.parse_known_args()
    use_sample_index = bool(args.use_sample_index)
    sample_index_root = Path(args.sample_index_root)
    
    try:
        # 1. 加载类别映射配置
        logger.info("正在加载类别映射配置...")
        cmapping = load_class_mapping()
        logger.info("类别映射配置加载完成")
        feature_stats = OnlineFeatureStats(n_features=10, scale=1e-4)
        climate_stats = OnlineClimateStats(n_features=3)
        
        # 2. 加载区块分割数据
        logger.info("正在加载区块分割数据...")
        try:
            blockpartition = gpd.read_file(shp_fn)
            # if blockpartition.crs != 'EPSG:4326':
            #     logger.info(f"正在转换坐标系: {blockpartition.crs} -> EPSG:4326")
            #     blockpartition = blockpartition.to_crs('EPSG:4326')
            logger.info(f"成功加载 {len(blockpartition)} 个区块")
        except Exception as e:
            logger.error(f"加载区块分割数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
            return
        
        # 3. 初始化处理统计变量
        pidx = 0  # 全局样本索引
        all_pindices = []  # 所有样本索引列表
        total_sites = len(sites)
        
        logger.info('========== 开始按站点处理数据 ==========')
        logger.info(f"计划处理 {total_sites} 个站点: {sites}")
        
        # 4. 按站点顺序处理（延迟加载优化）
        for site_idx, site in enumerate(sites, 1):
            logger.info(f"{'='*60}")
            logger.info(f"正在处理站点 {site} ({site_idx}/{total_sites})")
            logger.info(f"{'='*60}")
            
            # 4.1 检查该站点是否有对应的区块
            site_blocks = blockpartition[blockpartition['origin'] == site]
            if site_blocks.empty:
                logger.warning(f"站点 {site} 没有找到对应的区块，跳过处理")
                continue
                
            logger.info(f"站点 {site} 包含 {len(site_blocks)} 个区块")
            
            # 4.2 延迟加载：只加载当前站点的数据
            try:
                logger.info(f"开始加载站点 {site} 的数据...")
                
                # 加载CDL标签数据
                site_label = load_site_cdl_data(site, root, year)
                
                # 加载图像数据（S2 + Terra）
                site_images, site_doys, site_cond_imgs, site_scl_imags = load_site_image_data(site, root, year)
                
                # 将当前站点数据包装为字典格式（保持与原有process_block函数的兼容性）
                current_labels = {site: site_label}
                current_images = {site: site_images}
                current_doys = {site: site_doys}
                current_cond_imgs = {site: site_cond_imgs}
                current_scl_imgs = {site: site_scl_imags}
                
                logger.info(f"站点 {site} 数据加载完成")
                
            except Exception as e:
                logger.error(f"加载站点 {site} 数据失败: {str(e)}")
                import traceback
                traceback.print_exc()
                logger.error(f"跳过站点 {site} 的处理")
                continue
            
            # 4.3 处理当前站点的所有区块
            try:
                logger.info(f"开始处理站点 {site} 的 {len(site_blocks)} 个区块...")
                
                for block_idx, (idx, row) in enumerate(site_blocks.iterrows(), 1):
                    logger.info(f"\n--- 处理区块 {row.ID} ({block_idx}/{len(site_blocks)}) ---")
                    
                    # 检查区块是否在有效区域列表中
                    if row.origin not in ALL_REGIONS:
                        logger.warning(f"区块 {row.ID} ({row.origin}) 不在有效区域列表中，跳过")
                        continue
                    
                    # 获取区域特定配置
                    region_config = ALL_REGIONS[row.origin]
                    if not data_to_mapping:
                        max_samples = region_config.get('max_samples', None)
                    else:
                        max_samples = None
                    
                    if max_samples:
                        logger.info(f"区域 {row.origin} 设置最大样本数限制: {max_samples}")
                    
                    # 处理当前区块
                    try:
                        block_pindices, pidx = process_block2(
                            row, current_labels, current_images, current_doys, current_scl_imgs, 
                            cmapping, out_dir, year, pidx, current_cond_imgs, max_samples,
                            use_sample_index=use_sample_index, sample_index_root=sample_index_root,
                            stats=feature_stats,
                            climate_stats=climate_stats
                        )
                        
                        # 累积样本索引
                        all_pindices.extend(block_pindices)
                        
                        # 记录处理统计信息
                        if block_pindices:
                            logger.info(f"区块 {row.ID} 处理完成:")
                            logger.info(f"  - 新增样本索引: {len(block_pindices)}")
                        else:
                            logger.info(f"区块 {row.ID} 未产生有效样本")
                            
                    except Exception as e:
                        logger.error(f"处理区块 {row.ID} 时发生错误: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        logger.error(f"继续处理下一个区块...")
                        continue
                    
                    # 4.4 区块级内存清理（可选，用于处理超大区块）
                    if block_idx % 10 == 0:  # 每处理10个区块进行一次轻量级内存清理
                        import gc
                        gc.collect()
                        logger.debug(f"已处理 {block_idx} 个区块，执行内存清理")
                
                logger.info(f"站点 {site} 的所有区块处理完成")
                
            except Exception as e:
                logger.error(f"处理站点 {site} 的区块时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                logger.error(f"继续处理下一个站点...")
            
            finally:
                # 4.5 站点级内存清理（关键优化点）
                logger.info(f"正在释放站点 {site} 的内存...")
                
                try:
                    # 关闭并删除CDL数据集
                    if 'current_labels' in locals() and site in current_labels:
                        current_labels[site].close()
                        del current_labels[site]
                    
                    # 删除图像数据
                    if 'current_images' in locals() and site in current_images:
                        del current_images[site]
                    
                    # 删除DOY数据
                    if 'current_doys' in locals() and site in current_doys:
                        del current_doys[site]
                    
                    # 删除Terra数据
                    if 'current_cond_imgs' in locals() and site in current_cond_imgs:
                        del current_cond_imgs[site]
                    
                    # 删除整个字典对象
                    if 'current_labels' in locals():
                        del current_labels
                    if 'current_images' in locals():
                        del current_images
                    if 'current_doys' in locals():
                        del current_doys
                    if 'current_cond_imgs' in locals():
                        del current_cond_imgs
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    
                    logger.info(f"站点 {site} 内存释放完成")
                    
                except Exception as e:
                    logger.warning(f"释放站点 {site} 内存时发生警告: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # 5. 保存最终的索引文件
        logger.info("\n========== 保存处理结果 ==========")
        
        if all_pindices:
            # 从字典列表中提取需要的信息
            try:
                index_dir = out_dir / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold))
                out_fn = str(index_dir / 'index.csv')#'data/data-no-mask-cloud/US-dataset/2019_valid_kernel3_conf90/index.csv'
                
                # 将字典列表转换为DataFrame
                index_df = pd.DataFrame(all_pindices)
                index_df.to_csv(out_fn, index=False)
                
                logger.info(f"成功保存索引文件: {out_fn}")
                logger.info(f"总计处理样本数量: {len(all_pindices)}")
                
                # 输出处理统计信息
                logger.info("\n========== 处理统计信息 ==========")
                logger.info(f"处理站点数量: {len([s for s in sites if any(blockpartition['origin'] == s)])}")
                logger.info(f"总样本数量: {len(all_pindices)}")
                logger.info(f"索引文件路径: {out_fn}")
                
            except (KeyError, IndexError) as e:
                logger.error(f"处理索引数据时出错: {str(e)}")
                logger.error(f"all_pindices结构: {type(all_pindices)}, 长度: {len(all_pindices) if hasattr(all_pindices, '__len__') else 'unknown'}")
                if len(all_pindices) > 0:
                    logger.error(f"第一个元素类型: {type(all_pindices[0])}, 内容: {all_pindices[0]}")
                raise
                
        else:
            logger.warning("没有生成任何有效样本，未创建索引文件")
            # 创建一个空文件以表明处理完成
            index_dir = out_dir / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold))
            index_dir.mkdir(parents=True, exist_ok=True)
            out_fn = str(index_dir / 'index_empty.csv')
            pd.DataFrame(columns=['bin_path', 'sample_idx', 'label']).to_csv(out_fn, index=False)
            logger.info(f"创建了空索引文件: {out_fn}")
        
        # ===== 新增：按 mode 分片合并 block_*.bin，并重写索引 =====
        root_ds = out_dir / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold))
        index_df = pd.read_csv(str(root_ds / 'index.csv'))
        if 'index_df' in locals() and len(index_df) > 0 and 'mode' in index_df.columns:
            index_df['orig_bin_path'] = index_df['bin_path']
            mode_values = sorted(index_df['mode'].dropna().unique().tolist())

            def _block_sort_key(pstr):
                p = Path(pstr)#'data/data-no-mask-cloud/US-dataset/2019_valid_kernel3_conf90/Indiana/block_1.bin'
                try:
                    bid = int(p.name.split('_')[1].split('.')[0])
                except Exception:
                    bid = p.name
                return (p.parent.name, bid)

            shards_meta = {}
            for mode in mode_values:
                mode_mask = index_df['mode'] == mode
                mode_df = index_df.loc[mode_mask].copy()
                bin_files = mode_df['orig_bin_path'].unique().tolist()
                if not bin_files:
                    logger.warning(f'[{mode}] 未找到块文件，跳过分片合并。')
                    continue

                # 轻量：统计每个块的样本数，并按块排序
                counts_per_block = mode_df['orig_bin_path'].value_counts().to_dict()
                bin_files_sorted = sorted(bin_files, key=_block_sort_key)

                # 为每个块计算全局起始偏移（累计样本数）
                base_offsets = {}
                offset = 0
                for bp in bin_files_sorted:
                    base_offsets[bp] = offset
                    offset += int(counts_per_block.get(bp, 0))

                # 向量化更新每条样本的全局索引与分片位置
                mode_df['base_offset'] = mode_df['orig_bin_path'].map(base_offsets).astype(np.int64)
                mode_df['global_idx'] = (mode_df['base_offset'] + mode_df['sample_idx'].astype(np.int64))
                mode_df['shard_id'] = (mode_df['global_idx'] // SAMPLES_PER_SHARD).astype(np.int64)
                mode_df['sample_idx'] = (mode_df['global_idx'] % SAMPLES_PER_SHARD).astype(np.int64)
                mode_df['bin_path'] = mode_df['shard_id'].map(
                    lambda sid: str(root_ds / f'mode_{mode}_merged_shard{int(sid):03d}.bin')
                )

                # 回写到 index_df
                index_df.loc[mode_mask, ['bin_path', 'sample_idx']] = mode_df[['bin_path', 'sample_idx']].values
                index_df.loc[mode_mask, 'global_idx'] = mode_df['global_idx'].values
                index_df.loc[mode_mask, 'shard_id'] = mode_df['shard_id'].values

                # 物理分片合并（二进制流式复制，按样本边界切分）
                shard_id = 0
                current_shard_count = int(0)
                shard_path = root_ds / f'mode_{mode}_merged_shard{shard_id:03d}.bin'
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                out_f = open(shard_path, 'wb')

                try:
                    for bp in bin_files_sorted:
                        count = int(counts_per_block.get(bp, 0))
                        if count <= 0:
                            continue
                        with open(bp, 'rb') as in_f:
                            remaining = count
                            while remaining > 0:
                                if current_shard_count >= SAMPLES_PER_SHARD:
                                    out_f.close()
                                    shard_id += 1
                                    current_shard_count = 0
                                    shard_path = root_ds / f'mode_{mode}_merged_shard{shard_id:03d}.bin'
                                    out_f = open(shard_path, 'wb')

                                space = int(SAMPLES_PER_SHARD - current_shard_count)
                                to_write = int(min(space, remaining))
                                bytes_left = int(to_write * RECORD_SIZE_BYTES)

                                while bytes_left > 0:
                                    step_samples = int(min(CHUNK_SAMPLES, bytes_left // RECORD_SIZE_BYTES))
                                    step_bytes = step_samples * RECORD_SIZE_BYTES
                                    buf = in_f.read(step_bytes)
                                    if not buf:
                                        break
                                    out_f.write(buf)
                                    bytes_left -= step_bytes
                                    current_shard_count += step_samples
                                    remaining -= step_samples
                finally:
                    try:
                        out_f.close()
                    except Exception:
                        pass

                # 记录分片元数据
                shards_meta[mode] = {
                    'record_size_bytes': int(RECORD_SIZE_BYTES),
                    'samples_per_shard': int(SAMPLES_PER_SHARD),
                    'shards': sorted(index_df.loc[index_df['mode'] == mode]['bin_path'].unique().tolist()),
                }

            # 写回索引与分片元数据
            index_df = index_df.drop(columns=['orig_bin_path'])
            out_fn = str(root_ds / 'index.csv')
            index_df.to_csv(out_fn, index=False)
            with open(root_ds / 'shards_meta.json', 'w') as f:
                json.dump(shards_meta, f, indent=2, ensure_ascii=False)
            logger.info(f'已按 mode 分片合并并重写索引: {out_fn}')
        else:
            logger.warning('index_df 不存在或缺少 mode 列，跳过分片合并。')
        
        # # 合并所有站点的 block_*.bin 到一个全局二进制文件（流式拷贝，避免占用内存）
        # root_ds = out_dir / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold))
        # global_bin_path = root_ds / 'all_blocks_merged.bin'

        # # 递归查找所有块文件
        # bin_files = list(root_ds.rglob('block_*.bin'))
        # if not bin_files:
        #     logger.warning('未找到 block_*.bin 文件，跳过合并。')
        # else:
        #     # 为了确定性，按站点名和块ID排序（站点名->数字ID）
        #     def _block_sort_key(p):
        #         try:
        #             bid = int(p.name.split('_')[1].split('.')[0])
        #         except Exception:
        #             bid = p.name
        #         return (p.parent.name, bid)

        #     bin_files = sorted(bin_files, key=_block_sort_key)
        #     logger.info(f'将 {len(bin_files)} 个块文件合并到 {global_bin_path}')
        #     global_bin_path.parent.mkdir(parents=True, exist_ok=True)

        #     # 流式合并，16MB 缓冲
        #     with open(global_bin_path, 'wb') as out_f:
        #         for bp in bin_files:
        #             logger.info(f'合并 {bp}')
        #             with open(bp, 'rb') as in_f:
        #                 shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024)

        #     # 合并完毕，打印总大小
        #     total_bytes = os.path.getsize(global_bin_path)
        #     logger.info(f'合并完成: {global_bin_path} (总计 {total_bytes} 字节)')
        
        # ===== 写出均值/标准差到 norm_stats.json =====
        try:
            mean_x, std_x, cnt_x = feature_stats.finalize()
            if 'climate_stats' in locals() and climate_stats is not None:
                mean_c, std_c, cnt_c = climate_stats.finalize()
            else:
                mean_c, std_c, cnt_c = np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64), 0
            norm_out_dir = out_dir / (str(year)+'_valid_kernel'+str(kernel_size)+'_conf'+str(confidence_threshold))
            norm_out_dir.mkdir(parents=True, exist_ok=True)
            norm_stats = {
                "bands": bandnames,
                "mean": mean_x.tolist(),
                "std": std_x.tolist(),
                "count": int(cnt_x),
                "scale_applied": 1e-4,
                "climate_vars": ["tmn", "tmx", "srad"],
                "mean_c": mean_c.tolist(),
                "std_c": std_c.tolist(),
                "count_c": int(cnt_c),
                "note": "均值/标准差按特征维度统计，遥感排除置零时序（all-zeros rows），气候排除全零月份行；默认仅统计训练集样本。"
            }
            with open(norm_out_dir / "norm_stats.json", "w") as f:
                json.dump(norm_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"归一化统计已保存到 {norm_out_dir/'norm_stats.json'}")
            logger.info(f"RS mean: {norm_stats['mean']}")
            logger.info(f"RS std:  {norm_stats['std']}")
            logger.info(f"RS rows used in stats: {cnt_x}")
            logger.info(f"Climate mean: {norm_stats['mean_c']}")
            logger.info(f"Climate std:  {norm_stats['std_c']}")
            logger.info(f"Climate rows used in stats: {cnt_c}")
        except Exception as e:
            logger.warning(f"写出均值/标准差失败：{e}")

        logger.info("========== 主处理流程完成 ==========")
        
    except KeyboardInterrupt:
        logger.warning("用户中断了程序执行")
        return
    except Exception as e:
        logger.error(f"主处理流程发生严重错误: {str(e)}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()