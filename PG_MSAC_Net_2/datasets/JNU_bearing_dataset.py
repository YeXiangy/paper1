"""
JNU_bearing_dataset.py
JNU轴承数据集处理模块 - 优化版

优化内容:
1. 支持懒加载模式，减少内存使用
2. 改进错误处理和日志记录
3. 支持数据缓存机制
4. 更好的跨平台兼容性
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, List, Optional, Union
import warnings
import logging
from pathlib import Path
import pickle
import hashlib
import pandas as pd  # 添加这一行

warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)


class JNUBearingDataset(Dataset):
    """JNU轴承数据集类 - 优化版"""

    def __init__(self,
                 data_path: str,
                 speed: int = 800,
                 shot: int = 5,
                 sample_len: int = 1024,
                 normalize_type: str = 'mean~std',
                 overlap: float = 0.5,
                 mode: str = 'train',
                 fault_types: List[str] = None,
                 lazy_loading: bool = True,
                 cache_data: bool = True):
        """
        初始化JNU轴承数据集

        Args:
            data_path: 数据路径
            speed: 转速 (600, 800, 1000)
            shot: 每类样本数量
            sample_len: 样本长度
            normalize_type: 标准化类型 ('mean~std', 'min~max', 'none')
            overlap: 滑窗重叠率
            mode: 模式 ('train', 'test')
            fault_types: 故障类型列表
            lazy_loading: 是否使用懒加载
            cache_data: 是否缓存数据
        """
        self.data_path = Path(data_path)
        self.speed = speed
        self.shot = shot
        self.sample_len = sample_len
        self.normalize_type = normalize_type
        self.overlap = overlap
        self.mode = mode
        self.lazy_loading = lazy_loading
        self.cache_data = cache_data

        # 默认故障类型
        if fault_types is None:
            self.fault_types = ['Normal', 'InnerRace', 'OuterRace', 'Ball']
        else:
            self.fault_types = fault_types

        self.num_classes = len(self.fault_types)

        # 类别映射
        self.class_to_idx = {fault: idx for idx, fault in enumerate(self.fault_types)}
        self.idx_to_class = {idx: fault for fault, idx in self.class_to_idx.items()}

        # 缓存相关
        self.cache_dir = self.data_path / 'cache'
        self.cache_dir.mkdir(exist_ok=True)

        # 数据存储
        if self.lazy_loading:
            # 懒加载模式：只存储文件路径和索引
            self.sample_indices = self._build_sample_indices()
            self.data = None
            self.labels = None
        else:
            # 传统模式：加载所有数据到内存
            self.data, self.labels = self._load_data()

        logger.info(f"JNU Dataset initialized:")
        logger.info(f"  Speed: {self.speed} rpm")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Samples per class: {self.shot}")
        logger.info(f"  Total samples: {len(self)}")
        logger.info(f"  Sample length: {self.sample_len}")
        logger.info(f"  Classes: {self.fault_types}")
        logger.info(f"  Lazy loading: {self.lazy_loading}")

    def _build_sample_indices(self) -> List[Dict]:
        """构建样本索引（用于懒加载）"""
        sample_indices = []

        for fault_idx, fault_type in enumerate(self.fault_types):
            file_path = self._get_file_path(fault_type)

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            # 检查缓存
            cache_key = self._get_cache_key(file_path, fault_type)
            cached_indices = self._load_cached_indices(cache_key)

            if cached_indices is not None:
                logger.info(f"Loaded cached indices for {fault_type}")
                sample_indices.extend(cached_indices)
            else:
                # 生成新的索引
                try:
                    raw_data = self._load_raw_file(file_path)
                    if raw_data is not None:
                        indices = self._generate_sample_indices(
                            file_path, raw_data, fault_idx, fault_type
                        )
                        sample_indices.extend(indices)

                        # 缓存索引
                        if self.cache_data:
                            self._save_cached_indices(cache_key, indices)

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    continue

        return sample_indices

    def _generate_sample_indices(self, file_path: Path, raw_data: np.ndarray,
                                 fault_idx: int, fault_type: str) -> List[Dict]:
        """生成样本索引"""
        indices = []

        if len(raw_data) < self.sample_len:
            # 数据长度不足，进行重复
            repeat_times = self.sample_len // len(raw_data) + 1
            raw_data = np.tile(raw_data, repeat_times)

        # 计算步长
        step = int(self.sample_len * (1 - self.overlap)) if self.overlap > 0 else self.sample_len

        # 生成索引
        start = 0
        sample_count = 0

        while start + self.sample_len <= len(raw_data) and sample_count < self.shot:
            index_info = {
                'file_path': str(file_path),
                'start_idx': start,
                'end_idx': start + self.sample_len,
                'label': fault_idx,
                'fault_type': fault_type
            }
            indices.append(index_info)
            start += step
            sample_count += 1

        # 如果样本不够，进行随机采样补充
        while len(indices) < self.shot:
            random_start = np.random.randint(0, max(1, len(raw_data) - self.sample_len))
            index_info = {
                'file_path': str(file_path),
                'start_idx': random_start,
                'end_idx': random_start + self.sample_len,
                'label': fault_idx,
                'fault_type': fault_type
            }
            indices.append(index_info)

        # 如果样本太多，随机选择
        if len(indices) > self.shot:
            selected_indices = np.random.choice(len(indices), self.shot, replace=False)
            indices = [indices[i] for i in selected_indices]

        return indices

    def _get_cache_key(self, file_path: Path, fault_type: str) -> str:
        """生成缓存键"""
        # 使用文件路径、配置参数生成唯一键
        key_string = f"{file_path}_{self.speed}_{self.shot}_{self.sample_len}_{self.overlap}_{fault_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_cached_indices(self, cache_key: str) -> Optional[List[Dict]]:
        """加载缓存的索引"""
        if not self.cache_data:
            return None

        cache_file = self.cache_dir / f"indices_{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {str(e)}")
                # 删除损坏的缓存文件
                cache_file.unlink(missing_ok=True)

        return None

    def _save_cached_indices(self, cache_key: str, indices: List[Dict]):
        """保存索引到缓存"""
        if not self.cache_data:
            return

        cache_file = self.cache_dir / f"indices_{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(indices, f)
            logger.debug(f"Cached indices to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_file}: {str(e)}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据（传统模式）"""
        all_data = []
        all_labels = []

        for fault_idx, fault_type in enumerate(self.fault_types):
            file_path = self._get_file_path(fault_type)

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            try:
                fault_data = self._load_raw_file(file_path)
                if fault_data is None:
                    logger.warning(f"Failed to load data from {file_path}")
                    continue

                # 生成样本
                samples = self._generate_samples(fault_data)

                # 根据shot限制样本数量
                if len(samples) < self.shot:
                    # 如果样本不够，进行重复采样
                    indices = np.random.choice(len(samples), self.shot, replace=True)
                    samples = samples[indices]
                else:
                    # 随机选择shot个样本
                    indices = np.random.choice(len(samples), self.shot, replace=False)
                    samples = samples[indices]

                # 添加到总数据
                all_data.append(samples)
                all_labels.extend([fault_idx] * len(samples))

            except Exception as e:
                logger.error(f"Error processing {fault_type}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No valid data found in the dataset")

        # 合并数据
        data = np.vstack(all_data)
        labels = np.array(all_labels)

        # 数据标准化
        data = self._normalize_data(data)

        return data, labels

    def _get_file_path(self, fault_type: str) -> Path:
        """根据故障类型和转速获取文件路径 - 适配CSV命名格式"""

        # 新的故障类型映射（适配你的文件命名）
        fault_mapping = {
            'Normal': 'n',  # normal -> n
            'InnerRace': 'ib',  # inner race -> ib
            'OuterRace': 'ob',  # outer race -> ob
            'Ball': 'rb'  # rolling ball -> rb
        }

        fault_prefix = fault_mapping.get(fault_type, fault_type.lower())

        # 支持多种文件命名格式
        possible_filenames = []

        if fault_type == 'Normal':
            # 正常状态文件可能有不同后缀
            possible_filenames = [
                f"n{self.speed}_3_2.csv",
                f"n{self.speed}_2.csv",
                f"n{self.speed}.csv"
            ]
        else:
            # 故障状态文件
            possible_filenames = [
                f"{fault_prefix}{self.speed}_2.csv",
                f"{fault_prefix}{self.speed}.csv"
            ]

        # 查找存在的文件
        for filename in possible_filenames:
            file_path = self.data_path / filename
            if file_path.exists():
                return file_path

        # 如果都没找到，返回第一个作为默认值
        return self.data_path / possible_filenames[0]

        fault_str = fault_mapping.get(fault_type, fault_type.lower())

        # 支持多种文件命名格式
        possible_filenames = [
            f"JNU_{fault_str}_{self.speed}rpm.mat",
            f"JNU_{fault_str}_{self.speed}.mat",
            f"{fault_str}_{self.speed}rpm.mat",
            f"{fault_str}_{self.speed}.mat",
            f"{fault_type}_{self.speed}.mat"
        ]

        for filename in possible_filenames:
            file_path = self.data_path / filename
            if file_path.exists():
                return file_path

        # 如果都没找到，返回第一个作为默认值
        return self.data_path / possible_filenames[0]

    def _load_raw_file(self, file_path: Path) -> Optional[np.ndarray]:
        """加载原始文件数据 - 支持CSV和MAT格式"""
        try:
            if file_path.suffix.lower() == '.csv':
                # 加载CSV文件
                import pandas as pd

                try:
                    # 读取CSV文件
                    data = pd.read_csv(str(file_path), header=None)

                    # 如果只有一列，直接使用
                    if data.shape[1] == 1:
                        signal_data = data.iloc[:, 0].values
                    else:
                        # 如果有多列，通常取最后一列（信号数据）
                        signal_data = data.iloc[:, -1].values

                    # 确保数据是一维的
                    if len(signal_data.shape) > 1:
                        signal_data = signal_data.flatten()

                    # 移除可能的无效值
                    signal_data = signal_data[~np.isnan(signal_data)]

                    return signal_data.astype(np.float32)

                except Exception as e:
                    logger.error(f"Error reading CSV {file_path}: {str(e)}")
                    return None

            elif file_path.suffix.lower() == '.mat':
                # 保持原有的MAT文件加载逻辑
                import scipy.io as sio
                mat_data = sio.loadmat(str(file_path))

                # 尝试常见的变量名
                possible_keys = ['data', 'X', 'vibration', 'signal', 'DE_time', 'value']

                # 首先尝试自动检测
                for key in mat_data.keys():
                    if not key.startswith('_'):  # 忽略元数据
                        data = mat_data[key]
                        if isinstance(data, np.ndarray) and data.size > 1000:
                            if data.ndim > 1:
                                data = data.flatten()
                            return data.astype(np.float32)

                # 如果自动检测失败，尝试预定义的键
                for key in possible_keys:
                    if key in mat_data:
                        data = mat_data[key]
                        if data.ndim > 1:
                            data = data.flatten()
                        return data.astype(np.float32)

            logger.warning(f"Unsupported file format: {file_path}")
            return None

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None

            # 尝试常见的变量名
            possible_keys = ['data', 'X', 'vibration', 'signal', 'DE_time', 'value']

            # 首先尝试自动检测
            for key in mat_data.keys():
                if not key.startswith('_'):  # 忽略元数据
                    data = mat_data[key]
                    if isinstance(data, np.ndarray) and data.size > 1000:
                        # 确保是一维数据
                        if data.ndim > 1:
                            data = data.flatten()
                        return data.astype(np.float32)

            # 如果自动检测失败，尝试预定义的键
            for key in possible_keys:
                if key in mat_data:
                    data = mat_data[key]
                    if data.ndim > 1:
                        data = data.flatten()
                    return data.astype(np.float32)

            logger.warning(f"No suitable data found in {file_path}")
            return None

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None

    def _generate_samples(self, raw_data: np.ndarray) -> np.ndarray:
        """从原始数据生成固定长度样本"""
        if len(raw_data) < self.sample_len:
            # 如果数据长度不足，进行重复
            repeat_times = self.sample_len // len(raw_data) + 1
            raw_data = np.tile(raw_data, repeat_times)

        # 计算步长
        step = int(self.sample_len * (1 - self.overlap)) if self.overlap > 0 else self.sample_len

        # 生成样本
        samples = []
        start = 0

        while start + self.sample_len <= len(raw_data):
            sample = raw_data[start:start + self.sample_len]
            samples.append(sample)
            start += step

            # 限制样本数量，避免内存问题
            if len(samples) >= 1000:
                break

        if len(samples) == 0:
            # 如果没有生成任何样本，直接截取
            samples = [raw_data[:self.sample_len]]

        return np.array(samples, dtype=np.float32)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据标准化"""
        if self.normalize_type == 'none':
            return data

        original_shape = data.shape
        data_2d = data.reshape(-1, self.sample_len)

        if self.normalize_type == 'mean~std':
            # 逐样本标准化
            normalized_data = np.zeros_like(data_2d, dtype=np.float32)
            for i in range(data_2d.shape[0]):
                sample = data_2d[i]
                mean_val = np.mean(sample)
                std_val = np.std(sample)
                if std_val > 1e-8:
                    normalized_data[i] = (sample - mean_val) / std_val
                else:
                    normalized_data[i] = sample - mean_val

        elif self.normalize_type == 'min~max':
            # 逐样本min-max标准化
            normalized_data = np.zeros_like(data_2d, dtype=np.float32)
            for i in range(data_2d.shape[0]):
                sample = data_2d[i]
                min_val = np.min(sample)
                max_val = np.max(sample)
                if max_val > min_val:
                    normalized_data[i] = (sample - min_val) / (max_val - min_val)
                else:
                    normalized_data[i] = sample

        elif self.normalize_type == 'global_std':
            # 全局标准化
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data_2d).astype(np.float32)

        else:
            normalized_data = data_2d.astype(np.float32)

        return normalized_data.reshape(original_shape)

    def __len__(self) -> int:
        if self.lazy_loading:
            return len(self.sample_indices)
        else:
            return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        if self.lazy_loading:
            return self._get_lazy_item(idx)
        else:
            return self._get_cached_item(idx)

    def _get_lazy_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """懒加载模式获取样本"""
        index_info = self.sample_indices[idx]

        # 加载数据片段
        try:
            raw_data = self._load_raw_file(Path(index_info['file_path']))
            if raw_data is None:
                # 如果加载失败，返回零张量
                sample = np.zeros(self.sample_len, dtype=np.float32)
            else:
                start_idx = index_info['start_idx']
                end_idx = index_info['end_idx']
                sample = raw_data[start_idx:end_idx]

                # 标准化
                if self.normalize_type == 'mean~std':
                    mean_val = np.mean(sample)
                    std_val = np.std(sample)
                    if std_val > 1e-8:
                        sample = (sample - mean_val) / std_val
                    else:
                        sample = sample - mean_val
                elif self.normalize_type == 'min~max':
                    min_val = np.min(sample)
                    max_val = np.max(sample)
                    if max_val > min_val:
                        sample = (sample - min_val) / (max_val - min_val)
                    else:
                        sample = sample

        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {str(e)}")
            sample = np.zeros(self.sample_len, dtype=np.float32)

        # 转换为张量
        sample_tensor = torch.FloatTensor(sample).unsqueeze(0)  # [1, sample_len]
        label_tensor = torch.LongTensor([index_info['label']])[0]  # 标量

        return sample_tensor, label_tensor

    def _get_cached_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """缓存模式获取样本"""
        sample = self.data[idx]
        label = self.labels[idx]

        # 转换为张量并添加通道维度
        sample_tensor = torch.FloatTensor(sample).unsqueeze(0)  # [1, sample_len]
        label_tensor = torch.LongTensor([label])[0]  # 标量

        return sample_tensor, label_tensor

    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        distribution = {}

        if self.lazy_loading:
            for fault_type in self.fault_types:
                count = sum(1 for info in self.sample_indices if info['fault_type'] == fault_type)
                distribution[fault_type] = count
        else:
            for i, fault_type in enumerate(self.fault_types):
                count = np.sum(self.labels == i)
                distribution[fault_type] = count

        return distribution

    def clear_cache(self):
        """清空缓存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("Dataset cache cleared")


def create_jnu_dataloaders(
        data_path: str,
        source_speed: int = 800,
        target_speed: int = 1000,
        source_shot: int = 50,
        target_shot: int = 5,
        sample_len: int = 1024,
        batch_size: int = 64,
        normalize_type: str = 'mean~std',
        num_workers: int = 4,
        fault_types: List[str] = None,
        lazy_loading: bool = True
) -> Dict[str, DataLoader]:
    """
    创建JNU数据集的数据加载器（用于跨域实验）
    """
    # 检查数据路径
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # 创建源域训练集
    source_train_dataset = JNUBearingDataset(
        data_path=str(data_path),
        speed=source_speed,
        shot=source_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='train',
        fault_types=fault_types,
        lazy_loading=lazy_loading
    )

    # 创建源域测试集
    source_test_dataset = JNUBearingDataset(
        data_path=str(data_path),
        speed=source_speed,
        shot=target_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='test',
        fault_types=fault_types,
        lazy_loading=lazy_loading
    )

    # 创建目标域数据集
    target_dataset = JNUBearingDataset(
        data_path=str(data_path),
        speed=target_speed,
        shot=target_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='test',
        fault_types=fault_types,
        lazy_loading=lazy_loading
    )

    # 创建数据加载器
    dataloaders = {
        'source_train': DataLoader(
            source_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=num_workers > 0
        ),
        'source_test': DataLoader(
            source_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0
        ),
        'target': DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0
        )
    }

    logger.info(f"Created dataloaders:")
    logger.info(f"  Source train: {len(source_train_dataset)} samples")
    logger.info(f"  Source test: {len(source_test_dataset)} samples")
    logger.info(f"  Target: {len(target_dataset)} samples")

    return dataloaders


def create_single_domain_dataloader(
        data_path: str,
        speed: int = 800,
        train_shot: int = 40,
        test_shot: int = 10,
        sample_len: int = 1024,
        batch_size: int = 64,
        normalize_type: str = 'mean~std',
        num_workers: int = 4,
        fault_types: List[str] = None,
        lazy_loading: bool = True
) -> Dict[str, DataLoader]:
    """
    创建单域数据加载器（用于基础实验）
    """
    # 创建训练集
    train_dataset = JNUBearingDataset(
        data_path=data_path,
        speed=speed,
        shot=train_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='train',
        fault_types=fault_types,
        lazy_loading=lazy_loading
    )

    # 创建测试集
    test_dataset = JNUBearingDataset(
        data_path=data_path,
        speed=speed,
        shot=test_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='test',
        fault_types=fault_types,
        lazy_loading=lazy_loading
    )

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=num_workers > 0
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0
        )
    }

    return dataloaders


# 数据集信息获取函数
def get_dataset_info(data_path: str) -> Dict:
    """获取数据集信息"""
    data_path = Path(data_path)

    # 扫描可用文件
    available_files = []
    if data_path.exists():
        available_files = list(data_path.glob("*.mat"))

    info = {
        'dataset_name': 'JNU Bearing Dataset',
        'fault_types': ['Normal', 'InnerRace', 'OuterRace', 'Ball'],
        'available_speeds': [600, 800, 1000],
        'data_path': str(data_path),
        'available_files': [f.name for f in available_files],
        'file_format': '.mat files',
        'recommended_settings': {
            'sample_len': 1024,
            'normalize_type': 'mean~std',
            'overlap': 0.5,
            'batch_size': 64,
            'lazy_loading': True
        }
    }
    return info


# 测试函数
def test_jnu_dataset():
    """测试JNU数据集加载"""
    print("Testing optimized JNU Dataset...")

    # 使用当前目录作为测试路径
    data_path = "./data/JNU/"

    try:
        # 测试数据集信息
        dataset_info = get_dataset_info(data_path)
        print("✓ Dataset info retrieved")
        print(f"  Available files: {len(dataset_info['available_files'])}")

        # 如果没有真实数据，创建虚拟测试
        if not dataset_info['available_files']:
            print("  No real data found, using dummy dataset for testing")

            # 创建虚拟数据集测试
            from datasets.data_utils import BaseDataset

            dummy_data = np.random.randn(20, 1024).astype(np.float32)
            dummy_labels = np.random.randint(0, 4, 20)

            dummy_dataset = BaseDataset(dummy_data, dummy_labels)
            print(f"✓ Dummy dataset created: {len(dummy_dataset)} samples")

            # 测试数据加载
            sample, label = dummy_dataset[0]
            print(f"✓ Sample loading: {sample.shape}, label: {label}")

        else:
            print("  Real data found, testing with actual files")

            # 测试真实数据集
            try:
                dataset = JNUBearingDataset(
                    data_path=data_path,
                    speed=800,
                    shot=5,
                    sample_len=1024,
                    lazy_loading=True
                )

                print(f"✓ Real dataset created: {len(dataset)} samples")

                # 测试数据加载
                sample, label = dataset[0]
                print(f"✓ Real sample loading: {sample.shape}, label: {label}")

                # 测试类别分布
                distribution = dataset.get_class_distribution()
                print(f"✓ Class distribution: {distribution}")

            except Exception as e:
                print(f"✗ Real dataset test failed: {str(e)}")
                print("  This might be due to file format or path issues")

        return True

    except Exception as e:
        print(f"✗ Dataset test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 运行测试
    success = test_jnu_dataset()

    if success:
        print("\n🎉 Dataset module test completed successfully!")
    else:
        print("\n❌ Dataset module test failed!")

    # 显示数据集信息
    try:
        dataset_info = get_dataset_info("./data/JNU/")
        print("\nDataset Info:")
        for key, value in dataset_info.items():
            if key != 'available_files' or len(value) < 10:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {len(value)} files")
    except Exception as e:
        print(f"Failed to get dataset info: {e}")