"""
JNU_bearing.py
JNU轴承数据集处理模块

功能：
1. 加载JNU轴承故障数据
2. 数据预处理和标准化
3. 支持跨域实验设置
4. 兼容FIR-CCF的数据格式
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class JNUBearingDataset(Dataset):
    """JNU轴承数据集类"""

    def __init__(self,
                 data_path: str,
                 speed: int = 800,
                 shot: int = 5,
                 sample_len: int = 1024,
                 normalize_type: str = 'mean~std',
                 overlap: float = 0.5,
                 mode: str = 'train',
                 fault_types: List[str] = None):
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
        """
        self.data_path = data_path
        self.speed = speed
        self.shot = shot
        self.sample_len = sample_len
        self.normalize_type = normalize_type
        self.overlap = overlap
        self.mode = mode

        # 默认故障类型
        if fault_types is None:
            self.fault_types = ['Normal', 'InnerRace', 'OuterRace', 'Ball']
        else:
            self.fault_types = fault_types

        self.num_classes = len(self.fault_types)

        # 类别映射
        self.class_to_idx = {fault: idx for idx, fault in enumerate(self.fault_types)}
        self.idx_to_class = {idx: fault for fault, idx in self.class_to_idx.items()}

        # 加载数据
        self.data, self.labels = self._load_data()

        print(f"JNU Dataset loaded:")
        print(f"  Speed: {self.speed} rpm")
        print(f"  Mode: {self.mode}")
        print(f"  Samples per class: {self.shot}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Sample length: {self.sample_len}")
        print(f"  Classes: {self.fault_types}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据"""
        all_data = []
        all_labels = []

        for fault_idx, fault_type in enumerate(self.fault_types):
            # 构建文件路径
            file_pattern = self._get_file_pattern(fault_type)
            fault_data = self._load_fault_data(file_pattern)

            if fault_data is None:
                raise FileNotFoundError(f"No data found for {fault_type} at speed {self.speed}")

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

        # 合并数据
        data = np.vstack(all_data)
        labels = np.array(all_labels)

        # 数据标准化
        data = self._normalize_data(data)

        return data, labels

    def _get_file_pattern(self, fault_type: str) -> str:
        """根据故障类型和转速获取文件模式"""
        # 根据JNU数据集的命名规则调整
        speed_mapping = {
            600: '600rpm',
            800: '800rpm',
            1000: '1000rpm'
        }

        fault_mapping = {
            'Normal': 'normal',
            'InnerRace': 'inner',
            'OuterRace': 'outer',
            'Ball': 'ball'
        }

        speed_str = speed_mapping.get(self.speed, f'{self.speed}rpm')
        fault_str = fault_mapping.get(fault_type, fault_type.lower())

        # 构建文件路径
        file_pattern = os.path.join(
            self.data_path,
            f"JNU_{fault_str}_{speed_str}.mat"
        )

        return file_pattern

    def _load_fault_data(self, file_pattern: str) -> Optional[np.ndarray]:
        """加载特定故障类型的数据"""
        try:
            if os.path.exists(file_pattern):
                mat_data = sio.loadmat(file_pattern)

                # 尝试常见的变量名
                possible_keys = ['data', 'X', 'vibration', 'signal', 'DE_time']

                for key in mat_data.keys():
                    if not key.startswith('_'):  # 忽略元数据
                        data = mat_data[key]
                        if isinstance(data, np.ndarray) and data.size > 1000:
                            # 确保是一维数据
                            if data.ndim > 1:
                                data = data.flatten()
                            return data

                # 如果没找到合适的键，尝试第一个非元数据键
                for key in possible_keys:
                    if key in mat_data:
                        data = mat_data[key]
                        if data.ndim > 1:
                            data = data.flatten()
                        return data

            return None

        except Exception as e:
            print(f"Error loading {file_pattern}: {str(e)}")
            return None

    def _generate_samples(self, raw_data: np.ndarray) -> np.ndarray:
        """从原始数据生成固定长度样本"""
        if len(raw_data) < self.sample_len:
            # 如果数据长度不足，进行重复
            repeat_times = self.sample_len // len(raw_data) + 1
            raw_data = np.tile(raw_data, repeat_times)

        # 计算步长
        if self.overlap == 0:
            step = self.sample_len
        else:
            step = int(self.sample_len * (1 - self.overlap))

        # 生成样本
        samples = []
        start = 0

        while start + self.sample_len <= len(raw_data):
            sample = raw_data[start:start + self.sample_len]
            samples.append(sample)
            start += step

            # 限制样本数量，避免内存问题
            if len(samples) >= 1000:  # 最多生成1000个样本
                break

        if len(samples) == 0:
            # 如果没有生成任何样本，直接截取
            samples = [raw_data[:self.sample_len]]

        return np.array(samples)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据标准化"""
        if self.normalize_type == 'none':
            return data

        # 重塑数据为二维以便标准化
        original_shape = data.shape
        data_2d = data.reshape(-1, self.sample_len)

        if self.normalize_type == 'mean~std':
            # 逐样本标准化
            normalized_data = np.zeros_like(data_2d)
            for i in range(data_2d.shape[0]):
                sample = data_2d[i]
                mean_val = np.mean(sample)
                std_val = np.std(sample)
                if std_val > 0:
                    normalized_data[i] = (sample - mean_val) / std_val
                else:
                    normalized_data[i] = sample - mean_val

        elif self.normalize_type == 'min~max':
            # 逐样本min-max标准化
            normalized_data = np.zeros_like(data_2d)
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
            normalized_data = scaler.fit_transform(data_2d)

        else:
            normalized_data = data_2d

        return normalized_data.reshape(original_shape)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        # 获取数据和标签
        sample = self.data[idx]
        label = self.labels[idx]

        # 转换为张量并添加通道维度
        sample_tensor = torch.FloatTensor(sample).unsqueeze(0)  # [1, sample_len]
        label_tensor = torch.LongTensor([label])[0]  # 标量

        return sample_tensor, label_tensor

    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        distribution = {}
        for i, fault_type in enumerate(self.fault_types):
            count = np.sum(self.labels == i)
            distribution[fault_type] = count
        return distribution


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
        fault_types: List[str] = None
) -> Dict[str, DataLoader]:
    """
    创建JNU数据集的数据加载器（用于跨域实验）

    Args:
        data_path: 数据路径
        source_speed: 源域转速
        target_speed: 目标域转速
        source_shot: 源域每类样本数
        target_shot: 目标域每类样本数
        sample_len: 样本长度
        batch_size: 批大小
        normalize_type: 标准化类型
        num_workers: 数据加载进程数
        fault_types: 故障类型

    Returns:
        dataloaders: 数据加载器字典
    """

    # 创建源域训练集
    source_train_dataset = JNUBearingDataset(
        data_path=data_path,
        speed=source_speed,
        shot=source_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='train',
        fault_types=fault_types
    )

    # 创建源域测试集
    source_test_dataset = JNUBearingDataset(
        data_path=data_path,
        speed=source_speed,
        shot=target_shot,  # 测试集使用较少样本
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='test',
        fault_types=fault_types
    )

    # 创建目标域数据集
    target_dataset = JNUBearingDataset(
        data_path=data_path,
        speed=target_speed,
        shot=target_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='test',  # 目标域通常用作测试
        fault_types=fault_types
    )

    # 创建数据加载器
    dataloaders = {
        'source_train': DataLoader(
            source_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'source_test': DataLoader(
            source_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        ),
        'target': DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    }

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
        fault_types: List[str] = None
) -> Dict[str, DataLoader]:
    """
    创建单域数据加载器（用于基础实验）

    Args:
        data_path: 数据路径
        speed: 转速
        train_shot: 训练集每类样本数
        test_shot: 测试集每类样本数
        sample_len: 样本长度
        batch_size: 批大小
        normalize_type: 标准化类型
        num_workers: 数据加载进程数
        fault_types: 故障类型

    Returns:
        dataloaders: 数据加载器字典
    """

    # 创建训练集
    train_dataset = JNUBearingDataset(
        data_path=data_path,
        speed=speed,
        shot=train_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='train',
        fault_types=fault_types
    )

    # 创建测试集
    test_dataset = JNUBearingDataset(
        data_path=data_path,
        speed=speed,
        shot=test_shot,
        sample_len=sample_len,
        normalize_type=normalize_type,
        mode='test',
        fault_types=fault_types
    )

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    }

    return dataloaders


# 数据集信息获取函数
def get_dataset_info(data_path: str) -> Dict:
    """获取数据集信息"""
    info = {
        'dataset_name': 'JNU Bearing Dataset',
        'fault_types': ['Normal', 'InnerRace', 'OuterRace', 'Ball'],
        'available_speeds': [600, 800, 1000],
        'data_path': data_path,
        'file_format': '.mat files',
        'recommended_settings': {
            'sample_len': 1024,
            'normalize_type': 'mean~std',
            'overlap': 0.5,
            'batch_size': 64
        }
    }
    return info


# 测试函数
def test_jnu_dataset():
    """测试JNU数据集加载"""
    print("Testing JNU Dataset...")

    # 假设数据路径（需要根据实际情况修改）
    data_path = "./data/JNU/"

    try:
        # 测试单域数据加载器
        single_dataloaders = create_single_domain_dataloader(
            data_path=data_path,
            speed=800,
            train_shot=10,
            test_shot=5,
            sample_len=1024,
            batch_size=8
        )

        print("✓ Single domain dataloaders created successfully")

        # 测试数据形状
        for phase, dataloader in single_dataloaders.items():
            for batch_data, batch_labels in dataloader:
                print(f"  {phase} - Data: {batch_data.shape}, Labels: {batch_labels.shape}")
                break

        # 测试跨域数据加载器
        cross_dataloaders = create_jnu_dataloaders(
            data_path=data_path,
            source_speed=800,
            target_speed=1000,
            source_shot=10,
            target_shot=5,
            sample_len=1024,
            batch_size=8
        )

        print("✓ Cross-domain dataloaders created successfully")

        return True

    except Exception as e:
        print(f"✗ Dataset test failed: {str(e)}")
        return False


if __name__ == '__main__':
    # 运行测试
    test_jnu_dataset()

    # 显示数据集信息
    dataset_info = get_dataset_info("./data/JNU/")
    print("\nDataset Info:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")