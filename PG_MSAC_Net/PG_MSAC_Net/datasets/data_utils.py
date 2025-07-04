"""
数据处理工具类
包含数据预处理、增强、归一化等功能
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
import random


class SequenceNormalizer:
    """序列数据归一化器"""

    def __init__(self, method='mean_std'):
        """
        初始化归一化器

        Args:
            method: 归一化方法 ('mean_std', 'min_max', 'robust')
        """
        self.method = method
        self.fitted = False
        self.stats = {}

    def fit(self, data: np.ndarray) -> 'SequenceNormalizer':
        """
        拟合归一化参数

        Args:
            data: 训练数据 [N, C, L]
        """
        if self.method == 'mean_std':
            self.stats['mean'] = np.mean(data, axis=(0, 2), keepdims=True)
            self.stats['std'] = np.std(data, axis=(0, 2), keepdims=True) + 1e-8

        elif self.method == 'min_max':
            self.stats['min'] = np.min(data, axis=(0, 2), keepdims=True)
            self.stats['max'] = np.max(data, axis=(0, 2), keepdims=True)
            self.stats['range'] = self.stats['max'] - self.stats['min'] + 1e-8

        elif self.method == 'robust':
            self.stats['median'] = np.median(data, axis=(0, 2), keepdims=True)
            self.stats['q75'] = np.percentile(data, 75, axis=(0, 2), keepdims=True)
            self.stats['q25'] = np.percentile(data, 25, axis=(0, 2), keepdims=True)
            self.stats['iqr'] = self.stats['q75'] - self.stats['q25'] + 1e-8

        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        应用归一化

        Args:
            data: 输入数据 [N, C, L]

        Returns:
            归一化后的数据
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        if self.method == 'mean_std':
            return (data - self.stats['mean']) / self.stats['std']

        elif self.method == 'min_max':
            return (data - self.stats['min']) / self.stats['range']

        elif self.method == 'robust':
            return (data - self.stats['median']) / self.stats['iqr']

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并转换数据"""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """逆变换"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")

        if self.method == 'mean_std':
            return data * self.stats['std'] + self.stats['mean']

        elif self.method == 'min_max':
            return data * self.stats['range'] + self.stats['min']

        elif self.method == 'robust':
            return data * self.stats['iqr'] + self.stats['median']


class DataAugmentation:
    """数据增强类"""

    def __init__(self, aug_types: List[str] = None, aug_prob: float = 0.5):
        """
        初始化数据增强

        Args:
            aug_types: 增强类型列表
            aug_prob: 增强概率
        """
        if aug_types is None:
            aug_types = ['noise', 'scale', 'shift']

        self.aug_types = aug_types
        self.aug_prob = aug_prob

    def add_noise(self, data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def scale_amplitude(self, data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """幅值缩放"""
        scale_factor = np.random.uniform(*scale_range)
        return data * scale_factor

    def time_shift(self, data: np.ndarray, shift_range: int = 50) -> np.ndarray:
        """时间偏移"""
        shift = np.random.randint(-shift_range, shift_range + 1)
        if shift > 0:
            shifted_data = np.zeros_like(data)
            shifted_data[:, :, shift:] = data[:, :, :-shift]
        elif shift < 0:
            shifted_data = np.zeros_like(data)
            shifted_data[:, :, :shift] = data[:, :, -shift:]
        else:
            shifted_data = data.copy()

        return shifted_data

    def frequency_mask(self, data: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
        """频域掩码"""
        fft_data = np.fft.fft(data, axis=-1)
        mask_len = int(fft_data.shape[-1] * mask_ratio)
        mask_start = np.random.randint(0, fft_data.shape[-1] - mask_len)

        fft_data[:, :, mask_start:mask_start + mask_len] = 0

        return np.real(np.fft.ifft(fft_data, axis=-1))

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        if np.random.random() > self.aug_prob:
            return data

        augmented_data = data.copy()

        for aug_type in self.aug_types:
            if np.random.random() < 0.5:  # 每种增强有50%概率被应用
                if aug_type == 'noise':
                    augmented_data = self.add_noise(augmented_data)
                elif aug_type == 'scale':
                    augmented_data = self.scale_amplitude(augmented_data)
                elif aug_type == 'shift':
                    augmented_data = self.time_shift(augmented_data)
                elif aug_type == 'freq_mask':
                    augmented_data = self.frequency_mask(augmented_data)

        return augmented_data


class DataProcessor:
    """数据处理器"""

    def __init__(self, sample_len: int = 1024, overlap: float = 0.0):
        """
        初始化数据处理器

        Args:
            sample_len: 样本长度
            overlap: 重叠比例
        """
        self.sample_len = sample_len
        self.overlap = overlap
        self.step_size = int(sample_len * (1 - overlap))

    def segment_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        将长信号分割成固定长度的片段

        Args:
            signal: 输入信号 [length]

        Returns:
            分割后的信号 [num_segments, sample_len]
        """
        if len(signal) < self.sample_len:
            # 如果信号太短，进行零填充
            padded_signal = np.zeros(self.sample_len)
            padded_signal[:len(signal)] = signal
            return padded_signal.reshape(1, -1)

        # 计算可以分割的段数
        num_segments = (len(signal) - self.sample_len) // self.step_size + 1

        segments = []
        for i in range(num_segments):
            start_idx = i * self.step_size
            end_idx = start_idx + self.sample_len
            segments.append(signal[start_idx:end_idx])

        return np.array(segments)

    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        提取基础特征

        Args:
            signal: 输入信号 [sample_len]

        Returns:
            特征字典
        """
        features = {}

        # 时域特征
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['rms'] = np.sqrt(np.mean(signal ** 2))
        features['peak'] = np.max(np.abs(signal))
        features['kurtosis'] = self._compute_kurtosis(signal)
        features['skewness'] = self._compute_skewness(signal)
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0

        # 频域特征
        fft_signal = np.fft.fft(signal)
        magnitude = np.abs(fft_signal[:len(signal) // 2])
        freqs = np.fft.fftfreq(len(signal), 1.0)[:len(signal) // 2]

        features['spectral_centroid'] = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        features['spectral_energy'] = np.sum(magnitude ** 2)
        features['spectral_entropy'] = self._compute_spectral_entropy(magnitude)

        return features

    def _compute_kurtosis(self, signal: np.ndarray) -> float:
        """计算峭度"""
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if std_val == 0:
            return 0
        normalized = (signal - mean_val) / std_val
        return np.mean(normalized ** 4) - 3

    def _compute_skewness(self, signal: np.ndarray) -> float:
        """计算偏度"""
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if std_val == 0:
            return 0
        normalized = (signal - mean_val) / std_val
        return np.mean(normalized ** 3)

    def _compute_spectral_entropy(self, magnitude: np.ndarray) -> float:
        """计算谱熵"""
        # 归一化为概率分布
        prob = magnitude / (np.sum(magnitude) + 1e-8)
        # 计算熵
        entropy = -np.sum(prob * np.log(prob + 1e-8))
        return entropy


class CrossDomainDataSplitter:
    """跨域数据分割器"""

    def __init__(self, source_ratio: float = 0.8, random_state: int = 42):
        """
        初始化跨域数据分割器

        Args:
            source_ratio: 源域数据用作训练的比例
            random_state: 随机种子
        """
        self.source_ratio = source_ratio
        self.random_state = random_state

    def split_cross_domain_data(self,
                                source_data: np.ndarray,
                                source_labels: np.ndarray,
                                target_data: np.ndarray,
                                target_labels: np.ndarray,
                                target_shot: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        分割跨域数据

        Args:
            source_data: 源域数据 [N_s, C, L]
            source_labels: 源域标签 [N_s]
            target_data: 目标域数据 [N_t, C, L]
            target_labels: 目标域标签 [N_t]
            target_shot: 目标域每类标注样本数

        Returns:
            数据字典
        """
        np.random.seed(self.random_state)

        # 源域数据分割
        source_train_data, source_val_data, source_train_labels, source_val_labels = train_test_split(
            source_data, source_labels,
            train_size=self.source_ratio,
            stratify=source_labels,
            random_state=self.random_state
        )

        # 目标域数据分割
        target_labeled_data = []
        target_labeled_labels = []
        target_unlabeled_data = []
        target_unlabeled_labels = []

        unique_labels = np.unique(target_labels)

        for label in unique_labels:
            # 获取当前类别的所有样本
            class_indices = np.where(target_labels == label)[0]
            class_data = target_data[class_indices]
            class_labels = target_labels[class_indices]

            # 随机选择target_shot个样本作为标注数据
            if len(class_indices) >= target_shot:
                selected_indices = np.random.choice(
                    len(class_indices),
                    target_shot,
                    replace=False
                )

                # 标注数据
                target_labeled_data.append(class_data[selected_indices])
                target_labeled_labels.append(class_labels[selected_indices])

                # 剩余数据作为无标注数据
                remaining_indices = np.setdiff1d(np.arange(len(class_indices)), selected_indices)
                if len(remaining_indices) > 0:
                    target_unlabeled_data.append(class_data[remaining_indices])
                    target_unlabeled_labels.append(class_labels[remaining_indices])
            else:
                # 如果样本数不够，全部作为标注数据
                target_labeled_data.append(class_data)
                target_labeled_labels.append(class_labels)

        # 合并数据
        target_labeled_data = np.concatenate(target_labeled_data, axis=0)
        target_labeled_labels = np.concatenate(target_labeled_labels, axis=0)

        if target_unlabeled_data:
            target_unlabeled_data = np.concatenate(target_unlabeled_data, axis=0)
            target_unlabeled_labels = np.concatenate(target_unlabeled_labels, axis=0)
        else:
            target_unlabeled_data = np.empty((0, target_data.shape[1], target_data.shape[2]))
            target_unlabeled_labels = np.empty((0,), dtype=target_labels.dtype)

        # 目标域测试数据（使用部分标注数据）
        target_test_data, target_val_data, target_test_labels, target_val_labels = train_test_split(
            target_labeled_data, target_labeled_labels,
            test_size=0.5,
            stratify=target_labeled_labels,
            random_state=self.random_state
        )

        return {
            'source_train': (source_train_data, source_train_labels),
            'source_val': (source_val_data, source_val_labels),
            'target_labeled': (target_val_data, target_val_labels),
            'target_unlabeled': (target_unlabeled_data, target_unlabeled_labels),
            'target_test': (target_test_data, target_test_labels)
        }


class BaseDataset(Dataset):
    """基础数据集类"""

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 transform=None,
                 augmentation=None):
        """
        初始化数据集

        Args:
            data: 数据 [N, C, L]
            labels: 标签 [N]
            transform: 数据变换
            augmentation: 数据增强
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # 应用数据增强
        if self.augmentation is not None:
            sample = self.augmentation(sample.numpy())
            sample = torch.FloatTensor(sample)

        # 应用变换
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


class UnlabeledDataset(Dataset):
    """无标签数据集类"""

    def __init__(self, data: np.ndarray, transform=None, augmentation=None):
        """
        初始化无标签数据集

        Args:
            data: 数据 [N, C, L]
            transform: 数据变换
            augmentation: 数据增强
        """
        self.data = torch.FloatTensor(data)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 应用数据增强
        if self.augmentation is not None:
            sample = self.augmentation(sample.numpy())
            sample = torch.FloatTensor(sample)

        # 应用变换
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, -1  # 返回-1表示无标签


# 工具函数
def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    向信号添加高斯白噪声

    Args:
        signal: 原始信号
        snr_db: 信噪比(dB)

    Returns:
        加噪后的信号
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    return signal + noise


def compute_signal_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    计算信号统计特性

    Args:
        data: 信号数据 [N, C, L]

    Returns:
        统计特性字典
    """
    stats = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'shape': data.shape
    }

    return stats


def verify_data_integrity(data: np.ndarray, labels: np.ndarray) -> bool:
    """
    验证数据完整性

    Args:
        data: 数据
        labels: 标签

    Returns:
        数据是否有效
    """
    if len(data) != len(labels):
        return False

    if np.isnan(data).any() or np.isinf(data).any():
        return False

    if len(np.unique(labels)) < 2:
        return False

    return True