"""
MPIE: Multi-level Physical Information Encoder
多级物理信息编码器

创新点1: 将多域物理特征通过分层编码转换为深度网络可学习的表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicalFeatureExtractor:
    """物理特征提取器 - 从原始振动信号中提取物理特征"""

    def __init__(self):
        pass

    def extract_time_domain_features(self, signal):
        """提取时域特征 [8维]"""
        batch_size = signal.shape[0]
        time_features = []

        for i in range(batch_size):
            x = signal[i, 0, :]  # [sample_len]

            # 基本统计特征
            mean_val = torch.mean(x)
            std_val = torch.std(x)
            var_val = torch.var(x)

            # 高阶统计特征
            skewness = torch.mean(((x - mean_val) / (std_val + 1e-8)) ** 3)
            kurtosis = torch.mean(((x - mean_val) / (std_val + 1e-8)) ** 4) - 3

            # 幅值特征
            rms = torch.sqrt(torch.mean(x ** 2))
            peak_val = torch.max(torch.abs(x))

            # 形状因子
            clearance_factor = peak_val / (torch.mean(torch.sqrt(torch.abs(x) + 1e-8)) ** 2 + 1e-8)

            time_feat = torch.stack([
                mean_val, std_val, var_val, skewness,
                kurtosis, rms, peak_val, clearance_factor
            ])
            time_features.append(time_feat)

        return torch.stack(time_features)  # [B, 8]

    def extract_frequency_domain_features(self, signal):
        """提取频域特征 [7维]"""
        batch_size = signal.shape[0]
        freq_features = []

        for i in range(batch_size):
            x = signal[i, 0, :]

            # FFT变换
            fft_x = torch.fft.fft(x)
            magnitude = torch.abs(fft_x[:len(x) // 2])
            freqs = torch.linspace(0, 0.5, len(magnitude), device=x.device)

            # 频域统计特征
            total_power = torch.sum(magnitude ** 2)

            # 重心频率
            spectral_centroid = torch.sum(freqs * magnitude ** 2) / (total_power + 1e-8)

            # 频谱方差
            spectral_variance = torch.sum(((freqs - spectral_centroid) ** 2) * magnitude ** 2) / (total_power + 1e-8)

            # 频谱偏度
            spectral_skewness = torch.sum(((freqs - spectral_centroid) ** 3) * magnitude ** 2) / (
                        (total_power * spectral_variance ** 1.5) + 1e-8)

            # 频谱峭度
            spectral_kurtosis = torch.sum(((freqs - spectral_centroid) ** 4) * magnitude ** 2) / (
                        (total_power * spectral_variance ** 2) + 1e-8) - 3

            # 谱能量
            spectral_energy = total_power

            # 谱熵
            normalized_magnitude = magnitude / (torch.sum(magnitude) + 1e-8)
            spectral_entropy = -torch.sum(normalized_magnitude * torch.log(normalized_magnitude + 1e-8))

            # 谱滚降
            cumsum_magnitude = torch.cumsum(magnitude, dim=0)
            rolloff_idx = torch.where(cumsum_magnitude >= 0.85 * torch.sum(magnitude))[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

            freq_feat = torch.stack([
                spectral_centroid, spectral_variance, spectral_skewness,
                spectral_kurtosis, spectral_energy, spectral_entropy, spectral_rolloff
            ])
            freq_features.append(freq_feat)

        return torch.stack(freq_features)  # [B, 7]

    def extract_time_frequency_features(self, signal):
        """提取时频域特征 [5维] - 基于多频带能量分析"""
        batch_size = signal.shape[0]
        tf_features = []

        for i in range(batch_size):
            x = signal[i, 0, :]

            # FFT变换
            fft_x = torch.fft.fft(x)
            magnitude = torch.abs(fft_x[:len(x) // 2])

            # 分成5个频带
            band_size = len(magnitude) // 5
            band_energies = []

            for j in range(5):
                start_idx = j * band_size
                end_idx = (j + 1) * band_size if j < 4 else len(magnitude)
                band_energy = torch.sum(magnitude[start_idx:end_idx] ** 2)
                band_energies.append(band_energy)

            band_energies = torch.stack(band_energies)
            # 归一化为能量比
            total_energy = torch.sum(band_energies) + 1e-8
            tf_feat = band_energies / total_energy

            tf_features.append(tf_feat)

        return torch.stack(tf_features)  # [B, 5]


class MultiLevelPhysicalEncoder(nn.Module):
    """
    多级物理信息编码器

    功能：
    1. 从振动信号中提取多域物理特征
    2. 通过分层编码网络将物理特征转换为统一表示
    3. 输出可与CNN特征融合的物理编码张量
    """

    def __init__(self, config):
        super(MultiLevelPhysicalEncoder, self).__init__()

        self.config = config

        # 物理特征提取器
        self.feature_extractor = PhysicalFeatureExtractor()

        # 时域特征编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(config.time_features_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, config.time_hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 频域特征编码器
        self.freq_encoder = nn.Sequential(
            nn.Linear(config.freq_features_dim, 28),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(28, config.freq_hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 时频域特征编码器
        self.tf_encoder = nn.Sequential(
            nn.Linear(config.tf_features_dim, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(20, config.tf_hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 特征融合层
        total_encoded_dim = config.time_hidden_dim + config.freq_hidden_dim + config.tf_hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_encoded_dim, config.output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(config.output_dim * 2, config.output_dim),
            nn.Tanh()  # 限制输出范围到[-1, 1]
        )

        # 批归一化
        self.batch_norm = nn.BatchNorm1d(config.output_dim)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, signal):
        """
        前向传播

        Args:
            signal: 振动信号 [batch_size, 1, sample_len]

        Returns:
            physical_code: 物理编码张量 [batch_size, output_dim]
        """
        batch_size = signal.shape[0]

        # 1. 提取多域物理特征
        time_features = self.feature_extractor.extract_time_domain_features(signal)  # [B, 8]
        freq_features = self.feature_extractor.extract_frequency_domain_features(signal)  # [B, 7]
        tf_features = self.feature_extractor.extract_time_frequency_features(signal)  # [B, 5]

        # 2. 分层编码
        time_encoded = self.time_encoder(time_features)  # [B, 48]
        freq_encoded = self.freq_encoder(freq_features)  # [B, 40]
        tf_encoded = self.tf_encoder(tf_features)  # [B, 40]

        # 3. 特征融合
        combined_features = torch.cat([time_encoded, freq_encoded, tf_encoded], dim=1)  # [B, 128]
        physical_code = self.fusion_layer(combined_features)  # [B, output_dim]

        # 4. 批归一化
        physical_code = self.batch_norm(physical_code)

        return physical_code

    def get_feature_importance(self, signal):
        """
        获取不同域特征的重要性（用于可解释性分析）

        Args:
            signal: 振动信号 [batch_size, 1, sample_len]

        Returns:
            importance_dict: 特征重要性字典
        """
        with torch.no_grad():
            # 提取各域特征
            time_features = self.feature_extractor.extract_time_domain_features(signal)
            freq_features = self.feature_extractor.extract_frequency_domain_features(signal)
            tf_features = self.feature_extractor.extract_time_frequency_features(signal)

            # 编码后的特征
            time_encoded = self.time_encoder(time_features)
            freq_encoded = self.freq_encoder(freq_features)
            tf_encoded = self.tf_encoder(tf_features)

            # 计算各域特征的L2范数作为重要性指标
            time_importance = torch.norm(time_encoded, dim=1).mean().item()
            freq_importance = torch.norm(freq_encoded, dim=1).mean().item()
            tf_importance = torch.norm(tf_encoded, dim=1).mean().item()

            total_importance = time_importance + freq_importance + tf_importance

            return {
                'time_domain': time_importance / total_importance,
                'frequency_domain': freq_importance / total_importance,
                'time_frequency_domain': tf_importance / total_importance,
                'raw_values': {
                    'time': time_importance,
                    'frequency': freq_importance,
                    'time_frequency': tf_importance
                }
            }


# 测试代码
if __name__ == '__main__':
    # 测试MPIE模块
    from types import SimpleNamespace

    # 创建测试配置
    config = SimpleNamespace(
        time_features_dim=8,
        freq_features_dim=7,
        tf_features_dim=5,
        time_hidden_dim=48,
        freq_hidden_dim=40,
        tf_hidden_dim=40,
        output_dim=128
    )

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiLevelPhysicalEncoder(config).to(device)

    # 测试输入
    batch_size = 4
    sample_len = 1024
    test_input = torch.randn(batch_size, 1, sample_len).to(device)

    # 前向传播
    print("Testing MPIE module...")
    print(f"Input shape: {test_input.shape}")

    physical_code = model(test_input)
    print(f"Output shape: {physical_code.shape}")
    print(f"Output range: [{physical_code.min().item():.3f}, {physical_code.max().item():.3f}]")

    # 测试特征重要性
    importance = model.get_feature_importance(test_input)
    print("Feature importance:")
    for domain, imp in importance.items():
        if domain != 'raw_values':
            print(f"  {domain}: {imp:.3f}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("MPIE module test completed successfully!")