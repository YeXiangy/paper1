"""
AMSCNN: Adaptive Multi-Scale CNN
自适应多尺度卷积神经网络

创新点2: 物理知识引导的多尺度特征提取，通过调制机制实现物理约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleConvBlock(nn.Module):
    """多尺度卷积块"""

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7, 15]):
        super(MultiScaleConvBlock, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

        # 为每个尺度创建卷积分支
        self.conv_branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2, stride=2)
            )
            self.conv_branches.append(branch)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, in_channels, length]

        Returns:
            multi_scale_features: 多尺度特征列表
        """
        multi_scale_features = []

        for branch in self.conv_branches:
            branch_output = branch(x)
            multi_scale_features.append(branch_output)

        return multi_scale_features


class PhysicalModulator(nn.Module):
    """物理特征调制器"""

    def __init__(self, physical_dim, feature_dim):
        super(PhysicalModulator, self).__init__()

        self.modulation_network = nn.Sequential(
            nn.Linear(physical_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()  # 输出[0,1]范围的调制权重
        )

        # 自适应门控机制
        self.gate_network = nn.Sequential(
            nn.Linear(physical_dim, feature_dim),
            nn.Tanh()  # 输出[-1,1]范围的门控信号
        )

    def forward(self, physical_code):
        """
        计算物理调制权重

        Args:
            physical_code: 物理编码 [batch_size, physical_dim]

        Returns:
            modulation_weights: 调制权重 [batch_size, feature_dim]
            gate_weights: 门控权重 [batch_size, feature_dim]
        """
        modulation_weights = self.modulation_network(physical_code)
        gate_weights = self.gate_network(physical_code)

        return modulation_weights, gate_weights


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        计算通道注意力权重

        Args:
            x: 输入特征 [batch_size, channels, length]

        Returns:
            attention_weights: 注意力权重 [batch_size, channels, 1]
        """
        batch_size, channels, length = x.size()

        # 全局平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(batch_size, channels))
        max_out = self.fc(self.max_pool(x).view(batch_size, channels))

        # 融合并计算注意力权重
        attention = self.sigmoid(avg_out + max_out)
        attention_weights = attention.view(batch_size, channels, 1)

        return attention_weights


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        计算空间注意力权重

        Args:
            x: 输入特征 [batch_size, channels, length]

        Returns:
            attention_weights: 注意力权重 [batch_size, 1, length]
        """
        # 计算通道维度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并卷积
        combined = torch.cat([avg_out, max_out], dim=1)
        attention_weights = self.sigmoid(self.conv(combined))

        return attention_weights


class AdaptiveMultiScaleCNN(nn.Module):
    """
    自适应多尺度CNN

    功能：
    1. 多尺度并行卷积提取不同粒度特征
    2. 物理编码调制CNN特征权重
    3. 注意力机制增强重要特征
    """

    def __init__(self, config, physical_dim=128):
        super(AdaptiveMultiScaleCNN, self).__init__()

        self.config = config
        self.physical_dim = physical_dim

        # 多尺度卷积块
        self.multi_scale_conv = MultiScaleConvBlock(
            in_channels=1,
            out_channels=config.scale_channels,
            kernel_sizes=config.kernel_sizes
        )

        # 物理调制器
        self.physical_modulator = PhysicalModulator(
            physical_dim=physical_dim,
            feature_dim=config.physical_modulation_dim
        )

        # 特征融合卷积
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(config.physical_modulation_dim, config.conv_channels[0],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(config.conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )

        # 深度卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = config.conv_channels[0]

        for out_channels in config.conv_channels[1:]:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2, stride=2),
                nn.Dropout1d(config.dropout_rate)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels

        # 注意力机制
        self.channel_attention = ChannelAttention(config.conv_channels[-1])
        self.spatial_attention = SpatialAttention()

        # 全局自适应池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(config.conv_channels[-1], config.conv_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.conv_channels[-1] // 2, config.conv_channels[-1])
        )

        self.flatten = nn.Flatten()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, signal, physical_code):
        """
        前向传播

        Args:
            signal: 输入信号 [batch_size, 1, sample_len]
            physical_code: 物理编码 [batch_size, physical_dim]

        Returns:
            deep_features: 深度特征 [batch_size, feature_dim]
        """
        batch_size = signal.shape[0]

        # 1. 多尺度特征提取
        multi_scale_features = self.multi_scale_conv(signal)  # List of [B, 64, L/2]

        # 2. 拼接多尺度特征
        concatenated_features = torch.cat(multi_scale_features, dim=1)  # [B, 192, L/2]

        # 3. 物理特征调制
        modulation_weights, gate_weights = self.physical_modulator(physical_code)  # [B, 192]

        # 应用调制权重
        modulated_features = concatenated_features * modulation_weights.unsqueeze(-1)  # [B, 192, L/2]

        # 应用门控机制
        gated_features = modulated_features * (1 + gate_weights.unsqueeze(-1))  # [B, 192, L/2]

        # 4. 特征融合
        fused_features = self.feature_fusion(gated_features)  # [B, 64, L/4]

        # 5. 深度卷积处理
        deep_features = fused_features
        for conv_layer in self.conv_layers:
            deep_features = conv_layer(deep_features)  # 逐层降维和提取特征

        # 6. 注意力机制
        # 通道注意力
        channel_att = self.channel_attention(deep_features)
        deep_features = deep_features * channel_att

        # 空间注意力
        spatial_att = self.spatial_attention(deep_features)
        deep_features = deep_features * spatial_att

        # 7. 全局池化和全连接
        global_features = self.global_pool(deep_features)  # [B, 512, 1]
        global_features = self.flatten(global_features)  # [B, 512]

        output_features = self.fc(global_features)  # [B, 512]

        return output_features

    def get_attention_weights(self, signal, physical_code):
        """
        获取注意力权重（用于可视化分析）

        Args:
            signal: 输入信号 [batch_size, 1, sample_len]
            physical_code: 物理编码 [batch_size, physical_dim]

        Returns:
            attention_dict: 注意力权重字典
        """
        with torch.no_grad():
            # 前向传播到注意力层
            multi_scale_features = self.multi_scale_conv(signal)
            concatenated_features = torch.cat(multi_scale_features, dim=1)

            modulation_weights, gate_weights = self.physical_modulator(physical_code)
            modulated_features = concatenated_features * modulation_weights.unsqueeze(-1)
            gated_features = modulated_features * (1 + gate_weights.unsqueeze(-1))

            fused_features = self.feature_fusion(gated_features)

            deep_features = fused_features
            for conv_layer in self.conv_layers:
                deep_features = conv_layer(deep_features)

            # 获取注意力权重
            channel_att = self.channel_attention(deep_features)
            spatial_att = self.spatial_attention(deep_features)

            return {
                'modulation_weights': modulation_weights.cpu().numpy(),
                'gate_weights': gate_weights.cpu().numpy(),
                'channel_attention': channel_att.squeeze(-1).cpu().numpy(),
                'spatial_attention': spatial_att.squeeze(1).cpu().numpy()
            }


# 测试代码
if __name__ == '__main__':
    # 测试AMSCNN模块
    from types import SimpleNamespace

    # 创建测试配置
    config = SimpleNamespace(
        kernel_sizes=[3, 7, 15],
        scale_channels=64,
        conv_channels=[64, 128, 256, 512],
        physical_modulation_dim=192,
        dropout_rate=0.2
    )

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaptiveMultiScaleCNN(config, physical_dim=128).to(device)

    # 测试输入
    batch_size = 4
    sample_len = 1024
    physical_dim = 128

    test_signal = torch.randn(batch_size, 1, sample_len).to(device)
    test_physical = torch.randn(batch_size, physical_dim).to(device)

    # 前向传播
    print("Testing AMSCNN module...")
    print(f"Signal shape: {test_signal.shape}")
    print(f"Physical code shape: {test_physical.shape}")

    deep_features = model(test_signal, test_physical)
    print(f"Output shape: {deep_features.shape}")

    # 测试注意力权重
    attention_weights = model.get_attention_weights(test_signal, test_physical)
    print("Attention weights shapes:")
    for name, weights in attention_weights.items():
        print(f"  {name}: {weights.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("AMSCNN module test completed successfully!")