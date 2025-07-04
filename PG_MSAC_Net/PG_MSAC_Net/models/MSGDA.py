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
"""
MSGDA: Multi-Statistical-Guided Domain Adapter
多统计量引导域适应器

创新点3: 基于多维统计特征的智能域适应机制，实现动态权重调节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StatisticalComputer:
    """统计特征计算器"""

    @staticmethod
    def compute_kurtosis(x, dim=1):
        """计算峭度"""
        mean = torch.mean(x, dim=dim, keepdim=True)
        std = torch.std(x, dim=dim, keepdim=True)
        normalized = (x - mean) / (std + 1e-8)
        kurtosis = torch.mean(normalized ** 4, dim=dim) - 3
        return kurtosis

    @staticmethod
    def compute_skewness(x, dim=1):
        """计算偏度"""
        mean = torch.mean(x, dim=dim, keepdim=True)
        std = torch.std(x, dim=dim, keepdim=True)
        normalized = (x - mean) / (std + 1e-8)
        skewness = torch.mean(normalized ** 3, dim=dim)
        return skewness

    @staticmethod
    def compute_energy(x, dim=1):
        """计算能量"""
        energy = torch.mean(x ** 2, dim=dim)
        return energy

    @staticmethod
    def compute_correlation_mean(x):
        """计算特征间平均相关性"""
        batch_size, feature_dim = x.shape
        correlations = []

        for i in range(batch_size):
            # 计算第i个样本的特征相关矩阵
            sample = x[i:i + 1, :].T  # [feature_dim, 1]
            if feature_dim > 1:
                corr_matrix = torch.corrcoef(sample.squeeze())
                # 取上三角矩阵的平均值（排除对角线）
                mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1).bool()
                if mask.sum() > 0:
                    mean_corr = corr_matrix[mask].mean()
                else:
                    mean_corr = torch.tensor(0.0, device=x.device)
            else:
                mean_corr = torch.tensor(0.0, device=x.device)
            correlations.append(mean_corr)

        return torch.stack(correlations)


class DomainDiscriminator(nn.Module):
    """域判别器"""

    def __init__(self, feature_dim, hidden_dims=[256, 128], dropout_rate=0.2):
        super(DomainDiscriminator, self).__init__()

        layers = []
        input_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        前向传播

        Args:
            features: 输入特征 [batch_size, feature_dim]

        Returns:
            domain_pred: 域预测 [batch_size, 1]
        """
        return self.discriminator(features)


class AdaptiveWeightNetwork(nn.Module):
    """自适应权重网络"""

    def __init__(self, stat_dim=6, hidden_dims=[32, 16]):
        super(AdaptiveWeightNetwork, self).__init__()

        layers = []
        input_dim = stat_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # 确保权重在[0,1]范围内

        self.weight_network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, statistical_features):
        """
        计算自适应权重

        Args:
            statistical_features: 统计特征 [stat_dim]

        Returns:
            weight: 自适应权重 [1]
        """
        return self.weight_network(statistical_features)


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy损失"""

    def __init__(self, kernel_type='rbf', kernel_params=None):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {'sigma': 1.0}

    def gaussian_kernel(self, x, y, sigma):
        """高斯核函数"""
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        y_norm = torch.sum(y ** 2, dim=1, keepdim=True)

        # 计算欧氏距离的平方
        dist_sq = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())

        # 高斯核
        kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
        return kernel

    def forward(self, source_features, target_features, sigma=None):
        """
        计算MMD损失

        Args:
            source_features: 源域特征 [batch_size, feature_dim]
            target_features: 目标域特征 [batch_size, feature_dim]
            sigma: 核参数

        Returns:
            mmd_loss: MMD损失值
        """
        if sigma is None:
            sigma = self.kernel_params['sigma']

        # 计算核矩阵
        xx = self.gaussian_kernel(source_features, source_features, sigma)
        yy = self.gaussian_kernel(target_features, target_features, sigma)
        xy = self.gaussian_kernel(source_features, target_features, sigma)

        # 计算MMD
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd


class MultiStatGuidedDomainAdapter(nn.Module):
    """
    多统计量引导域适应器

    功能：
    1. 计算源域和目标域的多维统计特征差异
    2. 基于统计差异学习自适应权重
    3. 动态平衡MMD损失和对抗损失
    """

    def __init__(self, config, feature_dim=512):
        super(MultiStatGuidedDomainAdapter, self).__init__()

        self.config = config
        self.feature_dim = feature_dim

        # 统计特征计算器
        self.stat_computer = StatisticalComputer()

        # 域判别器
        self.domain_discriminator = DomainDiscriminator(
            feature_dim=feature_dim,
            hidden_dims=config.discriminator_hidden,
            dropout_rate=config.discriminator_dropout
        )

        # 自适应权重网络
        self.weight_network = AdaptiveWeightNetwork(
            stat_dim=config.num_statistical_features,
            hidden_dims=config.weight_net_hidden
        )

        # MMD损失
        self.mmd_loss = MMDLoss(
            kernel_type='rbf',
            kernel_params={'sigma': config.mmd_sigma}
        )

        # 梯度反转层的比例因子
        self.register_buffer('lambda_factor', torch.tensor(1.0))

    def compute_statistical_differences(self, source_features, target_features):
        """
        计算多维统计特征差异

        Args:
            source_features: 源域特征 [batch_size, feature_dim]
            target_features: 目标域特征 [batch_size, feature_dim]

        Returns:
            stat_differences: 统计差异特征 [6]
        """
        # 1. 均值差异
        src_mean = torch.mean(source_features, dim=0)
        tgt_mean = torch.mean(target_features, dim=0)
        mean_diff = torch.mean(torch.abs(src_mean - tgt_mean))

        # 2. 方差差异
        src_var = torch.var(source_features, dim=0)
        tgt_var = torch.var(target_features, dim=0)
        var_diff = torch.mean(torch.abs(src_var - tgt_var))

        # 3. 峭度差异
        src_kurtosis = self.stat_computer.compute_kurtosis(source_features, dim=0)
        tgt_kurtosis = self.stat_computer.compute_kurtosis(target_features, dim=0)
        kurtosis_diff = torch.mean(torch.abs(src_kurtosis - tgt_kurtosis))

        # 4. 偏度差异
        src_skewness = self.stat_computer.compute_skewness(source_features, dim=0)
        tgt_skewness = self.stat_computer.compute_skewness(target_features, dim=0)
        skewness_diff = torch.mean(torch.abs(src_skewness - tgt_skewness))

        # 5. 能量差异
        src_energy = self.stat_computer.compute_energy(source_features, dim=0)
        tgt_energy = self.stat_computer.compute_energy(target_features, dim=0)
        energy_diff = torch.mean(torch.abs(src_energy - tgt_energy))

        # 6. 相关性差异
        try:
            src_corr = self.stat_computer.compute_correlation_mean(source_features)
            tgt_corr = self.stat_computer.compute_correlation_mean(target_features)
            corr_diff = torch.mean(torch.abs(src_corr - tgt_corr))
        except:
            # 如果相关性计算失败，使用零值
            corr_diff = torch.tensor(0.0, device=source_features.device)

        # 组合统计特征
        stat_differences = torch.stack([
            mean_diff, var_diff, kurtosis_diff,
            skewness_diff, energy_diff, corr_diff
        ])

        return stat_differences

    def gradient_reverse_layer(self, x, alpha=1.0):
        """梯度反转层"""
        return GradientReverseFunction.apply(x, alpha)

    def forward(self, source_features, target_features, alpha=1.0):
        """
        前向传播

        Args:
            source_features: 源域特征 [batch_size, feature_dim]
            target_features: 目标域特征 [batch_size, feature_dim]
            alpha: 梯度反转参数

        Returns:
            domain_loss: 域适应损失
            adaptive_weight: 自适应权重
            loss_components: 损失组件字典
        """
        batch_size = source_features.shape[0]

        # 1. 计算统计特征差异
        stat_differences = self.compute_statistical_differences(source_features, target_features)

        # 2. 计算自适应权重
        adaptive_weight = self.weight_network(stat_differences)

        # 3. 计算MMD损失
        mmd_loss = self.mmd_loss(source_features, target_features)

        # 4. 计算对抗损失
        # 源域标签为1，目标域标签为0
        source_domain_features = self.gradient_reverse_layer(source_features, alpha)
        target_domain_features = self.gradient_reverse_layer(target_features, alpha)

        source_domain_pred = self.domain_discriminator(source_domain_features)
        target_domain_pred = self.domain_discriminator(target_domain_features)

        source_domain_labels = torch.ones_like(source_domain_pred)
        target_domain_labels = torch.zeros_like(target_domain_pred)

        source_adv_loss = F.binary_cross_entropy(source_domain_pred, source_domain_labels)
        target_adv_loss = F.binary_cross_entropy(target_domain_pred, target_domain_labels)
        adversarial_loss = source_adv_loss + target_adv_loss

        # 5. 自适应加权域损失
        domain_loss = adaptive_weight * mmd_loss + (1 - adaptive_weight) * adversarial_loss

        # 损失组件
        loss_components = {
            'mmd_loss': mmd_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'adaptive_weight': adaptive_weight.item(),
            'domain_loss': domain_loss.item(),
            'statistical_differences': stat_differences.detach().cpu().numpy()
        }

        return domain_loss, adaptive_weight, loss_components

    def get_domain_discriminator_accuracy(self, source_features, target_features):
        """
        计算域判别器的准确率（用于监控训练过程）

        Args:
            source_features: 源域特征
            target_features: 目标域特征

        Returns:
            accuracy: 域判别准确率
        """
        with torch.no_grad():
            source_pred = self.domain_discriminator(source_features)
            target_pred = self.domain_discriminator(target_features)

            source_acc = (source_pred > 0.5).float().mean()
            target_acc = (target_pred < 0.5).float().mean()

            overall_acc = (source_acc + target_acc) / 2

            return overall_acc.item()


class GradientReverseFunction(torch.autograd.Function):
    """梯度反转函数"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# 测试代码
if __name__ == '__main__':
    # 测试MSGDA模块
    from types import SimpleNamespace

    # 创建测试配置
    config = SimpleNamespace(
        num_statistical_features=6,
        weight_net_hidden=[32, 16],
        discriminator_hidden=[256, 128],
        discriminator_dropout=0.2,
        mmd_sigma=1.0
    )

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiStatGuidedDomainAdapter(config, feature_dim=512).to(device)

    # 测试输入
    batch_size = 32
    feature_dim = 512

    # 模拟源域和目标域特征
    source_features = torch.randn(batch_size, feature_dim).to(device)
    target_features = torch.randn(batch_size, feature_dim).to(device) * 1.5  # 添加域差异

    # 前向传播
    print("Testing MSGDA module...")
    print(f"Source features shape: {source_features.shape}")
    print(f"Target features shape: {target_features.shape}")

    domain_loss, adaptive_weight, loss_components = model(source_features, target_features)

    print(f"Domain loss: {domain_loss.item():.4f}")
    print(f"Adaptive weight: {adaptive_weight.item():.4f}")
    print("Loss components:")
    for key, value in loss_components.items():
        if key != 'statistical_differences':
            print(f"  {key}: {value:.4f}")

    # 测试域判别器准确率
    disc_acc = model.get_domain_discriminator_accuracy(source_features, target_features)
    print(f"Domain discriminator accuracy: {disc_acc:.4f}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("MSGDA module test completed successfully!")