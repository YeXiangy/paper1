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
            sample = x[i:i + 1, :].T
            if feature_dim > 1:
                try:
                    corr_matrix = torch.corrcoef(sample.squeeze())
                    if corr_matrix.numel() == 1:  # 处理只有一个元素的情况
                        mean_corr = torch.tensor(0.0, device=x.device)
                    else:
                        mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1).bool()
                        if mask.sum() > 0:
                            mean_corr = corr_matrix[mask].mean()
                        else:
                            mean_corr = torch.tensor(0.0, device=x.device)
                except:
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

        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
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

        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.weight_network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, statistical_features):
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
        dist_sq = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())
        kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
        return kernel

    def forward(self, source_features, target_features, sigma=None):
        if sigma is None:
            sigma = self.kernel_params['sigma']

        xx = self.gaussian_kernel(source_features, source_features, sigma)
        yy = self.gaussian_kernel(target_features, target_features, sigma)
        xy = self.gaussian_kernel(source_features, target_features, sigma)

        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd


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

        self.stat_computer = StatisticalComputer()

        self.domain_discriminator = DomainDiscriminator(
            feature_dim=feature_dim,
            hidden_dims=config.discriminator_hidden,
            dropout_rate=config.discriminator_dropout
        )

        self.weight_network = AdaptiveWeightNetwork(
            stat_dim=config.num_statistical_features,
            hidden_dims=config.weight_net_hidden
        )

        self.mmd_loss = MMDLoss(
            kernel_type='rbf',
            kernel_params={'sigma': config.mmd_sigma}
        )

        self.register_buffer('lambda_factor', torch.tensor(1.0))

    def compute_statistical_differences(self, source_features, target_features):
        """计算多维统计特征差异"""
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
            corr_diff = torch.tensor(0.0, device=source_features.device)

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