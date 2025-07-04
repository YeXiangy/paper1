"""
损失函数定义
包含PG-MSAC-Net所需的各种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class CrossDomainLoss(nn.Module):
    """
    跨域损失函数
    整合分类损失、物理一致性损失和域适应损失
    """

    def __init__(self,
                 num_classes: int,
                 loss_weights: Dict[str, float] = None,
                 label_smoothing: float = 0.1):
        super(CrossDomainLoss, self).__init__()

        self.num_classes = num_classes

        # 默认损失权重
        if loss_weights is None:
            self.loss_weights = {
                'classification': 1.0,
                'physical': 0.1,
                'domain': 0.01
            }
        else:
            self.loss_weights = loss_weights

        # 分类损失（带标签平滑）
        self.classification_loss = LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=label_smoothing
        )

        # 物理一致性损失
        self.physical_loss = PhysicalConsistencyLoss()

        # 记录各组件损失
        self.loss_components = {}

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                domain_loss: Optional[torch.Tensor] = None,
                physical_code: Optional[torch.Tensor] = None,
                deep_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失

        Args:
            outputs: 模型输出字典，包含logits
            targets: 真实标签
            domain_loss: 域适应损失
            physical_code: 物理编码
            deep_features: 深度特征

        Returns:
            total_loss: 总损失
            loss_dict: 损失组件字典
        """
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs

        # 1. 分类损失
        cls_loss = self.classification_loss(logits, targets)
        self.loss_components['classification'] = cls_loss.item()

        # 2. 物理一致性损失
        if physical_code is not None and deep_features is not None:
            phy_loss = self.physical_loss(physical_code, deep_features)
            self.loss_components['physical'] = phy_loss.item()
        else:
            phy_loss = torch.tensor(0.0, device=logits.device)
            self.loss_components['physical'] = 0.0

        # 3. 域适应损失
        if domain_loss is not None:
            self.loss_components['domain'] = domain_loss.item()
        else:
            domain_loss = torch.tensor(0.0, device=logits.device)
            self.loss_components['domain'] = 0.0

        # 4. 总损失
        total_loss = (
                self.loss_weights['classification'] * cls_loss +
                self.loss_weights['physical'] * phy_loss +
                self.loss_weights['domain'] * domain_loss
        )

        self.loss_components['total'] = total_loss.item()

        return total_loss, self.loss_components.copy()

    def update_weights(self, epoch: int, max_epochs: int):
        """动态调整损失权重"""
        # 域适应权重随训练进行逐渐增加
        progress = epoch / max_epochs
        self.loss_weights['domain'] = 0.01 * (1 + progress)

        # 物理一致性权重在前期较大，后期减小
        self.loss_weights['physical'] = 0.1 * (2 - progress)


class PhysicalConsistencyLoss(nn.Module):
    """物理一致性损失"""

    def __init__(self, consistency_type: str = 'mse'):
        super(PhysicalConsistencyLoss, self).__init__()
        self.consistency_type = consistency_type

    def forward(self, physical_code: torch.Tensor, deep_features: torch.Tensor) -> torch.Tensor:
        """
        计算物理特征与深度特征的一致性损失

        Args:
            physical_code: 物理编码 [batch_size, physical_dim]
            deep_features: 深度特征 [batch_size, feature_dim]

        Returns:
            consistency_loss: 一致性损失
        """
        # 维度对齐：将深度特征投影到物理特征空间
        if physical_code.shape[1] != deep_features.shape[1]:
            # 使用线性投影对齐维度
            projection = nn.Linear(
                deep_features.shape[1],
                physical_code.shape[1]
            ).to(deep_features.device)

            projected_features = projection(deep_features)
        else:
            projected_features = deep_features

        # 计算一致性损失
        if self.consistency_type == 'mse':
            # 均方误差
            consistency_loss = F.mse_loss(projected_features, physical_code)
        elif self.consistency_type == 'cosine':
            # 余弦相似度损失
            cos_sim = F.cosine_similarity(projected_features, physical_code, dim=1)
            consistency_loss = 1 - cos_sim.mean()
        elif self.consistency_type == 'kl':
            # KL散度损失
            p_norm = F.softmax(physical_code, dim=1)
            q_norm = F.softmax(projected_features, dim=1)
            consistency_loss = F.kl_div(q_norm.log(), p_norm, reduction='batchmean')
        else:
            consistency_loss = F.mse_loss(projected_features, physical_code)

        return consistency_loss


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss

        Args:
            inputs: 预测logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]

        Returns:
            focal_loss: Focal损失
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑损失

        Args:
            inputs: 预测logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]

        Returns:
            smoothed_loss: 平滑损失
        """
        log_probs = F.log_softmax(inputs, dim=1)

        # 创建平滑标签
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # 计算损失
        loss = torch.sum(-true_dist * log_probs, dim=1)

        return loss.mean()


class AdversarialLoss(nn.Module):
    """对抗损失"""

    def __init__(self, loss_type: str = 'bce'):
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self,
                source_domain_pred: torch.Tensor,
                target_domain_pred: torch.Tensor) -> torch.Tensor:
        """
        计算对抗损失

        Args:
            source_domain_pred: 源域判别结果 [batch_size, 1]
            target_domain_pred: 目标域判别结果 [batch_size, 1]

        Returns:
            adversarial_loss: 对抗损失
        """
        # 源域标签为1，目标域标签为0
        source_labels = torch.ones_like(source_domain_pred)
        target_labels = torch.zeros_like(target_domain_pred)

        if self.loss_type == 'bce':
            source_loss = F.binary_cross_entropy_with_logits(
                source_domain_pred, source_labels
            )
            target_loss = F.binary_cross_entropy_with_logits(
                target_domain_pred, target_labels
            )
        else:
            source_loss = F.mse_loss(source_domain_pred, source_labels)
            target_loss = F.mse_loss(target_domain_pred, target_labels)

        return source_loss + target_loss


class CenterLoss(nn.Module):
    """Center Loss - 增强类内紧密性"""

    def __init__(self, num_classes: int, feature_dim: int, alpha: float = 0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha

        # 可学习的类中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算Center Loss

        Args:
            features: 特征向量 [batch_size, feature_dim]
            labels: 标签 [batch_size]

        Returns:
            center_loss: Center损失
        """
        batch_size = features.size(0)

        # 计算特征到对应类中心的距离
        centers_batch = self.centers[labels]
        center_loss = F.mse_loss(features, centers_batch)

        return center_loss


# 辅助函数
def get_loss_function(loss_config: Dict) -> nn.Module:
    """
    根据配置获取损失函数

    Args:
        loss_config: 损失函数配置

    Returns:
        loss_function: 损失函数实例
    """
    loss_type = loss_config.get('type', 'cross_domain')

    if loss_type == 'cross_domain':
        return CrossDomainLoss(
            num_classes=loss_config.get('num_classes', 4),
            loss_weights=loss_config.get('weights', None),
            label_smoothing=loss_config.get('label_smoothing', 0.1)
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('alpha', 1.0),
            gamma=loss_config.get('gamma', 2.0)
        )
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(
            num_classes=loss_config.get('num_classes', 4),
            smoothing=loss_config.get('smoothing', 0.1)
        )
    else:
        return nn.CrossEntropyLoss()


# 测试函数
if __name__ == '__main__':
    # 测试损失函数
    print("Testing loss functions...")

    batch_size = 8
    num_classes = 4
    feature_dim = 512
    physical_dim = 128

    # 创建测试数据
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    physical_code = torch.randn(batch_size, physical_dim)
    deep_features = torch.randn(batch_size, feature_dim)
    domain_loss = torch.tensor(0.1)

    # 测试CrossDomainLoss
    cross_domain_loss = CrossDomainLoss(num_classes=num_classes)

    outputs = {'logits': logits}
    total_loss, loss_components = cross_domain_loss(
        outputs, targets, domain_loss, physical_code, deep_features
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print("Loss components:")
    for name, value in loss_components.items():
        print(f"  {name}: {value:.4f}")

    print("✓ Loss functions test completed!")