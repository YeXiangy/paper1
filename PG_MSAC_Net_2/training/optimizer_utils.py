"""
优化器工具函数
提供各种优化器和学习率调度策略
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, CyclicLR, OneCycleLR
)
import math
from typing import Dict, List, Optional, Union


class WarmupLRScheduler:
    """带预热的学习率调度器"""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_scheduler: torch.optim.lr_scheduler._LRScheduler,
                 warmup_epochs: int = 10,
                 warmup_factor: float = 0.1):
        """
        初始化预热调度器

        Args:
            optimizer: 优化器
            base_scheduler: 基础调度器
            warmup_epochs: 预热轮数
            warmup_factor: 预热因子
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.current_epoch = 0

        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            warmup_lr_factor = self.warmup_factor + (1.0 - self.warmup_factor) * \
                               (self.current_epoch / self.warmup_epochs)

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_lr_factor
        else:
            # 正常调度
            self.base_scheduler.step()

    def get_last_lr(self):
        """获取最后的学习率"""
        if self.current_epoch < self.warmup_epochs:
            warmup_lr_factor = self.warmup_factor + (1.0 - self.warmup_factor) * \
                               (self.current_epoch / self.warmup_epochs)
            return [lr * warmup_lr_factor for lr in self.base_lrs]
        else:
            return self.base_scheduler.get_last_lr()


class CosineAnnealingWarmupRestarts:
    """带预热和重启的余弦退火调度器"""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_0: int = 50,
                 T_mult: int = 2,
                 eta_max: float = 1e-3,
                 eta_min: float = 1e-6,
                 warmup_epochs: int = 10):
        """
        初始化调度器

        Args:
            optimizer: 优化器
            T_0: 第一次重启的周期
            T_mult: 重启周期倍数
            eta_max: 最大学习率
            eta_min: 最小学习率
            warmup_epochs: 预热轮数
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs

        self.T_cur = 0
        self.T_i = T_0
        self.restart_count = 0

        # 设置初始学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = eta_min

    def step(self):
        """更新学习率"""
        if self.T_cur < self.warmup_epochs:
            # 预热阶段
            lr = self.eta_max * (self.T_cur / self.warmup_epochs)
        else:
            # 余弦退火阶段
            effective_T_cur = self.T_cur - self.warmup_epochs
            effective_T_i = self.T_i - self.warmup_epochs

            lr = self.eta_min + (self.eta_max - self.eta_min) * \
                 (1 + math.cos(math.pi * effective_T_cur / effective_T_i)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.T_cur += 1

        # 检查是否需要重启
        if self.T_cur >= self.T_i:
            self.restart_count += 1
            self.T_cur = 0
            self.T_i = self.T_0 * (self.T_mult ** self.restart_count)

    def get_last_lr(self):
        """获取最后的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


def create_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    创建优化器

    Args:
        model: 模型
        config: 优化器配置

    Returns:
        optimizer: 优化器实例
    """
    optimizer_type = config.get('type', 'adam').lower()
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)

    # 获取参数组
    if config.get('use_differential_lr', False):
        # 使用分层学习率
        param_groups = []

        # 物理编码器参数
        if hasattr(model, 'physical_encoder'):
            param_groups.append({
                'params': model.physical_encoder.parameters(),
                'lr': lr * config.get('physical_lr_mult', 2.0),
                'name': 'physical_encoder'
            })

        # CNN主干参数
        if hasattr(model, 'adaptive_cnn'):
            param_groups.append({
                'params': model.adaptive_cnn.parameters(),
                'lr': lr,
                'name': 'adaptive_cnn'
            })

        # 域适应器参数
        if hasattr(model, 'domain_adapter'):
            param_groups.append({
                'params': model.domain_adapter.parameters(),
                'lr': lr * config.get('domain_lr_mult', 0.5),
                'name': 'domain_adapter'
            })

        # 分类器参数
        if hasattr(model, 'classifier'):
            param_groups.append({
                'params': model.classifier.parameters(),
                'lr': lr,
                'name': 'classifier'
            })

        params = param_groups
    else:
        # 使用统一学习率
        params = model.parameters()

    # 创建优化器
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            params,
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=config.get('nesterov', False)
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            params,
            lr=lr,
            alpha=config.get('alpha', 0.99),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay,
            momentum=config.get('momentum', 0)
        )
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(
            params,
            lr=lr,
            lr_decay=config.get('lr_decay', 0),
            weight_decay=weight_decay,
            eps=config.get('eps', 1e-10)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[object]:
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        config: 调度器配置

    Returns:
        scheduler: 调度器实例
    """
    if config is None or config.get('type') is None:
        return None

    scheduler_type = config.get('type', 'none').lower()

    if scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [50, 100]),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'max'),
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 10),
            threshold=config.get('threshold', 1e-4),
            verbose=config.get('verbose', True)
        )
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(
            optimizer,
            base_lr=config.get('base_lr', 1e-5),
            max_lr=config.get('max_lr', 1e-2),
            step_size_up=config.get('step_size_up', 20),
            mode=config.get('mode', 'triangular')
        )
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 1e-2),
            total_steps=config.get('total_steps', 1000),
            pct_start=config.get('pct_start', 0.3),
            anneal_strategy=config.get('anneal_strategy', 'cos')
        )
    elif scheduler_type == 'warmup':
        # 带预热的调度器
        base_scheduler_config = config.get('base_scheduler', {'type': 'cosine'})
        base_scheduler = create_scheduler(optimizer, base_scheduler_config)

        scheduler = WarmupLRScheduler(
            optimizer,
            base_scheduler,
            warmup_epochs=config.get('warmup_epochs', 10),
            warmup_factor=config.get('warmup_factor', 0.1)
        )
    elif scheduler_type == 'cosine_warmup_restarts':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            T_0=config.get('T_0', 50),
            T_mult=config.get('T_mult', 2),
            eta_max=config.get('eta_max', 1e-3),
            eta_min=config.get('eta_min', 1e-6),
            warmup_epochs=config.get('warmup_epochs', 10)
        )
    else:
        scheduler = None

    return scheduler


def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict:
    """
    获取优化器信息

    Args:
        optimizer: 优化器

    Returns:
        info: 优化器信息字典
    """
    info = {
        'type': optimizer.__class__.__name__,
        'param_groups': len(optimizer.param_groups),
        'total_params': sum(len(group['params']) for group in optimizer.param_groups)
    }

    # 添加参数组详细信息
    for i, group in enumerate(optimizer.param_groups):
        group_info = {
            'lr': group['lr'],
            'weight_decay': group.get('weight_decay', 0),
            'params_count': len(group['params'])
        }

        if 'name' in group:
            group_info['name'] = group['name']

        info[f'group_{i}'] = group_info

    return info


# 测试函数
if __name__ == '__main__':
    import torch.nn as nn


    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.physical_encoder = nn.Linear(10, 20)
            self.adaptive_cnn = nn.Conv1d(1, 64, 3)
            self.domain_adapter = nn.Linear(64, 32)
            self.classifier = nn.Linear(32, 4)

        def forward(self, x):
            return x


    model = TestModel()

    # 测试优化器创建
    optimizer_config = {
        'type': 'adam',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'use_differential_lr': True,
        'physical_lr_mult': 2.0,
        'domain_lr_mult': 0.5
    }

    optimizer = create_optimizer(model, optimizer_config)
    print("Optimizer created successfully")
    print(f"Optimizer info: {get_optimizer_info(optimizer)}")

    # 测试调度器创建
    scheduler_config = {
        'type': 'cosine_warmup_restarts',
        'T_0': 50,
        'warmup_epochs': 10
    }

    scheduler = create_scheduler(optimizer, scheduler_config)
    print(f"Scheduler created: {scheduler.__class__.__name__}")

    print("✓ Optimizer utils test completed!")