"""
Training模块
包含训练相关的所有组件
"""

from .trainer import PGMSACTrainer
from .loss_functions import (
    CrossDomainLoss,
    PhysicalConsistencyLoss,
    FocalLoss,
    LabelSmoothingLoss
)

__all__ = [
    'PGMSACTrainer',
    'CrossDomainLoss',
    'PhysicalConsistencyLoss',
    'FocalLoss',
    'LabelSmoothingLoss'
]

# 版本信息
__version__ = '1.0.0'
__description__ = 'Training components for PG-MSAC-Net'

# 默认训练配置
DEFAULT_TRAINING_CONFIG = {
    'batch_size': 64,
    'max_epochs': 200,
    'base_lr': 1e-3,
    'weight_decay': 1e-4,
    'scheduler_type': 'cosine',
    'warmup_epochs': 10,
    'loss_weights': {
        'classification': 1.0,
        'physical': 0.1,
        'domain': 0.01
    }
}

def get_default_config():
    """获取默认训练配置"""
    return DEFAULT_TRAINING_CONFIG.copy()