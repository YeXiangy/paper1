"""
评估模块
包含模型评估相关的功能
"""

from .evaluator import (
    ModelEvaluator,
    CrossDomainEvaluator,
    evaluate_single_domain,
    evaluate_cross_domain,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves
)

__all__ = [
    'ModelEvaluator',
    'CrossDomainEvaluator',
    'evaluate_single_domain',
    'evaluate_cross_domain',
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_roc_curves'
]

# 版本信息
__version__ = '1.0.0'
__description__ = 'Evaluation module for PG-MSAC-Net bearing fault diagnosis'

# 支持的评估指标
SUPPORTED_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc_roc',
    'confusion_matrix',
    'classification_report'
]

# 默认类别名称
DEFAULT_CLASS_NAMES = ['Normal', 'Inner_Race', 'Outer_Race', 'Rolling_Element']

def get_supported_metrics():
    """获取支持的评估指标列表"""
    return SUPPORTED_METRICS.copy()

def get_default_class_names():
    """获取默认类别名称"""
    return DEFAULT_CLASS_NAMES.copy()