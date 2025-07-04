"""
工具模块
包含可视化、模型保存加载等实用功能
"""

from .visualization import (
    plot_training_curves,
    plot_confusion_matrix_advanced,
    plot_feature_tsne,
    plot_attention_weights,
    plot_domain_adaptation_analysis,
    save_model_architecture_diagram,
    create_training_dashboard
)

from .checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    save_best_model,
    load_best_model,
    create_experiment_folder,
    backup_code_files
)

__all__ = [
    # 可视化工具
    'plot_training_curves',
    'plot_confusion_matrix_advanced',
    'plot_feature_tsne',
    'plot_attention_weights',
    'plot_domain_adaptation_analysis',
    'save_model_architecture_diagram',
    'create_training_dashboard',

    # 模型管理工具
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
    'save_best_model',
    'load_best_model',
    'create_experiment_folder',
    'backup_code_files'
]

# 版本信息
__version__ = '1.0.0'
__description__ = 'Utility functions for PG-MSAC-Net project'

# 默认配置
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 300
DEFAULT_FONT_SIZE = 12


def get_default_style():
    """获取默认的matplotlib样式设置"""
    return {
        'figure.figsize': DEFAULT_FIGURE_SIZE,
        'figure.dpi': DEFAULT_DPI,
        'font.size': DEFAULT_FONT_SIZE,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    }


def setup_matplotlib_style():
    """设置matplotlib样式"""
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8')  # 使用seaborn样式
    plt.rcParams.update(get_default_style())