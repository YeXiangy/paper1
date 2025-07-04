"""
Configuration file for PG-MSAC-Net
物理引导的多尺度自适应跨域轴承故障诊断网络配置文件
"""

import torch
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = 'JNU_bearing_dataset'
    data_root: str = r"C:\Users\33075\Desktop\Paper_1\PG_MSAC_Net_1\data\JNU"
    sample_len: int = 1024
    num_classes: int = 3  # 改为3类，去掉Ball类别
    in_channels: int = 1
    normalize_type: str = 'mean~std'
    source_speed: int = 800
    target_speed: int = 1000
    source_shot: int = 50
    target_shot: int = 5
    target_unlabeled: int = 200
    class_names: List[str] = field(default_factory=lambda: ['Normal', 'InnerRace', 'OuterRace'])

    # 文件名映射配置 - 适配你的CSV文件命名
    file_name_mapping: Dict[str, str] = field(default_factory=lambda: {
        'Normal_600': 'n600_3_2.csv',
        'Normal_800': 'n800_3_2.csv',
        'Normal_1000': 'n1000_3_2.csv',
        'InnerRace_600': 'ib600_2.csv',
        'InnerRace_800': 'ib800_2.csv',
        'InnerRace_1000': 'ib1000_2.csv',
        'OuterRace_600': 'ob600_2.csv',
        'OuterRace_800': 'ob800_2.csv',
        'OuterRace_1000': 'ob1000_2.csv',
        # 如果以后有滚动体故障数据，可以添加：
        # 'Ball_600': 'rb600_2.csv',
        # 'Ball_800': 'rb800_2.csv',
        # 'Ball_1000': 'rb1000_2.csv',
    })


@dataclass
class MPIEConfig:
    """MPIE配置 - 多级物理信息编码器"""
    time_features_dim: int = 8
    freq_features_dim: int = 7
    tf_features_dim: int = 5
    total_physical_dim: int = 20
    time_hidden_dim: int = 48
    freq_hidden_dim: int = 40
    tf_hidden_dim: int = 40
    output_dim: int = 128
    activation: str = 'relu'


@dataclass
class AMSCNNConfig:
    """AMSCNN配置 - 自适应多尺度CNN"""
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 15])
    scale_channels: int = 64
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    pool_size: int = 2
    dropout_rate: float = 0.2
    physical_modulation_dim: int = 192


@dataclass
class MSGDAConfig:
    """MSGDA配置 - 多统计量引导域适应器"""
    num_statistical_features: int = 6
    weight_net_hidden: List[int] = field(default_factory=lambda: [32, 16])
    discriminator_hidden: List[int] = field(default_factory=lambda: [256, 128])
    discriminator_dropout: float = 0.2
    mmd_sigma: float = 1.0


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    max_epochs: int = 200
    base_lr: float = 1e-3
    physical_encoder_lr: float = 2e-3
    domain_adapter_lr: float = 5e-4
    classifier_lr: float = 1e-3
    lr_scheduler: str = 'cosine'
    lr_step_size: int = 50
    lr_gamma: float = 0.5
    optimizer: str = 'adam'
    weight_decay: float = 1e-4
    momentum: float = 0.9
    lambda_classification: float = 1.0
    lambda_physical: float = 0.1
    lambda_domain: float = 0.01
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    grad_clip: bool = True
    max_grad_norm: float = 1.0
    save_dir: str = './checkpoints'
    log_interval: int = 50
    val_interval: int = 1
    save_interval: int = 10
    domain_adaptation: bool = True
    alpha_schedule: str = 'constant'


@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1_score', 'auc'])
    save_confusion_matrix: bool = True
    save_tsne_plot: bool = True
    save_loss_curves: bool = True
    save_feature_visualization: bool = True
    tsne_perplexity: int = 30
    tsne_n_iter: int = 1000
    normalize_confusion_matrix: bool = True


class Config:
    """主配置类"""

    def __init__(self):
        # 基础设置
        self.project_name = "PG_MSAC_Net"
        self.model_name = "PG_MSAC_Net"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 8 if torch.cuda.is_available() else 4
        self.pin_memory = True if self.device.type == 'cuda' else False
        self.seed = 42

        # 各模块配置
        self.data = DataConfig()
        self.mpie = MPIEConfig()
        self.amscnn = AMSCNNConfig()
        self.msgda = MSGDAConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()

        # 路径配置
        self.base_dir = os.getcwd()
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.models_dir = os.path.join(self.results_dir, 'models')
        self.logs_dir = os.path.join(self.results_dir, 'logs')
        self.figures_dir = os.path.join(self.results_dir, 'figures')

        # 实验特定路径
        experiment_name = f"{self.data.dataset_name}_{self.data.source_speed}to{self.data.target_speed}"
        self.experiment_dir = os.path.join(self.results_dir, experiment_name)

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.results_dir, self.models_dir, self.logs_dir,
            self.figures_dir, self.experiment_dir, self.training.save_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_experiment_name(self):
        """获取实验名称"""
        return f"{self.model_name}_{self.data.dataset_name}_{self.data.source_speed}to{self.data.target_speed}_shot{self.data.target_shot}"

    def print_config(self):
        """打印配置信息"""
        print("=" * 50)
        print(f"PG-MSAC-Net Configuration")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.data.dataset_name}")
        print(f"Data root: {self.data.data_root}")
        print(f"Cross-domain: {self.data.source_speed}rpm → {self.data.target_speed}rpm")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.training.batch_size}")
        print(f"Max epochs: {self.training.max_epochs}")
        print(f"Learning rate: {self.training.base_lr}")
        print(f"Sample length: {self.data.sample_len}")
        print(f"Target shot: {self.data.target_shot}")
        print(f"Number of classes: {self.data.num_classes}")
        print(f"Class names: {self.data.class_names}")
        print("=" * 50)

    def update_data_path(self, new_path: str):
        """更新数据路径"""
        self.data.data_root = new_path
        print(f"Data path updated to: {new_path}")

    def update_training_params(self, **kwargs):
        """更新训练参数"""
        for key, value in kwargs.items():
            if hasattr(self.training, key):
                setattr(self.training, key, value)
                print(f"Updated training.{key} to {value}")
            else:
                print(f"Warning: Unknown training parameter {key}")


# 全局配置实例
config = Config()


def update_config_from_args():
    """从命令行参数更新配置"""
    import argparse

    parser = argparse.ArgumentParser(description='PG-MSAC-Net Training')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default=config.data.dataset_name,
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default=config.data.data_root,
                        help='Data root directory')
    parser.add_argument('--source_speed', type=int, default=config.data.source_speed,
                        help='Source domain speed')
    parser.add_argument('--target_speed', type=int, default=config.data.target_speed,
                        help='Target domain speed')
    parser.add_argument('--shot', type=int, default=config.data.target_shot,
                        help='Number of samples per class in target domain')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=config.training.batch_size,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.training.max_epochs,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=config.training.base_lr,
                        help='Learning rate')

    # 损失权重
    parser.add_argument('--lambda_physical', type=float, default=config.training.lambda_physical,
                        help='Physical consistency loss weight')
    parser.add_argument('--lambda_domain', type=float, default=config.training.lambda_domain,
                        help='Domain adaptation loss weight')

    # 实验设置
    parser.add_argument('--experiment_type', type=str, default='quick_test',
                        choices=['cross_domain', 'ablation', 'hyperparameter', 'quick_test'],
                        help='Type of experiment to run')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')

    args = parser.parse_args()

    # 更新配置
    config.data.dataset_name = args.dataset
    config.data.data_root = args.data_root
    config.data.source_speed = args.source_speed
    config.data.target_speed = args.target_speed
    config.data.target_shot = args.shot
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.epochs
    config.training.base_lr = args.lr
    config.training.lambda_physical = args.lambda_physical
    config.training.lambda_domain = args.lambda_domain

    # 设置GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        config.device = torch.device(f'cuda:{args.gpu}')

    return args


def load_config_from_file(config_path: str):
    """从文件加载配置（为将来的YAML支持预留）"""
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default config")
        return config

    # 这里可以添加YAML配置文件加载逻辑
    print(f"Loading config from {config_path}")
    return config


if __name__ == '__main__':
    # 测试配置
    config.print_config()
    print(f"Experiment name: {config.get_experiment_name()}")

    # 测试配置更新
    config.update_data_path("./new_data_path")
    config.update_training_params(batch_size=32, max_epochs=100)

    # 显示文件名映射
    print("\nFile name mapping:")
    for key, value in config.data.file_name_mapping.items():
        print(f"  {key}: {value}")