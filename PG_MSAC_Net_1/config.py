"""
Configuration file for PG-MSAC-Net
物理引导的多尺度自适应跨域轴承故障诊断网络配置文件
"""

import torch
import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = 'JNU_bearing_dataset'
    data_root: str = './data/JNU'
    sample_len: int = 1024
    num_classes: int = 4
    in_channels: int = 1
    normalize_type: str = 'mean~std'
    source_speed: int = 800
    target_speed: int = 1000
    source_shot: int = 50
    target_shot: int = 5
    target_unlabeled: int = 200
    class_names: List[str] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['Normal', 'InnerRace', 'OuterRace', 'Ball']


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
    kernel_sizes: List[int] = None
    scale_channels: int = 64
    conv_channels: List[int] = None
    pool_size: int = 2
    dropout_rate: float = 0.2
    physical_modulation_dim: int = 192

    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 7, 15]
        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256, 512]


@dataclass
class MSGDAConfig:
    """MSGDA配置 - 多统计量引导域适应器"""
    num_statistical_features: int = 6
    weight_net_hidden: List[int] = None
    discriminator_hidden: List[int] = None
    discriminator_dropout: float = 0.2
    mmd_sigma: float = 1.0

    def __post_init__(self):
        if self.weight_net_hidden is None:
            self.weight_net_hidden = [32, 16]
        if self.discriminator_hidden is None:
            self.discriminator_hidden = [256, 128]


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
    metrics: List[str] = None
    save_confusion_matrix: bool = True
    save_tsne_plot: bool = True
    save_loss_curves: bool = True
    save_feature_visualization: bool = True
    tsne_perplexity: int = 30
    tsne_n_iter: int = 1000
    normalize_confusion_matrix: bool = True

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']


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
        print(f"Cross-domain: {self.data.source_speed}rpm → {self.data.target_speed}rpm")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.training.batch_size}")
        print(f"Max epochs: {self.training.max_epochs}")
        print(f"Learning rate: {self.training.base_lr}")
        print(f"Sample length: {self.data.sample_len}")
        print(f"Target shot: {self.data.target_shot}")
        print("=" * 50)


# 全局配置实例
config = Config()


def update_config_from_args():
    """从命令行参数更新配置"""
    import argparse

    parser = argparse.ArgumentParser(description='PG-MSAC-Net Training')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default=config.data.dataset_name,
                        help='Dataset name')
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
    parser.add_argument('--experiment_type', type=str, default='cross_domain',
                        choices=['cross_domain', 'ablation', 'hyperparameter'],
                        help='Type of experiment to run')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')

    args = parser.parse_args()

    # 更新配置
    config.data.dataset_name = args.dataset
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


if __name__ == '__main__':
    # 测试配置
    config.print_config()
    print(f"Experiment name: {config.get_experiment_name()}")