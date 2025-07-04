"""
简化的配置文件
"""

import torch
import os
from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
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
    class_names: List[str] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['Normal', 'Inner', 'Ball', 'Outer']


@dataclass  
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 200
    base_lr: float = 1e-3
    optimizer: str = 'adam'
    weight_decay: float = 1e-4
    lambda_classification: float = 1.0
    lambda_physical: float = 0.1
    lambda_domain: float = 0.01
    save_dir: str = './checkpoints'


class Config:
    def __init__(self):
        self.project_name = "PG_MSAC_Net"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        self.seed = 42

        self.data = DataConfig()
        self.training = TrainingConfig()

        # 创建目录
        os.makedirs(self.training.save_dir, exist_ok=True)

    def print_config(self):
        print("=" * 50)
        print("PG-MSAC-Net Configuration")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.training.batch_size}")
        print(f"Max epochs: {self.training.max_epochs}")
        print("=" * 50)


config = Config()
