"""
Datasets module for PG-MSAC-Net
数据集模块
"""

from .JNU_bearing_dataset import JNUBearingDataset, create_jnu_dataloaders, create_single_domain_dataloader
from .data_utils import (
    DataProcessor,
    CrossDomainDataSplitter,
    DataAugmentation,
    SequenceNormalizer,
    BaseDataset,
    UnlabeledDataset
)

__all__ = [
    'JNUBearingDataset',
    'create_jnu_dataloaders',
    'create_single_domain_dataloader',
    'DataProcessor',
    'CrossDomainDataSplitter',
    'DataAugmentation',
    'SequenceNormalizer',
    'BaseDataset',
    'UnlabeledDataset'
]

# 数据集信息
DATASET_INFO = {
    'JNU_bearing': {
        'num_classes': 4,
        'class_names': ['Normal', 'InnerRace', 'OuterRace', 'Ball'],
        'available_speeds': [600, 800, 1000],
        'sample_length': 1024,
        'channels': 1,
        'file_format': '.mat files',
        'recommended_settings': {
            'sample_len': 1024,
            'normalize_type': 'mean~std',
            'overlap': 0.5,
            'batch_size': 64
        }
    }
}

def get_dataset_info(dataset_name):
    """获取数据集信息"""
    return DATASET_INFO.get(dataset_name, None)

def list_available_datasets():
    """列出可用的数据集"""
    return list(DATASET_INFO.keys())