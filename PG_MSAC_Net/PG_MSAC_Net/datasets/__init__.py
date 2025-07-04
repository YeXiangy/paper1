"""
Datasets module for PG-MSAC-Net
数据集模块
"""

from .JNU_bearing import JNU_bearing_dataset, JNUBearingDataset
from .data_utils import (
    DataProcessor,
    CrossDomainDataSplitter,
    DataAugmentation,
    SequenceNormalizer
)

__all__ = [
    'JNU_bearing_dataset',
    'JNUBearingDataset',
    'DataProcessor',
    'CrossDomainDataSplitter',
    'DataAugmentation',
    'SequenceNormalizer'
]

# 数据集信息
DATASET_INFO = {
    'JNU_bearing': {
        'num_classes': 4,
        'class_names': ['Normal', 'Inner', 'Ball', 'Outer'],
        'sample_rates': [800, 1000, 1200],  # 可用的转速
        'sample_length': 1024,
        'channels': 1
    }
}

def get_dataset_info(dataset_name):
    """获取数据集信息"""
    return DATASET_INFO.get(dataset_name, None)

def list_available_datasets():
    """列出可用的数据集"""
    return list(DATASET_INFO.keys())