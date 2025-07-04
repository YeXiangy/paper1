"""
PG-MSAC-Net模型模块
包含所有模型组件的导入
"""

from .PG_MSAC_Net import PG_MSAC_Net, create_pg_msac_net
from .MPIE import MultiLevelPhysicalEncoder
from .AMSCNN import AdaptiveMultiScaleCNN
from .MSGDA import MultiStatGuidedDomainAdapter
from .model_utils import (
    count_parameters,
    get_model_size,
    save_model_checkpoint,
    load_model_checkpoint,
    print_model_summary,
    initialize_weights,
    freeze_layers,
    unfreeze_layers,
    get_learning_rate_groups,
    validate_model_config,
    model_inference
)

__all__ = [
    # 主要模型类
    'PG_MSAC_Net',
    'MultiLevelPhysicalEncoder',
    'AdaptiveMultiScaleCNN',
    'MultiStatGuidedDomainAdapter',

    # 工厂函数
    'create_pg_msac_net',

    # 工具函数
    'count_parameters',
    'get_model_size',
    'save_model_checkpoint',
    'load_model_checkpoint',
    'print_model_summary',
    'initialize_weights',
    'freeze_layers',
    'unfreeze_layers',
    'get_learning_rate_groups',
    'validate_model_config',
    'model_inference'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

# 模型信息
MODEL_INFO = {
    'name': 'PG-MSAC-Net',
    'full_name': 'Physical-Guided Multi-Scale Adaptive Cross-Domain Network',
    'description': 'A deep learning model for cross-domain bearing fault diagnosis',
    'components': {
        'MPIE': 'Multi-level Physical Information Encoder',
        'AMSCNN': 'Adaptive Multi-Scale CNN',
        'MSGDA': 'Multi-Statistical-Guided Domain Adapter'
    },
    'features': [
        'Multi-level physical feature encoding',
        'Adaptive multi-scale convolutional processing',
        'Statistical-guided domain adaptation',
        'Cross-domain fault diagnosis capability'
    ]
}


def get_model_info():
    """获取模型信息"""
    return MODEL_INFO


def create_model(config):
    """
    工厂函数：根据配置创建模型

    Args:
        config: 配置对象

    Returns:
        model: PG_MSAC_Net模型实例
    """
    # 验证配置
    if not validate_model_config(config):
        raise ValueError("Invalid model configuration")

    # 创建模型
    model = create_pg_msac_net(config)

    # 打印模型摘要
    print_model_summary(model, input_shape=(1, 1, config.data.sample_len))

    return model


def quick_test():
    """快速测试所有模型组件"""
    print("Running quick test for all model components...")

    # 这里可以添加简单的测试代码来验证模型组件工作正常
    try:
        from types import SimpleNamespace
        import torch

        # 创建测试配置
        mpie_config = SimpleNamespace(
            time_features_dim=8, freq_features_dim=7, tf_features_dim=5,
            time_hidden_dim=48, freq_hidden_dim=40, tf_hidden_dim=40,
            output_dim=128
        )

        amscnn_config = SimpleNamespace(
            kernel_sizes=[3, 7, 15], scale_channels=64,
            conv_channels=[64, 128, 256, 512],
            physical_modulation_dim=192, dropout_rate=0.2
        )

        msgda_config = SimpleNamespace(
            num_statistical_features=6, weight_net_hidden=[32, 16],
            discriminator_hidden=[256, 128], discriminator_dropout=0.2,
            mmd_sigma=1.0
        )

        # 测试模型创建
        device = torch.device('cpu')  # 使用CPU避免GPU依赖
        model = PG_MSAC_Net(
            num_classes=4, sample_len=1024,
            mpie_config=mpie_config,
            amscnn_config=amscnn_config,
            msgda_config=msgda_config
        ).to(device)

        # 测试前向传播
        test_input = torch.randn(2, 1, 1024)
        output = model(test_input)

        assert output.shape == (2, 4), f"Expected output shape (2, 4), got {output.shape}"

        print("✓ All model components working correctly!")
        return True

    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")
        return False

# 在模块导入时运行快速测试（可选）
# quick_test()