"""
测试脚本 - 验证修复效果
"""

def test_imports():
    """测试导入是否正常"""
    print("Testing imports...")

    try:
        from datasets.JNU_bearing_dataset import JNUBearingDataset
        print("✅ JNUBearingDataset import OK")
    except ImportError as e:
        print(f"❌ JNUBearingDataset import failed: {e}")

    try:
        from evaluation.metrics import calculate_metrics
        print("✅ calculate_metrics import OK")
    except ImportError as e:
        print(f"❌ calculate_metrics import failed: {e}")

    try:
        from models.PG_MSAC_Net import PG_MSAC_Net
        print("✅ PG_MSAC_Net import OK")
    except ImportError as e:
        print(f"❌ PG_MSAC_Net import failed: {e}")


def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")

    try:
        import torch
        from types import SimpleNamespace

        # 创建测试配置
        mpie_config = SimpleNamespace(
            time_features_dim=8, freq_features_dim=7, tf_features_dim=5,
            time_hidden_dim=48, freq_hidden_dim=40, tf_hidden_dim=40, output_dim=128
        )

        amscnn_config = SimpleNamespace(
            kernel_sizes=[3, 7, 15], scale_channels=64,
            conv_channels=[64, 128, 256, 512],
            physical_modulation_dim=192, dropout_rate=0.2
        )

        msgda_config = SimpleNamespace(
            num_statistical_features=6, weight_net_hidden=[32, 16],
            discriminator_hidden=[256, 128], discriminator_dropout=0.2, mmd_sigma=1.0
        )

        from models.PG_MSAC_Net import PG_MSAC_Net
        model = PG_MSAC_Net(
            num_classes=4, sample_len=1024,
            mpie_config=mpie_config, amscnn_config=amscnn_config, msgda_config=msgda_config
        )

        # 测试前向传播
        test_input = torch.randn(2, 1, 1024)
        output = model(test_input)
        print(f"✅ Model forward pass OK, output shape: {output.shape}")

    except Exception as e:
        print(f"❌ Model creation/forward failed: {e}")


if __name__ == '__main__':
    test_imports()
    test_model_creation()
    print("\n🎉 Testing completed!")
