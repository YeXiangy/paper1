# 创建测试文件 test_dataset.py
from datasets.JNU_bearing_dataset import JNUBearingDataset

# 测试数据集加载
data_path = "C:\\Users\\33075\\Desktop\\Paper_1\\PG_MSAC_Net_1\\data\JNU"  # 你的数据路径

try:
    dataset = JNUBearingDataset(
        data_path=data_path,
        speed=800,
        shot=5,
        sample_len=1024,
        fault_types=['Normal', 'InnerRace', 'OuterRace']
    )

    print(f"✅ 数据集创建成功: {len(dataset)} 个样本")

    # 测试加载一个样本
    sample, label = dataset[0]
    print(f"✅ 样本加载成功: 形状 {sample.shape}, 标签 {label}")

except Exception as e:
    print(f"❌ 错误: {e}")