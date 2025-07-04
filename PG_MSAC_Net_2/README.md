# PG-MSAC-Net: 物理引导的多尺度自适应跨域轴承故障诊断网络

## 项目简介
基于多级物理编码的自适应多尺度跨域轴承故障诊断方法

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 快速测试：`python test_simple.py`
3. 开始训练：`python main.py`

## 项目结构
- `models/`: 网络模型模块
- `utils/`: 工具函数
- `configs/`: 配置文件
- `datasets/`: 数据处理
- `checkpoints/`: 模型保存
- `results/`: 结果保存

## 创新点
1. 多级物理信息编码器 (MPIE)
2. 自适应多尺度CNN (AMSCNN)  
3. 多统计量引导域适应器 (MSGDA)
