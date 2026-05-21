# NYC Taxi Demand Prediction (Dual-ConvMLP)

基于双分支卷积-MLP网络的纽约出租车出行需求预测 —— 数据挖掘课程大作业。

## 项目结构

```
├── data/
│   ├── raw/                 # 原始数据
│   └── output/              # 预处理后的 .npy 文件（成员A产出）
├── src/                     # 成员A：数据预处理脚本
├── members/
│   ├── B_HA_XGBoost/        # 成员B：历史均值 + XGBoost 基线
│   ├── C_LSTM/              # 成员C：LSTM 深度学习基线
│   ├── D_ConvEncoder/       # 成员D：Dual-ConvMLP 卷积编码器
│   └── E_Fusion/            # 成员E：融合 MLP + 全量实验
├── reports/                 # 实验报告
└── README.md
```

## 数据集

- **来源**：NYC Yellow Taxi Trip Data (2023年1月) + NOAA 天气数据
- **样本数**：训练456 / 验证96 / 测试120
- **输入**：历史需求热力图 (20×20 网格) + 外部特征（温度、降水量、小时、星期、节假日）
- **输出**：未来1小时上下车需求热力图 (20×20×2)

## 成员分工

| 角色 | 成员 | 任务 |
|------|------|------|
| 数据负责人 | A - 张皓淞 | 数据清洗、网格聚合、滑动窗口、标准化、输出 .npy |
| 传统方法基线 | B | 历史均值 (HA) + XGBoost |
| 深度学习基线 | C | LSTM 序列预测 |
| 核心模型编码器 | D | Dual-ConvMLP 双分支卷积编码器 |
| 模型集成 + 实验 | E | 融合 MLP + 全量评估 + 报告 |
