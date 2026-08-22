# 代码

## 目录约定

```
03_代码/
├── baseline/      # UCI #437 基准复现（Rafiei&Adeli 2018 + 单一模型对照）
├── models/        # 混合学习模型实现（stacking / GA-ANN / PSO-SVR / CNN+tree）
├── utils/         # 数据加载、特征工程、评估指标(MAPE/R²/RMSE)、绘图
└── README.md
```

## 环境

- Python ≥ 3.10
- 核心依赖：`scikit-learn`、`xgboost`、`lightgbm`、`ngboost`、`torch`(可选 CNN)、`shap`、`pandas`、`numpy`、`matplotlib`
- 建议固定版本：`requirements.txt`（实验定稿后生成）

## 可复现性要求（投稿期刊与审稿人关注）

- 所有实验固定 `random_state`，记录种子。
- 数据划分（train/val/test）脚本化，保存索引。
- 每次实验输出到 `04_实验结果/` 带时间戳的目录，含配置 JSON。
- 混合学习的每个组件模型单独可跑，便于消融。

## 运行顺序

1. `utils/build_features.py` — 特征工程
2. `baseline/run_baselines.py` — 单一模型 + 既有基线复现
3. `models/run_hybrid.py` — 混合学习模型
4. `utils/evaluate.py` — 汇总指标 + 生成图表
