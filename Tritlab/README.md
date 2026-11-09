# TritLab —— 个人三值量化实验室

> 目标：从"跟做教程"到"自己的模型、自己的研究"。三值 = {-1, 0, +1}，每个权重 1.58 bit。

## 实验室宪章（三条铁规矩）

1. **一切进 config**：脚本里不允许出现魔法数字。所有超参写在 `experiments/EX-xxx/config.yaml`。
2. **先写假设再跑**：每个实验先填 `hypothesis.md`（假设/对照组/判断标准/风险），跑完不许改假设。
3. **实验编号 + 登记簿**：结果统一追加到 `logs/experiments.jsonl`，每个实验目录里有 `results.md`。

## 目录结构

```
tritlab/
├── tritlab/            # 可复用库（武器库）
│   ├── quant/          # 量化器：三值权重 + int8 激活（STE）
│   ├── layers/         # TritLinear / TritQKV / 无参 RMSNorm
│   ├── train/          # QAT 训练循环（warmup + grad clip + bf16）
│   ├── eval/           # 固定评测：滑动窗口 perplexity
│   └── utils/          # 种子 / 实验日志 / 画图
├── experiments/
│   └── EX-001-gpt2-ternary-qat/    # 第一个正式实验（逐层消融）
└── logs/experiments.jsonl          # 实验登记簿
```

## Quickstart

```bash
pip install -r requirements.txt

# 0. 骨架自检（不需要 transformers，几十秒）
python experiments/smoke_test.py

# 1. 快速预实验：本地 CPU 验证 L1-B（per-channel vs per-tensor）
python experiments/smoke_test.py --sweep

# 2. EX-001 正式实验（需要 GPU，建议 Kaggle/Colab）
cd experiments/EX-001-gpt2-ternary-qat
python run_experiment.py            # 按 config.yaml 跑全部组 × 3 种子
python run_experiment.py --group b_all --seed 0 --max-steps 20   # 管线烟雾测试
```

## 依赖

- 核心库：torch + pyyaml（`experiments/smoke_test.py` 只需这两个）
- EX-001：`transformers`, `datasets`, `tqdm`（仅训练时需要，Kaggle 预装）

## 两条路线

- **模型线（主线）**：造自己的三值模型——`models/trit_gpt.py`（TritGPT，原生三值 LM，
  自有架构：ReLU² / sub-norm / GQA / RoPE 全是设计变量），从零预训练，
  产出自己的 checkpoint 系列。
- **实验线（支撑）**：改造与消融（EX-001）回答"哪些设计重要"，结论反哺模型线。

## 实验索引

| 编号 | 课题 | 路线 | 状态 |
|---|---|---|---|
| EX-001 | GPT-2-small 三值化逐层消融（哪些层不能三值化） | 实验线 | 设计中 |
| EX-002 | TritGPT v0 从零预训练（38M，配方验证：三值 vs FP 差距 <15%） | 模型线 | 待跑 |
