# EX-002：原生三值 LM v0（TritGPT）

## 假设

按 BitNet 训练配方（无参 RMSNorm + ReLU² + sub-norm + absmean 三值权重 + int8 激活），
从零训练的 ~38M 三值 LM 在 TinyStories 验证集上的 ppl，与同规模 FP 基线
（同架构、同数据、只关量化）的差距 < 15%。

## 对照组

| 组 | 配置 | 目的 |
|---|---|---|
| n_fp | weight_mode=none, quantize_input=false | FP 基线（同架构同数据） |
| n_all | weight_mode=absmean, quantize_input=true | 全三值（attention + MLP） |

注意：FP 基线也保留 ReLU²/sub-norm/RoPE——只隔离"量化"这一个变量。

## 判断标准

- gap = (ppl(n_all) - ppl(n_fp)) / ppl(n_fp)
- gap < 15% → 配方成立，进入 EX-003（自训 tokenizer + 50M + scaling）
- gap > 30% → 配方需要调参，每次只改一个变量（lr / warmup / Δ 系数），
  全部记入 config 和登记簿，不许同时改多个

## 风险

1. 原生三值训练不稳定（loss spike）——降 lr / 加长 warmup，按 BitNet 官方建议执行
2. 38M 远小于文献的 3B，三值化损失可能比文献报道大——这本身是 L3-A scaling 的数据点
3. TinyStories 分布窄——v0 只验证"配方能训起来"，不追求通用能力
