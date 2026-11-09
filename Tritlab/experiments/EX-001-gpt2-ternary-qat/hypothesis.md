# EX-001：GPT-2-small 三值化逐层消融

## 假设

GPT-2-small 全三值化后的 perplexity 损失主要来自注意力投影层；
若只把 o_proj（和/或 q,k）保留为 FP，可挽回大部分精度损失。

## 对照组

| 组 | 三值化范围 | 目的 |
|---|---|---|
| a_fp | 无 | baseline perplexity |
| b_all | q,k,v,o,fc,proj | 全三值化的总损失 |
| c_mlp_only | fc,proj | attention 全 FP 时的损失 |
| d_attn_only | q,k,v,o | MLP 全 FP 时的损失 |
| e_all_keep_o | 全部，但 o 保持 FP | 检验 o_proj 敏感性 |
| f_all_keep_qk | 全部，但 q,k 保持 FP | 检验 q,k（内积对距离敏感） |

（GPT-2 的 MLP 没有 gate/up，对应关系：fc -> "gate/up"，proj -> "down"。）

## 判断标准

设 gap = ppl(b_all) - ppl(a_fp)。
- 若 ppl(e_all_keep_o) - ppl(b_all) 的降幅 > 30% × gap → **"o_proj 是主要敏感层"成立**；
- 若 f 组降幅 > e 组 → 假设被推翻/修正，q,k 更敏感，按 f 的结论记录。

## 风险

1. GPT-2 无 sub-norm、激活是 GELU，全三值化 ppl 可能暴涨（>2 倍），
   此时 a/b 两组仍有效，但 e/f 的相对降幅解释要谨慎；
2. wikitext-2 验证集较小（~240k tokens），3 种子 std 可能掩盖小差异；
3. 三值微调 lr=2e-5 是保守值，若 loss 不降，先扫 {5e-5, 1e-4}（记入 config，不偷偷改）。
