# Pairwise Preference Test Results (2026-04-21)

## 目的
验证 Qwen3-30B 能否通过 pairwise comparison 正确判断 plan 质量（以 Opus eval 为 ground truth），评估 DPO 可行性。

## Step 1: Opus Ground Truth

从 v7-B4 buffer 分层采样 14 个 plans + reference，用 Opus 评估（4 维度 × 1-10 = /40）。

| Plan ID | Tier | Qwen | Opus /40 | D | M | F | G |
|---|---|---|---|---|---|---|---|
| 0 | low | 0.360 | 10 | 3 | 3 | 3 | 1 |
| 1 | low | 0.380 | 13 | 4 | 4 | 4 | 1 |
| 2 | mid_low | 0.410 | 14 | 4 | 4 | 4 | 2 |
| 3 | mid_low | **0.580** | **22** | 6 | 6 | 7 | 3 |
| 4 | mid_low | 0.550 | 14 | 4 | 4 | 4 | 2 |
| 5 | mid | 0.610 | 16 | 4 | 4 | 4 | 4 |
| 6 | mid | 0.770 | 19 | 5 | 5 | 5 | 4 |
| 7 | mid | 0.715 | 18 | 5 | 4 | 5 | 4 |
| 8 | high | 0.850 | 19 | 5 | 5 | 5 | 4 |
| 9 | high | 0.805 | 18 | 5 | 4 | 5 | 4 |
| 10 | high | 0.840 | 15 | 4 | 4 | 4 | 3 |
| 11 | top | **0.940** | 17 | 5 | 4 | 4 | 4 |
| 12 | top | 0.915 | 19 | 5 | 5 | 5 | 4 |
| 13 | top | 0.910 | 17 | 4 | 5 | 5 | 3 |
| ref | ref | — | **31** | 8 | 8 | 7 | 8 |

**Qwen-Opus Spearman ρ = 0.571 (p=0.033)**

关键发现：Qwen 0.580 的 plan (ID=3) 拿到最高 non-ref Opus 分 (22/40)，而 Qwen 0.940 的 plan (ID=11) 只拿了 17/40。高分区域 Qwen 和 Opus 严重脱钩。

## Step 2: Pairwise Preference Test

### 原始 CPR Prompt ("Answer with just A or B")
- **Accuracy: 50% — 100% position bias (永远选 A)**
- 结论：不可用

### 改良 Prompt (分维度分析 + "VERDICT: X/Y")

**大 gap (Opus 差 >= 5):**
- Accuracy: **85% (17/20)**
- Position bias: X=9, Y=11 (已消除)
- Consistency: **90%** (A/B 互换一致)

**小 gap (Opus 差 3-4):**
- Accuracy: **65% (13/20)**
- Position bias: balanced
- Consistency: **90%**
- 饱和区 (both Qwen >= 0.7): **50% (2/4)** — 样本太少，但趋势是随机

## DPO 可行性评估

### 利好
- 模型有内在的质量判别能力（大 gap 85%）
- Position bias 可通过 prompt 设计解决

### 关键问题

**Qwen pairwise preference 跟随 Qwen scalar preference，不跟 Opus。**

例：Plan 3 (Qwen 0.58, Opus 22) vs Plan 9 (Qwen 0.805, Opus 18) → 模型选 Plan 9（Qwen 更高的那个），但 Opus 说 Plan 3 更好。

这意味着：**用 Qwen 同时做 policy 和 preference model = 和 RL 用 Qwen reward 完全一样的 Goodhart 风险。** DPO 不会比 REINFORCE 好，因为 preference signal 来自同一个模型。

### 结论
DPO 要真正优于 RL，需要**外部 preference signal**（Opus、人类、或更强的模型）——不能用 Qwen 自己给自己打分。

## 附：实验结果

| 实验 | Fresh Mean | BufMax |
|---|---|---|
| Exp-A: Frozen, no CR (seed=42) | 0.416 | 0.655 |
| v7-FreshOnly: RL, no CR (seed=0) | 0.406 | 0.625 |
| Exp-B: RL + scores-only CR | 0.436 | 0.915 |
| b4-scores-only: no RL + SO CR | 0.409 | 0.880 |

Exp-A 揭示了之前的 v7-Frozen 数据 (fresh mean=0.047) 是 seed-specific 的极端值。冻结模型在不同 seed 下 fresh mean 可达 0.416，和 RL 训练后几乎相同。**RL 对 fresh plan quality 的提升远比之前认为的小。**
