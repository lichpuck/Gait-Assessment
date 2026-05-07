# F_Description

`F_Description` 将 `outputs/E_Extraction` 中的 walk/turn 定量特征映射为中文、半结构化的描述 JSON。

## 输入

- 单序列输入来自 `outputs/E_Extraction/<subset>/<stem>.json`
- subset 参考分布来自：
  - `outputs/E_Extraction/<subset>/walk_features.csv`
  - `outputs/E_Extraction/<subset>/turn_features.csv`

## 输出

- 输出根目录：`outputs/F_Description/<subset>/`
- 每个序列输出一个 JSON：`<stem>.json`

JSON 结构采用精简交付格式，主要包含：

- `sequence`: 序列标识与源路径
- `quality_summary`: walk/turn 段数量与 warning
- `description.text_summary_zh`: 中文汇总描述
- `description.walk`: walk 语义域描述
- `description.turn`: turn 语义域描述

## 当前语义域

### Walk

- `pace`
- `cadence`
- `step_amplitude`
- `rhythm`
- `asymmetry`
- `posture`
- `stability`
- `coordination`

### Turn

- `extent`
- `speed`
- `efficiency`
- `hesitation`
- `smoothness`
- `compactness`
- `reorientation`
- `postural_control`

每个语义域当前输出：

- `label`: 机器可读标签
- `zh_phrase`: 中文短语
- `policy`: `fixed` 或 `subset_percentile`
- `evidence`: 触发该描述的指标和值

## 脚本入口

单序列：

```bash
/opt/anaconda3/envs/hymotion/bin/python scripts/F_Description/run_one_sequence.py \
  --subset BMCLab \
  --subject-id SUB11 \
  --trial-id SUB11_off_walk_8
```

单个 subset：

```bash
/opt/anaconda3/envs/hymotion/bin/python scripts/F_Description/run_subset_batch.py \
  --subset PD-GaM \
  --max-trials 2
```

全部 subset：

```bash
/opt/anaconda3/envs/hymotion/bin/python scripts/F_Description/run_all_subsets.py \
  --max-trials-per-subset 2
```

## Smoke test

推荐至少验证两个样本：

- 仅含 walk：`BMCLab / SUB11 / SUB11_off_walk_8`
- 同时含 walk + turn：`PD-GaM / 027 / 027-15-004635_wid00_0`

## 已知限制

- walk 映射优先复用现有 `F_Text_Translation` 的离散化思路，但已经改为中文输出。
- turn 映射为第一版规则实现，当前混合使用固定阈值和 subset 分位映射。
- 当前交付优先保证单序列 JSON；未额外生成 txt、csv 汇总表。
