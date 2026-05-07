# F_Description 设计说明

## 目标

`F_Description` 的目标是把 `E_Extraction` 输出的 walk/turn 定量特征，映射为中文、半结构化、可供下游直接消费的描述 JSON。

本阶段不再沿用 `care_pd_pipeline/F_Text_Translation` 的研究型输出格式，而是保留更精简的正式交付 schema。

## 输入依赖

- 主输入：`outputs/E_Extraction/<subset>/<stem>.json`
- 参考分布：
  - `outputs/E_Extraction/<subset>/walk_features.csv`
  - `outputs/E_Extraction/<subset>/turn_features.csv`

单序列 JSON 提供：

- sequence metadata
- walk/turn segment 级 `features`
- `missing_features`
- `quality_summary`
- 上游输入路径

subset CSV 用于构建 subset 内的相对分位参考，从而避免所有语义域都依赖硬编码阈值。

## 输出 schema

每个序列输出一个 JSON，核心字段如下：

```json
{
  "module": "F_Description",
  "version": "1.0.0",
  "language": "zh-CN",
  "sequence": {},
  "quality_summary": {},
  "description": {
    "text_summary_zh": "...",
    "walk": {
      "available": true,
      "segment_count": 1,
      "descriptors": {},
      "omitted_descriptors": []
    },
    "turn": {
      "available": false,
      "segment_count": 0,
      "descriptors": {},
      "omitted_descriptors": []
    }
  },
  "outputs": {}
}
```

## 规则设计

### Walk

walk 使用两类策略：

- 固定阈值：`pace`、`cadence`、`step_amplitude`、`posture`
- subset 分位映射：`rhythm`、`asymmetry`、`stability`、`coordination`

当前组合逻辑：

- `rhythm` 由 `step_time_variability_cv_percent` 和 `stride_time_variability_cv_percent` 组合
- `asymmetry` 由 `step_length_asymmetry_percent` 和 `stride_length_asymmetry_percent` 组合
- `stability` 由 `double_support_percent` 和 `trunk_ml_sway_m` 组合
- `coordination` 由 `arm_swing_amplitude_mean_deg`、`arm_swing_asymmetry_percent`、`pelvis_rotation_rom_deg` 组合

### Turn

turn 的第一版描述同时覆盖：

- `extent`
- `speed`
- `efficiency`
- `hesitation`
- `smoothness`
- `compactness`
- `reorientation`
- `postural_control`

其中：

- 固定阈值优先用于 `extent`、`speed`、`hesitation`、`reorientation`
- subset 分位映射优先用于 `efficiency`、`smoothness`、`compactness`、`postural_control`

## 文本生成

最终汇总描述采用短句模板：

- `步行表现为...。`
- `转身表现为...。`

其目标是：

- 中文
- 低熵
- 保持语义域可追溯
- 让 JSON 既可被人读，也可被程序消费

## Smoke test 约定

本阶段只做 smoke test，不跑全量：

- walk-only: `BMCLab / SUB11 / SUB11_off_walk_8`
- walk + turn: `PD-GaM / 027 / 027-15-004635_wid00_0`

## 后续可扩展项

- 将 turn 规则进一步对齐临床文献阈值
- 为中文输出增加更严格的模板与词表控制
- 增加 subset 级 summary table 导出
- 增加自动化测试，覆盖无 turn、缺失特征、warnings 传播等场景
