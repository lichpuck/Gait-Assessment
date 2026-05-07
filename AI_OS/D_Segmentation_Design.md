# D_Segmentation Design

## 1. 模块定位

`D_Segmentation` 是 CARE-PD 流水线中的动作语义分割层。它读取
`outputs/C_Representation/<subset>/<stem>.npz` 与同名 `.json`，基于
C 阶段已经标准化的运动学、朝向、接触、质量信号，对整条序列做逐帧动作语义判定，并导出段级结果。

该模块是 rule-based segmentation，不使用学习模型。它的目标不是重新估计人体姿态或步态事件，而是在 C 阶段给出的时序表示之上，给出可解释、可追溯、互斥完备的动作标签。

## 2. 标签体系

主标签为 6 类，逐帧互斥：

- `stand_to_sit`
- `sit`
- `sit_to_stand`
- `turn`
- `walk`
- `adjust`

辅助标签为 1 类：

- `hesitation`

`hesitation` 不能单独存在，只能叠加在 `walk` 或 `turn` 帧上。它不改变主标签，只作为辅助语义和质量提示写入段级结果。

## 3. 输入契约

D 只支持 strict `C_Representation` v2 输出，不兼容 legacy C 字段别名。

每条序列必须有：

- `fps`
- `joints_can`
- `root_pos_m`
- `root_speed_xy_mps`
- `heading_deg`
- `heading_unwrapped_deg`
- `yaw_rate_deg_s`
- `pelvis_height_m`
- `valid_frame_mask`
- `representation_quality_score`
- `left_foot_pos_m`, `right_foot_pos_m`
- `left_foot_speed_mps`, `right_foot_speed_mps`
- `left_foot_contact`, `right_foot_contact`
- `left_heel_strike`, `right_heel_strike`
- `left_toe_off`, `right_toe_off`

序列至少 2 帧，`fps` 必须为有限正数。核心数组必须与帧数 `T` 对齐。

## 4. 派生信号

### 4.1 速度信号

`pelvis_speed_mps` 直接来自 C v2 的 `root_speed_xy_mps`，并在 D 中按 `speed_sigma_sec` 做轻量平滑。它用于：

- `turn` 起点回填
- `walk` 中的 `hesitation` 判定

双足速度 `foot_speed_mean_mps` 由 `left_foot_speed_mps` 与 `right_foot_speed_mps` 平滑后取均值，目前主要作为诊断/扩展信号保留。

### 4.2 骨盆高度归一化

`pelvis_height_norm` 从 `pelvis_height_m` 派生：

- 低参考值：`pelvis_height_m` 的 2% 分位数
- 高参考值：`pelvis_height_m` 的 98% 分位数
- 若高低跨度 `< 0.16 m`，整段高度归一化为 `0.5`
- 否则线性映射到 `[0, 1]`

该信号是 `stand_to_sit`、`sit_to_stand`、`sit` 判定的核心。

### 4.3 朝向与转身信号

D 使用 C v2 的 `heading_unwrapped_deg` 和 `yaw_rate_deg_s`：

- `heading_smooth_deg`：对 `heading_unwrapped_deg` 平滑后 wrap 回 `[-180, 180)`
- `turn_speed_deg_s`：`abs(yaw_rate_deg_s)` 平滑后得到
- `turn_angle_from_start_deg`：当前平滑朝向相对整条序列第 0 帧的绝对角度差，范围 `[0, 180]`

注意：转身角度基准是整条序列第 0 帧，不是每个允许区间的起点。

## 5. 主标签判定总流程

动作判定遵循固定优先级：

1. 先检测整条序列中的 `stand_to_sit` 和 `sit_to_stand`
2. 根据 STS 事件拓扑填充 `sit`
3. 仅在非 STS / 非 `sit` 区间检测 `turn`
4. 剩余所有未覆盖帧填充为 `walk`
5. 最后将 `turn -> walk -> stand_to_sit` 模式中的中间 `walk` 改写为 `adjust`

最终主标签逐帧互斥且覆盖全序列。

## 6. STS 与 sit 规则

### 6.1 stand_to_sit

`stand_to_sit` 基于 `pelvis_height_norm` 的下降穿越判定。

核心区间：

- 核心起点：骨盆归一化高度向下穿越 `0.8`
- 核心终点：随后继续向下穿越 `0.2`

边界扩展：

- 实际起点：从核心起点向前回溯到局部最大骨盆高度
- 实际终点：从核心终点向后扩展到局部最小骨盆高度

该规则将“开始下坐前的最高站立点”到“落座后的最低点”视为完整 `stand_to_sit` 段。

### 6.2 sit_to_stand

`sit_to_stand` 基于 `pelvis_height_norm` 的上升穿越判定。

核心区间：

- 核心起点：骨盆归一化高度向上穿越 `0.2`
- 核心终点：随后继续向上穿越 `0.8`

边界扩展：

- 实际起点：从核心起点向前回溯到局部最小骨盆高度
- 实际终点：从核心终点向后扩展到局部最大骨盆高度

该规则将“离座前的最低坐姿点”到“站起后的最高稳定点”视为完整 `sit_to_stand` 段。

### 6.3 sit

`sit` 不直接由速度或姿态阈值单独触发，而由 STS 事件拓扑填充。

填充规则：

- 如果序列开头先出现 `sit_to_stand`，则从第 0 帧到该 `sit_to_stand` 起点前一帧标为 `sit`
- 任意相邻 `stand_to_sit -> sit_to_stand` 之间的空隙标为 `sit`
- 如果序列最后一个 STS 事件是 `stand_to_sit`，则该事件结束后到序列末尾标为 `sit`

`sit` 会排除已经属于 `stand_to_sit` 或 `sit_to_stand` 的帧。

## 7. turn 规则

`turn` 仅在允许运动区间内检测：

```text
locomotion_allowed = not (stand_to_sit or sit_to_stand or sit)
```

在每个允许区间内，统一检索向外转身和回转转身两类模式，并采用“最早有效候选优先”的策略。

### 7.1 outward_turn

核心区间：

- 核心起点：`turn_angle_from_start_deg` 向上穿越 `15°`
- 核心终点：随后向上穿越 `165°`

只有当起点和终点都在当前允许区间内时，该候选才有效。

### 7.2 return_turn

核心区间：

- 核心起点：`turn_angle_from_start_deg` 向下穿越 `165°`
- 核心终点：随后向下穿越 `15°`

如果找不到有效核心终点，则以当前允许区间末尾作为该回转事件的截断终点。

### 7.3 turn 边界校准

核心区间确定后，统一做实际边界校准。

实际起点：

- 从核心起点向前回溯
- 选择首个 `pelvis_speed_mps > 0.05 m/s` 的帧
- 如果没有找到，则保留核心起点

实际终点：

- 从核心终点向后检索
- 选择首个满足以下任一条件的帧：
  - `turn_speed_deg_s <= 10 deg/s`
  - `turn_angle_from_start_deg` 位于边界角范围 `0°~5°` 或 `175°~180°`
- 如果没有找到，则保留核心终点

每识别一个 turn 事件后，检测游标跳到该事件核心终点之后，继续搜索下一个事件。

## 8. walk 与 adjust 规则

### 8.1 walk

`walk` 是残差类，不单独进行显式检测。

在 `stand_to_sit`、`sit`、`sit_to_stand`、`turn` 判定完成后，所有剩余帧都标为 `walk`。

### 8.2 adjust

`adjust` 是后处理重标规则，用于描述“转身后、坐下前”的短时姿态调整或碎步。

在最终主标签段序列中，如果某个内部 `walk` 段满足：

```text
previous segment = turn
current segment  = walk
next segment     = stand_to_sit
```

则该 `walk` 段整体重命名为 `adjust`。

`adjust` 仍然是主标签之一，与其他主标签互斥。

## 9. hesitation 辅助标签

`hesitation` 只允许叠加在 `walk` 或 `turn` 上。

候选规则：

- 在 `walk` 帧中，若 `pelvis_speed_mps < 0.10 m/s`，则为 hesitation 候选
- 在 `turn` 帧中，若 `turn_speed_deg_s < 10 deg/s`，则为 hesitation 候选

时长过滤：

- 最小时长阈值为 `0.20 s`
- 短于该时长的连续 hesitation 候选片段会被移除

`hesitation` 不改变主标签，只在段级记录中体现为 overlap。

## 10. 输出

每条序列输出到：

```text
outputs/D_Segmentation/<subset>/<stem>.json
outputs/D_Segmentation/<subset>/<stem>.csv
```

批处理额外输出：

```text
outputs/D_Segmentation/<subset>/segmentation_summary.csv
```

### 10.1 JSON 结构

顶层包含：

- `module`
- `version`
- `sequence`
- `schema`
- `quality_summary`
- `segments`
- `outputs`

每个 segment 至少包含：

- `segment_id`
- `label`
- `start_frame`
- `end_frame`
- `used_for_extraction`
- `quality`

### 10.2 段级质量规则

每个 segment 根据 C 阶段质量与 hesitation 重叠情况给出 `quality` 和 `used_for_extraction`。

`quality.status = bad` 当：

- 段内 `valid_frame_ratio < 0.8`
- 或段内平均 `representation_quality_score < 0.6`

`quality.status = warning` 当：

- 没有达到 `bad` 条件
- 但存在 hesitation overlap 或 gait/contact warning

`quality.status = good` 当：

- 没有 bad 条件
- 没有 warning 条件

`used_for_extraction = false` 当：

- `quality.status = bad`
- 或 `hesitation_overlap_ratio >= 0.5`

其余情况 `used_for_extraction = true`。

## 11. 成功判定与质量约束

一条序列的 D 分割需要满足：

- 输入契约合法
- 信号数组有限
- contact mask 形状正确
- gait event 索引合法且非降序
- 主标签索引合法且逐帧有且只有一个主标签
- `hesitation` 不出现在非 `walk` / 非 `turn` 区域
- segments 无缝覆盖 `[0, T-1]`

满足以上核心约束时，`segmentation_success = true`。

## 12. 关键参数

最影响语义边界的参数如下：

- STS 高度阈值：`0.8 / 0.2`
- pelvis height 归一化分位数：`2% / 98%`
- 最小高度跨度：`0.16 m`
- turn 核心角度阈值：`15° / 165°`
- turn 起点回填速度阈值：`0.05 m/s`
- turn 结束角速度阈值：`10 deg/s`
- walk hesitation 速度阈值：`0.10 m/s`
- turn hesitation 角速度阈值：`10 deg/s`
- hesitation 最小时长：`0.20 s`
- segment valid frame ratio 阈值：`0.8`
- segment representation quality 阈值：`0.6`
- segment hesitation 排除阈值：`0.5`
