# D_Segmentation 设计说明（动作语义判定规则）

## 1. 模块目标

D_Segmentation 基于 C_Representation 的时序特征与接触信号，对整段序列逐帧给出动作语义标签，并生成段级摘要。

- 主标签（互斥）：stand_to_sit, sit, sit_to_stand, turn, walk, adjust
- 辅助标签（可叠加）：hesitation

该模块是规则驱动（rule-based）分割，不依赖学习模型。

---

## 2. 输入契约

输入来自 outputs/C_Representation/`<subset>`/*.npz 与同名 .json。

只支持 strict C_Representation v2，不兼容 legacy C 别名字段。必须包含以下 NPZ 字段：

- joints_can: (T, 24, 3)
- root_pos_m: (T, 3)
- root_speed_xy_mps: (T,)
- heading_deg / heading_unwrapped_deg: (T,)
- yaw_rate_deg_s: (T,)
- pelvis_height_m: (T,)
- valid_frame_mask: (T,)
- representation_quality_score: (T,)
- left_foot_pos_m / right_foot_pos_m: (T, 3)
- left_foot_speed_mps / right_foot_speed_mps: (T,)
- left_foot_contact / right_foot_contact: (T,)
- left_heel_strike / right_heel_strike: (T,)
- left_toe_off / right_toe_off: (T,)
- fps: 标量

序列至少 2 帧，fps 必须为有限正数。

---

## 3. 信号构造与预处理

## 3.1 平滑参数（默认）

- posture_sigma_sec = 0.20
- speed_sigma_sec = 0.12
- heading_sigma_sec = 0.18

通过 seconds_to_frames 按 fps 映射为帧域高斯核 sigma。

## 3.2 关键派生信号

1. pelvis_speed_mps（骨盆平面速度）
   - 优先使用 C v2 的 root_speed_xy_mps
   - 在 D 内按 speed_sigma_sec 做轻量平滑

2. foot_speed_mean_mps（双足速度均值）
   - 左右足速度分别平滑后取均值

3. pelvis_height_norm（姿态高度归一化）
   - 以 pelvis_height_m 的 2% 与 98% 分位作为低高参考
   - 若高低跨度 < 0.16 m，则整段置为 0.5（避免噪声放大）
   - 否则线性归一化并截断到 [0, 1]

4. heading_smooth_deg / turn_speed_deg_s / turn_angle_from_start_deg
   - heading_unwrapped_deg 平滑后 wrap 回 [-180, 180)
   - turn_speed_deg_s 使用 C v2 yaw_rate_deg_s 的绝对值并轻量平滑
   - turn_angle_from_start_deg 为相对整条序列首帧朝向差的绝对值（范围 [0, 180]）

5. distance_from_start_m
   - 骨盆 XY 到首帧位置的欧氏距离

---

## 4. 主标签判定规则

## 4.1 总体顺序与互斥策略

先检测并生成各类事件/掩码，再按优先级写入最终主标签。

写入顺序：

1. walk（默认底色）
2. turn
3. sit
4. stand_to_sit
5. sit_to_stand
6. adjust（后处理改写）

后写入会覆盖先写入，因此 STS 相关标签优先级高于 sit，sit 高于 turn，turn 高于 walk。

## 4.2 stand_to_sit 与 sit_to_stand（STS）

基于 pelvis_height_norm 的阈值穿越和局部极值扩展：

- stand_to_sit 核心触发：高度先向下穿越 0.8，再向下穿越 0.2
- sit_to_stand 核心触发：高度先向上穿越 0.2，再向上穿越 0.8

区间边界：

- 起点从核心开始点向前回溯到局部极值（stand_to_sit 回溯到局部高点；sit_to_stand 回溯到局部低点）
- 终点从核心结束点向后扩展到局部极值（stand_to_sit 到局部低点；sit_to_stand 到局部高点）

得到两类 STS 区间后，转为逐帧 mask。

## 4.3 sit（坐姿静段填充）

sit 不直接由阈值触发，而由 STS 拓扑关系填充：

- 若序列起始前先出现 sit_to_stand，则 [0, sit_to_stand_start-1] 记为 sit
- 对任意相邻 stand_to_sit -> sit_to_stand，二者之间空隙记为 sit
- 若最后事件为 stand_to_sit，则其结束后到序列末尾记为 sit

随后执行去重：sit = sit 且非（stand_to_sit 或 sit_to_stand）。

## 4.4 turn（转身）

仅在 locomotion_allowed 区域检测转身：

locomotion_allowed = 非（stand_to_sit 或 sit_to_stand 或 sit）

在每个允许区间内，搜索两类模式：

1. outward_turn
   - 核心开始：turn_angle_from_start_deg 向上穿越 15°
   - 核心结束：随后向上穿越 165°

2. return_turn
   - 核心开始：turn_angle_from_start_deg 向下穿越 165°
   - 核心结束：随后向下穿越 15°
   - 若未找到结束点，可截断到当前允许区间末尾

边界修正：

- 起点回填：从核心开始向前找首个 walk_speed > 0.05 m/s 的帧，作为事件起点
- 终点前推：从核心结束向后找满足以下任一条件的首帧
  - turn_speed_deg_s <= 10 deg/s
  - turn_angle_from_start_deg 位于边界角（0~5° 或 175~180°）

搜索采用“最早可成立候选优先”。每识别一个事件后，游标推进到该事件核心结束之后继续。

## 4.5 walk（残差类）

walk 不显式检测，作为默认标签：

- 初始全帧赋值 walk
- 再被 turn/sit/STS/adjust 规则覆盖
- 最终未被覆盖的帧保持 walk

## 4.6 adjust（后处理重标）

在主标签段序列上做模式重写：

- 对内部 walk 段（非首段、非尾段）
- 若其前一段是 turn，后一段是 stand_to_sit
- 则该整段 walk 改写为 adjust

该规则用于标记“转身后、落座前”的短时姿态调整/碎步。

---

## 5. 辅助标签 hesitation 判定

hesitation 只允许出现在 walk 或 turn 内：

- 在 walk 帧上，若 pelvis_speed_mps < 0.10 m/s，则候选 hesitation
- 在 turn 帧上，若 turn_speed_deg_s < 10 deg/s，则候选 hesitation

候选掩码经过最小时长过滤：

- hesitation_min_duration_sec = 0.20
- 低于该时长的连续真值片段被移除

最终 hesitation 与主标签并存，不改变主标签类别。

---

## 6. 事件与步态接触代理（质量侧）

模块使用 C v2 的 left/right contact 与 heel-strike/toe-off mask：

- heel strike：接触由 0->1 的上升沿
- toe off：接触由 1->0 的下降沿

并计算 contact_stability_score（0~1）用于质量评估，综合：

- 左右 heel strike 交替性
- 双侧占空比与目标占空比（0.60）的接近程度
- 左右事件数量平衡度

当事件过少、单侧缺失或稳定度过低时写入 warning，但不直接改写主标签。

---

## 7. 质量约束与成功判定

分割结果需满足以下关键约束：

- 关键信号数组有限（无 NaN/Inf）
- 接触掩码尺寸正确
- gait event 索引合法且非降序
- 主标签逐帧互斥且完备（每帧恰一类）
- hesitation 不得出现在非 walk/turn 区域
- 段级记录必须无缝覆盖全帧

满足核心约束则 segmentation_success = true。

---

## 8. 输出语义

每条序列输出：

- `<stem>`.json：顶层包含 module、version、sequence、schema、quality_summary、segments、outputs
- `<stem>`.csv：段级表，包含 segment_id、label、start_frame、end_frame、used_for_extraction、quality 等字段

批处理额外保留 subset 级 segmentation_summary.csv。

段级 quality 规则：

- valid_frame_ratio < 0.8 或 representation_quality_score_mean < 0.6 时为 bad
- 质量通过但存在 warning 或 hesitation overlap 时为 warning
- 否则为 good
- bad 段或 hesitation_overlap_ratio >= 0.5 的段 used_for_extraction = false

其中 source_rule 可追踪每帧/每段主要来源：

- stand_to_sit_rule
- sit_to_stand_rule
- sit_fill
- turn_rule
- walk_residual
- adjust_relabel

---

## 9. 可调参数与语义敏感点

最影响动作语义边界的参数：

- STS 阈值：0.8 / 0.2
- turn 角度阈值：15° / 165°
- turn 结束条件：turn_speed_end_deg_s = 10
- adjust 触发模式：turn -> walk -> stand_to_sit
- hesitation 速度阈值与最小时长：0.10 m/s、10 deg/s、0.20 s

若要迁移到新数据域，建议优先调上述参数，并用 diagnostic 图核对边界是否符合人工语义。
