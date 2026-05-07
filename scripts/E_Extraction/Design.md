# 总体原则

对于 CARE-PD 这种以帕金森患者为主、且序列中同时包含直行和转身的 SMPL 步态数据，建议：

- walk 主要看：
 时序节律异常 + 振幅缩小 + 左右不对称 + 周期间不稳定

- turn 主要看：
 转身变慢 + 步数增多 + 旋转分段化 + 姿势控制不足 + 转身前后衔接异常
这和近年的 PD 步态综述基本一致：PD 常见的量化异常包括速度下降、步长缩短、双支撑时间增加、变异性增加；而转身方面，困难往往表现为转身时间变长、转身步数变多、角速度下降、转身更碎片化。【39860708】【32580330】【36413901】

## Walk段核心参数

### 时序参数

gait_speed
 cadence
 step_time_mean
 stride_time_mean
 stance_time_mean
 swing_time_mean
 double_support_time_mean
 double_support_percent
 stride_time_variability
 step_time_variability

### 运动学参数

step_length_mean
 stride_length_mean
 step_length_asymmetry
 hip_ROM_mean
 knee_ROM_mean
 ankle_ROM_mean
 arm_swing_amplitude
 arm_swing_asymmetry

### 姿势学参数

trunk_flexion_mean
 trunk_flexion_ROM
 pelvis_rotation_ROM
 trunk_ML_sway

## Turn段核心参数

### Turn 时序参数

turn_angle
 turn_duration
 turn_step_count
 mean_step_time_during_turn
 pre_turn_hesitation_time

### Turn 运动学参数

mean_turn_angular_velocity
 peak_turn_angular_velocity
 turn_angular_velocity_variability
 turn_path_compactness / radius
 mean_step_length_during_turn

### Turn 姿势学参数

trunk_yaw_ROM_during_turn
 pelvis_yaw_ROM_during_turn
 head-trunk-pelvis reorientation delay
 en_bloc_index
 trunk_lateral_lean_during_turn
 pelvis_ML_excursion_during_turn

## 当前实现约定

### 输入依赖

- 主输入来自 `outputs/C_Representation/<subset>/<stem>.npz/.json`
- 段级标签来自 `outputs/D_Segmentation/<subset>/<stem>.json`
- `E_Extraction` 对 segment 边界完全信任 `D_Segmentation`
- `E_Extraction` 严格遵守 `D_Segmentation.segments[*].used_for_extraction`
- 对大多数特征，直接使用 `C_Representation` 的 canonical joints、heading、contact 和 trunk/pelvis 派生信号
- 对 `head-trunk-pelvis reorientation delay` 与 `en_bloc_index`，允许回退读取 `outputs/B_Canonicalization/<subset>/<stem>.npz` 中的 `pose_raw`，以恢复 canonical 下的 pelvis/trunk/head 全局 yaw；若该信息缺失，则这些特征写 `NaN`，并在 JSON 中记录原因

### 输出组织

- 输出根目录：`outputs/E_Extraction/<subset>/`
- 每个 subset 写两个汇总 CSV：
  - `walk_features.csv`
  - `turn_features.csv`
- CSV 为 segment 级，一行对应一个 `walk` 或 `turn` segment
- 每一行保留 `subset`、`subject_id`、`trial_id`、`sequence_name`、`segment_id` 等标识字段，因此可以回溯该 segment 所属序列
- 每个序列写一个 JSON：`<stem>.json`
- JSON 只保存 segment 级摘要，不保存逐帧时间序列

### 过滤与缺失值策略

- 仅当 `label in {walk, turn}` 且 `used_for_extraction = true` 时，该段才进入 CSV
- 非 `walk/turn` 段，或 `used_for_extraction = false` 的段，不进入 CSV，但会在序列 JSON 中保留其摘要与跳过原因
- 对于可定义但因事件不足、姿态信号不足、或可选 pose fallback 缺失而无法计算的特征：
  - CSV 写 `NaN`
  - JSON 的 `missing_features` 记录逐特征原因

## 特征定义细则

### 通用统计约定

- `variability` 统一定义为变异系数 `CV = std / mean * 100%`
- `asymmetry` 统一定义为左右均值的归一化绝对差：
 `abs(left - right) / max((abs(left) + abs(right)) / 2, eps) * 100%`
- 各类 `ROM`/`amplitude` 统一采用稳健范围 `P95 - P5`，而不是直接 `max - min`
- 连续角度量在统计范围前，若存在 `-180/180` 包裹问题，应先做 `unwrap`

### Walk 特征定义

- `gait_speed_mps`
 骨盆在水平面上的累计路径长度除以 segment 时长
- `cadence_steps_per_min`
 segment 内 heel strike 总数除以时长，再乘 60
- `step_time_mean_sec`
 相邻、且左右脚交替的 heel strike 间隔均值
- `stride_time_mean_sec`
 同侧相邻 heel strike 间隔均值
- `stance_time_mean_sec`
 heel strike 到下一次 toe off 的时长均值
- `swing_time_mean_sec`
 toe off 到下一次 heel strike 的时长均值
- `double_support_time_mean_sec`
 `gait_phase_global == 3` 的连续片段时长均值
- `double_support_percent`
 segment 内 double support 帧占比
- `step_length`
 在 heel strike 帧，前后脚沿 canonical `+X` 方向的足间距
- `stride_length`
 同一只脚相邻 heel strike 之间，足部沿 canonical `+X` 的位移
- `hip_ROM`
 thigh 向量相对于 trunk-down 向量在矢状面的角度稳健范围
- `knee_ROM`
 hip-knee 与 ankle-knee 两向量在矢状面的屈曲角稳健范围
- `ankle_ROM`
 foot-ankle 相对于 knee-ankle 在矢状面的相对角稳健范围
- `arm_swing_amplitude`
 shoulder-wrist 向量相对于 trunk-down 向量在矢状面的摆动角稳健范围
- `trunk_flexion_mean_deg`
 直接取 `C_Representation.trunk_forward_flexion_deg` 的有效帧均值
- `trunk_flexion_rom_deg`
 `trunk_forward_flexion_deg` 的稳健范围
- `pelvis_rotation_rom_deg`
 `pelvis_yaw_deg` 做 `unwrap` 后的稳健范围
- `trunk_ml_sway_m`
 `neck` 在 `Y` 方向位移去线性趋势后的标准差

### Turn 特征定义

- `turn_angle_deg`
 `heading_unwrapped_deg[end] - heading_unwrapped_deg[start]` 的绝对值
- `turn_duration_sec`
 直接沿用 `D_Segmentation` 的 segment 时长
- `turn_step_count`
 segment 内 heel strike 总数
- `mean_step_time_during_turn_sec`
 turn 段内交替 heel strike 间隔均值
- `pre_turn_hesitation_time_sec`
 turn 起点前最多 1 秒内，`root_speed_xy_mps < 0.10` 的紧邻连续时长
- `mean_turn_angular_velocity_deg_s`
 `abs(yaw_rate_deg_s)` 的有效帧均值
- `peak_turn_angular_velocity_deg_s`
 `abs(yaw_rate_deg_s)` 的有效帧峰值
- `turn_angular_velocity_variability_cv_percent`
 `abs(yaw_rate_deg_s)` 的变异系数
- `turn_path_radius_m`
 骨盆水平路径长度除以转身角度对应的弧度
- `turn_path_compactness_deg_per_m`
 转身角度除以骨盆水平路径长度
- `mean_step_length_during_turn_m`
 turn 段内 heel strike 时刻左右足在水平面的欧氏距离均值
- `trunk_yaw_rom_during_turn_deg`
 `trunk_yaw_deg` 做 `unwrap` 后的稳健范围
- `pelvis_yaw_rom_during_turn_deg`
 `pelvis_yaw_deg` 做 `unwrap` 后的稳健范围
- `head_trunk_pelvis_reorientation_delay_sec`
 将 pelvis/trunk/head 的 yaw 起转时间定义为达到该段总转角 `10%` 的首时刻，主指标取 `head_onset - pelvis_onset`
- `trunk_pelvis_reorientation_delay_sec`
 `trunk_onset - pelvis_onset`
- `head_trunk_reorientation_delay_sec`
 `head_onset - trunk_onset`
- `en_bloc_index`
 `1 - mean(|head-trunk|, |trunk-pelvis|, |head-pelvis|) / total_turn_angle`，结果裁剪到 `[0, 1]`
- `trunk_lateral_lean_during_turn_deg`
 `abs(trunk_lateral_lean_deg)` 的有效帧均值
- `pelvis_ml_excursion_during_turn_m`
 `root_pos_m[:, 1]` 的稳健范围

## 当前脚本入口

- `scripts/E_Extraction/run_one_sequence.py`
- `scripts/E_Extraction/run_subset_batch.py`
- `scripts/E_Extraction/run_all_subsets.py`

## Smoke Test 约定

- 仅做 smoke test，不跑全量数据
- 推荐先验证：
  - 一个只含 `walk` 的序列
  - 一个同时含 `walk + turn` 的序列
