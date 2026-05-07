# A_Audition: 坐标轴语义初步标准化

## Summary

`scripts/A_Audition` 是 raw SMPL 数据的轻量审计/转换阶段。它遍历 `raw_data/*.pkl`，跳过 `<3s` 序列，用 `SMPL_NEUTRAL.pkl` 通过 SMPLX/SMPL FK 得到 24 个关节，然后只做坐标轴语义标准化：

- `+X = forward`
- `+Y = subject-left / lateral`
- `+Z = vertical / up`
- 右手系

本阶段不做坡度修正、地面拟合、floor leveling、轨迹归零、heading canonicalization 或 SMPL pose 改写。

## Processing Logic

单条序列处理流程：

1. 读取并校验 `pose=(T,72)`、`trans=(T,3)`、`beta=(1,10)`、`fps`；若时长 `<3s`，只在 summary 中记录 `duration_lt_3s`。
2. 用 `SMPL_NEUTRAL.pkl` 前向展开得到 `joints_raw=(T,24,3)`。
3. 对 `trans_raw` 三个原始轴计算 `robust_range=P95-P5`。
4. 用 `median(head - pelvis)` 判断 vertical 原始轴与正负号；必要时 fallback 到 `median(neck - pelvis)`。
5. vertical 优先；若最大 robust range 轴与 vertical 冲突，不丢弃序列，而是在剩余两个原始轴中选择 robust range 更大的轴作为 forward，另一个作为 lateral。
6. lateral 符号由 `median(left_hip - right_hip)` 决定，使 `+Y` 指向 subject-left。
7. forward 符号从候选 forward 轴的平滑 `trans` 投影中选最长近似单调段，使该单程段朝 `+X` 增长。
8. 组合 signed permutation 得到 `R_total`；若 `det(R_total)<0`，翻转 forward 符号以保持右手系，并记录 warning。
9. 输出 `joints_3d=R_total @ joints_raw` 与 `trans_canonical=R_total @ trans_raw`；只旋转，保留原点。
10. 在新语义坐标系下，从左右脚踝和左右脚掌四个点中逐帧选择 `Z` 最低的点作为支撑点。

低置信但无硬冲突的序列仍输出，并在 JSON 与 summary 中记录 warning。

## Output Contract

每个有效序列输出到 `outputs/A_Audition/<subset>/`，文件名前缀为 `<subset>__<subject_id>__<trial_id>`。

`.npz` 固定字段：

- `joints_3d`: `(T,24,3)`，新语义坐标系下的 24 关节坐标
- `trans_canonical`: `(T,3)`，新语义坐标系下的根节点轨迹
- `pose_raw`: `(T,72)`，原始 SMPL pose
- `trans_raw`: `(T,3)`，原始 trans
- `beta`: `(1,10)`
- `fps`: scalar
- `R_total`: `(3,3)`，raw-to-semantic rotation
- `support_points`: `(T,3)`，每帧左右脚踝、左右脚掌四个点中 canonical `Z` 最低点的三维坐标

`.json` 至少记录：

- `subset`、`subject_id`、`trial_id`、`source_path`
- `input_shapes`、`output_shapes`
- `raw_axes`：原始轴顺序、raw range、robust range、推断语义
- `canonical_axes`
- `axis_mapping`：forward/lateral/vertical 对应的原始轴、符号、一致性、`det(R_total)`
- `forward_segment`：最长近似单调段的起止帧、位移、方向
- `R_total`
- `support_points` 指标，包括四个候选关节的选中次数/比例与 finite ratio
- `quality_flags` / `warnings`
- 原始 metadata：`medication`、`UPDRS`、`other` 等

`.png` 固定为三个诊断子图：

- `trans_canonical` 在标准 `XY` 平面上的轨迹，标记 start/end
- `support_points` 与根节点 `trans_canonical` 在标准 `XZ` 平面上的轨迹
- 人物骨架朝向与 `+X` 前进轴夹角绝对值随时间变化，范围为 `0-180°`

批处理维护 `outputs/A_Audition/audition_summary.csv`，记录：

- `subset`、`subject_id`、`trial_id`
- `num_frames`、`fps`、`duration_sec`
- `status` (`success` / `skipped` / `failed`)
- `skip_reason` / `error_message`
- warning 数量、轴判定摘要、支撑点 finite ratio
- `output_npz`、`output_json`、`output_png`

## Public Interfaces

- `run_one_sequence.py`：按 `subset + subject-id + trial-id` 处理单条序列。
- `run_subset_batch.py`：处理单个 `.pkl` 子集，支持 subject/trial 过滤与 `--max-trials`。
- `run_all_subsets.py`：扫描 `raw_data/*.pkl` 全量运行。
- 推荐命令形式：`conda run -n hymotion python scripts/A_Audition/...`。

## Test Plan

1. 只读确认 9 个 raw pkl 均按 `pose=(T,72)`、`trans=(T,3)` 读取。
2. 单序列 smoke test：生成 NPZ/JSON/PNG，检查新字段、shape、finite 值、`det(R_total)>0`。
3. 坐标语义测试：转换后 head 高于 pelvis，left hip 相对 right hip 主方向为 `+Y`，主要单程段沿 `+X` 增长。
4. 跳过/失败测试：`<3s` 只写 summary；forward/vertical 同轴样本应输出；`det(R_total)>0`。
5. 支撑点测试：`support_points.shape==(T,3)`，NPZ 中不存在 `support_mask`，每帧点等于四个候选关节里 canonical `Z` 最低者。
6. PNG 测试：确认三个子图分别为 XY 根轨迹、XZ 根节点+支撑点轨迹、骨架朝向与 `+X` 夹角曲线。
