# C_Representation输出

**C_Representation** 输出两个核心文件：一个用于存储高维数组的 `.npz` 和一个用于存储配置与元数据的 `.json`。

## 1. C_representation.npz 输出清单

该文件存储所有 **逐帧（Frame-wise）** 的数据，数组的第一维通常为时间步 $T$。

### 1.1 基础时间与有效帧信息

|**字段名**|**形状 (Shape)**|**单位**|**现实意义**|
|---|---|---|---|
|**fps**|scalar|Hz|原始序列帧率，是所有时间参数的基础|
|**time_s**|$[T]$|s|每一帧对应的时间戳|
|**frame_index**|$[T]$|frame|原始帧编号，方便回溯原始视频序列|
|**valid_frame_mask**|$[T]$|bool|标记哪些帧有效，避免异常帧参与分割和特征提取|

### 1.2 标准化后的三维运动信息

|**字段名**|**形状**|**单位**|**现实意义**|
|---|---|---|---|
|**joints_can**|$[T, J, 3]$|m|标准坐标系下的三维关节序列（后续分析的基础）|
|**trans_can**|$[T, 3]$|m|标准坐标系下的 SMPL 全局平移|
|**root_pos_m**|$[T, 3]$|m|root / pelvis 的全局位置（用于速度、轨迹、路径计算）|

> **注**：标准坐标系固定为 $+X$=forward, $+Y$=left, $+Z$=up。

### 1.3 root / pelvis 运动信号

|**字段名**|**形状**|**单位**|**现实意义**|
|---|---|---|---|
|**root_velocity_mps**|$[T, 3]$|m/s|root 三维速度，用于判断移动状态|
|**root_speed_xy_mps**|$[T]$|m/s|水平移动速度，是 walk / pause / adjust 判断的核心依据|
|**root_acceleration_mps2**|$[T, 3]$|m/s²|可选，用于识别突然加减速、异常抖动|
|**pelvis_height_m**|$[T]$|m|骨盆高度，是 sit / STS 动作判断的核心依据|
|**pelvis_vertical_velocity_mps**|$[T]$|m/s|骨盆垂直速度，用于判断起立/坐下过程|

### 1.4 heading / yaw / 转身基础信号

|**字段名**|**形状**|**单位**|**现实意义**|
|---|---|---|---|
|**heading_deg**|$[T]$|deg|人体朝向角，范围通常为 -180° 到 180°|
|**heading_unwrapped_deg**|$[T]$|deg|连续展开后的朝向角，避免 179° 到 -179° 的跳变|
|**yaw_rate_deg_s**|$[T]$|deg/s|朝向角速度，是 **turn 检测** 的核心信号|
|**yaw_acceleration_deg_s2**|$[T]$|deg/s²|可选，用于识别转身起止和角速度突变|

### 1.5 左右脚运动信号

|**字段名**|**形状**|**单位**|**现实意义**|
|---|---|---|---|
|**left_foot_pos_m**|$[T, 3]$|m|左脚位置，用于步长、轨迹、清除高度计算|
|**right_foot_pos_m**|$[T, 3]$|m|右脚位置|
|**left_foot_velocity_mps**|$[T, 3]$|m/s|左脚速度，用于足接触和摆动相判断|
|**right_foot_velocity_mps**|$[T, 3]$|m/s|右脚速度|
|**left_foot_speed_mps**|$[T]$|m/s|左脚速度模长，用于判断足部是否稳定接触地面|
|**right_foot_speed_mps**|$[T]$|m/s|右脚速度模长|
|**left_foot_height_m**|$[T]$|m|左脚离地高度，用于足清除高度、拖步分析|
|**right_foot_height_m**|$[T]$|m|右脚离地高度|

### 1.6 足接触信号

|**字段名**|**形状**|**单位**|**现实意义**|
|---|---|---|---|
|**left_foot_contact_prob**|$[T]$|0-1|左脚接触地面的概率|
|**right_foot_contact_prob**|$[T]$|0-1|右脚接触地面的概率|
|**left_foot_contact**|$[T]$|bool|左脚是否接触地面|
|**right_foot_contact**|$[T]$|bool|右脚是否接触地面|
|**contact_confidence**|$[T]$ 或 $[T,2]$|0-1|足接触判断的置信度|

### 1.7 步态事件信号

|**字段名**|**形状**|**单位**|**现实意义**|
|---|---|---|---|
|**left_heel_strike**|$[T]$|bool|左脚 heel strike（落脚）事件|
|**right_heel_strike**|$[T]$|bool|右脚 heel strike（落脚）事件|
|**left_toe_off**|$[T]$|bool|左脚 toe off（离地）事件|
|**right_toe_off**|$[T]$|bool|右脚 toe off（离地）事件|
|**left_gait_phase**|$[T]$|phase|左脚步态相位|
|**right_gait_phase**|$[T]$|phase|右脚步态相位|
|**gait_phase_global**|$[T]$|phase|整体步态相位（可选）|

### 1.8 姿势学基础信号

|**字段名**|**形状**|**单位**|**现实意义**|
|---|---|---|---|
|**trunk_forward_flexion_deg**|$[T]$|deg|躯干前屈角（评估前倾姿势）|
|**trunk_lateral_lean_deg**|$[T]$|deg|躯干左右倾斜角（评估侧向不稳）|
|**trunk_lean_angle_deg**|$[T]$|deg|躯干总倾斜角（稳定性指标）|
|**pelvis_pitch_deg**|$[T]$|deg|骨盆前后倾|
|**pelvis_roll_deg**|$[T]$|deg|骨盆左右倾|
|**pelvis_yaw_deg**|$[T]$|deg|骨盆水平旋转|
|**trunk_pitch_deg**|$[T]$|deg|躯干 pitch|
|**trunk_roll_deg**|$[T]$|deg|躯干 roll|
|**trunk_yaw_deg**|$[T]$|deg|躯干 yaw|

### 1.9 质量控制信号

|**字段名**|**形状**|**单位**|**现实意义**|
|---|---|---|---|
|**joint_nan_mask**|$[T]$|bool|标记 joints 是否存在 NaN|
|**velocity_outlier_mask**|$[T]$|bool|标记速度异常帧|
|**representation_quality_score**|scalar 或 $[T]$|score|中间表示整体或逐帧质量得分|
|**valid_frame_mask**|$[T]$|bool|总体有效帧标记（C, D, E 模块统一使用）|

---

## 2. C_representation_meta.json 清单

该文件存储生成过程中的**非数组**解释信息。

- **基本信息**：模块名(`module`)、版本(`version`)、关联的 B 模块输入文件路径、序列总帧数、帧率。

- **坐标系说明**：

  - `coordinate_system.x/y/z`：各轴语义（forward/left/up）。
  - `unit/angle_unit/time_unit`：物理单位（meter/degree/second）。

- **字段说明 (Fields)**：逐一说明 `.npz` 中每个字段的 shape、unit 和详细 description。

- **派生信号生成配置 (Derivation Config)**：

  - 记录平滑方法、速度计算方法（如中央差分）、足接触判断阈值、步态事件检测方法、Heading 计算逻辑等。

- **质量汇总 (Quality Info)**：

  - `valid_frame_ratio`（有效帧比例）。

  - `num_invalid_frames`（无效帧总数）。

  - `contact_quality`（足接触信号质量）。

  - `warnings`（异常情况记录，如速度异常、接触置信度低、关节缺失等）。
