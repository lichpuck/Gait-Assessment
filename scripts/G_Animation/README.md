# G_Animation

`G_Animation` is the animation presentation layer for the CARE-PD pipeline.

It consumes existing `B_Canonicalization`, `C_Representation`, `D_Segmentation`, and `F_Description` artifacts and renders one composite MP4 per sequence.

## Inputs

- `outputs/B_Canonicalization/<subset>/<stem>.npz`
- `outputs/B_Canonicalization/<subset>/<stem>.json`
- `outputs/C_Representation/<subset>/<stem>.npz`
- `outputs/C_Representation/<subset>/<stem>.json`
- `outputs/D_Segmentation/<subset>/<stem>.json`
- `outputs/F_Description/<subset>/<stem>.json`

The animation stage uses:

- canonical 24-joint 3D coordinates from B for skeleton motion
- framewise kinematic signals from C for the six curves
- primary motion segments from D for per-frame coloring
- overall Chinese summary text from F for the description panel

If `F_Description` does not provide `description.text_summary_zh`, the animation still renders and leaves the text panel empty.

## Outputs

- `outputs/G_Animation/<subset>/<stem>.mp4`
- `outputs/G_Animation/<subset>/<stem>.json`

The MP4 contains one composite view with:

- a fixed friendly 3D skeleton view
- visible `X/Y/Z` coordinate axes
- skeleton color driven by the active D primary label
- one overall summary text panel with abnormal clauses highlighted in red
- a dedicated right-side status block for `fps`, `duration`, and summary availability
- a dedicated right-side segment legend so the label color chips do not overlap the status block
- six full-sequence curves with a moving frame cursor

The JSON sidecar records source files, metric summaries, render settings, and output paths.

## Six Curves

1. `稳定性：躯干倾斜角度` -> `abs(trunk_lean_angle_deg)`
2. `平衡性：骨盆侧倾角度` -> `abs(pelvis_roll_deg)`
3. `对称性：左右脚相位差` -> `abs(left_gait_phase - right_gait_phase)`
4. `协调性：左右脚速度一致性` -> rolling correlation of `left_foot_speed_mps` and `right_foot_speed_mps`
5. `移动能力：前进速度` -> `root_speed_xy_mps`
6. `控制能力：运动平滑度` -> jerk norm derived from `root_acceleration_mps2`

## Commands

Standalone render from existing B/C/D/F outputs:

```bash
python scripts/G_Animation/run_one_sequence.py \
  --subset BMCLab \
  --subject-id SUB01 \
  --trial-id SUB01_off_walk_1
```

Integrated single-sequence run from raw SMPL input:

```bash
/opt/anaconda3/envs/hymotion/bin/python scripts/run_smpl_to_description.py \
  --input-pkl raw_data/test.pkl
```

Use the `hymotion` interpreter for the integrated command because `A_Audition` requires `smplx`.

## Notes

- No new Python dependency is required beyond the current repository stack.
- MP4 writing uses matplotlib's ffmpeg writer, so an `ffmpeg` binary must be available on `PATH`.
- On macOS the renderer prefers locally available CJK-capable fonts such as `Hiragino Sans GB`, `STHeiti`, `Songti SC`, or `Arial Unicode MS` for the Chinese summary panel.
- Overall summary highlighting uses stable clause-based rendering: short clauses containing abnormal Chinese phrases such as `轻度异常`、`中度异常`、`重度异常` are emphasized in red.
