# C_Representation Design

## Summary

`scripts/C_Representation` is the strict analysis-representation layer between
`B_Canonicalization` and downstream segmentation / parameter extraction.

It consumes the current B-stage file contract, computes frame-aligned kinematic,
postural, gait, contact, and quality-control signals, and writes only per-sequence
`.npz` and `.json` artifacts.

C does not emit final action labels and does not emit final clinical parameters.
Version 2 is a strict output contract aligned with
`scripts/C_Representation/outputs_demand.md`; it intentionally does not write
legacy D compatibility aliases.

## Input Contract

Each sequence is read from:

```text
outputs/B_Canonicalization/<subset>/<stem>.npz
outputs/B_Canonicalization/<subset>/<stem>.json
```

The NPZ must contain exactly:

- `pose_raw`: original SMPL pose axis-angle array, shape `(T, 72)`.
- `trans_raw`: original SMPL translation array, shape `(T, 3)`.
- `joints_can`: canonicalized SMPL 24-joint coordinates, shape `(T, 24, 3)`.
- `trans_can`: canonicalized SMPL global translation, shape `(T, 3)`.

Legacy B fields such as `pose_canonical`, `trans_canonical`, `betas`, old contact
masks, and old quality scores are not accepted.

The JSON must provide:

- top-level `subset`, `subject_id`, `trial_id`.
- `metadata.fps`.
- top-level `R_global`, the B-stage rigid transform from A semantic coordinates
  to B canonical coordinates.
- `metadata.audition_metadata.R_total`, the A-stage transform from raw SMPL
  coordinates to A semantic coordinates.
- upstream source paths and B transform diagnostics for provenance.

`R_global` and `R_total` are required because C computes canonical pelvis/root
pitch, roll, and yaw from `R_global @ R_total @ pose_raw[:, :3]`. Missing or
malformed rotation metadata is a hard input-contract failure.

## Coordinate System

C assumes B has already established the canonical frame:

- `+X`: forward / progression.
- `+Y`: subject-left / lateral.
- `+Z`: vertical up.
- right-handed axes.
- positions in meters.
- angles in degrees.
- time in seconds.
- velocities in meters/second or degrees/second.
- accelerations in meters/second^2 or degrees/second^2.

## Algorithm

1. Validate the B NPZ contract, JSON identity, fps, `R_global`, and upstream
   `R_total`.
2. Use `joints_can[:, pelvis]` as `root_pos_m`; keep `trans_can` as the canonical
   SMPL global translation signal.
3. Compute `time_s` and `frame_index` from `fps` and frame count.
4. Compute root velocity, root horizontal speed, root acceleration, pelvis height,
   and pelvis vertical velocity with central differences.
5. Estimate body heading from canonical joint geometry:
   - combine hip span and shoulder span as the body-left axis.
   - use pelvis-to-neck as the trunk-up axis.
   - compute body-forward as `body_left x trunk_up`.
   - project to the horizontal plane.
   - when body heading is degenerate, fall back to root horizontal velocity.
   - fill invalid gaps from nearest valid heading.
6. Compute `heading_unwrapped_deg`, `yaw_rate_deg_s`, and
   `yaw_acceleration_deg_s2` with central differences.
7. Compute foot positions, velocities, speeds, and heights from left/right SMPL
   foot joints.
8. Estimate side-specific support points by choosing the lower of foot and ankle
   for each side.
9. Compute foot contact probability from low support height and low support
   horizontal speed.
10. Threshold and clean contact masks with minimum contact/swing duration.
11. Derive heel-strike and toe-off masks from contact onsets/offsets.
12. Encode gait phase:
    - `left_gait_phase` / `right_gait_phase`: `0=swing`, `1=stance`.
    - `gait_phase_global`: `0=no_contact`, `1=left_stance`, `2=right_stance`,
      `3=double_support`.
13. Compute `contact_confidence` as a `(T, 2)` left/right confidence score for the
    binary contact decisions.
14. Compute pelvis/root orientation:
    - convert `pose_raw[:, :3]` from axis-angle to rotation matrices.
    - apply the canonical transform `R_global @ R_total`.
    - extract canonical pelvis/root pitch, roll, and yaw.
15. Compute trunk posture from canonical joints:
    - `trunk_forward_flexion_deg`, signed sagittal flexion.
    - `trunk_lateral_lean_deg`, signed lateral lean.
    - `trunk_lean_angle_deg`, unsigned lean from vertical.
    - `trunk_pitch_deg`, `trunk_roll_deg`, `trunk_yaw_deg` from pelvis-neck and
      shoulder geometry.
16. Build quality-control signals:
    - `joint_nan_mask`.
    - `velocity_outlier_mask`.
    - `valid_frame_mask`.
    - per-frame `representation_quality_score` in `[0, 1]`.

## Output Contract

Each successful sequence writes:

```text
outputs/C_Representation/<subset>/<stem>.npz
outputs/C_Representation/<subset>/<stem>.json
```

No formal PNG or summary CSV is emitted by C.

The NPZ contains exactly these strict fields, in stable order:

- `fps`
- `time_s`
- `frame_index`
- `valid_frame_mask`
- `joints_can`
- `trans_can`
- `root_pos_m`
- `root_velocity_mps`
- `root_speed_xy_mps`
- `root_acceleration_mps2`
- `pelvis_height_m`
- `pelvis_vertical_velocity_mps`
- `heading_deg`
- `heading_unwrapped_deg`
- `yaw_rate_deg_s`
- `yaw_acceleration_deg_s2`
- `left_foot_pos_m`, `right_foot_pos_m`
- `left_foot_velocity_mps`, `right_foot_velocity_mps`
- `left_foot_speed_mps`, `right_foot_speed_mps`
- `left_foot_height_m`, `right_foot_height_m`
- `left_foot_contact_prob`, `right_foot_contact_prob`
- `left_foot_contact`, `right_foot_contact`
- `contact_confidence`
- `left_heel_strike`, `right_heel_strike`
- `left_toe_off`, `right_toe_off`
- `left_gait_phase`, `right_gait_phase`
- `gait_phase_global`
- `trunk_forward_flexion_deg`
- `trunk_lateral_lean_deg`
- `trunk_lean_angle_deg`
- `pelvis_pitch_deg`
- `pelvis_roll_deg`
- `pelvis_yaw_deg`
- `trunk_pitch_deg`
- `trunk_roll_deg`
- `trunk_yaw_deg`
- `joint_nan_mask`
- `velocity_outlier_mask`
- `representation_quality_score`

The NPZ intentionally does not contain these legacy aliases:

- `joints_world`
- `pelvis_pos`
- `left_foot_pos`
- `right_foot_pos`
- `upstream_left_contact`
- `upstream_right_contact`
- `heading_rate_deg_per_sec`
- old `gait_phase`

Current `scripts/D_Segmentation` expects those legacy aliases, so D must be
adapted before it can consume C v2 outputs. C v2 does not preserve backwards
compatibility in the NPZ payload.

The JSON contains:

- `module` and `version`.
- sequence metadata and B source file paths.
- input contract and legacy-field policy.
- coordinate-system and unit conventions.
- SMPL 24-joint index definitions.
- per-NPZ-field shape, dtype, unit, and description.
- derivation config, including velocity/acceleration, heading, contact, phase,
  pelvis orientation, and trunk orientation methods.
- gait summary and phase mappings.
- `quality_info`: valid-frame ratio, invalid-frame count, contact quality, and
  warnings.
- `quality_diagnostics`: success flag, quality flags, and detailed metrics.
- upstream B metadata for provenance.

## Quality Policy

C fails fast when:

- the B NPZ does not contain exactly the required four fields.
- core arrays have malformed shapes.
- core arrays contain non-finite values.
- `fps` is missing, non-finite, or non-positive.
- `R_global` or upstream `R_total` is missing, malformed, or non-finite.

For computed outputs, `representation_success` requires:

- valid input contract.
- finite input arrays and rotation metadata.
- frame-shape consistency.
- finite floating-point feature arrays.
- at least one usable heading source.
- available pelvis/root orientation.
- finite contact probabilities.
- at least one valid frame.

`valid_frame_mask` is stricter than `representation_success` and is intended for
downstream filtering on a per-frame basis. It combines finite feature checks,
speed sanity gates, heading availability, and contact-quality checks.

`representation_quality_score` is a per-frame `[T]` score in `[0, 1]` combining
finite-frame status, speed sanity, heading availability, and contact confidence.

## Public Entry Points

```bash
conda run -n hymotion python scripts/C_Representation/run_one_sequence.py \
  --input-dir outputs/B_Canonicalization \
  --output-dir outputs/C_Representation \
  --subset 3DGait \
  --subject-id 34 \
  --trial-id vid0137_0170
```

```bash
conda run -n hymotion python scripts/C_Representation/run_subset_batch.py \
  --input-dir outputs/B_Canonicalization \
  --output-dir outputs/C_Representation \
  --subset 3DGait \
  --max-trials 5
```

```bash
conda run -n hymotion python scripts/C_Representation/run_all_subsets.py \
  --input-dir outputs/B_Canonicalization \
  --output-dir outputs/C_Representation \
  --max-trials-per-subset 1
```

## Smoke Test

Use existing B outputs and write C outputs to a temporary directory:

```bash
conda run -n hymotion python scripts/C_Representation/run_one_sequence.py \
  --input-dir outputs/B_Canonicalization \
  --output-dir /tmp/care_pd_c_smoke_C_strict_v2 \
  --subset 3DGait \
  --subject-id 34 \
  --trial-id vid0137_0170
```

Acceptance checks:

- exactly two C artifacts are written: `.npz` and `.json`.
- no C `.png` or summary `.csv` is produced.
- NPZ keys exactly match the strict output field list.
- no legacy D alias fields are present.
- frame-aligned arrays share first dimension `T`.
- contact/event masks and `valid_frame_mask` are boolean arrays with shape `(T,)`.
- `contact_confidence` has shape `(T, 2)` and values in `[0, 1]`.
- `representation_quality_score` has shape `(T,)` and values in `[0, 1]`.
- floating-point arrays are finite.
- JSON includes field descriptions, derivation config, source files, gait summary,
  quality info, and quality diagnostics.
