# B_Canonicalization Design

## Summary

`scripts/B_Canonicalization` is a simple rigid canonicalization stage built on top of
`A_Audition`.

It does not read raw `.pkl` files, does not run SMPL forward kinematics, and does not
rewrite SMPL pose. Its only job is to apply one sequence-level rigid rotation and one
translation to the already semantic A-stage coordinates so that:

- `+X` is the main forward/progression axis.
- `+Y` is subject-left/lateral.
- `+Z` is vertical up from the fitted support-floor normal.
- the subject is first oriented from body-joint semantics.
- the low foot/ankle support plane is leveled when the fit is reliable.
- the first root position is at the XY origin.
- the low foot/ankle support height is approximately at `Z=0`.

`R_global`, `s_global`, and `t_global` are defined as the B-stage transform from
A_Audition semantic coordinates to B_Canonicalization coordinates:

```text
point_B = s_global * (R_global @ point_A) + t_global
```

## Input Contract

Each sequence is read from:

```text
outputs/A_Audition/<subset>/<subset>__<subject_id>__<trial_id>.npz
outputs/A_Audition/<subset>/<subset>__<subject_id>__<trial_id>.json
```

The A-stage NPZ must contain:

- `joints_3d`: `(T, 24, 3)`, semantic A coordinates.
- `trans_canonical`: `(T, 3)`, root trajectory in semantic A coordinates.
- `pose_raw`: `(T, 72)`, original SMPL pose, preserved unchanged.
- `trans_raw`: `(T, 3)`, original SMPL translation, preserved unchanged.
- `fps`: scalar.

The A-stage JSON is copied into the B JSON metadata as upstream provenance.

## Algorithm

1. Estimate the body semantic frame from the first `5` frames of `joints_3d`.
   - `up = normalize(median(mid_shoulder - mid_hip))`.
   - `left = median(left_hip - right_hip, left_shoulder - right_shoulder)`.
   - Orthogonalize `left` against `up`, normalize it, and compute
     `forward = normalize(cross(left, up))`.
   - `R_body` uses row vectors `[forward, left, up]` and must be right-handed.

2. Estimate one per-sequence global body scale from a robust long-bone subset.
   - Apply `R_body` to `joints_3d` and `trans_canonical`.
   - Use the body-frame leg long bones `left_hip-left_knee`, `left_knee-left_ankle`,
     `right_hip-right_knee`, and `right_knee-right_ankle`.
   - Compute robust observed bone lengths from the body-frame joints.
   - Generate target bone lengths from neutral SMPL in a standard static pose with
     `beta = 0` using the same 24-joint layout.
   - Use the median target/observed ratio as `s_global_raw`.
   - Clip `s_global_raw` to the configured plausible range `[0.75, 1.35]`.
   - If the estimate is under-supported or invalid, skip scaling, use `s_global = 1.0`,
     and record `scale_alignment.skipped_reason`.

3. Level the fitted support floor when reliable.
   - Apply `R_body` to `trans_canonical`.
   - Apply `R_body` to `joints_3d`.
   - Apply `s_global` to the body-frame joints and root trajectory before floor fitting.
   - Select low foot/ankle support candidates from `left_foot`, `left_ankle`,
     `right_foot`, and `right_ankle`.
   - Fit `Z = aX + bY + c` from a robust low-support cloud.
   - Compute `R_floor` that maps the fitted floor normal to `[0, 0, 1]`.
   - Enable floor leveling only when the floor tilt is at most `60.0 deg` and the
     median absolute plane residual is at most `0.08m`.
   - If support points are insufficient, the XY design is degenerate, values are
     non-finite, tilt is too high, or residuals are too high, use `R_floor = I` and
     record `floor_alignment.skipped_reason`.

4. Choose optional horizontal yaw from floor-corrected pelvis/root motion.
  - Apply `R_floor` to the scaled body-frame root trajectory.
   - Run PCA on the root XY trajectory.
   - Enable yaw only when the first PCA direction explains at least `0.90` of XY
     variance and its `P95-P5` robust range is at least `0.20m`.
   - If `net_displacement / path_length >= 0.15`, use net displacement to choose the
     PCA direction sign; otherwise keep the sign closest to body forward.
   - If the motion is not stable enough, use identity yaw and record the skipped reason.

5. Compose the B-stage rotation.

```text
R_global = R_yaw @ R_floor @ R_body
```

6. Solve the B-stage translation.
  - Rotate and scale the root and all `left_foot`, `left_ankle`, `right_foot`,
    `right_ankle` samples with `s_global * (R_global @ point_A)`.
   - Set `t_global[:2]` so the first root frame lands at `(0, 0)` in XY.
   - Set `t_global[2]` so the 5th percentile of all rotated foot/ankle `Z` samples
     lands at `0`.

7. Apply the transform.
  - `joints_can = s_global * (R_global @ joints_3d) + t_global`
  - `trans_can = s_global * (R_global @ trans_canonical) + t_global`
   - `pose_raw` and `trans_raw` are copied unchanged.

Final `+Z` prioritizes the fitted floor normal. The original body-up vector after
`R_global` is retained as a diagnostic angle, not as a hard zero-error invariant.

## Output Contract

Each successful sequence writes exactly three formal artifacts:

```text
outputs/B_Canonicalization/<subset>/<stem>.npz
outputs/B_Canonicalization/<subset>/<stem>.json
outputs/B_Canonicalization/<subset>/<stem>.png
```

The NPZ contains exactly these four arrays:

- `pose_raw`: original SMPL pose from A.
- `trans_raw`: original SMPL translation from A.
- `joints_can`: canonicalized 24-joint coordinates.
- `trans_can`: canonicalized root trajectory.

The JSON contains:

- sequence identifiers and upstream A metadata.
- input and output shapes.
- canonical axis convention.
- `R_global`, `s_global`, `t_global`.
- `body_frame_alignment`: `R_body`, raw-space forward/left/up axes, frames used,
  aggregation method, determinant, and degeneracy metrics.
- `scale_alignment`: `enabled`, `skipped_reason`, `method`, `s_global`,
  `s_global_raw`, `clip_applied`, `clip_range`, `aggregation`, `bones_used`,
  `bone_count_valid`, `quality_flag`, and per-bone target/observed/ratio diagnostics.
- `floor_alignment`: `enabled`, `skipped_reason`, `R_floor`, floor normals before and
  after leveling, plane coefficients, tilt before/after, robust support cloud count,
  median absolute residual, and reliability thresholds.
- `ground_alignment`: foot/ankle joint order, percentile, ground height, sample count.
- `yaw_alignment`: enabled/skipped status, yaw angle, PCA direction, variance ratio,
  robust range, net/path metrics, and thresholds.
- validation checks and warnings.

The PNG contains three rows:

1. root `X` and `Y` values over time.
2. root `Z` and support-point `Z` values over time.
3. absolute body-facing angle to `+X`, bounded to `0-180 deg`.

No summary CSV is part of the new B formal contract.

## Failure Policy

If the input is malformed or the first-frame body axes are degenerate, the sequence is
treated as failed. The run script reports the error, and the pipeline does not emit
formal NPZ/JSON/PNG artifacts for that sequence. If scale fitting is unreliable, the
sequence still succeeds with `s_global = 1.0` and a recorded skipped reason. If floor
fitting is unreliable, the sequence still succeeds with `R_floor = I` and a recorded
skipped reason.

Batch commands continue to the next sequence and report processed/failed counts on
stdout.

## Public Entry Points

```bash
conda run -n hymotion python scripts/B_Canonicalization/run_one_sequence.py \
  --input-dir outputs/A_Audition \
  --output-dir outputs/B_Canonicalization \
  --subset 3DGait \
  --subject-id 34 \
  --trial-id vid0137_0169

conda run -n hymotion python scripts/B_Canonicalization/run_subset_batch.py \
  --input-dir outputs/A_Audition \
  --output-dir outputs/B_Canonicalization \
  --subset 3DGait \
  --max-trials 5

conda run -n hymotion python scripts/B_Canonicalization/run_all_subsets.py \
  --input-dir outputs/A_Audition \
  --output-dir outputs/B_Canonicalization \
  --max-trials-per-subset 1
```

## Smoke Test

Use a temporary output directory to avoid overwriting historical B artifacts:

```bash
conda run -n hymotion python scripts/B_Canonicalization/run_one_sequence.py \
  --input-dir outputs/A_Audition \
  --output-dir /tmp/care_pd_b_smoke_body_axes \
  --subset PD-GaM \
  --subject-id 004 \
  --trial-id 004-13-002558_wid00_0
```

Current floor-leveling smoke set:

```bash
conda run -n hymotion python scripts/B_Canonicalization/run_subset_batch.py \
  --input-dir outputs/A_Audition \
  --output-dir /tmp/care_pd_b_floor_smoke \
  --subset PD-GaM \
  --subject-id 004 \
  --trial-id 004-13-002558_wid00_0 \
  --trial-id 004-13-002558_wid00_1 \
  --trial-id 004-13-003768_wid00_0 \
  --trial-id 004-13-003768_wid05_0
```

Acceptance checks:

- NPZ, JSON, and PNG exist.
- NPZ keys are exactly `pose_raw`, `trans_raw`, `joints_can`, `trans_can`.
- all NPZ arrays are finite.
- `trans_can[0, :2]` is approximately `(0, 0)`.
- foot/ankle `Z` 5th percentile is approximately `0`.
- `det(R_global)` is approximately `1`.
- JSON contains `body_frame_alignment`, `scale_alignment`, `floor_alignment`,
  `ground_alignment`, and `yaw_alignment.enabled/skipped_reason`.
- `scale_alignment.s_global` is finite, and disabled scale falls back cleanly to `1.0`.
- For the four-sequence floor smoke set, `floor_alignment.enabled` is `true`, floor
  tilt before leveling is about `44-50 deg`, and tilt after leveling is approximately
  `0 deg`.

## Downstream Note

The current `scripts/C_Representation` implementation still expects the old B fields
such as `pose_canonical`, `trans_canonical`, contact masks, and quality scores. Updating
C to consume the new four-field B NPZ contract is out of scope for this task.
