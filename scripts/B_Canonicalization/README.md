# CARE-PD B_Canonicalization

This package canonicalizes `A_Audition` outputs with one sequence-level similarity transform.

The stage is independent of the removed `care_pd_pipeline` package.  Its only SMPL
forward use is the neutral beta=0 long-bone target calculation for scale alignment,
and that call is routed through the local `scripts.A_Audition.smpl_forward` SMPLX
backend.

It reads:

```text
outputs/A_Audition/<subset>/<stem>.npz
outputs/A_Audition/<subset>/<stem>.json
```

and writes:

```text
outputs/B_Canonicalization/<subset>/<stem>.npz
outputs/B_Canonicalization/<subset>/<stem>.json
outputs/B_Canonicalization/<subset>/<stem>.png
```

## Transform

`R_global`, `s_global`, and `t_global` are the B-stage transform from A semantic
coordinates to B canonical coordinates:

```text
point_B = s_global * (R_global @ point_A) + t_global
```

The final coordinate frame is right-handed:

- `+X`: progression / forward
- `+Y`: subject-left / lateral
- `+Z`: vertical up

The body frame is estimated from the first frames of `joints_3d`: torso up initializes
`+Z`, subject-left initializes `+Y`, and `left x up` defines `+X`. The stage then
estimates one per-sequence global body scale from a robust lower-limb bone subset by
comparing observed body-frame bone lengths against neutral SMPL `beta=0` target lengths.
Those target lengths are computed from `body_models/smpl/SMPL_NEUTRAL.pkl` through the
shared local SMPLX wrapper used by A_Audition, including the NumPy 2.x/chumpy pickle
compatibility patch.
If scale estimation is unreliable, the stage keeps `s_global = 1.0` and records a
skipped reason in JSON. Floor fitting then runs on the scaled body-frame joints. If the
fitted plane is reliable, `R_floor` levels that plane so its normal maps to final `+Z`;
otherwise `R_floor` is identity and the JSON records the skipped reason. If the corrected
pelvis/root trajectory is a stable horizontal line, a yaw about `+Z` maps that motion
direction to final `+X`.

```text
R_global = R_yaw @ R_floor @ R_body
```

The sequence is translated so the first root frame is at the XY origin and the low
percentile of all scaled foot/ankle `Z` samples is at `0`. Final `+Z` prioritizes the
fitted floor normal, so body-up is reported as a diagnostic angle rather than a hard invariant.

## NPZ Contract

The output NPZ contains exactly:

- `pose_raw`
- `trans_raw`
- `joints_can`
- `trans_can`

`pose_raw` and `trans_raw` are copied from A unchanged. `joints_can` and `trans_can`
are the canonicalized joint and root trajectories.

## JSON Contract

The JSON records identifiers, upstream A metadata, shapes, `R_global`, `s_global`,
`t_global`, body-frame alignment details, scale diagnostics, floor-plane leveling
details, ground-height translation details, yaw details, checks, warnings, and output paths.

## PNG Diagnostic

The diagnostic figure has three rows:

1. root `X` and `Y` over time
2. root/support `Z` over time
3. body-facing angle to `+X` in `0-180 deg`

## Failure Policy

If the input is malformed or the body-frame vectors are degenerate, the sequence fails.
If scale estimation is unreliable, scaling is skipped with `s_global = 1.0` and the
skipped reason is written to JSON; this does not fail the sequence.
If floor-plane fitting is unreliable, floor leveling is skipped with `R_floor = I` and
the skipped reason is written to JSON; this does not fail the sequence. Batch runs
continue with the next sequence.

No summary CSV is written by this stage.

## Commands

Run in the `hymotion` environment:

```bash
conda run -n hymotion python scripts/B_Canonicalization/run_one_sequence.py \
  --input-dir outputs/A_Audition \
  --output-dir outputs/B_Canonicalization \
  --subset 3DGait \
  --subject-id 34 \
  --trial-id vid0137_0169
```

```bash
conda run -n hymotion python scripts/B_Canonicalization/run_subset_batch.py \
  --input-dir outputs/A_Audition \
  --output-dir outputs/B_Canonicalization \
  --subset 3DGait \
  --max-trials 5
```

```bash
conda run -n hymotion python scripts/B_Canonicalization/run_all_subsets.py \
  --input-dir outputs/A_Audition \
  --output-dir outputs/B_Canonicalization \
  --max-trials-per-subset 1
```

## Smoke Test

Use a temporary output directory when checking the stage:

```bash
conda run -n hymotion python scripts/B_Canonicalization/run_one_sequence.py \
  --input-dir outputs/A_Audition \
  --output-dir /tmp/care_pd_b_smoke_body_axes \
  --subset PD-GaM \
  --subject-id 004 \
  --trial-id 004-13-002558_wid00_0
```

For the current floor-leveling smoke set:

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

Expected checks:

- exactly three artifacts are written
- NPZ keys are exactly `pose_raw`, `trans_raw`, `joints_can`, `trans_can`
- arrays are finite
- first root XY is approximately zero
- foot/ankle `Z` 5th percentile is approximately zero
- `det(R_global)` is approximately `1`
- JSON contains `body_frame_alignment`, `scale_alignment`, `floor_alignment`,
  `ground_alignment`, and `yaw_alignment.enabled/skipped_reason`
- `scale_alignment.s_global` is finite, and either `scale_alignment.enabled` is `true`
  or the block records a non-null `skipped_reason` with `s_global = 1.0`
- for the floor-leveling smoke set, `floor_alignment.enabled` is `true`, tilt before
  leveling is about `44-50 deg`, and tilt after leveling is approximately `0 deg`

Default floor reliability thresholds are:

- `floor_max_tilt_deg = 60.0`
- `floor_max_median_abs_residual_m = 0.08`

## Downstream Note

The current `scripts/C_Representation` still expects the old B output fields
(`pose_canonical`, `trans_canonical`, contact masks, quality scores). Updating C for this
new B contract is a separate task.
