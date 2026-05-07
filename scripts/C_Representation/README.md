# CARE-PD C_Representation

`scripts/C_Representation` converts `B_Canonicalization` outputs into the strict
frame-wise representation described in `outputs_demand.md`.

## Input

Each sequence is read from:

```text
outputs/B_Canonicalization/<subset>/<stem>.npz
outputs/B_Canonicalization/<subset>/<stem>.json
```

The B-stage NPZ must contain exactly:

- `pose_raw`
- `trans_raw`
- `joints_can`
- `trans_can`

The B-stage JSON must provide top-level `R_global` and upstream
`metadata.audition_metadata.R_total`. C uses these matrices with the SMPL root
axis-angle in `pose_raw[:, :3]` to compute canonical pelvis/root pitch, roll, and
yaw. Missing rotation metadata is a hard input-contract failure.

## Output

Each sequence writes only:

```text
outputs/C_Representation/<subset>/<stem>.npz
outputs/C_Representation/<subset>/<stem>.json
```

No diagnostic PNG and no summary CSV are part of the C contract.

The NPZ is strict and contains only the fields in `outputs_demand.md`: time/index
signals, canonical joints and translations, root/pelvis kinematics, heading/yaw
signals, left/right foot signals, contact probabilities and confidence, gait
events and phases, trunk/pelvis posture angles, QC masks, and per-frame
`representation_quality_score`.

Legacy D aliases are intentionally not written:

- `joints_world`
- `pelvis_pos`
- `left_foot_pos`
- `right_foot_pos`
- `upstream_left_contact`
- `upstream_right_contact`
- `heading_rate_deg_per_sec`
- old `gait_phase`

Current `scripts/D_Segmentation` still expects those aliases, so D must be
adapted before it can consume new C v2 outputs.

The JSON metadata includes module/version, B source paths, coordinate-system and
unit conventions, SMPL joint indices, per-field shape/unit/description,
derivation config, gait summary, quality info, warnings, and upstream B metadata.

## Commands

Run one sequence:

```bash
conda run -n hymotion python scripts/C_Representation/run_one_sequence.py \
  --input-dir outputs/B_Canonicalization \
  --output-dir outputs/C_Representation \
  --subset 3DGait \
  --subject-id 34 \
  --trial-id vid0137_0170
```

Run a subset:

```bash
conda run -n hymotion python scripts/C_Representation/run_subset_batch.py \
  --input-dir outputs/B_Canonicalization \
  --output-dir outputs/C_Representation \
  --subset 3DGait \
  --max-trials 5
```

Run all subsets:

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
  --output-dir /tmp/care_pd_c_smoke_C \
  --subset 3DGait \
  --subject-id 34 \
  --trial-id vid0137_0170
```

Expected checks:

- only `.npz` and `.json` are written
- NPZ keys exactly match the strict `outputs_demand.md` contract
- no legacy D alias fields are present
- frame-aligned arrays have first dimension `T`
- contact/event masks and `valid_frame_mask` are boolean arrays with shape `(T,)`
- `contact_confidence` has shape `(T, 2)`
- `representation_quality_score` has shape `(T,)` and values in `[0, 1]`
- JSON includes field descriptions, derivation config, quality info, source files,
  and gait summary
