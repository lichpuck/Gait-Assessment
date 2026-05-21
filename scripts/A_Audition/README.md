# A_Audition

`A_Audition` converts raw CARE-PD SMPL sequences into first-pass semantic-coordinate outputs. It only standardizes coordinate-axis semantics:

- `+X`: forward
- `+Y`: subject-left / lateral
- `+Z`: up / vertical

It does not fit a floor, correct slope, level the ground, recenter trajectories, or canonicalize SMPL pose.

The raw I/O and SMPL forward layers are implemented locally for A_Audition; this module does not import or require the removed `care_pd_pipeline` package.

## Outputs

Each valid sequence writes three files under `outputs/A_Audition/<subset>/`.

`.npz` fields:

- `joints_3d`: `(T, 24, 3)`, semantic-coordinate SMPL joints
- `trans_canonical`: `(T, 3)`, `R_total @ trans_raw`, with the original origin preserved
- `pose_raw`: `(T, 72)`
- `trans_raw`: `(T, 3)`
- `beta`: `(1, 10)`
- `fps`: scalar
- `R_total`: `(3, 3)`, raw-to-semantic axis rotation
- `support_points`: `(T, 3)`, the lowest point per frame among left/right ankles and left/right feet in the semantic frame

`.json` records source identifiers, metadata, raw axis semantics, robust ranges, axis mapping, longest monotonic forward segment, `R_total`, lowest-support-point metrics, warnings, and output paths.

`.png` contains three diagnostic plots:

- root `trans_canonical` trajectory in the `XY` plane
- per-frame support point and root `trans_canonical` trajectories in the `XZ` plane
- absolute skeleton-facing angle to the `+X` forward axis over frames, constrained to `0-180` degrees

Sequences shorter than `3s` are skipped and recorded in `outputs/A_Audition/audition_summary.csv` with `status=skipped` and `skip_reason=duration_lt_3s`. If the largest robust-range axis conflicts with the inferred vertical axis, the sequence is still processed by prioritizing vertical and selecting the forward axis from the two remaining raw axes. Handedness is corrected by flipping the forward sign when needed and recorded as a warning.

## Commands

Single sequence:

```bash
conda run -n hymotion python scripts/A_Audition/run_one_sequence.py \
  --subset BMCLab \
  --subject-id SUB01 \
  --trial-id SUB01_on_walk_1
```

One subset:

```bash
conda run -n hymotion python scripts/A_Audition/run_subset_batch.py \
  --subset BMCLab \
  --max-trials 5
```

All subsets:

```bash
conda run -n hymotion python scripts/A_Audition/run_all_subsets.py \
  --max-trials-per-subset 3
```

## Inputs and SMPL Forward

Raw subset files are read from `raw_data/*.pkl` as nested datasets:

```text
{
  subject_id: {
    trial_id: {
      "pose": (T, 72),
      "trans": (T, 3),
      "beta": (1, 10) | (10,) | constant (T, 10),
      "fps": scalar,
      ...metadata
    }
  }
}
```

The loader first attempts `joblib.load`, which supports the current compressed raw CARE-PD files, then falls back to standard `pickle.load` for plain pickle inputs. `beta` is normalized to `(1, 10)`; per-frame `(T, 10)` beta is accepted only when every row is constant within floating-point tolerance.

SMPL forward kinematics are run locally through `smplx` and `body_models/smpl/SMPL_NEUTRAL.pkl`. Because the official SMPL `.pkl` files can contain legacy chumpy objects, the local wrapper patches the small set of NumPy 2.x aliases needed during model loading before calling `smplx`.

## Axis Inference

The raw coordinate array order is always treated as `[X, Y, Z]`, but its semantics are inferred per sequence:

1. Compute `robust_range = P95 - P5` for each raw `trans` axis.
2. Use the dominant component of median `head - pelvis` to choose vertical and its sign, with `neck - pelvis` as fallback.
3. Choose forward from the two non-vertical raw axes by larger robust range; the other non-vertical axis becomes lateral.
4. Lateral sign is chosen from median `left_hip - right_hip`.
5. Forward sign is chosen from the longest approximately monotonic segment of the smoothed raw forward-axis trajectory.
6. If the signed axis mapping is not right-handed, flip the forward sign and record `forward_sign_flipped_for_right_handed_frame`.
