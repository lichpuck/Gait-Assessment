# CARE-PD Raw Root Trajectory Visualization

Plot raw SMPL root translation trajectories from `raw_data` pickle files.

## Raw Root Run

```bash
python scripts/Visualization/plot_raw_root_trajectory.py
```

Equivalent explicit command:

```bash
python scripts/Visualization/plot_raw_root_trajectory.py --raw-data raw_data --output-dir outputs/Visualization
```

The script recursively scans `raw_data` for `.pkl` files and writes PNGs to:

```text
outputs/Visualization/raw_sequence/<subset_name>/<subject_id>__<trial_id>.png
```

## Raw Root Input Format

Each raw subset pickle is expected to contain:

```text
subject_id -> trial_id -> sequence_payload
```

The sequence payload must contain:

- `trans`: root translation with shape `(T, 3)`.
- `fps` or `FPS`: sequence frame rate.

The plotted coordinates are exactly the values in `trans`:

- `trans[:, 0]` is X.
- `trans[:, 1]` is Y.
- `trans[:, 2]` is Z.

## Raw Root Filtering

Sequences shorter than 3 seconds are skipped:

```text
duration_sec = T / fps
```

Malformed sequences are skipped with a warning so the rest of the batch can continue.

## Raw Root Figure Content

Each output PNG contains six subplots arranged as `3 x 2`.

The left column contains spatial plane trajectories:

1. XY plane: X on the horizontal axis, Y on the vertical axis.
2. YZ plane: Y on the horizontal axis, Z on the vertical axis.
3. XZ plane: X on the horizontal axis, Z on the vertical axis.

Each left-column subplot includes one trajectory line plus start and end markers.

The right column contains raw translation components over time:

1. X(t): `trans[:, 0]` over `t`.
2. Y(t): `trans[:, 1]` over `t`.
3. Z(t): `trans[:, 2]` over `t`.

Time is computed in seconds from `fps`:

```text
t = frame_index / fps
```

The right-column subplots contain only the component line. Axis labels use `unitless` because raw `trans` values are treated as dimensionless.

## Raw Root Useful Options

```bash
python scripts/Visualization/plot_raw_root_trajectory.py --skip-existing
python scripts/Visualization/plot_raw_root_trajectory.py --max-sequences-per-subset 5
python scripts/Visualization/plot_raw_root_trajectory.py --min-duration-sec 3.0
```

## CARE-PD Canonical Sequence Feature Visualization

Plot canonical C_Representation time-series features from `outputs/C_Representation`.

### Canonical Feature Run

```bash
python scripts/Visualization/plot_can_sequence_features.py
```

Equivalent explicit command:

```bash
python scripts/Visualization/plot_can_sequence_features.py --input-dir outputs/C_Representation --output-dir outputs/Visualization
```

The script scans each subset folder under `outputs/C_Representation` and writes PNGs to:

```text
outputs/Visualization/can_sequence/<subset_name>/<sequence_stem>.png
```

### Canonical Feature Input Format

Each C_Representation sequence is expected to have a `.json` metadata file and a matching `.npz` array file. The JSON is used for subset/sequence metadata and the NPZ path; the NPZ provides the actual time-series arrays.

Required NPZ fields:

- `root_pos_m`: root/pelvis position with shape `(T, 3)`.
- `joints_can`: canonical SMPL joints with shape `(T, 24, 3)`.
- `heading_deg`: wrapped body-facing yaw angle with shape `(T,)`.

Time is read from `time_s` when available. Otherwise it is computed as `frame_index / fps`. FPS is read from `npz["fps"]`, then JSON `basic_info.fps`, then JSON `metadata.fps`; if none are valid, the script logs a warning and uses 30 FPS.

### Canonical Feature Figure Content

Each output PNG contains three rows:

1. Root X position over time.
2. Root Z position plus left/right wrist Z positions over time.
3. Absolute wrapped heading difference from the first frame over time, in degrees.

The heading difference is wrapped to `[-180, 180]` before taking the absolute value, so it stays in the `0-180` degree range.

Malformed sequences are skipped with a warning so the rest of the batch can continue.

### Canonical Feature Useful Options

```bash
python scripts/Visualization/plot_can_sequence_features.py --skip-existing
python scripts/Visualization/plot_can_sequence_features.py --max-sequences-per-subset 5
```

## CARE-PD Segmentation Check Visualization

Render three-row inspection plots for `outputs/D_Segmentation` sequences using motion signals from `outputs/C_Representation`.

### Segment Check Run

```bash
conda run -n hymotion python scripts/Visualization/segment_check.py
```

Equivalent explicit command:

```bash
conda run -n hymotion python scripts/Visualization/segment_check.py --segmentation-dir outputs/D_Segmentation --representation-dir outputs/C_Representation --output-dir outputs/Visualization
```

The script scans each subset folder under `outputs/D_Segmentation` and writes PNGs to:

```text
outputs/Visualization/<subset_name>/<sequence_stem>.png
```

### Segment Check Figure Content

Each output PNG contains three rows:

1. Root X displacement relative to the first frame, with `walk` and `adjust` spans overlaid.
2. Absolute heading difference from the first frame, with `turn` spans and precise `hesitation` spans overlaid.
3. `pelvis_height_norm`, with `stand_to_sit`, `sit`, and `sit_to_stand` spans overlaid.

The first and third rows reuse D_Segmentation-compatible signal definitions. In particular, `pelvis_height_norm` uses the same 2%/98% percentile normalization and minimum-span guard as `scripts/D_Segmentation`.

### Segment Check Useful Options

```bash
conda run -n hymotion python scripts/Visualization/segment_check.py --subset E-LC
conda run -n hymotion python scripts/Visualization/segment_check.py --sequence-stem E-LC__4__FOG-004-offmed-TUG-standard1-TP
conda run -n hymotion python scripts/Visualization/segment_check.py --skip-existing
```

## CARE-PD B First-Frame Summary Export

Render the first frame of each `outputs/B_Canonicalization` sequence as a skeleton PNG for visual inspection.

### B First-Frame Run

```bash
conda run -n hymotion python scripts/Visualization/export_b_first_frame_summary.py
```

Equivalent explicit command:

```bash
conda run -n hymotion python scripts/Visualization/export_b_first_frame_summary.py --input-dir outputs/B_Canonicalization --output-dir outputs/Visualization
```

The script scans each subset folder under `outputs/B_Canonicalization` and writes JSON files to:

```text
outputs/Visualization/first_frame/<subset_name>/<sequence_stem>.png
```

### B First-Frame Input Format

Each B_Canonicalization sequence is expected to have a `.json` metadata file and a matching `.npz` file.

Required NPZ fields:

- `joints_can`: canonical SMPL joints with shape `(T, 24, 3)`.
- `trans_can`: canonical root trajectory with shape `(T, 3)`.

### B First-Frame Output Content

Each PNG visualizes the first frame in the canonical B coordinate system:

1. 24 joints with SMPL skeleton connections.
2. Three projections: XY (top), XZ (side), YZ (front).
3. Body-facing direction arrow in XY and first-frame pelvis/root metrics in the footer text.

### B First-Frame Useful Options

```bash
conda run -n hymotion python scripts/Visualization/export_b_first_frame_summary.py --subset BMCLab --subset PD-GaM --subset T-SDU-PD
conda run -n hymotion python scripts/Visualization/export_b_first_frame_summary.py --skip-existing
```
