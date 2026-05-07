# Raw SMPL Root Trajectory Visualization Design

## Goal

Generate PNG diagnostics for raw CARE-PD SMPL sequences by plotting the root translation trajectory stored in each sequence's `trans` array.

The visualization intentionally uses the raw coordinate values without canonicalization, forward kinematics, filtering, smoothing, or unit conversion.

## Input Contract

The script recursively scans `raw_data` for `.pkl` files. Each subset pickle is expected to use this hierarchy:

```text
subject_id -> trial_id -> sequence_payload
```

Each `sequence_payload` must contain:

- `trans`: root translation array with shape `(T, 3)`.
- `fps` or `FPS`: positive frame rate used to compute sequence duration.

Other fields such as `pose`, `beta`, `UPDRS_GAIT`, `medication`, and `other` may be present but are not required for this visualization.

## Sequence Filtering

Sequences shorter than 3 seconds are discarded:

```text
duration_sec = trans.shape[0] / fps
```

If `duration_sec < 3.0`, no PNG is generated for that sequence.

Invalid sequences are skipped with a warning, and the batch continues. Invalid cases include missing `trans`, missing `fps/FPS`, non-positive FPS, non-finite values, or a `trans` array whose shape is not `(T, 3)`.

## Output Layout

PNG files are written to:

```text
outputs/Visualization/raw_sequence/<subset_name>/<subject_id>__<trial_id>.png
```

`subset_name` is derived from the `.pkl` file stem. For example:

```text
raw_data/3DGait.pkl
outputs/Visualization/raw_sequence/3DGait/<subject_id>__<trial_id>.png
```

Subject and trial identifiers are sanitized only enough to make safe filenames.

## Figure Layout

Each PNG contains a `3 x 2` subplot layout with six views of the same raw root trajectory.

The left column contains spatial plane trajectories:

1. XY plane: horizontal axis `X (unitless)`, vertical axis `Y (unitless)`.
2. YZ plane: horizontal axis `Y (unitless)`, vertical axis `Z (unitless)`.
3. XZ plane: horizontal axis `X (unitless)`, vertical axis `Z (unitless)`.

Each left-column subplot contains one trajectory line plus explicit start and end markers.

The right column contains raw translation components over time:

1. X(t): horizontal axis `t (s)`, vertical axis `X (unitless)`.
2. Y(t): horizontal axis `t (s)`, vertical axis `Y (unitless)`.
3. Z(t): horizontal axis `t (s)`, vertical axis `Z (unitless)`.

Time is computed as `t = np.arange(T) / fps`. The right-column subplots contain only the component line and do not include start/end markers.

All plots include axis labels. They do not include titles or grid lines.

## Entry Point

```bash
python scripts/Visualization/plot_raw_root_trajectory.py --raw-data raw_data --output-dir outputs/Visualization
```

Default behavior is equivalent to the command above.
