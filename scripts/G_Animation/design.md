# G_Animation Design

## Goal

`G_Animation` is the presentation layer added after `F_Description`.

After this stage, the repository-level integrated flow is:

`SMPL -> physical features + semi-structured description + skeleton animation`

The stage is intentionally file-contract based and does not recompute upstream semantic segmentation or descriptive text.

## Data Dependencies

### B_Canonicalization

- Uses `joints_can` and `trans_can` as the authoritative normalized 3D motion sequence.
- Keeps the animation aligned with the canonical frame: `+X` forward, `+Y` subject-left, `+Z` up.

### C_Representation

- Supplies the six framewise curves.
- Avoids duplicating kinematic derivations already formalized in C.

### D_Segmentation

- Supplies the per-frame primary label by expanding segment intervals.
- The six D primary labels are the authoritative color semantics for the skeleton.

### F_Description

- Supplies only the sequence-level overall summary text.
- Missing summary text is non-fatal and renders as an empty panel.

## Visual Layout

The MP4 is a single composite canvas:

1. Left: 3D skeleton animation with fixed friendly camera.
2. Top-right left block: overall summary text from `F_Description`, split into short clauses for stable rendering.
3. Top-right right block: a status subpanel for `fps`, `duration`, and summary presence, plus a dedicated segment legend panel.
4. Right-lower: six full-sequence curves, each with a moving cursor showing the current frame.

The 3D panel includes:

- static world coordinate axes
- fixed viewing angle
- static scene bounds across the whole sequence to avoid camera jitter
- thin root trajectory trail for motion context

## Color Policy

Each D primary label has its own color:

- `stand_to_sit`
- `sit`
- `sit_to_stand`
- `turn`
- `walk`
- `adjust`

The active frame label drives skeleton color. The same label palette is also used as light background spans behind the six curves.

The legend is rendered in a dedicated panel on the right so it does not overlap the status text.

## Curve Definitions

1. Stability
   - Display title: `稳定性：躯干倾斜角度`
   - Definition: `abs(trunk_lean_angle_deg)`
   - Source: C

2. Balance
   - Display title: `平衡性：骨盆侧倾角度`
   - Definition: `abs(pelvis_roll_deg)`
   - Source: C

3. Symmetry
   - Display title: `对称性：左右脚相位差`
   - Definition: `abs(left_gait_phase - right_gait_phase)`
   - Source: C
   - Current C contract encodes gait phase as binary stance/swing, so the resulting curve is a binary asymmetry proxy.

4. Coordination
   - Display title: `协调性：左右脚速度一致性`
   - Definition: centered rolling Pearson correlation between `left_foot_speed_mps` and `right_foot_speed_mps`
   - Source: C
   - Uses a short configurable time window to produce a framewise approximation.

5. Mobility
   - Display title: `移动能力：前进速度`
   - Definition: `root_speed_xy_mps`
   - Source: C

6. Control
   - Display title: `控制能力：运动平滑度`
   - Definition: jerk norm, where jerk is the time derivative of `root_acceleration_mps2`
   - Source: derived in G from C acceleration

## Overall Assessment Rendering

- The summary is split into a neutral header and short clauses, rather than rendered as one uninterrupted paragraph.
- Clauses containing stable negative phrases from current `F_Description` outputs, especially `轻度异常`, `中度异常`, and `重度异常`, are emphasized in red.
- Neutral clauses such as `正常` remain in the default text color.
- This clause-based strategy is used deliberately because it is more stable in matplotlib than arbitrary inline rich-text mixing.

## Output Contract

Per sequence:

- one MP4: `outputs/G_Animation/<subset>/<stem>.mp4`
- one JSON sidecar: `outputs/G_Animation/<subset>/<stem>.json`

The JSON records:

- sequence identity and duration
- exact upstream source artifact paths
- whether summary text was present
- metric summaries
- render configuration
- final output paths

## Integration Strategy

`scripts/run_smpl_to_description.py` now invokes G after F.

The integrated JSON under `outputs/G_Integrated` now includes:

- `quality_summary.stage_status.G_Animation`
- an `animation` block with render metadata and paths
- animation paths in the top-level `outputs` block

## Validation Strategy

- Standalone validation: render one known existing B/C/D/F sample.
- Integrated smoke test: run only `raw_data/test.pkl` through `scripts/run_smpl_to_description.py`.
- No full-batch validation is part of this module delivery.
