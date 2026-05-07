# CARE-PD D_Segmentation

`scripts/D_Segmentation` rebuilds the D-stage motion segmentation directly from
strict `outputs/C_Representation` v2 artifacts.

It:

- reads per-sequence `.npz` and `.json` from `outputs/C_Representation/<subset>/`
- consumes C v2 root speed, heading/yaw-rate, contact/event, valid-frame, and quality arrays
- assigns exclusive primary labels `stand_to_sit / sit / sit_to_stand / turn / walk / adjust`
- assigns auxiliary `hesitation` only on `walk` or `turn`
- writes formal outputs to `outputs/D_Segmentation/<subset>/`

Key STS robustness rule:

- `pelvis_height_norm` uses the 2% and 98% pelvis-height percentiles as low/high references
- if the resulting height span is smaller than `0.16 m`, the whole sequence is normalized to `0.5`
- this avoids low-quality non-STS sequences being over-amplified into false sit-to-stand evidence

Formal outputs per sequence:

- `<stem>.json`
- `<stem>.csv`

Formal subset output:

- `segmentation_summary.csv`

Entry points:

```bash
/opt/anaconda3/envs/hymotion/bin/python scripts/D_Segmentation/cleanup_legacy_c_outputs.py --input-dir outputs/C_Representation
/opt/anaconda3/envs/hymotion/bin/python scripts/D_Segmentation/cleanup_legacy_c_outputs.py --input-dir outputs/C_Representation --apply
/opt/anaconda3/envs/hymotion/bin/python scripts/D_Segmentation/run_one_sequence.py --subset BMCLab --subject-id SUB01 --trial-id SUB01_off_walk_1
/opt/anaconda3/envs/hymotion/bin/python scripts/D_Segmentation/run_subset_batch.py --subset BMCLab --max-trials 2
/opt/anaconda3/envs/hymotion/bin/python scripts/D_Segmentation/run_all_subsets.py --max-trials-per-subset 1
```

This package is intentionally file-contract based: it depends on
strict `C_Representation` v2 outputs, not on older D/B segmentation modules or
legacy C aliases.
