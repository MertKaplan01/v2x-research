# Evaluation protocol and metric definitions

## Scope and input firewall

All final results concern 400 TUMTraf-V2X R4 Mini **training** frames. Validation and test data were not opened. Six timestamp-continuity groups are descriptive development partitions, not leakage-safe independent scenes.

The operational baseline consumes Vehicle and RSU JPEGs, static calibration, fixed YOLO11s weights, class-height priors, and a per-frame LiDAR-derived `vehicle_lidar_robosense_to_s110_lidar_ouster_south` transform. The transform is privileged pose, so this is not strict camera-only fusion. GT cuboids, positions, and IDs are accessed only after operational records are frozen, for post-hoc scoring.

## Detection coverage

The final eligible population contains 2,956 supported-class GT objects whose projected cuboid intersects at least one Vehicle or South2 image. Post-hoc matching is one-to-one, label-compatible image IoU at 0.3.

| Metric | Value | Definition |
|---|---:|---|
| Vehicle-only | 902/2,956 (30.51%) | Unique GT IDs matched in Vehicle |
| RSU-only increment | 1,011 | RSU matched minus Vehicle matched |
| Oracle union | 1,913/2,956 (64.72%) | GT-ID-deduplicated union; not deployable |
| Hungarian outputs | 1,833/2,956 (62.01%) | Unique GT IDs represented after operational merging |
| Selective outputs | 1,902/2,956 (64.34%) | 1x1 component rule; failed acceptance gate |

The 624 outside-FOV and 387 inside-FOV split applies to RSU-only GT objects under projected Vehicle FOV. It is not a physical-occlusion measurement.

## Association quality

The 509 co-visible opportunities are GT identities post-hoc matched in both views. Hungarian predicts 895 pairs: 235 correct, 102 incorrect, 558 unverifiable. Conditional recall is `235/509 = 46.17%`; verifiable precision is `235/(235+102) = 69.73%`; confirmed-correct fraction over all pairs is `235/895 = 26.26%`.

Unverifiable does not mean wrong. The 4,721/6,704 post-hoc-unmatched Hungarian outputs are not confirmed false positives; annotation gaps, class mismatch, and the fixed IoU protocol can contribute.

## Reproducibility layers

`python3 -m pytest -q tests` validates a synthetic fixture without proprietary assets. Full data-dependent scripts require independently obtained data, calibration, and weights and write ignored local outputs. `scripts/evaluate_coverage_and_audit.py` verifies archived Hungarian rows only when those local records exist; it does not fabricate coverage from aggregates. The safe final report and figures are in `docs/research/` and `figures/`.
