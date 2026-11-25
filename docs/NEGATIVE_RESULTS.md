# Negative results and stopping decisions

## Calibration and image state

The direct official Vehicle projection was not plausible on raw Vehicle-LiDAR overlays. Official-plus-canonical-axis became geometrically plausible but lacked independent candidate-blind correspondence validation. The historical provisional projection is retained only as a baseline. JPEG rectification state was undocumented; no intrinsic interchange or calibration optimization was performed.

## Association methods

- **Hungarian, fixed gate:** 1,833/2,956 output coverage, 102 incorrect merges, 80 coverage losses relative to concatenation, and 150 confirmed duplicates.
- **Assumed-covariance Mahalanobis Hungarian:** 136/509 correct opportunities (26.72%), below Hungarian’s 235/509; rejected without tuning.
- **Isolated-component selective fusion:** 1,902/2,956 coverage and 12 wrong merges, but 81.54% verifiable precision and only 66 duplicate removals; it failed two pre-registered criteria.
- **RSU ground-point → Vehicle-image reprojection:** all 509 true pairs projected validly, but rank-first was 374/509 (73.48%), below the 80% gate. Pedestrian rank-first was 24.00%; Stage B was not implemented.

These results do not prove RGB association impossible. They establish that the tested geometry-only rules cannot meet their safety/utility requirements under current pose, lifting, and calibration assumptions.

## Pose-free alignment

The GT-blind pose-free object-graph route had adequate candidate coverage but only 22.27% verifiable appearance precision. Its SE(2) estimates had 48.528 m median translation and 75.394° median yaw error against offline reference pose. It was rejected; LiDAR-derived pose remains an unresolved runtime dependency.

Classical V2X-Sim visual geometry and a mini-scene cooperation ceiling were also stopped under their recorded gates. They are not evidence about all future real-world RGB approaches. See the complete [technical report](research/TUMTRAF_FINAL_TECHNICAL_REPORT.md).
