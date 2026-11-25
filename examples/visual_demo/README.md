# Representative Visual Examples

These four representative outputs come from the TUMTraf-V2X research prototype. Each PNG is an existing composite showing a Vehicle camera view with frozen YOLO detections, an RSU camera view with frozen YOLO detections, and estimated positions in a shared S110 bird’s-eye view (BEV). They are research outputs, not raw-image/prediction pairs.

- **Frame 264 - mixed association outcomes:** the operational associations were assessed post hoc as 3 correct, 2 incorrect, and 1 unverifiable. It is not an all-correct success case.
- **Frame 135 - complementary Vehicle/RSU detections:** complementary observations remain unmatched. This example does not claim confirmed recovery of a ground-truth object.
- **Frame 271 - association limitation:** post-hoc assessment found 0 correct, 3 incorrect, and 3 unverifiable associations.
- **Reference-pose demo:** the original 1-second clip contains ten adjacent frames at 10 Hz.

Color and panel interpretation is in the images themselves: the upper panels show camera detections and the lower panel shows estimated shared-frame positions. Prediction marks represent operational outputs. Correctness counts are post-hoc ground-truth interpretation, not inference inputs; unverifiable does not mean incorrect.

**System scope:** the prototype is reference-pose-assisted. It uses a LiDAR-derived per-frame Vehicle-to-infrastructure reference pose and provisional Vehicle calibration. It is not strict RGB-only cooperative perception, and the results are train-development-only.

**Use and attribution:** these selected materials are included with permission for non-commercial academic and research presentation. Source dataset: [TUMTraf-V2X](https://tum-traffic-dataset.github.io/tumtraf-v2x/), Walter Zimmer et al., CVPR 2024, Technical University of Munich and Fraunhofer IVI. See the [official TUM Traffic Dataset Development Kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit). The full dataset, annotations, LiDAR files, calibration archives, and model weights are not included.
