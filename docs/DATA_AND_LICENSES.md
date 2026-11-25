# Data, code, and licensing boundaries

## TUMTraf-V2X

Official TUMTraf sources currently contain a licensing discrepancy: the dataset landing page states **CC BY-NC-SA 4.0**, while the official development-kit README states **CC BY-NC-ND 4.0**. Until the dataset maintainers clarify the applicable term, this project treats raw images, labels, point clouds, per-frame transformations, calibration files, rendered frames, and videos as third-party material that must not be redistributed. Users must obtain the dataset independently from the official source.

Official references:

- [TUMTraf-V2X paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zimmer_TUMTraf_V2X_Cooperative_Perception_Dataset_CVPR_2024_paper.pdf)
- [TUMTraf-V2X dataset landing page](https://tum-traffic-dataset.github.io/tumtraf-v2x/)
- [TUM Traffic Dataset Development Kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)

The local `external/tumtraf-devkit/LICENSE` is MIT. Preserve its copyright notice whenever devkit code is copied or distributed.

## Ultralytics YOLO

The local baseline calls Ultralytics YOLO11s. Ultralytics describes its open-source option as AGPL-3.0 and offers a separate Enterprise license for closed or commercial deployment. Model weights are excluded from this release. Before publishing source that depends on Ultralytics code or models, review the applicable license and decide whether the project will comply with AGPL-3.0 or obtain appropriate terms.

Official reference: [Ultralytics licensing](https://www.ultralytics.com/license).

## Project license

The release candidate includes the GNU Affero General Public License v3.0 text in `LICENSE`. This prepares an open public release for code that uses the Ultralytics licensing path. It does not license TUMTraf data, calibration, transformations, model weights, or media. Those materials remain excluded.

## Local data layout

The release scripts expect local, ignored assets such as:

```text
data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip
external/tumtraf-devkit/calib/
models/yolo11s.pt
```

These paths are examples for local regeneration only. They must not be committed.
