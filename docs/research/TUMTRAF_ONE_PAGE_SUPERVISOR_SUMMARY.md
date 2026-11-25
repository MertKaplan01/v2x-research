# TUMTraf-V2X cooperative perception: research summary

## Question

Can an infrastructure camera provide useful complementary perception to a moving Vehicle camera, and can the two RGB streams be associated safely enough for future strict camera-only cooperative perception?

## What was implemented

A reproducible 400-frame TUMTraf-V2X mini-train prototype used frozen YOLO11s detections, Vehicle class-size-prior monocular lifting, RSU ground-plane lifting, a provisional Vehicle camera projection, and a per-frame LiDAR-registration pose supplied only as a privileged reference-pose input. GT cuboids and identities were used only after inference for evaluation.

## Main measured finding: the RSU is useful

- Vehicle-only detects 902/2,956 eligible GT objects (**30.51%**).
- The RSU detects **1,011** additional GT objects missed by Vehicle.
- Post-hoc GT-deduplicated Vehicle+RSU union covers 1,913/2,956 (**64.72%**).
- Of RSU-only detections, 624 are outside projected Vehicle FOV; 387 are inside FOV but missed. This is a detection observation, not a physical-occlusion claim.

This establishes meaningful sensing complementarity in the train-development population.

## Main negative finding: association is not yet safe

| Method | Unique GT coverage | Key limitation |
|---|---:|---|
| Naïve concatenation | 1,913 | 509 confirmed cross-view duplicates |
| 4 m Hungarian | 1,833 | 102 incorrect merges; 80 coverage losses |
| Conservative 1×1 graph rule | 1,902 | only 66 duplicates removed; 81.54% merge precision |

Hungarian achieves 235/509 correct co-detected associations (46.17%) with 69.73% verifiable precision, but forced merging discards valid object evidence. The conservative rule is safer but failed its pre-registered precision and utility gate. Its result supports a coverage-preserving policy: uncertain observations should be retained rather than merged destructively.

## Geometry and localization evidence

Vehicle size-prior lifting improves conditional median Vehicle XY error from 25.599 m for bottom-center lifting to **3.053 m** on 902 post-hoc matched Vehicle detections. A privileged oracle ablation shows Vehicle lifting was the dominant bottleneck in an earlier configuration: correct associations rose from 7/509 to 372/509 when only Vehicle locations were replaced by GT diagnostic locations.

However, the current system remains dependent on LiDAR-derived Vehicle pose and a provisional camera calibration. RSU-ground-point reprojection into Vehicle images was valid for all 509 co-detected pairs but ranked the correct Vehicle box first only 374/509 (**73.48%**, below an 80% gate). Cars performed well (92.65%), while pedestrians failed (24.00%; 0.67% containment). The conditional image-space matcher was therefore not implemented.

## Status and next research question

This is a rigorous prototype and diagnostic foundation, not a deployable strict RGB-only system and not held-out evidence. The tested RGB-only object-graph pose route was rejected (48.528 m median translation and 75.394° yaw error).

The coherent master’s question is: **Can static-scene temporal RGB evidence provide a relative-pose distribution that enables coverage-preserving, uncertainty-aware cross-view association with principled abstention?**

A credible next study requires scene-disjoint sequential evaluation, independently validated image/calibration conventions, an RGB-only inference firewall, offline-only reference pose, and pre-registered reporting of pose error, merge precision, coverage loss, abstention, duplicates, and false merges.

**Bottom line:** RSU sensing benefit is demonstrated; reliable RGB-only alignment and safe cross-view association are unresolved research problems.
