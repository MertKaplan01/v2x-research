"""Public, calibration-free core of the reference-pose-assisted baseline.

This module intentionally contains no TUMTraf image, annotation, transform, or
camera-calibration value. A user who independently obtains authorized data must
supply camera/pose matrices at runtime. The archived numeric evaluation lives
in ``results/summary_metrics.json`` and is audited without model inference.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

HEIGHT_PRIOR_M = {"person": 1.70, "car": 1.50, "bus": 3.20, "truck": 3.25}

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    base_xy: tuple[float, float] | None

def semantic_group(label: str) -> str:
    return "person" if label == "person" else "vehicle"

def hungarian_association(local: list[Detection], remote: list[Detection], gate_m: float = 4.0) -> list[tuple[int, int, float]]:
    """Apply the archived class-compatible 4 m association rule.

    Inputs must already be localized into the same S110-compatible 2-D frame.
    This function neither accesses pose, calibration, GT, images, nor IDs.
    """
    cost = np.full((len(local), len(remote)), 1e9, dtype=float)
    for i, vehicle in enumerate(local):
        for j, rsu in enumerate(remote):
            if vehicle.base_xy is None or rsu.base_xy is None:
                continue
            if semantic_group(vehicle.label) != semantic_group(rsu.label):
                continue
            distance = float(np.linalg.norm(np.asarray(vehicle.base_xy) - np.asarray(rsu.base_xy)))
            limit = gate_m + (1.5 if vehicle.label in {"bus", "truck"} or rsu.label in {"bus", "truck"} else 0.0)
            if distance <= limit:
                cost[i, j] = distance
    row, col = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i, j])) for i, j in zip(row, col) if cost[i, j] < 1e8]
