#!/usr/bin/env python3
"""Deterministic audit of archived TUMTraf-V2X frozen-baseline outputs.

This release script never runs YOLO or fusion. It verifies that the cached
Hungarian per-frame aggregation exactly matches its frozen summary. Detection
coverage needs cached per-object detections and assignments; when those files
are absent, the script reports that metric block as unavailable rather than
reconstructing it with a new model run.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

HUNGARIAN = "B_hungarian"
FIELDS = ("frames", "vehicle_detections", "rsu_detections", "spatial_associations",
          "correct_association_pairs", "wrong_association_pairs",
          "unverifiable_association_pairs", "co_visible_opportunities",
          "successful_opportunities", "unmatched_vehicle_by_association",
          "unmatched_rsu_by_association")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def aggregate_hungarian(rows: list[dict]) -> dict:
    total = Counter()
    for row in rows:
        for key in FIELDS[1:]: total[key] += int(row[key])
    total["frames"] = len(rows)
    return dict(total)

def association_metrics(total: dict) -> dict:
    pairs = total["spatial_associations"]
    verified = total["correct_association_pairs"] + total["wrong_association_pairs"]
    opportunities = total["co_visible_opportunities"]
    return {
        "conditional_association_recall": total["successful_opportunities"] / opportunities if opportunities else None,
        "verifiable_association_precision": total["correct_association_pairs"] / verified if verified else None,
        "confirmed_correct_fraction_all_predicted_pairs": total["correct_association_pairs"] / pairs if pairs else None,
        "all_predicted_pairs_treatment": "Unverifiable pairs remain in the denominator and do not count as confirmed correct.",
    }

def load_hungarian_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["condition"] == HUNGARIAN]
    if not rows: raise ValueError(f"No {HUNGARIAN} rows in {path}")
    return rows

def cache_inventory(root: Path) -> dict:
    # Required to derive coverage / stage failures without re-running YOLO.
    candidates = {
        "frozen_yolo_detections": root / "outputs/v2x_uncertainty_association/frozen_yolo_detections.jsonl",
        "frozen_lifted_positions": root / "outputs/v2x_uncertainty_association/frozen_lifted_positions.jsonl",
        "frozen_hungarian_assignments": root / "outputs/v2x_uncertainty_association/frozen_hungarian_assignments.jsonl",
    }
    present = {name: str(path.relative_to(root)) for name, path in candidates.items() if path.is_file()}
    missing = [name for name, path in candidates.items() if not path.is_file()]
    return {"present": present, "missing": missing, "coverage_reconstructible": not missing}

def run(root: Path) -> dict:
    assoc_dir = root / "outputs/v2x_uncertainty_association"
    baseline_path = root / "outputs/v2x_vehicle_size_prior_baseline/summary.json"
    archived_path = assoc_dir / "summary.json"
    per_frame = assoc_dir / "per_frame.csv"
    for path in (baseline_path, archived_path, per_frame):
        if not path.is_file(): raise FileNotFoundError(path)
    archived = json.loads(archived_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    rows = load_hungarian_rows(per_frame)
    recomputed = aggregate_hungarian(rows)
    frozen = archived["results"][HUNGARIAN]
    regression = {key: {"recomputed": recomputed[key], "frozen": frozen[key], "equal": recomputed[key] == frozen[key]} for key in FIELDS}
    passed = all(item["equal"] for item in regression.values())
    baseline_sha = sha256(baseline_path)
    source_sha = archived.get("frozen_size_prior_sha256")
    transform_ok = source_sha == baseline_sha
    inventory = cache_inventory(root)
    zip_path = root / "data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip"
    labels = None
    if zip_path.is_file():
        with ZipFile(zip_path) as archive:
            labels = sum(name.startswith("train/labels_point_clouds/") and name.endswith(".json") for name in archive.namelist())
    coverage = {
        "available": False,
        "reason": "Frozen per-object YOLO detections, lifted positions, and Hungarian assignments are absent. No YOLO/fusion rerun was performed.",
        "requested_metrics": ["Vehicle-only GT coverage", "RSU-only GT coverage", "GT-deduplicated union oracle coverage", "operational fused-output coverage", "RSU-only missed-by-Vehicle FOV split"],
    }
    failures = {
        "available": False,
        "supported_counts": {
            "correct_associations": recomputed["correct_association_pairs"],
            "incorrect_associations": recomputed["wrong_association_pairs"],
            "unverifiable_associations": recomputed["unverifiable_association_pairs"],
            "unmatched_vehicle_after_association": recomputed["unmatched_vehicle_by_association"],
            "unmatched_rsu_after_association": recomputed["unmatched_rsu_by_association"],
        },
        "unavailable_breakdown": ["missed detections", "invalid lifting", "out-of-gate counterparts", "per-object false merges"],
        "reason": "The archived aggregate CSV lacks per-object detection, lift-validity, candidate-gate, and assignment provenance.",
    }
    status = "PASS" if passed and transform_ok else "FAIL"
    return {
        "status": status,
        "scope": "Train-development archive audit only. The script never opens validation/test data or runs YOLO.",
        "inputs": {"archived_summary": str(archived_path.relative_to(root)), "archived_per_frame": str(per_frame.relative_to(root)), "size_prior_summary": str(baseline_path.relative_to(root)), "train_label_members_visible": labels},
        "frozen_baseline_provenance": {"archived_size_prior_sha256": source_sha, "current_size_prior_sha256": baseline_sha, "matches": transform_ok},
        "association_regression": {"condition": HUNGARIAN, "passed": passed, "fields": regression, "metrics": association_metrics(recomputed)},
        "detection_coverage": coverage,
        "failure_breakdown": failures,
        "cache_inventory": inventory,
        "uncertainty": {"independent_scene_intervals": "Not reported. Six train-development scene groups are not declared leakage-safe independent evaluation scenes.", "descriptive_only": True},
        "release_implication": "Association baseline is reproducible from archived aggregate records; coverage and stage-level metrics are unavailable from cached data and must not be reported as reconstructed.",
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(); root = args.root.resolve()
    result = run(root)
    out = args.out or root / "results/summary_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"], "out": str(out), "regression_passed": result["association_regression"]["passed"]}, indent=2))

if __name__ == "__main__": main()
