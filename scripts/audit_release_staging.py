#!/usr/bin/env python3
"""Audit the staged public release allowlist without changing staged content."""
from __future__ import annotations
import hashlib, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_PREFIXES = ("data/", "external/", "models/", "outputs/", ".venv/", ".v2xsim-venv/", ".coopscenes")
BLOCKED_SUFFIXES = (".zip", ".pt", ".pth", ".onnx", ".jpg", ".jpeg", ".png", ".mp4", ".avi", ".pcd", ".ply")
ALLOWED_AGGREGATE_FIGURES = {
    "docs/figures/association_outcomes.png",
    "docs/figures/coverage_dedup_tradeoff.png",
    "docs/figures/localization_oracle_evidence.png",
    "docs/figures/reprojection_class_disparity.png",
}
MAX_STAGED_FILE_BYTES = 1_000_000
SECRET_PATTERNS = (rb"AKIA[0-9A-Z]{16}", rb"ghp_[A-Za-z0-9]{20,}", rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", rb"(?i)(api[_-]?key|password)\s*[:=]\s*[^\s]{8,}")

def staged_paths() -> list[str]:
    raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], cwd=ROOT)
    return [x.decode("utf8") for x in raw.split(b"\0") if x]

def main() -> None:
    paths = staged_paths(); audited=[x for x in paths if x != "results/release_staging_audit.json"]; findings=[]; manifest=[]
    for rel in audited:
        path=ROOT/rel; data=path.read_bytes()
        blocked_suffix = rel.lower().endswith(BLOCKED_SUFFIXES) and rel not in ALLOWED_AGGREGATE_FIGURES
        bad = rel.startswith(BLOCKED_PREFIXES) or blocked_suffix
        if bad: findings.append({"path":rel,"kind":"blocked_asset_path"})
        if len(data) > MAX_STAGED_FILE_BYTES: findings.append({"path":rel,"kind":"file_exceeds_size_limit","bytes":len(data)})
        unix_home = b"/home" + b"/mert"
        windows_home = b"C:" + b"\\Users\\Mert"
        if unix_home in data or windows_home in data: findings.append({"path":rel,"kind":"absolute_local_path"})
        for pattern in SECRET_PATTERNS:
            if re.search(pattern,data): findings.append({"path":rel,"kind":"secret_like_pattern"}); break
        manifest.append({"path":rel,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    report={"status":"PASS" if not findings else "FAIL","staged_file_count":len(paths),"audited_file_count":len(audited),"excluded_self_manifest":"results/release_staging_audit.json","findings":findings,"files":manifest,"checks":{"blocked_assets":True,"max_file_bytes":MAX_STAGED_FILE_BYTES,"absolute_paths":True,"secret_patterns":True}}
    out=ROOT/"results/release_staging_audit.json";out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"staged_file_count":len(paths),"findings":findings},indent=2))
    if findings: raise SystemExit(1)
if __name__=="__main__": main()
