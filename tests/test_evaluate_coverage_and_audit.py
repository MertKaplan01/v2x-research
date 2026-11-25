import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_coverage_and_audit.py"
spec = importlib.util.spec_from_file_location("audit", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def test_aggregate_fixture_and_metrics():
    rows = json.loads((Path(__file__).parent / "fixtures" / "association_rows.json").read_text())
    total = audit.aggregate_hungarian(rows)
    assert total["frames"] == 2
    assert total["spatial_associations"] == 5
    assert total["correct_association_pairs"] == 2
    assert total["wrong_association_pairs"] == 1
    assert total["unverifiable_association_pairs"] == 2
    metrics = audit.association_metrics(total)
    assert metrics["conditional_association_recall"] == 2 / 5
    assert metrics["verifiable_association_precision"] == 2 / 3
    assert metrics["confirmed_correct_fraction_all_predicted_pairs"] == 2 / 5
