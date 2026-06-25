"""Contract tests for promotion_gates.yaml and gate decisions."""
from __future__ import annotations

from pathlib import Path

import yaml

GATES_YAML = Path("configs/promotion_gates.yaml")
GATES_DOC = Path("docs/gates.md")


def _load_gates() -> dict:
    return yaml.safe_load(GATES_YAML.read_text(encoding="utf-8"))


def test_defect_coverage_min_coverage_threshold_is_095() -> None:
    gates = _load_gates()
    assert "defect_coverage" in gates, "section defect_coverage manquante"
    assert gates["defect_coverage"]["min_coverage"] == 0.95


def test_quality_max_regression_covers_the_four_business_metrics() -> None:
    gates = _load_gates()
    quality = gates["feature_ae"]["quality_max_regression"]
    for metric in ("pixel_aupimo_1e-5_1e-3", "pixel_ap", "image_ap", "image_auroc"):
        assert metric in quality, f"seuil de regression manquant pour {metric}"
        # Non-regression = max drop tolere, pas un seuil absolu bloquant.
        assert 0.0 <= float(quality[metric]) < 1.0


def test_roi_model_version_is_frozen_v001() -> None:
    gates = _load_gates()
    assert "roi" in gates, "section roi manquante"
    assert gates["roi"]["model_version"] == "roi_segmenter_v001_fixed"
    assert gates["roi"]["frozen"] is True


def test_gates_doc_records_roi_frozen_and_defect_coverage_gate() -> None:
    assert GATES_DOC.exists(), "docs/gates.md manquant"
    content = GATES_DOC.read_text(encoding="utf-8")
    assert "roi_segmenter_v001_fixed" in content, "ROI figé non documenté"
    assert "defect_coverage" in content, "gate defect_coverage non documenté"
    assert "0.95" in content, "seuil 0.95 non documenté"
